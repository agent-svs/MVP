"""
Canonical Entity Store — resolves surface text to backend IDs.

Prototype implementation with an in-memory roster/catalogue. In production
this would be backed by the clinic's practice-management system; the resolver
interface is deliberately the only thing callers depend on.

Matching policy (deliberately conservative):
  1. exact match on the canonical name or an alias
  2. token-level match (surface tokens ⊆ candidate tokens, or vice versa)
  3. typo tolerance via SequenceMatcher, but ONLY above a high ratio AND with
     a matching first character and similar length.

The previous implementation used ratio >= 0.70 with no other constraint,
which resolved "Bones" and "Jonas" to Dr. Jones and "Smyth" to Dr. Smith.
For a medical booking system a wrong-but-confident match is worse than no
match, so an ambiguous surface form now returns None and the agent asks.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TITLE_RE = re.compile(r'^\s*(dr\.?|doctor|mr\.?|mrs\.?|ms\.?)\s+', re.IGNORECASE)
_NON_WORD_RE = re.compile(r'[^\w\s]')

# Typo tolerance. 0.86 accepts "Kulkarni"/"Kulkarni." and one-character slips,
# but rejects "Bones"→"Jones" (0.80) and "Smyth"→"Smith" (0.80).
_TYPO_RATIO = 0.86
_MIN_TYPO_LEN = 5
# Shared-prefix length that counts as the same domain term.
_MIN_STEM_LEN = 6


def _normalize(text: str) -> str:
    text = _TITLE_RE.sub('', text or '')
    text = _NON_WORD_RE.sub(' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


class EntityStore:
    """Canonical entity store for resolving surface text to backend IDs."""

    def __init__(self) -> None:
        self.doctors: List[Dict[str, Any]] = [
            {"id": "DOC_001", "name": "Dr. Kulkarni", "specialty": "General Practice",
             "aliases": ["kulkarni"]},
            {"id": "DOC_002", "name": "Dr. Smith", "specialty": "Pediatrics",
             "aliases": ["smith"]},
            {"id": "DOC_003", "name": "Dr. Jones", "specialty": "Cardiology",
             "aliases": ["jones"]},
        ]

        self.services: List[Dict[str, Any]] = [
            {"id": "SRV_001", "name": "General Checkup",
             "aliases": ["checkup", "check up", "check-up", "general checkup",
                         "physical", "physical exam", "annual physical", "routine"]},
            {"id": "SRV_002", "name": "Consultation",
             "aliases": ["consult", "consultation", "advice", "initial visit", "new patient"]},
            {"id": "SRV_003", "name": "Follow-up",
             "aliases": ["follow up", "followup", "follow-up", "review"]},
            {"id": "SRV_004", "name": "Vaccination",
             "aliases": ["vaccine", "vaccination", "shot", "flu shot", "immunization"]},
            {"id": "SRV_005", "name": "Cardiology",
             "aliases": ["cardiology", "cardiac", "heart", "cardiologist"]},
            {"id": "SRV_006", "name": "Dermatology",
             "aliases": ["dermatology", "skin", "dermatologist"]},
            {"id": "SRV_007", "name": "Pediatrics",
             "aliases": ["pediatrics", "paediatrics", "child", "children",
                         "pediatrician", "paediatrician"]},
            {"id": "SRV_008", "name": "Orthopedics",
             "aliases": ["orthopedics", "orthopaedics", "ortho", "bone",
                         "orthopedist", "orthopaedist"]},
            {"id": "SRV_009", "name": "Neurology",
             "aliases": ["neurology", "neuro", "neurologist"]},
            {"id": "SRV_010", "name": "ENT",
             "aliases": ["ent", "ear nose throat", "ear nose and throat",
                         "otolaryngology", "otolaryngologist"]},
            {"id": "SRV_011", "name": "Dental",
             "aliases": ["dental", "dentist", "teeth", "tooth", "dental checkup",
                         "dental cleaning", "teeth cleaning", "root canal", "dentistry"]},
            {"id": "SRV_012", "name": "Ophthalmology",
             "aliases": ["ophthalmology", "eye", "eye exam", "vision test", "eye test",
                         "ophthalmologist", "optometrist", "optometry"]},
            {"id": "SRV_013", "name": "Psychiatry",
             "aliases": ["psychiatry", "mental health", "therapy",
                         "psychiatrist", "psychologist", "psychology"]},
            {"id": "SRV_014", "name": "Gastroenterology",
             "aliases": ["gastroenterology", "gastro", "stomach", "gastroenterologist"]},
            {"id": "SRV_015", "name": "Gynecology",
             "aliases": ["gynecology", "gynaecology", "gyno",
                         "gynecologist", "gynaecologist"]},
            {"id": "SRV_016", "name": "Blood Test",
             "aliases": ["blood test", "blood work", "lab work", "labs"]},
            {"id": "SRV_017", "name": "Imaging",
             "aliases": ["x-ray", "xray", "mri", "ct scan", "ultrasound", "scan"]},
        ]

    # ── matching ────────────────────────────────────────────
    @staticmethod
    def _match_score(surface: str, candidate: str) -> float:
        """Return a 0..1 confidence that `surface` refers to `candidate`."""
        s, c = _normalize(surface), _normalize(candidate)
        if not s or not c:
            return 0.0
        if s == c:
            return 1.0

        s_tokens, c_tokens = set(s.split()), set(c.split())
        # Full token containment, e.g. "kulkarni" ⊆ "dr kulkarni".
        if s_tokens and (s_tokens <= c_tokens or c_tokens <= s_tokens):
            return 0.95
        # Substantial token overlap for multiword names.
        if len(s_tokens) > 1 and len(c_tokens) > 1:
            overlap = len(s_tokens & c_tokens) / max(len(s_tokens), len(c_tokens))
            if overlap >= 0.75:
                return 0.9

        # Morphological match: "cardiologist" vs "cardiology" share a long
        # stem but only score 0.82 on raw similarity, so typo tolerance alone
        # rejected every practitioner noun. A long shared prefix is strong
        # evidence for domain terms while staying safe for short surnames
        # ("Bones"/"Jones" and "Smyth"/"Smith" share almost no prefix).
        if len(s) >= _MIN_STEM_LEN and len(c) >= _MIN_STEM_LEN:
            shared = 0
            for a, b in zip(s, c):
                if a != b:
                    break
                shared += 1
            if shared >= _MIN_STEM_LEN:
                return 0.9

        # Typo tolerance — tightly gated so near-miss surnames don't collide.
        if (
            len(s) >= _MIN_TYPO_LEN
            and len(c) >= _MIN_TYPO_LEN
            and s[0] == c[0]
            and abs(len(s) - len(c)) <= 2
        ):
            ratio = SequenceMatcher(None, s, c).ratio()
            if ratio >= _TYPO_RATIO:
                return ratio
        return 0.0

    def _best(self, surface_text: str, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not surface_text or not surface_text.strip():
            return None

        scored = []
        for rec in records:
            candidates = [rec["name"]] + list(rec.get("aliases", []))
            best = max((self._match_score(surface_text, c) for c in candidates), default=0.0)
            if best > 0.0:
                scored.append((best, rec))

        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_rec = scored[0]

        # Ambiguous between two records → refuse rather than guess.
        if len(scored) > 1 and abs(scored[1][0] - top_score) < 0.02:
            logger.debug(
                f"Ambiguous entity '{surface_text}': "
                f"{top_rec['name']} vs {scored[1][1]['name']} — refusing to guess"
            )
            return None

        return {
            "id": top_rec["id"],
            "canonical_name": top_rec["name"],
            "match_confidence": round(top_score, 3),
        }

    # ── public API ──────────────────────────────────────────
    def resolve_doctor(self, surface_text: str) -> Optional[Dict[str, Any]]:
        """Resolve a raw doctor name to a canonical doctor entity."""
        return self._best(surface_text, self.doctors)

    def resolve_service(self, surface_text: str) -> Optional[Dict[str, Any]]:
        """Resolve a raw service request to a canonical service entity."""
        return self._best(surface_text, self.services)

    def resolve_entity(self, slot_type: str, surface_text: str) -> Optional[Dict[str, Any]]:
        """Resolve any supported entity type. Returns None for free-form slots."""
        if slot_type == "doctor_name":
            return self.resolve_doctor(surface_text)
        if slot_type == "service_type":
            return self.resolve_service(surface_text)
        return None

    # ── introspection (used by prompts and clarification text) ──
    def doctor_names(self) -> List[str]:
        return [d["name"] for d in self.doctors]

    def service_names(self) -> List[str]:
        return [s["name"] for s in self.services]

    def doctor_roster_text(self) -> str:
        return ", ".join(f"{d['name']} ({d['specialty']})" for d in self.doctors)


entity_store = EntityStore()
