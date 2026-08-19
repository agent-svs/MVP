"""
Slot Extractor — BERT NER + DateUtil + Custom Patterns

Extracts structured entity slots from user utterances:
  patient_name / doctor_name  → dslim/bert-base-NER (+ Dr. prefix handling)
  date                        → relative keywords, weekday names, dateutil
  time                        → 12h / 24h / o'clock / natural language
  phone                       → digit patterns (incl. spoken digit words)
  service_type                → word-boundary catalogue match
  reason                      → purpose phrase extraction

Used directly by the legacy (separate-model) NLU path, and reused by the
joint NLU path for date/time normalisation.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from dateutil import parser as dateutil_parser

from config import cfg
from core.nlu.entities import entity_store

logger = logging.getLogger(__name__)

# ── BERT NER Model (lazy-loaded, thread-safe) ───────────────
_ner_pipeline = None
_ner_lock = threading.Lock()
_ner_failed = False


def _load_ner() -> bool:
    """Load the NER pipeline once. Returns True if usable.

    Never raises: a missing model or absent network degrades the slot
    extractor to its regex paths rather than killing the conversation turn.
    """
    global _ner_pipeline, _ner_failed
    if _ner_pipeline is not None:
        return True
    if _ner_failed:
        return False

    with _ner_lock:
        if _ner_pipeline is not None:
            return True
        if _ner_failed:
            return False
        try:
            from transformers import pipeline

            logger.info(f"Loading NER model: {cfg.nlu.ner_model}")
            t0 = time.time()
            _ner_pipeline = pipeline(
                "ner",
                model=cfg.nlu.ner_model,
                aggregation_strategy="simple",
                device=cfg.hf_pipeline_device,
            )
            logger.info(f"NER model loaded in {time.time() - t0:.1f}s")
            return True
        except Exception as e:
            _ner_failed = True
            logger.error(f"NER model unavailable ({e}) — falling back to regex-only slots")
            return False


def _run_ner(text: str) -> list[dict]:
    """Run NER, returning [] on any failure."""
    if not _load_ner():
        return []
    try:
        with _ner_lock:
            return list(_ner_pipeline(text))
    except Exception as e:
        logger.warning(f"NER inference failed: {e}")
        return []


# ── Date patterns ───────────────────────────────────────────
# NOTE: ordered longest-first at match time. "day after tomorrow" contains
# "tomorrow" as a substring, so a naive dict scan resolved it to +1 day.
_RELATIVE_DATES = {
    "day after tomorrow": 2,
    "the day after tomorrow": 2,
    "next week": 7,
    "tomorrow": 1,
    "today": 0,
    "tonight": 0,
}

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_MONTH_NAME_RE = re.compile(rf'\b(?:{_MONTHS})\b', re.IGNORECASE)

_NUMERIC_DATE_RE = re.compile(r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b')

# Only these shapes are allowed to reach dateutil's fuzzy parser.
_DATE_TOKEN_RE = re.compile(
    r'\b(?:'
    rf'\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})'          # 15 March
    rf'|(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?'                    # March 15
    r'|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?'                          # 3/14, 3-14-26
    r'|\d{4}-\d{2}-\d{2}'                                           # 2026-08-19
    r')\b',
    re.IGNORECASE,
)

_DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}

# ── Time patterns ───────────────────────────────────────────
_TIME_12H_RE = re.compile(
    r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b', re.IGNORECASE
)
_TIME_24H_RE = re.compile(r'\b([01]?\d|2[0-3]):([0-5]\d)\b')
_TIME_OCLOCK_RE = re.compile(r'\b(\d{1,2})\s*o\'?\s?clock\b', re.IGNORECASE)
_MORNING_HINT_RE = re.compile(r'\b(morning|a\.?m\.?)\b', re.IGNORECASE)
_EVENING_HINT_RE = re.compile(r'\b(afternoon|evening|night|p\.?m\.?)\b', re.IGNORECASE)

# ── Phone patterns ──────────────────────────────────────────
# Requires a full 10-digit (optionally +1 prefixed) number. The previous
# pattern made the area code optional, so it matched any bare 7 digits.
_PHONE_RE = re.compile(
    r'(?:\+?1[-.\s]?)?'
    r'\(?\d{3}\)?[-.\s]?'
    r'\d{3}[-.\s]?'
    r'\d{4}(?!\d)'
)
_SPOKEN_DIGITS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

_DOCTOR_PREFIX_RE = re.compile(r'\b(?:dr\.?|doctor)\s+', re.IGNORECASE)

# ── Service catalogue ───────────────────────────────────────
# Matched on WORD BOUNDARIES. The old substring scan turned "examine" into
# "exam", "repair the trim" into "repair", and "check up on my booking" into
# a booked check-up.
_SERVICES = [
    # Dental
    "dental checkup", "dental cleaning", "teeth cleaning", "teeth whitening",
    "root canal", "dental filling", "dental crown", "dental",
    # Medical
    "general checkup", "annual physical", "physical exam", "consultation",
    "checkup", "check-up", "check up", "follow-up", "follow up", "followup",
    "eye exam", "vision test", "eye test", "blood test", "blood work", "lab work",
    "x-ray", "xray", "ct scan", "mri", "ultrasound",
    "vaccination", "vaccine", "flu shot", "immunization",
    "initial visit", "new patient",
    # Specialties (these are what the joint model was trained on)
    "cardiology", "dermatology", "pediatrics", "paediatrics", "orthopedics",
    "orthopaedics", "neurology", "ophthalmology", "psychiatry",
    "gastroenterology", "gynecology", "gynaecology", "ent",
    # Practitioner nouns — callers routinely name the doctor, not the
    # specialty ("change it to cardiologist"). The entity store maps these
    # onto the canonical service.
    "cardiologist", "dermatologist", "pediatrician", "paediatrician",
    "orthopedist", "orthopaedist", "neurologist", "ophthalmologist",
    "optometrist", "psychiatrist", "psychologist", "gastroenterologist",
    "gynecologist", "gynaecologist", "dentist", "dentistry",
]
# Precompiled longest-first so "dental checkup" wins over "dental".
_SERVICE_RES = [
    (s, re.compile(r'(?<!\w)' + re.escape(s).replace(r'\ ', r'[\s-]') + r'(?!\w)', re.IGNORECASE))
    for s in sorted(_SERVICES, key=len, reverse=True)
]

# A catalogue hit immediately followed by these words is phrasal-verb usage
# ("check up ON my booking"), not a service name.
_SERVICE_VERB_SUFFIX_RE = re.compile(r'\s+(?:on|with|about)\b', re.IGNORECASE)

_REASON_PATTERNS = re.compile(
    r'\b(?:for|about|regarding|because of|due to)\s+'
    r'(?:my|a|an|the)?\s*'
    r'(\w[\w\s]{2,30}?)'
    r'(?:\.|,|$|\b(?:on|at|tomorrow|next|this))',
    re.IGNORECASE,
)

_NAME_FALSE_POSITIVES = {
    "i", "me", "my", "here", "there", "fine", "good", "well", "okay", "ok",
    "yes", "no", "thanks", "thank you", "hello", "hi", "hey", "sure",
}


# ── Extractors ──────────────────────────────────────────────
def _extract_doctor_name(text: str) -> Optional[tuple[str, float]]:
    """Extract a doctor name (Dr./Doctor + name)."""
    match = _DOCTOR_PREFIX_RE.search(text)
    if not match:
        return None

    after_prefix = text[match.end():].strip()
    if not after_prefix:
        return None

    entities = _run_ner(after_prefix)
    person_parts, total = [], 0.0
    for ent in entities:
        if ent.get("entity_group") == "PER" and float(ent.get("score", 0)) > 0.5:
            person_parts.append(str(ent.get("word", "")).strip())
            total += float(ent["score"])

    if person_parts:
        name = _clean_name(" ".join(person_parts))
        if len(name) >= 2:
            return (name, total / len(person_parts))

    # Regex fallback: leading capitalised words after the prefix.
    name_parts = []
    for w in after_prefix.split()[:3]:
        clean = re.sub(r'[^\w]', '', w)
        if clean and clean[0].isupper():
            name_parts.append(clean)
        else:
            break
    return (" ".join(name_parts), 0.5) if name_parts else None


def _clean_name(name: str) -> str:
    name = name.replace(" ##", "").replace("##", "")
    return re.sub(r'\s+', ' ', name).strip()


def _extract_name_ner(text: str) -> Optional[tuple[str, float]]:
    """Extract a person name using BERT NER."""
    entities = _run_ner(text)
    if not entities:
        return None

    person_parts, total = [], 0.0
    for ent in entities:
        if ent.get("entity_group") == "PER" and float(ent.get("score", 0)) > 0.6:
            person_parts.append(str(ent.get("word", "")).strip())
            total += float(ent["score"])

    if not person_parts:
        return None

    name = _clean_name(" ".join(person_parts))
    if name.lower() in _NAME_FALSE_POSITIVES or len(name) < 2:
        return None
    return (name, total / len(person_parts))


def _resolve_weekday(text_lower: str, today) -> Optional[tuple[str, float]]:
    """Resolve 'Monday' / 'next Monday' / 'this Friday' to a concrete date.

    'Monday' means the next upcoming Monday (never today's date in the past).
    'next Monday' means that same day — NOT a week further. The old code added
    7 on top of the roll-forward, landing a full week late.
    """
    for day_name, weekday in _DAY_NAMES.items():
        if not re.search(rf'\b{day_name}\b', text_lower):
            continue

        days_ahead = (weekday - today.weekday()) % 7
        if days_ahead == 0:
            # Same weekday as today → assume next week's occurrence.
            days_ahead = 7

        # "next Monday" only shifts an extra week when today IS that weekday
        # would otherwise be ambiguous; the common English reading of
        # "next Monday" is simply the upcoming Monday.
        if re.search(rf'\bweek\s+(?:after|from)\b.*\b{day_name}\b', text_lower):
            days_ahead += 7

        return ((today + timedelta(days=days_ahead)).isoformat(), 0.9)
    return None


def _extract_date(text: str) -> Optional[tuple[str, float]]:
    """Extract and normalise a date to YYYY-MM-DD. Never returns a past date."""
    if not text:
        return None
    text_lower = text.lower()
    today = datetime.now().date()

    # Relative keywords, longest phrase first ("day after tomorrow" before "tomorrow").
    for keyword in sorted(_RELATIVE_DATES, key=len, reverse=True):
        if re.search(rf'(?<!\w){re.escape(keyword)}(?!\w)', text_lower):
            return ((today + timedelta(days=_RELATIVE_DATES[keyword])).isoformat(), 0.95)

    weekday_hit = _resolve_weekday(text_lower, today)
    if weekday_hit:
        return weekday_hit

    # dateutil fuzzy parsing — only when the text contains a real calendar
    # token. Unconstrained fuzzy parsing turned "I want 2 appointments" into a
    # date, so the trigger is restricted to month names and explicit numeric
    # date formats.
    if not _DATE_TOKEN_RE.search(text_lower):
        return None

    try:
        parsed = dateutil_parser.parse(text, fuzzy=True, default=datetime.now())
    except (ValueError, OverflowError, TypeError):
        return None

    parsed_date = parsed.date()
    if parsed_date >= today:
        return (parsed_date.isoformat(), 0.8)

    # The date landed in the past. dateutil defaults missing components to
    # "now", so "March 15" in August means next March, and "the 5th" late in
    # the month means next month. Roll forward on the coarsest missing unit.
    from dateutil.relativedelta import relativedelta

    # An explicit month (name or numeric M/D) means the user meant next year;
    # a bare day-of-month ("the 5th") means next month.
    has_explicit_month = bool(
        _MONTH_NAME_RE.search(text_lower) or _NUMERIC_DATE_RE.search(text_lower)
    )
    try:
        rolled = parsed_date + (relativedelta(years=1) if has_explicit_month
                                else relativedelta(months=1))
    except Exception:
        return None

    if rolled < today:
        logger.debug(f"Date {parsed_date.isoformat()} is in the past — ignoring")
        return None

    # Range validation deliberately lives in guardrails, not here: an
    # out-of-range date should produce a spoken explanation ("that's further
    # ahead than we book"), not a silently dropped slot.
    return (rolled.isoformat(), 0.65)


def _extract_time(text: str) -> Optional[tuple[str, float]]:
    """Extract and normalise a time to HH:MM (24h)."""
    if not text:
        return None

    m = _TIME_12H_RE.search(text)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        ampm = m.group(3).lower().replace('.', '')
        if not 1 <= hour <= 12 or minute > 59:
            return None
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return (f"{hour:02d}:{minute:02d}", 0.95)

    m = _TIME_24H_RE.search(text)
    if m:
        return (f"{int(m.group(1)):02d}:{int(m.group(2)):02d}", 0.9)

    # "X o'clock" — resolve AM/PM from surrounding words, then from business
    # hours, instead of the old arbitrary "1-7 means PM" rule which made
    # "7 o'clock" 19:00 (always outside opening hours) but "8 o'clock" 08:00.
    m = _TIME_OCLOCK_RE.search(text)
    if m:
        hour = int(m.group(1))
        if not 1 <= hour <= 12:
            return None
        if _EVENING_HINT_RE.search(text) and hour != 12:
            hour += 12
        elif _MORNING_HINT_RE.search(text):
            pass
        elif hour < cfg.business.hours_start and hour + 12 < cfg.business.hours_end:
            # e.g. "3 o'clock" with hours 8-18 → 15:00
            hour += 12
        return (f"{hour:02d}:00", 0.85)

    # Natural-language windows, clamped into opening hours so they can never
    # produce a time the guardrails must immediately reject.
    text_lower = text.lower()
    open_s, open_e = cfg.business.hours_start, cfg.business.hours_end
    windows = {
        "first thing": open_s,
        "early morning": open_s,
        "morning": max(9, open_s),
        "noon": 12,
        "midday": 12,
        "lunchtime": 12,
        "early afternoon": 13,
        "afternoon": 14,
        "late afternoon": 16,
        "evening": min(17, open_e - 1),
        "tonight": min(17, open_e - 1),
        "night": min(17, open_e - 1),
        "end of day": open_e - 1,
    }
    for keyword in sorted(windows, key=len, reverse=True):
        if re.search(rf'(?<!\w){re.escape(keyword)}(?!\w)', text_lower):
            hour = min(max(windows[keyword], open_s), open_e - 1)
            return (f"{hour:02d}:00", 0.75)

    return None


def _digits_from_spoken(text: str) -> str:
    """Convert spoken digit words to digits ('five five five' → '555')."""
    out = []
    for tok in re.findall(r"[a-z]+|\d", text.lower()):
        if tok.isdigit():
            out.append(tok)
        elif tok in _SPOKEN_DIGITS:
            out.append(_SPOKEN_DIGITS[tok])
    return "".join(out)


def _extract_phone(text: str) -> Optional[tuple[str, float]]:
    """Extract a 10-digit phone number, including spoken-digit form."""
    m = _PHONE_RE.search(text)
    if m:
        digits = re.sub(r'\D', '', m.group())
        if len(digits) >= 10:
            return (digits[-10:], 0.95)

    spoken = _digits_from_spoken(text)
    if len(spoken) >= 10:
        return (spoken[-10:], 0.85)
    return None


def _extract_service(text: str) -> Optional[tuple[str, float]]:
    """Extract a service type using word-boundary matching."""
    for service, pattern in _SERVICE_RES:
        m = pattern.search(text)
        if not m:
            continue
        # "check up on my booking" / "follow up on that" are status queries,
        # not service requests.
        if _SERVICE_VERB_SUFFIX_RE.match(text[m.end():]):
            continue
        return (service, 0.85)
    return None


def _extract_reason(text: str) -> Optional[tuple[str, float]]:
    """Extract the appointment reason/purpose."""
    m = _REASON_PATTERNS.search(text)
    if not m:
        return None
    reason = m.group(1).strip()
    if len(reason) >= 3 and reason.lower() not in (
        "it", "that", "this", "me", "you", "him", "her",
        "appointment", "booking", "visit",
    ):
        return (reason, 0.8)
    return None


def extract_slots(text: str) -> dict[str, Any]:
    """Extract structured slots from an utterance.

    Returns {slot_key: {"value": x, "confidence": y, ...}}.
    """
    if not text or not text.strip():
        return {}

    t0 = time.time()
    slots: dict[str, Any] = {}

    def add_slot(key: str, result: Optional[tuple[str, float]]) -> None:
        if not result:
            return
        val, conf = result
        slot_data: dict[str, Any] = {"value": val, "confidence": conf}
        canonical = entity_store.resolve_entity(key, val)
        if canonical:
            slot_data["canonical_id"] = canonical["id"]
            slot_data["canonical_name"] = canonical["canonical_name"]
            slot_data["value"] = canonical["canonical_name"]
        slots[key] = slot_data

    doc_res = _extract_doctor_name(text)
    add_slot("doctor_name", doc_res)

    name_res = _extract_name_ner(text)
    if name_res:
        doctor_val = doc_res[0] if doc_res else None
        if not doctor_val or name_res[0].lower() != doctor_val.lower():
            add_slot("patient_name", name_res)

    add_slot("date", _extract_date(text))
    add_slot("time", _extract_time(text))
    add_slot("phone", _extract_phone(text))
    add_slot("service_type", _extract_service(text))
    add_slot("reason", _extract_reason(text))

    elapsed = time.time() - t0
    if slots:
        logger.info(f"Slots extracted ({elapsed:.3f}s): {list(slots)}")
    else:
        logger.debug(f"No slots found ({elapsed:.3f}s)")
    return slots
