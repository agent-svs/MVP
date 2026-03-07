"""
Slot Extractor — BERT NER + DateUtil + Custom Patterns (v3)

Extracts structured entity slots from user utterances:
- PERSON → name (via dslim/bert-base-NER)
- DOCTOR → doctor_name (Dr./Doctor prefix + NER)
- DATE → date (via python-dateutil fuzzy parsing, validated not-past)
- TIME → time (via python-dateutil + regex, validated business hours)
- PHONE → phone (regex pattern)
- SERVICE → service_type (keyword matching, expanded list)
- REASON → reason (appointment purpose extraction)

SpaCy was incompatible with Python 3.14, so we use HuggingFace
transformers NER instead. Future migration to fine-tuned BERT NER
only requires swapping the model name.
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta

from config import cfg

logger = logging.getLogger(__name__)

# Import entity resolver
from core.nlu.entities import entity_store

# ── BERT NER Model (lazy-loaded) ───────────────────────────
_ner_pipeline = None
_NER_MODEL = "dslim/bert-base-NER"


def _load_ner():
    """Load the NER pipeline (once)."""
    global _ner_pipeline
    if _ner_pipeline is not None:
        return

    from transformers import pipeline

    logger.info(f"Loading NER model: {_NER_MODEL}")
    t0 = time.time()

    _ner_pipeline = pipeline(
        "ner",
        model=_NER_MODEL,
        aggregation_strategy="simple",
        device=-1,  # CPU
    )

    elapsed = time.time() - t0
    logger.info(f"NER model loaded in {elapsed:.1f}s")


# ── Date/Time Patterns ─────────────────────────────────────
# Relative date keywords
_RELATIVE_DATES = {
    "today": 0,
    "tomorrow": 1,
    "day after tomorrow": 2,
    "next week": 7,
}

# Day name → weekday number (0=Mon, 6=Sun)
_DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}

# Time patterns
_TIME_12H_RE = re.compile(
    r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b',
    re.IGNORECASE
)
_TIME_24H_RE = re.compile(r'\b(\d{1,2}):(\d{2})\b')
_TIME_OCLOCK_RE = re.compile(r'\b(\d{1,2})\s*o\'?clock\b', re.IGNORECASE)

# Phone patterns
_PHONE_RE = re.compile(
    r'\b(?:\+?1[-.\s]?)?'
    r'(?:\(?\d{3}\)?[-.\s]?)?'
    r'\d{3}[-.\s]?\d{4}\b'
)

# Doctor name prefix pattern
_DOCTOR_PREFIX_RE = re.compile(
    r'\b(?:dr\.?|doctor)\s+',
    re.IGNORECASE
)

# Service types (receptionist domain) — expanded v3
_SERVICES = [
    # Dental
    "dental checkup", "dental cleaning", "teeth cleaning", "teeth whitening",
    "root canal", "dental filling", "dental crown",
    # Medical
    "consultation", "checkup", "check-up", "check up",
    "physical exam", "physical", "exam", "annual physical",
    "eye exam", "vision test", "eye test",
    "blood test", "blood work", "lab work",
    "x-ray", "xray", "scan", "mri", "ct scan",
    "vaccination", "vaccine", "flu shot", "immunization",
    # Wellness / Beauty
    "haircut", "trim", "coloring", "hair color", "highlights",
    "massage", "facial", "spa", "manicure", "pedicure",
    # Auto
    "oil change", "tune-up", "repair", "brake check", "tire rotation",
    # General
    "general", "follow-up", "follow up", "followup",
    "initial visit", "new patient", "routine",
]

# Reason keywords — to capture appointment purpose
_REASON_PATTERNS = re.compile(
    r'\b(?:for|about|regarding|because of|due to)\s+'
    r'(?:my|a|an|the)?\s*'
    r'(\w[\w\s]{2,30}?)'
    r'(?:\.|,|$|\b(?:on|at|tomorrow|next|this))',
    re.IGNORECASE
)

# Business hours for validation (8 AM to 8 PM)
_BUSINESS_HOUR_START = 8
_BUSINESS_HOUR_END = 20


def _extract_doctor_name(text: str) -> Optional[tuple[str, float]]:
    """Extract doctor name from text (Dr./Doctor + name)."""
    match = _DOCTOR_PREFIX_RE.search(text)
    if not match:
        return None

    after_prefix = text[match.end():].strip()
    if not after_prefix:
        return None

    _load_ner()
    try:
        entities = _ner_pipeline(after_prefix)
    except Exception:
        words = after_prefix.split()
        name_parts = []
        for w in words[:3]:
            if w[0].isupper() or len(name_parts) > 0:
                clean = re.sub(r'[^\w]', '', w)
                if clean and clean[0].isupper():
                    name_parts.append(clean)
                else:
                    break
            else:
                break
        return (" ".join(name_parts), 0.6) if name_parts else None

    person_parts = []
    avg_score = 0.0
    for ent in entities:
        if ent["entity_group"] == "PER" and float(ent["score"]) > 0.5:
            person_parts.append(ent["word"].strip())
            avg_score += float(ent["score"])

    if person_parts:
        name = " ".join(person_parts)
        name = name.replace(" ##", "").replace("##", "")
        name = re.sub(r'\s+', ' ', name).strip()
        if len(name) >= 2:
            return (name, avg_score / len(person_parts))

    words = after_prefix.split()
    name_parts = []
    for w in words[:3]:
        clean = re.sub(r'[^\w]', '', w)
        if clean and clean[0].isupper():
            name_parts.append(clean)
        else:
            break
    return (" ".join(name_parts), 0.5) if name_parts else None


def _extract_reason(text: str) -> Optional[tuple[str, float]]:
    """Extract appointment reason/purpose from text."""
    match = _REASON_PATTERNS.search(text)
    if match:
        reason = match.group(1).strip()
        if len(reason) >= 3 and reason.lower() not in (
            "it", "that", "this", "me", "you", "him", "her",
            "appointment", "booking", "visit",
        ):
            return (reason, 0.8)
    return None


def extract_slots(text: str) -> dict[str, Any]:
    """
    Extract structured entity slots from user utterance.

    Uses BERT NER for person names, dateutil + regex for dates/times,
    and pattern matching for phone numbers and service types.

    Args:
        text: User utterance

    Returns:
        Dict of extracted slots with confidence: 
        {slot_key: {"value": x, "confidence": y}}
    """
    if not text or not text.strip():
        return {}

    t0 = time.time()
    slots: dict[str, Any] = {}

    # Helper to clean up the code below
    def add_slot(key: str, result: Optional[tuple[str, float]]):
        if result:
            val, conf = result
            slot_data = {"value": val, "confidence": conf}
            
            # Attempt canonical resolution
            canonical = entity_store.resolve_entity(key, val)
            if canonical:
                slot_data["canonical_id"] = canonical["id"]
                slot_data["canonical_name"] = canonical["canonical_name"]
                
            slots[key] = slot_data

    # 1. Extract doctor name
    doc_res = _extract_doctor_name(text)
    add_slot("doctor_name", doc_res)

    # 2. Extract person name
    name_res = _extract_name_ner(text)
    if name_res:
        name_val, name_conf = name_res
        doctor_val = doc_res[0] if doc_res else None
        if not doctor_val or name_val.lower() != doctor_val.lower():
            add_slot("name", name_res)

    # 3. Extract date
    add_slot("date", _extract_date(text))

    # 4. Extract time
    add_slot("time", _extract_time(text))

    # 5. Extract phone number
    add_slot("phone", _extract_phone(text))

    # 6. Extract service type
    add_slot("service_type", _extract_service(text))

    # 7. Extract reason/purpose
    add_slot("reason", _extract_reason(text))

    elapsed = time.time() - t0
    if slots:
        logger.info(f"Slots extracted ({elapsed:.3f}s): {slots}")
    else:
        logger.debug(f"No slots found ({elapsed:.3f}s)")

    return slots
def _extract_name_ner(text: str) -> Optional[tuple[str, float]]:
    """Extract person name using BERT NER."""
    _load_ner()

    try:
        entities = _ner_pipeline(text)
    except Exception as e:
        logger.warning(f"NER failed: {e}")
        return None

    # Collect PER entities
    person_parts = []
    avg_score = 0.0
    for ent in entities:
        if ent["entity_group"] == "PER" and float(ent["score"]) > 0.6:
            person_parts.append(ent["word"].strip())
            avg_score += float(ent["score"])

    if not person_parts:
        return None

    # Join adjacent person parts (first + last name)
    name = " ".join(person_parts)

    # Clean up tokenization artifacts
    name = name.replace(" ##", "").replace("##", "")
    name = re.sub(r'\s+', ' ', name).strip()

    # Filter out common false positives
    if name.lower() in ("i", "me", "my", "here", "there", "fine", "good", "well", "okay"):
        return None

    # Require at least 2 characters
    if len(name) < 2:
        return None

    return (name, avg_score / len(person_parts))


def _extract_date(text: str) -> Optional[tuple[str, float]]:
    """Extract and normalize date from text to YYYY-MM-DD."""
    text_lower = text.lower()
    today = datetime.now().date()

    # Check relative dates first
    for keyword, days_offset in _RELATIVE_DATES.items():
        if keyword in text_lower:
            target = today + timedelta(days=days_offset)
            return (target.isoformat(), 0.95)

    # Check day names (e.g., "next Monday", "this Friday")
    for day_name, weekday in _DAY_NAMES.items():
        if day_name in text_lower:
            days_ahead = weekday - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            if f"next {day_name}" in text_lower:
                days_ahead += 7
            target = today + timedelta(days=days_ahead)
            return (target.isoformat(), 0.9)

    # Try dateutil fuzzy parsing
    try:
        parsed = dateutil_parser.parse(text, fuzzy=True, default=datetime.now())
        if parsed.date() != today or any(
            kw in text_lower for kw in ["today", "tonight", "this evening"]
        ):
            parsed_date = parsed.date()
            if parsed_date < today:
                logger.info(f"Date '{parsed_date.isoformat()}' is in the past — skipping")
                return None
            return (parsed_date.isoformat(), 0.8)
    except (ValueError, OverflowError):
        pass

    return None


def _extract_time(text: str) -> Optional[tuple[str, float]]:
    """Extract and normalize time from text to HH:MM format."""
    # Try 12-hour format first (3 PM, 3:30 pm)
    match = _TIME_12H_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        ampm = match.group(3).lower().replace('.', '')

        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0

        return (f"{hour:02d}:{minute:02d}", 0.95)

    # Try 24-hour format (15:00)
    match = _TIME_24H_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour < 24 and 0 <= minute < 60:
            return (f"{hour:02d}:{minute:02d}", 0.9)

    # Try "X o'clock"
    match = _TIME_OCLOCK_RE.search(text)
    if match:
        hour = int(match.group(1))
        if 1 <= hour <= 7:
            hour += 12
        return (f"{hour:02d}:00", 0.85)

    # Try natural language time
    text_lower = text.lower()
    time_keywords = {
        "morning": "09:00", "noon": "12:00", "afternoon": "14:00",
        "evening": "17:00", "night": "19:00",
    }
    for keyword, time_val in time_keywords.items():
        if keyword in text_lower:
            return (time_val, 0.8)

    return None


def _extract_phone(text: str) -> Optional[tuple[str, float]]:
    """Extract phone number from text."""
    match = _PHONE_RE.search(text)
    if match:
        digits = re.sub(r'\D', '', match.group())
        if len(digits) >= 10:
            return (digits[-10:], 0.95)
    return None


def _extract_service(text: str) -> Optional[tuple[str, float]]:
    """Extract service type from text."""
    text_lower = text.lower()
    for service in sorted(_SERVICES, key=len, reverse=True):
        if service in text_lower:
            return (service, 0.85)
    return None


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from config import cfg
    cfg.setup_logging()

    print("🏷️  BERT NER Slot Extractor — Interactive Test")
    print(f"   Model: {_NER_MODEL}")
    print("   Loading model...")
    _load_ner()
    print("   Ready!\n")

    test_phrases = [
        "I want to book an appointment for tomorrow at 3 PM",
        "My name is Vishesh Gupta",
        "Can I come in next Monday at 10 AM?",
        "I am pretty good",
        "Schedule a dental checkup for Friday",
        "Call me at 555-123-4567",
        "I need to see Dr. Smith on March 15th at 2:30 PM",
        "Cancel my appointment",
        "I'm available in the afternoon",
    ]

    print("── Batch test ──────────────────────────────\n")
    for phrase in test_phrases:
        slots = extract_slots(phrase)
        print(f"  \"{phrase}\"")
        if slots:
            for k, v in slots.items():
                print(f"    {k}: {v}")
        else:
            print("    (no slots)")
        print()

    print("── Interactive mode ────────────────────────\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        slots = extract_slots(text)
        if slots:
            for k, v in slots.items():
                print(f"  {k}: {v}")
        else:
            print("  (no slots)")
        print()
