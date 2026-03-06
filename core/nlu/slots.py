"""
Slot Extractor — BERT NER + DateUtil + Custom Patterns (v2)

Extracts structured entity slots from user utterances:
- PERSON → name (via dslim/bert-base-NER)
- DATE → date (via python-dateutil fuzzy parsing)
- TIME → time (via python-dateutil + regex patterns)
- PHONE → phone (regex pattern)
- SERVICE → service_type (keyword matching)

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

logger = logging.getLogger(__name__)

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

# Service types (receptionist domain)
_SERVICES = [
    "dental checkup", "dental cleaning", "teeth cleaning",
    "consultation", "checkup", "check-up", "check up",
    "physical exam", "physical", "exam",
    "eye exam", "vision test",
    "haircut", "trim", "coloring",
    "massage", "facial", "spa",
    "oil change", "tune-up", "repair",
    "general", "follow-up", "follow up",
]


def extract_slots(text: str) -> dict[str, Any]:
    """
    Extract structured entity slots from user utterance.

    Uses BERT NER for person names, dateutil + regex for dates/times,
    and pattern matching for phone numbers and service types.

    Args:
        text: User utterance

    Returns:
        Dict of extracted slots: {name, date, time, phone, service_type}
    """
    if not text or not text.strip():
        return {}

    t0 = time.time()
    slots: dict[str, Any] = {}

    # 1. Extract person name via BERT NER
    name = _extract_name_ner(text)
    if name:
        slots["name"] = name

    # 2. Extract date
    date = _extract_date(text)
    if date:
        slots["date"] = date

    # 3. Extract time
    extracted_time = _extract_time(text)
    if extracted_time:
        slots["time"] = extracted_time

    # 4. Extract phone number
    phone = _extract_phone(text)
    if phone:
        slots["phone"] = phone

    # 5. Extract service type
    service = _extract_service(text)
    if service:
        slots["service_type"] = service

    elapsed = time.time() - t0
    if slots:
        logger.info(f"Slots extracted ({elapsed:.3f}s): {slots}")
    else:
        logger.debug(f"No slots found ({elapsed:.3f}s)")

    return slots


def _extract_name_ner(text: str) -> Optional[str]:
    """Extract person name using BERT NER."""
    _load_ner()

    try:
        entities = _ner_pipeline(text)
    except Exception as e:
        logger.warning(f"NER failed: {e}")
        return None

    # Collect PER entities
    person_parts = []
    for ent in entities:
        if ent["entity_group"] == "PER" and ent["score"] > 0.6:
            person_parts.append(ent["word"].strip())

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

    return name


def _extract_date(text: str) -> Optional[str]:
    """Extract and normalize date from text to YYYY-MM-DD."""
    text_lower = text.lower()
    today = datetime.now().date()

    # Check relative dates first
    for keyword, days_offset in _RELATIVE_DATES.items():
        if keyword in text_lower:
            target = today + timedelta(days=days_offset)
            return target.isoformat()

    # Check day names (e.g., "next Monday", "this Friday")
    for day_name, weekday in _DAY_NAMES.items():
        if day_name in text_lower:
            # Calculate next occurrence of this weekday
            days_ahead = weekday - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7  # Next week
            # Check if "next" is specified — add another week
            if f"next {day_name}" in text_lower:
                days_ahead += 7
            target = today + timedelta(days=days_ahead)
            return target.isoformat()

    # Try dateutil fuzzy parsing
    try:
        parsed = dateutil_parser.parse(text, fuzzy=True, default=datetime.now())
        # Only return if the parsed date is different from "now" (meaning it found something)
        if parsed.date() != today or any(
            kw in text_lower for kw in ["today", "tonight", "this evening"]
        ):
            return parsed.date().isoformat()
    except (ValueError, OverflowError):
        pass

    return None


def _extract_time(text: str) -> Optional[str]:
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

        return f"{hour:02d}:{minute:02d}"

    # Try 24-hour format (15:00)
    match = _TIME_24H_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour < 24 and 0 <= minute < 60:
            return f"{hour:02d}:{minute:02d}"

    # Try "X o'clock"
    match = _TIME_OCLOCK_RE.search(text)
    if match:
        hour = int(match.group(1))
        # Assume PM for business hours (9-12 could be AM)
        if 1 <= hour <= 7:
            hour += 12
        return f"{hour:02d}:00"

    # Try natural language time
    text_lower = text.lower()
    time_keywords = {
        "morning": "09:00", "noon": "12:00", "afternoon": "14:00",
        "evening": "17:00", "night": "19:00",
    }
    for keyword, time_val in time_keywords.items():
        if keyword in text_lower:
            return time_val

    return None


def _extract_phone(text: str) -> Optional[str]:
    """Extract phone number from text."""
    match = _PHONE_RE.search(text)
    if match:
        # Normalize: keep only digits
        digits = re.sub(r'\D', '', match.group())
        if len(digits) >= 10:
            return digits[-10:]  # Last 10 digits
    return None


def _extract_service(text: str) -> Optional[str]:
    """Extract service type from text."""
    text_lower = text.lower()
    for service in _SERVICES:
        if service in text_lower:
            return service
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
