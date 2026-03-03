"""
Slot Extractor — Regex + dateutil

Extracts structured slots from user utterances:
- date, time, name, phone, service_type, duration

Uses regex patterns and python-dateutil for date/time parsing.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Optional

from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

# ── Slot Definitions ────────────────────────────────────────

# Day-of-week mapping
_DAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3,
    "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
}

# Relative date patterns
_RELATIVE_DATES = {
    "today": 0, "tomorrow": 1, "day after tomorrow": 2,
    "next week": 7, "in a week": 7,
}

# Phone regex (various formats)
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
)

# Time patterns
_TIME_12H_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b", re.IGNORECASE
)
_TIME_24H_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_TIME_OCLOCK_RE = re.compile(r"\b(\d{1,2})\s*o'?clock\b", re.IGNORECASE)

# Name patterns — trigger is case-insensitive, but name must be Capitalized
_NAME_TRIGGERS = re.compile(
    r"(?:my\s+name\s+is|i'?m|i\s+am|this\s+is|name\s+is|call\s+me)\s+",
    re.IGNORECASE
)
_CAPITALIZED_NAME = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
)

# Service types
_SERVICES = [
    "dental", "checkup", "check-up", "general", "cleaning",
    "eye exam", "consultation", "physical", "screening",
    "vaccination", "follow-up", "follow up", "therapy",
]


def extract_slots(text: str) -> dict[str, Any]:
    """
    Extract all recognizable slots from user text.

    Returns dict with found slots. Keys: date, time, name, phone, service_type
    """
    if not text:
        return {}

    slots: dict[str, Any] = {}
    text_lower = text.lower().strip()

    # ── Date extraction ──
    date = _extract_date(text_lower)
    if date:
        slots["date"] = date

    # ── Time extraction ──
    time_val = _extract_time(text_lower)
    if time_val:
        slots["time"] = time_val

    # ── Name extraction ──
    name = _extract_name(text)
    if name:
        slots["name"] = name

    # ── Phone extraction ──
    phone = _extract_phone(text)
    if phone:
        slots["phone"] = phone

    # ── Service type ──
    service = _extract_service(text_lower)
    if service:
        slots["service_type"] = service

    if slots:
        logger.debug(f"Extracted slots: {slots}")
    return slots


def _extract_date(text: str) -> Optional[str]:
    """Extract a date from text, return as YYYY-MM-DD string."""
    today = datetime.now().date()

    # Check relative dates first
    for phrase, offset in _RELATIVE_DATES.items():
        if phrase in text:
            result = today + timedelta(days=offset)
            return result.isoformat()

    # Check day-of-week names ("next Monday", "on Friday")
    for day_name, day_num in _DAYS.items():
        pattern = rf"\b(?:next\s+)?{day_name}\b"
        if re.search(pattern, text):
            # Find next occurrence of this day
            days_ahead = day_num - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            result = today + timedelta(days=days_ahead)
            return result.isoformat()

    # Try dateutil as fallback
    try:
        parsed = dateutil_parser.parse(text, fuzzy=True, default=datetime.now())
        if parsed.date() >= today:
            return parsed.date().isoformat()
    except (ValueError, OverflowError):
        pass

    return None


def _extract_time(text: str) -> Optional[str]:
    """Extract time from text, return as HH:MM string (24h format)."""
    # 12-hour format: "3 pm", "10:30 am"
    match = _TIME_12H_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        period = match.group(3).lower().replace(".", "")
        if period == "pm" and hour != 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    # O'clock: "3 o'clock"
    match = _TIME_OCLOCK_RE.search(text)
    if match:
        hour = int(match.group(1))
        # Assume PM for business hours (after noon context)
        if hour < 8:
            hour += 12
        return f"{hour:02d}:00"

    # 24-hour format: "15:00"
    match = _TIME_24H_RE.search(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    return None


def _extract_name(text: str) -> Optional[str]:
    """Extract a person's name from text (must be Capitalized)."""
    trigger = _NAME_TRIGGERS.search(text)
    if trigger:
        remainder = text[trigger.end():]
        name_match = _CAPITALIZED_NAME.match(remainder)
        if name_match:
            name = name_match.group(1).strip()
            # Filter out common false positives
            if name.lower() not in ("here", "there", "fine", "good", "well", "okay"):
                return name
    return None


def _extract_phone(text: str) -> Optional[str]:
    """Extract a phone number from text."""
    match = _PHONE_RE.search(text)
    if match:
        # Clean to digits only
        digits = re.sub(r"\D", "", match.group())
        if 10 <= len(digits) <= 13:
            return digits
    return None


def _extract_service(text: str) -> Optional[str]:
    """Extract a service type from text."""
    for service in _SERVICES:
        if service in text:
            return service.replace("-", " ")
    return None
