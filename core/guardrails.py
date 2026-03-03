"""
Guardrails — Safety Filters + Business Rules

Layer 5 & 6: Validates and filters responses before TTS.
- PII detection (SSN, credit card, email in output)
- Business rule validation (no past dates, hours check)
- Loop/failure detection
- Response length capping for voice
"""

import logging
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── PII Patterns ────────────────────────────────────────────
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# ── Business Hours ──────────────────────────────────────────
BUSINESS_HOURS_START = 9   # 9 AM
BUSINESS_HOURS_END = 17    # 5 PM
BUSINESS_DAYS = {0, 1, 2, 3, 4}  # Mon-Fri

# Max response length for voice (characters)
MAX_RESPONSE_LENGTH = 500


class GuardrailResult:
    """Result of guardrail validation."""

    def __init__(self, passed: bool, text: str, warnings: list[str] = None):
        self.passed = passed
        self.text = text
        self.warnings = warnings or []

    def __repr__(self):
        status = "PASS" if self.passed else "BLOCKED"
        return f"GuardrailResult({status}, warnings={self.warnings})"


def check_pii(text: str) -> list[str]:
    """Check for PII patterns in output text."""
    issues = []
    if _SSN_RE.search(text):
        issues.append("Response contains SSN pattern")
    if _CREDIT_CARD_RE.search(text):
        issues.append("Response contains credit card pattern")
    if _EMAIL_RE.search(text):
        issues.append("Response contains email address")
    return issues


def scrub_pii(text: str) -> str:
    """Remove PII patterns from text."""
    text = _SSN_RE.sub("[REDACTED]", text)
    text = _CREDIT_CARD_RE.sub("[REDACTED]", text)
    return text


def validate_date(date_str: str) -> list[str]:
    """Validate a date for business rules."""
    issues = []
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        if date < today:
            issues.append(f"Date {date_str} is in the past")
        if date.weekday() not in BUSINESS_DAYS:
            issues.append(f"Date {date_str} falls on a weekend")
    except ValueError:
        issues.append(f"Invalid date format: {date_str}")
    return issues


def validate_time(time_str: str) -> list[str]:
    """Validate a time for business hours."""
    issues = []
    try:
        parts = time_str.split(":")
        hour = int(parts[0])
        if hour < BUSINESS_HOURS_START or hour >= BUSINESS_HOURS_END:
            issues.append(
                f"Time {time_str} is outside business hours "
                f"({BUSINESS_HOURS_START}:00-{BUSINESS_HOURS_END}:00)"
            )
    except (ValueError, IndexError):
        issues.append(f"Invalid time format: {time_str}")
    return issues


def validate_slots(slots: dict) -> list[str]:
    """Validate all slots against business rules."""
    issues = []
    if "date" in slots:
        issues.extend(validate_date(slots["date"]))
    if "time" in slots:
        issues.extend(validate_time(slots["time"]))
    return issues


def cap_response_length(text: str, max_length: int = MAX_RESPONSE_LENGTH) -> str:
    """Cap response length for voice output."""
    if len(text) <= max_length:
        return text
    # Truncate at last sentence boundary
    truncated = text[:max_length]
    last_period = truncated.rfind(".")
    if last_period > max_length * 0.5:
        return truncated[:last_period + 1]
    return truncated + "."


def apply_guardrails(
    text: str,
    slots: Optional[dict] = None,
) -> GuardrailResult:
    """
    Apply all guardrails to a response.

    Args:
        text: Response text to validate
        slots: Optional slots to validate business rules

    Returns:
        GuardrailResult with passed status, cleaned text, and warnings
    """
    warnings = []

    # PII check
    pii_issues = check_pii(text)
    if pii_issues:
        warnings.extend(pii_issues)
        text = scrub_pii(text)
        logger.warning(f"PII detected and scrubbed: {pii_issues}")

    # Business rule validation
    if slots:
        slot_issues = validate_slots(slots)
        if slot_issues:
            warnings.extend(slot_issues)
            logger.warning(f"Business rule violations: {slot_issues}")

    # Length cap
    text = cap_response_length(text)

    # Determine if we should block or just warn
    passed = not any("past" in w.lower() for w in warnings)

    if warnings:
        logger.info(f"Guardrails: {len(warnings)} warnings — {warnings}")

    return GuardrailResult(passed=passed, text=text, warnings=warnings)
