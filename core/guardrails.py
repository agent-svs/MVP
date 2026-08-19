"""
Guardrails — outbound response safety.

Scope is deliberately narrow: make the text about to be spoken safe.
  * strip PII that must never be read aloud or logged
  * strip markdown/control artefacts that sound wrong in speech
  * cap length for a voice channel

Business-rule validation of SLOTS lives in core/nlu/validation.py and is
handled by the DST/router as a dialogue repair. Mixing the two here meant an
invalid slot replaced the agent's clarification question with a refusal, and
the caller could never correct the value — the call deadlocked.

`apply_guardrails` therefore no longer returns passed=False for a business
rule. It blocks only for content that is genuinely unsafe to speak.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── PII patterns ────────────────────────────────────────────
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Medical record numbers / policy numbers, e.g. MRN 1234567
_MRN_RE = re.compile(r"\b(?:mrn|record\s*(?:no|number)|policy\s*(?:no|number))\s*[:#]?\s*\w{4,}\b",
                     re.IGNORECASE)
# Date of birth spoken as a full date
_DOB_RE = re.compile(r"\b(?:dob|date of birth)\s*[:#]?\s*[\d/\-\.]{6,10}\b", re.IGNORECASE)
# A bare 10+ digit run that is not a formatted phone number we chose to speak
_LONG_DIGITS_RE = re.compile(r"\b\d{11,}\b")

_PII_RULES = [
    (_SSN_RE, "SSN", "[redacted]"),
    (_CREDIT_CARD_RE, "credit card number", "[redacted]"),
    (_EMAIL_RE, "email address", "[redacted]"),
    (_MRN_RE, "record number", "[redacted]"),
    (_DOB_RE, "date of birth", "[redacted]"),
    (_LONG_DIGITS_RE, "long digit sequence", "[redacted]"),
]

# Artefacts that sound wrong read aloud.
_MARKDOWN_RE = re.compile(r"[*_`#>|]{1,3}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

MAX_RESPONSE_LENGTH = 500


class GuardrailResult:
    """Result of guardrail validation."""

    def __init__(
        self,
        passed: bool,
        text: str,
        warnings: Optional[list[str]] = None,
        redactions: Optional[list[str]] = None,
    ) -> None:
        self.passed = passed
        self.text = text
        self.warnings = warnings or []
        self.redactions = redactions or []

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "BLOCKED"
        return f"GuardrailResult({status}, warnings={self.warnings})"


def check_pii(text: str) -> list[str]:
    """Return a list of PII categories present in `text`."""
    return [label for pattern, label, _ in _PII_RULES if pattern.search(text or "")]


def scrub_pii(text: str) -> str:
    """Replace every PII pattern with a redaction marker."""
    out = text or ""
    for pattern, _, replacement in _PII_RULES:
        out = pattern.sub(replacement, out)
    return out


def clean_for_speech(text: str) -> str:
    """Remove markdown/control characters and normalise whitespace."""
    out = _CONTROL_RE.sub(" ", text or "")
    out = _MARKDOWN_RE.sub("", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def cap_response_length(text: str, max_length: int = MAX_RESPONSE_LENGTH) -> str:
    """Cap length for voice output, preferring a sentence boundary."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_period = truncated.rfind(".")
    if last_period > max_length * 0.5:
        return truncated[: last_period + 1]
    return truncated.rstrip() + "."


def apply_guardrails(text: str, slots: Optional[dict] = None) -> GuardrailResult:
    """Make an outbound response safe to speak.

    `slots` is accepted for signature compatibility and is no longer used to
    block responses — see core/nlu/validation.py.
    """
    warnings: list[str] = []
    redactions: list[str] = []

    if not text or not text.strip():
        return GuardrailResult(
            passed=False,
            text="Sorry, I didn't catch that. Could you say it again?",
            warnings=["empty response"],
        )

    found = check_pii(text)
    if found:
        redactions = found
        warnings.extend(f"Response contained {f} — redacted" for f in found)
        text = scrub_pii(text)
        logger.warning(f"PII redacted from response: {found}")

    text = clean_for_speech(text)
    text = cap_response_length(text)

    if not text.strip():
        return GuardrailResult(
            passed=False,
            text="Sorry, I didn't catch that. Could you say it again?",
            warnings=warnings + ["response empty after sanitisation"],
            redactions=redactions,
        )

    return GuardrailResult(passed=True, text=text, warnings=warnings, redactions=redactions)


# ── Back-compat shims ───────────────────────────────────────
# Business-rule validation moved to core/nlu/validation.py. These thin wrappers
# keep older call sites and tests working.
def extract_val(v):
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def validate_date(date_str: str) -> list[str]:
    from core.nlu.validation import validate_date as _v
    return [p.message for p in _v(date_str)]


def validate_time(time_str: str) -> list[str]:
    from core.nlu.validation import validate_time as _v
    return [p.message for p in _v(time_str)]


def validate_doctor(doctor_name: str) -> list[str]:
    from core.nlu.validation import validate_doctor as _v
    return [p.message for p in _v(doctor_name)]


def validate_service(service_type: str) -> list[str]:
    from core.nlu.validation import validate_service as _v
    return [p.message for p in _v(service_type)]


def validate_slots(slots: dict) -> list[str]:
    from core.nlu.validation import validate_slots as _v
    return [p.message for p in _v(slots).problems]
