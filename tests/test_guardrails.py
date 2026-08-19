"""Guardrails — response sanitisation only."""

from __future__ import annotations

import pytest

from core.guardrails import (
    apply_guardrails, cap_response_length, check_pii, clean_for_speech, scrub_pii,
)


# ── the deadlock regression ─────────────────────────────────
def test_business_rules_no_longer_block_a_clarification():
    """A weekend date in slots used to replace the agent's question with a
    refusal, so the caller could never correct it. Guardrails must not care."""
    r = apply_guardrails("What time works best for you?",
                         {"date": "2026-08-22", "time": "15:00"})
    assert r.passed is True
    assert r.text == "What time works best for you?"


def test_unknown_doctor_does_not_block_the_response():
    r = apply_guardrails("Which doctor would you like?", {"doctor_name": "Dr. Pastor"})
    assert r.passed is True


def test_doctor_named_pastor_does_not_trip_a_past_keyword():
    r = apply_guardrails("Dr. Pastor is not one of our doctors.", {})
    assert r.passed is True


# ── PII ─────────────────────────────────────────────────────
@pytest.mark.parametrize("text,label", [
    ("Your SSN is 123-45-6789", "SSN"),
    ("Card 4111 2222 3333 4444", "credit card number"),
    ("Email me at a.b@example.com", "email address"),
    ("MRN: 99AB2231", "record number"),
    ("DOB: 01/02/1988", "date of birth"),
])
def test_pii_detected_and_redacted(text, label):
    assert label in check_pii(text)
    assert "[redacted]" in scrub_pii(text)
    r = apply_guardrails(text)
    assert "[redacted]" in r.text and r.redactions


def test_clean_text_passes_untouched():
    r = apply_guardrails("Your appointment is confirmed.")
    assert r.passed and r.warnings == [] and r.text == "Your appointment is confirmed."


# ── speech shaping ──────────────────────────────────────────
def test_markdown_stripped_for_speech():
    assert "*" not in clean_for_speech("Your **appointment** is `confirmed`")


def test_length_capped():
    assert cap_response_length("Hello there.") == "Hello there."
    assert len(cap_response_length("A" * 600)) <= 501


def test_empty_response_is_blocked_with_a_recovery_line():
    r = apply_guardrails("")
    assert r.passed is False and r.text


def test_legacy_validate_helpers_still_work():
    from core.guardrails import validate_date, validate_time
    assert validate_date("2020-01-01")
    assert validate_time("23:00")
    assert validate_time("14:00") == []
