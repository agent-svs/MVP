"""
Tests for Guardrails — PII, Business Rules, Length Cap.
"""

from datetime import datetime, timedelta


def test_pii_detects_ssn():
    from core.guardrails import check_pii
    issues = check_pii("Your SSN is 123-45-6789")
    assert any("SSN" in i for i in issues)


def test_pii_detects_credit_card():
    from core.guardrails import check_pii
    issues = check_pii("Card: 4111-2222-3333-4444")
    assert any("credit card" in i for i in issues)


def test_pii_clean_text():
    from core.guardrails import check_pii
    issues = check_pii("Your appointment is at 3 PM")
    assert issues == []


def test_scrub_pii():
    from core.guardrails import scrub_pii
    result = scrub_pii("SSN: 123-45-6789 is private")
    assert "123-45-6789" not in result
    assert "[REDACTED]" in result


def test_validate_date_past():
    from core.guardrails import validate_date
    yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
    issues = validate_date(yesterday)
    assert any("past" in i.lower() for i in issues)


def test_validate_date_future():
    from core.guardrails import validate_date
    # Use next Monday (always a weekday)
    today = datetime.now().date()
    days_ahead = 7 - today.weekday()  # next Monday
    if days_ahead <= 0:
        days_ahead += 7
    next_monday = (today + timedelta(days=days_ahead)).isoformat()
    issues = validate_date(next_monday)
    assert not any("past" in i.lower() for i in issues)


def test_validate_time_outside_hours():
    from core.guardrails import validate_time
    issues = validate_time("22:00")
    assert any("outside" in i.lower() for i in issues)


def test_validate_time_within_hours():
    from core.guardrails import validate_time
    issues = validate_time("14:00")
    assert issues == []


def test_cap_response_length():
    from core.guardrails import cap_response_length
    short = "Hello there."
    assert cap_response_length(short) == short

    long = "A" * 600
    result = cap_response_length(long)
    assert len(result) <= 501  # 500 + possible period


def test_apply_guardrails_clean():
    from core.guardrails import apply_guardrails
    result = apply_guardrails("Your appointment is confirmed.")
    assert result.passed is True
    assert result.warnings == []


def test_apply_guardrails_pii():
    from core.guardrails import apply_guardrails
    result = apply_guardrails("Your SSN is 123-45-6789")
    assert "REDACTED" in result.text
    assert len(result.warnings) > 0
