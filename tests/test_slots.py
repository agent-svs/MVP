"""Slot extraction — regression tests for the date/time/service/phone bugs."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.nlu.slots import (
    _extract_date, _extract_phone, _extract_service, _extract_time,
)


def _today():
    return datetime.now().date()


# ── dates ───────────────────────────────────────────────────
def test_tomorrow():
    assert _extract_date("come in tomorrow")[0] == (_today() + timedelta(days=1)).isoformat()


def test_day_after_tomorrow_is_not_tomorrow():
    """'day after tomorrow' contains 'tomorrow'; a naive scan resolved it to +1."""
    got = _extract_date("the day after tomorrow")[0]
    assert got == (_today() + timedelta(days=2)).isoformat()


def test_next_weekday_is_not_a_week_late():
    """'next Monday' must be the upcoming Monday, not the one after it."""
    got = _extract_date("next Monday")[0]
    bare = _extract_date("Monday")[0]
    assert got == bare
    d = datetime.strptime(got, "%Y-%m-%d").date()
    assert d.weekday() == 0
    assert 1 <= (d - _today()).days <= 7


def test_weekday_never_resolves_to_today_or_past():
    for day in ("Monday", "Tuesday", "Friday", "Saturday"):
        d = datetime.strptime(_extract_date(day)[0], "%Y-%m-%d").date()
        assert d > _today()


def test_no_date_from_arbitrary_digits():
    """Unconstrained fuzzy parsing turned any number into a date."""
    assert _extract_date("I want 2 appointments") is None
    assert _extract_date("book 2 slots") is None
    assert _extract_date("at 3 pm") is None


def test_month_name_rolls_to_next_year_when_past():
    got = _extract_date("March 15")
    assert got is not None
    d = datetime.strptime(got[0], "%Y-%m-%d").date()
    assert d >= _today() and d.month == 3 and d.day == 15


# ── times ───────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("at 3 pm please", "15:00"),
    ("10:30 am", "10:30"),
    ("at 15:00", "15:00"),
    ("noon", "12:00"),
    ("12 am", "00:00"),
    ("12 pm", "12:00"),
])
def test_time_formats(text, expected):
    assert _extract_time(text)[0] == expected


def test_oclock_uses_business_hours_not_an_arbitrary_rule():
    """'7 o'clock' used to become 19:00 (always closed) while '8 o'clock' was 08:00."""
    assert _extract_time("come at 3 o'clock")[0] == "15:00"
    assert _extract_time("come at 9 o'clock")[0] == "09:00"


def test_natural_language_times_stay_within_opening_hours():
    """'at night' produced 19:00, which the guardrails then always rejected."""
    from config import cfg

    for phrase in ("in the evening", "at night", "in the afternoon", "in the morning"):
        got = _extract_time(phrase)
        assert got is not None, phrase
        hour = int(got[0].split(":")[0])
        assert cfg.business.hours_start <= hour < cfg.business.hours_end, phrase


# ── services ────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "I need to examine my options",
    "please repair the trim",
    "that's a general question",
    "scan the room for me",
    "check up on my booking",
])
def test_service_no_substring_false_positives(text):
    """Substring matching booked an 'exam' for 'examine'."""
    assert _extract_service(text) is None


@pytest.mark.parametrize("text,expected", [
    ("I need a dental cleaning", "dental cleaning"),
    ("cardiology please", "cardiology"),
    ("just a check up", "check up"),
])
def test_service_still_matches_real_requests(text, expected):
    assert _extract_service(text)[0] == expected


# ── phones ──────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("call 5551234567", "5551234567"),
    ("my number is 555-123-4567", "5551234567"),
    ("(555) 123 4567", "5551234567"),
    ("nine eight seven six five four three two one zero", "9876543210"),
])
def test_phone_extraction(text, expected):
    assert _extract_phone(text)[0] == expected


def test_phone_rejects_short_numbers():
    assert _extract_phone("my number is 1234567") is None
