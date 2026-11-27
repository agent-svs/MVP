"""Business-rule slot validation produces recoverable repairs."""

from __future__ import annotations

from datetime import datetime, timedelta

from config import cfg
from core.nlu.validation import validate_slots


def _next_weekday(offset_days: int = 1):
    d = datetime.now().date() + timedelta(days=offset_days)
    while d.weekday() not in cfg.business.open_days:
        d += timedelta(days=1)
    return d.isoformat()


def test_valid_slots_pass():
    assert validate_slots({"date": _next_weekday(), "time": "15:00",
                           "doctor_name": "Dr. Jones", "service_type": "Cardiology"}).ok


def test_weekend_is_a_repair_not_a_dead_end():
    r = validate_slots({"date": "2026-08-22"})  # a Saturday
    assert not r.ok
    p = r.problems[0]
    assert p.reason == "closed_day"
    assert p.slot == "date" and p.clear_slot
    # The message must invite a new value, not just refuse.
    assert "?" in p.message


def test_past_date_rejected():
    r = validate_slots({"date": "2020-01-01"})
    assert r.problems[0].reason == "past_date"


def test_closed_hours_message_states_the_hours():
    r = validate_slots({"time": "21:00"})
    assert r.problems[0].reason == "closed_hours"
    assert str(cfg.business.hours_start) in r.problems[0].message


def test_unknown_doctor_lists_the_roster():
    r = validate_slots({"doctor_name": "Dr. Pastor"})
    assert r.problems[0].reason == "unknown_doctor"
    assert "Dr. Jones" in r.problems[0].message


def test_slots_to_clear_lets_the_dst_recover():
    r = validate_slots({"date": "2020-01-01", "time": "23:00"})
    assert set(r.slots_to_clear) == {"date", "time"}


def test_accepts_dict_shaped_slots():
    r = validate_slots({"date": {"value": "2020-01-01", "confidence": 0.9}})
    assert r.problems[0].reason == "past_date"


def test_empty_slots_are_skipped():
    assert validate_slots({"date": None, "time": "", "doctor_name": None}).ok
