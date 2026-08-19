"""Dialogue State Tracker — the state machine and its regressions."""

from __future__ import annotations

import pytest

from core.nlu.dst import DialogueStateTracker, State
from tests.conftest import slot


def _run(dst, engine, turns):
    """Feed (intent, slots) pairs and return the final result."""
    result = None
    for text, intent, slots in turns:
        engine.queue(intent, slots=slots)
        result = dst.process_turn(text, joint_nlu_engine=engine)
    return result


# ── confirmation gate ───────────────────────────────────────
def test_all_slots_filled_asks_for_confirmation_first(joint_dst):
    """The agent used to execute the instant the last slot arrived."""
    dst, engine = joint_dst
    r = _run(dst, engine, [
        ("book an appointment", "book_appointment", {}),
        ("cardiology", None, {"service_type": slot("Cardiology")}),
        ("next monday", None, {"date": slot("2026-08-24")}),
        ("3 pm", None, {"time": slot("15:00")}),
        ("john smith", None, {"patient_name": slot("John Smith")}),
    ])
    assert r["missing_slots"] == []
    assert r["state"] == State.CONFIRMING.value, "must confirm before executing"


def test_yes_executes_exactly_once(joint_dst):
    """After executing, the DST must reach DONE so the action can't re-fire."""
    dst, engine = joint_dst
    _run(dst, engine, [
        ("book", "book_appointment",
         {"service_type": slot("Cardiology"), "date": slot("2026-08-24"),
          "time": slot("15:00"), "patient_name": slot("John Smith")}),
    ])
    assert dst.state == State.CONFIRMING

    engine.queue("confirmation")
    r = dst.process_turn("yes", joint_nlu_engine=engine)
    assert r["state"] == State.EXECUTING.value

    dst.mark_action_complete("book_appointment")

    # Any further turn must NOT return to EXECUTING.
    engine.queue("confirmation")
    r2 = dst.process_turn("yes", joint_nlu_engine=engine)
    assert r2["state"] != State.EXECUTING.value
    assert dst.completed_action == "book_appointment"


# ── the caller can say no ───────────────────────────────────
def test_bare_no_is_a_denial_not_a_continuation(joint_dst):
    """`_last_classified_intent` was force-set to 'continuation', so denial
    never reached the router and the caller could not say no."""
    dst, engine = joint_dst
    _run(dst, engine, [
        ("book", "book_appointment",
         {"service_type": slot("Cardiology"), "date": slot("2026-08-24"),
          "time": slot("15:00"), "patient_name": slot("John Smith")}),
    ])
    # Model misreads bare "no" as slot_answer — the lexical override must win.
    engine.queue(None)
    r = dst.process_turn("no", joint_nlu_engine=engine)
    assert r["intent"] == "denial"
    assert r["state"] == State.COLLECTING.value
    assert dst.declined_confirmation is True


def test_bare_yes_is_a_confirmation(joint_dst):
    dst, engine = joint_dst
    _run(dst, engine, [
        ("book", "book_appointment",
         {"service_type": slot("Cardiology"), "date": slot("2026-08-24"),
          "time": slot("15:00"), "patient_name": slot("John Smith")}),
    ])
    engine.queue(None)  # model says slot_answer
    r = dst.process_turn("yes please", joint_nlu_engine=engine)
    assert r["intent"] == "confirmation"
    assert r["state"] == State.EXECUTING.value


def test_denial_then_change_re_confirms(joint_dst):
    dst, engine = joint_dst
    _run(dst, engine, [
        ("book", "book_appointment",
         {"service_type": slot("Cardiology"), "date": slot("2026-08-24"),
          "time": slot("15:00"), "patient_name": slot("John Smith")}),
    ])
    engine.queue(None)
    dst.process_turn("no", joint_nlu_engine=engine)
    assert dst.state == State.COLLECTING

    engine.queue(None, slots={"date": slot("2026-08-25")})
    r = dst.process_turn("make it Tuesday", joint_nlu_engine=engine)
    assert r["slots"]["date"] == "2026-08-25"
    assert r["state"] == State.CONFIRMING.value


def test_correction_does_not_switch_task_mid_booking(joint_dst):
    """'actually make it Tuesday' reads as reschedule_appointment to the model;
    mid-booking it must stay a slot correction."""
    dst, engine = joint_dst
    _run(dst, engine, [
        ("book", "book_appointment",
         {"service_type": slot("Cardiology"), "date": slot("2026-08-24"),
          "time": slot("15:00"), "patient_name": slot("John Smith")}),
    ])
    engine.queue("reschedule_appointment", slots={"date": slot("2026-08-25")})
    r = dst.process_turn("actually make it Tuesday", joint_nlu_engine=engine)
    assert r["_active_intent"] == "book_appointment"
    assert r["slots"]["date"] == "2026-08-25"


# ── invalid slots are recoverable ───────────────────────────
def test_invalid_slot_is_cleared_and_staged_for_repair(joint_dst):
    dst, engine = joint_dst
    engine.queue("book_appointment", slots={"date": slot("2026-08-22")})  # Saturday
    r = dst.process_turn("book me in on Saturday", joint_nlu_engine=engine)
    assert r["repair_message"] and "closed" in r["repair_message"].lower()
    assert "date" not in r["slots"], "invalid slot must be cleared so it can be re-asked"
    assert r["state"] == State.COLLECTING.value


def test_emergency_overrides_everything(joint_dst):
    """A context-primed model must never cause a missed emergency."""
    dst, engine = joint_dst
    engine.queue("repeat", confidence=0.99)
    r = dst.process_turn("my father is having a heart attack", joint_nlu_engine=engine)
    assert r["intent"] == "emergency_request"


# ── slot hygiene ────────────────────────────────────────────
def test_low_confidence_slots_are_dropped(joint_dst):
    dst, engine = joint_dst
    engine.queue("book_appointment", slots={"patient_name": slot("Mumble", 0.2)})
    r = dst.process_turn("book", joint_nlu_engine=engine)
    assert "patient_name" not in r["slots"]


def test_empty_slot_values_do_not_count_as_filled(joint_dst):
    dst, engine = joint_dst
    engine.queue("book_appointment", slots={"patient_name": slot("")})
    r = dst.process_turn("book", joint_nlu_engine=engine)
    assert "patient_name" in r["missing_slots"]


def test_bare_string_slots_do_not_crash(joint_dst):
    """_build_result flattens slots to strings; feeding them back used to
    raise AttributeError on .get('value')."""
    dst, engine = joint_dst
    dst.slots = {"date": "2026-08-24"}          # bare string, not a dict
    dst.intent = "book_appointment"
    engine.queue(None, slots={"time": slot("15:00")})
    r = dst.process_turn("3 pm", joint_nlu_engine=engine)
    assert r["slots"]["date"] == "2026-08-24"
    assert r["slots"]["time"] == "15:00"


def test_reset_clears_everything(joint_dst):
    dst, engine = joint_dst
    engine.queue("book_appointment", slots={"date": slot("2026-08-24")})
    dst.process_turn("book", joint_nlu_engine=engine)
    dst.reset()
    assert dst.state == State.IDLE and dst.intent is None and dst.slots == {}


def test_joint_enabled_without_engine_raises_clearly(monkeypatch):
    from config import cfg

    monkeypatch.setattr(cfg.nlu, "joint_nlu_enabled", True)
    with pytest.raises(RuntimeError, match="joint_nlu_engine"):
        DialogueStateTracker().process_turn("hello")


def test_bare_no_with_nothing_pending_is_not_a_declined_confirmation(joint_dst):
    """"No thanks" after an FAQ is the caller closing the call, not rejecting
    a read-back — it must not trigger "what would you like to change?"."""
    dst, engine = joint_dst
    engine.queue("clinic_hours")
    dst.process_turn("what are your hours", joint_nlu_engine=engine)

    engine.queue(None)
    r = dst.process_turn("no thanks", joint_nlu_engine=engine)
    assert r["intent"] == "denial"
    assert r["declined_confirmation"] is False


def test_a_question_is_never_swallowed_as_a_confirmation(joint_dst):
    """After an action the system says "anything else?", which primes the
    model towards confirmation; a real question must still be answered."""
    dst, engine = joint_dst
    engine.queue("confirm_appointment", confidence=0.9)
    r = dst.process_turn("what are your opening hours", joint_nlu_engine=engine)
    assert r["intent"] not in ("confirmation", "confirm_appointment")


def test_canonical_resolution_rescues_an_over_extended_span(joint_dst):
    """The tagger often over-extends a span ("to book a checkup"), dragging
    mean token confidence below the accept threshold even though the entity
    store resolved it exactly."""
    dst, engine = joint_dst
    engine.queue("book_appointment", slots={
        "service_type": {"value": "General Checkup", "confidence": 0.71,
                         "canonical_id": "SRV_001", "canonical_name": "General Checkup"},
    })
    r = dst.process_turn("I want to book a checkup", joint_nlu_engine=engine)
    assert r["slots"].get("service_type") == "General Checkup"
    assert "service_type" not in r["missing_slots"]
