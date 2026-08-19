"""Router — path selection, confirmation flow, escalation."""

from __future__ import annotations

import pytest

from core.router import Path, route


def nlu(**over):
    base = {
        "intent": "book_appointment", "_active_intent": "book_appointment",
        "confidence": 0.9, "slots": {}, "state": "collecting",
        "missing_slots": [], "turn_count": 1, "failed_turns": 0,
        "repair_message": None, "pending_slot": None,
        "completed_action": None, "declined_confirmation": False,
    }
    base.update(over)
    return base


FULL = {"service_type": "Cardiology", "date": "2026-08-24",
        "time": "15:00", "patient_name": "John Smith"}


# ── confirmation flow ───────────────────────────────────────
def test_complete_booking_asks_to_confirm_not_books():
    r = route(nlu(slots=FULL, missing_slots=[], state="confirming"))
    assert r.path is Path.A_DIRECT
    assert r.execute_action is None, "must not commit before the caller says yes"
    assert "?" in r.response
    assert r.metadata.get("awaiting_confirmation") is True


def test_executing_state_commits_the_action():
    r = route(nlu(slots=FULL, missing_slots=[], state="executing"))
    assert r.path is Path.A_DIRECT
    assert r.execute_action == "book_appointment"
    assert "John Smith" in r.response


def test_missing_slot_clarifies():
    r = route(nlu(slots={"date": "2026-08-24"}, missing_slots=["time", "patient_name"]))
    assert r.path is Path.B_CLARIFY
    assert r.needs_llm is False
    assert "time" in r.response.lower()
    assert r.metadata["missing_slot"] == "time"


def test_repair_message_is_spoken_and_keeps_the_dialogue_open():
    """An invalid slot used to blank out the agent's question entirely."""
    msg = "We're closed on Saturdays. Monday the 24th works — shall I use that?"
    r = route(nlu(repair_message=msg, missing_slots=["date"]))
    assert r.path is Path.B_CLARIFY
    assert r.response == msg


def test_denial_after_completion_says_goodbye():
    r = route(nlu(intent="denial", state="done", completed_action="book_appointment",
                  slots=FULL, missing_slots=[]))
    assert r.path is Path.A_DIRECT
    assert r.end_conversation is True


def test_denial_at_confirmation_asks_what_to_change():
    r = route(nlu(intent="denial", state="collecting", slots=FULL,
                  missing_slots=[], declined_confirmation=True))
    assert r.path is Path.B_CLARIFY
    assert "change" in r.response.lower()


# ── other paths ─────────────────────────────────────────────
def test_faq_goes_to_llm():
    r = route(nlu(intent="clinic_hours", _active_intent="clinic_hours"))
    assert r.path is Path.D_RAG_LLM and r.needs_llm is True and r.filler


def test_doctor_availability_is_handled_not_dropped():
    """This intent matched no branch and silently fell through."""
    r = route(nlu(intent="doctor_availability", _active_intent="doctor_availability"))
    assert r.path is Path.D_RAG_LLM


def test_out_of_domain_goes_to_llm():
    r = route(nlu(intent="out_of_domain", _active_intent="out_of_domain", confidence=0.3))
    assert r.path is Path.C_LLM and r.needs_llm is True


@pytest.mark.parametrize("intent", ["complaint", "transfer_to_human", "emergency_request"])
def test_escalations(intent):
    r = route(nlu(intent=intent, _active_intent=intent))
    assert r.path is Path.E_ESCALATE and r.end_conversation is True


def test_emergency_tells_the_caller_to_dial_911():
    r = route(nlu(intent="emergency_request", _active_intent="emergency_request"))
    assert "911" in r.response


def test_loop_escalates():
    r = route(nlu(failed_turns=3))
    assert r.path is Path.E_ESCALATE
    assert r.metadata["reason"] == "loop_detected"


def test_greeting_and_goodbye():
    assert route(nlu(intent="greeting", _active_intent="greeting")).path is Path.A_DIRECT
    g = route(nlu(intent="goodbye", _active_intent="goodbye"))
    assert g.path is Path.A_DIRECT and g.end_conversation is True


def test_repeat_replays_the_last_response():
    r = route(nlu(intent="repeat", _active_intent="repeat"), last_response="Your slot is 3 PM.")
    assert r.response == "Your slot is 3 PM."


def test_readonly_intent_executes_without_confirmation():
    r = route(nlu(intent="check_my_appointments", _active_intent="check_my_appointments",
                  slots={"patient_name": "John"}, missing_slots=[]))
    assert r.execute_action == "check_my_appointments"


def test_denial_with_no_task_in_flight_closes_the_call():
    """"No thanks" after an FAQ answer must not ask "what would you like to
    change?" — there is nothing to change."""
    r = route(nlu(intent="denial", _active_intent="clinic_hours",
                  state="collecting", missing_slots=[], declined_confirmation=False))
    assert r.path is Path.A_DIRECT
    assert r.end_conversation is True
