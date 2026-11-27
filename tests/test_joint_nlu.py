"""Joint NLU — exercises the REAL model, not a mock.

The previous version of this file replaced the model with a MagicMock, so
tokenisation, offset mapping, BIO decoding and length capping had zero
coverage. The mock also emitted the slot key "name" while the real model emits
"patient_name", so the tests passed against a shape that can never occur.
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.slow


def test_model_loads_with_a_fast_tokenizer(joint_engine):
    """Offset mapping — and therefore slot spans — needs a fast tokenizer."""
    assert joint_engine.is_loaded
    assert joint_engine.tokenizer.is_fast


def test_special_turn_tokens_exist(joint_engine):
    tok = joint_engine.tokenizer
    for marker in ("[SYS]", "[USR]"):
        assert tok.convert_tokens_to_ids(marker) != tok.unk_token_id, marker


# ── intents ─────────────────────────────────────────────────
@pytest.mark.parametrize("utterance,expected", [
    ("I want to book an appointment", "book_appointment"),
    ("please cancel my appointment", "cancel_appointment"),
    ("what are your opening hours", "clinic_hours"),
    ("where are you located", "clinic_location"),
    ("I want to speak to a real person", "transfer_to_human"),
])
def test_core_intents(joint_engine, utterance, expected):
    result = joint_engine.predict("", utterance)
    assert result["intent"] == expected
    assert result["intent_confidence"] > 0.5


def test_bare_answer_is_a_slot_answer_not_a_new_intent(joint_engine):
    result = joint_engine.predict("What service do you need?", "Cardiology")
    assert result["intent"] is None
    assert result["raw_nlu_intent"] == "slot_answer"


# ── slot spans (the previously untested part) ───────────────
def test_slot_keys_match_what_the_dst_requires(joint_engine):
    """REQUIRED_SLOTS uses patient_name; a mock emitting "name" hid this."""
    from core.nlu.dst import REQUIRED_SLOTS

    result = joint_engine.predict("May I have the name?", "John Smith")
    assert "patient_name" in result["slots"]
    assert "patient_name" in REQUIRED_SLOTS["book_appointment"]


def test_multi_token_name_is_not_truncated(joint_engine):
    """I- continuation tags must extend the span, not start a new one."""
    result = joint_engine.predict("And the patient's name?", "Priya Sharma")
    assert result["slots"]["patient_name"]["value"] == "Priya Sharma"


def test_date_is_normalised_to_iso(joint_engine):
    result = joint_engine.predict("What date?", "next Monday")
    value = result["slots"]["date"]["value"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)


def test_time_is_normalised_to_24h(joint_engine):
    assert joint_engine.predict("What time?", "3 pm")["slots"]["time"]["value"] == "15:00"


def test_phone_is_normalised_to_digits(joint_engine):
    """The joint path returned the raw span while the legacy path returned
    digits — two contracts for the same slot."""
    value = joint_engine.predict("Your phone number?", "555-123-4567")["slots"]["phone"]["value"]
    assert value == "5551234567"


def test_entities_are_canonicalised(joint_engine):
    """The joint path never called the entity store, so canonical IDs never
    resolved in production."""
    result = joint_engine.predict("", "I want to see Dr. Kulkarni")
    doctor = result["slots"]["doctor_name"]
    assert doctor["canonical_id"] == "DOC_001"
    assert doctor["canonical_name"] == "Dr. Kulkarni"


def test_multi_slot_utterance(joint_engine):
    result = joint_engine.predict(
        "", "Book Dr. Kulkarni tomorrow at 4pm for a checkup, my name is Priya Sharma"
    )
    assert result["intent"] == "book_appointment"
    for key in ("doctor_name", "date", "time", "patient_name"):
        assert key in result["slots"], f"missing {key}"


# ── robustness ──────────────────────────────────────────────
def test_empty_utterance_is_safe(joint_engine):
    result = joint_engine.predict("", "")
    assert result["intent"] is None and result["slots"] == {}


def test_overlong_input_does_not_overflow(joint_engine):
    """Length capping must trim context first and keep the utterance whole."""
    long_context = "please tell me more about that " * 60
    result = joint_engine.predict(long_context, "3 pm")
    assert result["slots"]["time"]["value"] == "15:00"


def test_unloaded_model_raises_a_typed_error():
    from core.nlu.joint_nlu import JointNLU, JointNLUError

    with pytest.raises(JointNLUError):
        JointNLU().predict("", "hello")


def test_missing_model_path_raises_a_typed_error(monkeypatch):
    from core.nlu.joint_nlu import JointNLU, JointNLUError

    monkeypatch.setattr("config.cfg.nlu.joint_nlu_hub_id", "definitely/not-a-real-repo-xyz")
    with pytest.raises(JointNLUError):
        JointNLU().load("/nonexistent/path/to/model")


# ── end to end through the DST ──────────────────────────────
def test_full_booking_reaches_confirmation_then_done(joint_engine, monkeypatch):
    from config import cfg
    from core.nlu.dst import DialogueStateTracker, State
    from core.router import route

    monkeypatch.setattr(cfg.nlu, "joint_nlu_enabled", True)
    dst, last = DialogueStateTracker(), ""

    for text in ("I want to book an appointment", "Cardiology",
                 "next Monday", "3 pm", "John Smith"):
        nlu = dst.process_turn(text, system_context=last, joint_nlu_engine=joint_engine)
        last = route(nlu, last_response=last).response

    assert dst.state == State.CONFIRMING
    assert dst.missing_slots == []

    nlu = dst.process_turn("yes", system_context=last, joint_nlu_engine=joint_engine)
    assert nlu["intent"] == "confirmation"
    routing = route(nlu, last_response=last)
    assert routing.execute_action == "book_appointment"

    dst.mark_action_complete(routing.execute_action)
    assert dst.state == State.DONE
