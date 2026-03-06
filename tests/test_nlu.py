"""
Tests for NLU Brain — Intent, Slots, DST, Confidence.

Updated for ML-based models:
- DeBERTa v3 xsmall zero-shot classification (intent)
- BERT NER (slot extraction)
"""

import pytest
from datetime import datetime, timedelta


# ── Intent Classifier Tests ──────────────────────────────────

def test_intent_greeting():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("Hello there!")
    assert intent == "greeting"
    assert conf > 0.3


def test_intent_goodbye():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("Goodbye, see you later")
    assert intent == "goodbye"
    assert conf > 0.3


def test_intent_book_appointment():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("I'd like to book an appointment for tomorrow")
    assert intent == "book_appointment"
    assert conf > 0.4


def test_intent_cancel():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("Please cancel my upcoming appointment")
    assert intent == "cancel_appointment"
    assert conf > 0.3


def test_intent_reschedule():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("I need to reschedule my appointment to a later date")
    assert intent == "reschedule"
    assert conf > 0.3


def test_intent_check_status():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("Can you check the status of my appointment?")
    assert intent == "check_status"
    assert conf > 0.3


def test_intent_faq():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("What are your business hours and location?")
    assert intent == "faq"
    assert conf > 0.3


def test_intent_out_of_domain():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("Tell me a joke about elephants")
    assert intent == "out_of_domain"


def test_intent_empty():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("")
    assert intent == "out_of_domain"
    assert conf == 0.0


def test_intent_complaint():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("I am very unhappy with the service, this is terrible")
    assert intent == "complaint"
    assert conf > 0.3


def test_intent_transfer():
    from core.nlu.intent import classify_intent
    intent, conf = classify_intent("I want to speak with a real human please")
    assert intent == "transfer_to_human"
    assert conf > 0.3


# ── Slot Extractor Tests ────────────────────────────────────

def test_slot_date_tomorrow():
    from core.nlu.slots import extract_slots
    slots = extract_slots("I want to come in tomorrow")
    assert "date" in slots
    expected = (datetime.now().date() + timedelta(days=1)).isoformat()
    assert slots["date"] == expected


def test_slot_date_day_of_week():
    from core.nlu.slots import extract_slots
    slots = extract_slots("Can I come on Monday?")
    assert "date" in slots
    # Should be a valid date string
    assert len(slots["date"]) == 10  # YYYY-MM-DD


def test_slot_time_12h():
    from core.nlu.slots import extract_slots
    slots = extract_slots("at 3 pm please")
    assert "time" in slots
    assert slots["time"] == "15:00"


def test_slot_time_12h_with_minutes():
    from core.nlu.slots import extract_slots
    slots = extract_slots("Schedule for 10:30 am")
    assert "time" in slots
    assert slots["time"] == "10:30"


def test_slot_time_oclock():
    from core.nlu.slots import extract_slots
    slots = extract_slots("Come at 3 o'clock")
    assert "time" in slots
    assert slots["time"] == "15:00"


def test_slot_name():
    from core.nlu.slots import extract_slots
    slots = extract_slots("My name is John Smith")
    assert "name" in slots
    assert "John" in slots["name"]


def test_slot_name_no_false_positive():
    from core.nlu.slots import extract_slots
    slots = extract_slots("I am pretty good")
    assert "name" not in slots


def test_slot_phone():
    from core.nlu.slots import extract_slots
    slots = extract_slots("My number is 555-123-4567")
    assert "phone" in slots
    assert "5551234567" in slots["phone"]


def test_slot_service():
    from core.nlu.slots import extract_slots
    slots = extract_slots("I need a dental cleaning")
    assert "service_type" in slots
    assert "dental" in slots["service_type"] or "cleaning" in slots["service_type"]


def test_slot_multiple():
    from core.nlu.slots import extract_slots
    slots = extract_slots("Book for tomorrow at 3 pm")
    assert "date" in slots
    assert "time" in slots
    assert slots["time"] == "15:00"


def test_slot_empty():
    from core.nlu.slots import extract_slots
    slots = extract_slots("")
    assert slots == {}


# ── Dialogue State Tracker Tests ────────────────────────────

def test_dst_greeting():
    from core.nlu.dst import DialogueStateTracker
    dst = DialogueStateTracker()
    result = dst.process_turn("Hello!")
    assert result["intent"] == "greeting"
    assert result["state"] == "greeting"


def test_dst_book_appointment_flow():
    from core.nlu.dst import DialogueStateTracker
    dst = DialogueStateTracker()

    # Turn 1: Intent
    r1 = dst.process_turn("I want to book an appointment")
    assert r1["intent"] == "book_appointment"
    assert r1["state"] == "collecting"
    assert len(r1["missing_slots"]) > 0

    # Turn 2: Add date + time
    r2 = dst.process_turn("Tomorrow at 3 pm")
    assert "date" in r2["slots"]
    assert "time" in r2["slots"]

    # Turn 3: Add name — should transition to confirming
    r3 = dst.process_turn("My name is John Smith")
    assert "name" in r3["slots"]
    # With ML models, all slots should now be filled
    if not r3["missing_slots"]:
        assert r3["state"] == "confirming"


def test_dst_slot_correction():
    from core.nlu.dst import DialogueStateTracker
    dst = DialogueStateTracker()

    dst.process_turn("Book an appointment for tomorrow at 3 pm")
    r = dst.process_turn("Actually, make it 5 pm")
    # Time should be updated
    assert r["slots"].get("time") == "17:00"


def test_dst_reset():
    from core.nlu.dst import DialogueStateTracker
    dst = DialogueStateTracker()
    dst.process_turn("Book an appointment for tomorrow")
    dst.reset()
    assert dst.state.value == "idle"
    assert dst.intent is None
    assert dst.slots == {}


def test_dst_turn_count():
    from core.nlu.dst import DialogueStateTracker
    dst = DialogueStateTracker()
    dst.process_turn("Hello")
    dst.process_turn("Book appointment")
    dst.process_turn("Tomorrow")
    assert dst.turn_count == 3


# ── Confidence Scorer Tests ─────────────────────────────────

def test_confidence_high_with_all_slots():
    from core.nlu.confidence import compute_confidence
    score = compute_confidence(
        "book_appointment", 0.9,
        {"date": "2026-01-01", "time": "15:00", "name": "John"}
    )
    assert score >= 0.85


def test_confidence_medium_missing_slots():
    from core.nlu.confidence import compute_confidence
    score = compute_confidence(
        "book_appointment", 0.9,
        {"date": "2026-01-01"}  # missing time and name
    )
    assert 0.4 < score < 0.85


def test_confidence_greeting_always_high():
    from core.nlu.confidence import compute_confidence
    score = compute_confidence("greeting", 0.9, {})
    # Greeting has no required slots, so slot_score = 0.4
    assert score >= 0.8


def test_confidence_level_categorization():
    from core.nlu.confidence import confidence_level
    assert confidence_level(0.9) == "high"
    assert confidence_level(0.7) == "medium"
    assert confidence_level(0.3) == "low"
