import pytest
from unittest.mock import MagicMock
from core.nlu.dst import DialogueStateTracker, State
from config import cfg

@pytest.fixture
def mock_joint_nlu():
    mock = MagicMock()
    # Mocking a predict function
    def override_predict(context, text):
        t = text.lower()
        if "cardiology" == t:
            return {"intent": "slot_answer", "intent_confidence": 0.9, "slots": {"service_type": {"value": "cardiology", "confidence": 0.95}}}
        elif "book" in t:
            return {"intent": "book_appointment", "intent_confidence": 0.95, "slots": {"service_type": {"value": "cardiology", "confidence": 0.88}}}
        elif "john smith" == t:
            return {"intent": "slot_answer", "intent_confidence": 0.9, "slots": {"name": {"value": "John Smith", "confidence": 0.99}}}
        elif "okay" == t:
            return {"intent": "confirmation", "intent_confidence": 0.9, "slots": {}}
        elif "friday" in t and "no" in t:
            return {"intent": "denial", "intent_confidence": 0.9, "slots": {"date": {"value": "Friday", "confidence": 0.9}}}
        elif "not want to cancel" in t:
            return {"intent": "denial", "intent_confidence": 0.85, "slots": {}}
        elif "hello" == t:
            return {"intent": "greeting", "intent_confidence": 0.99, "slots": {}}
        elif "heart attack" in t:
            return {"intent": "emergency_request", "intent_confidence": 0.95, "slots": {}}
        elif "nine eight" in t:
            return {"intent": "slot_answer", "intent_confidence": 0.88, "slots": {"phone": {"value": "9876543210", "confidence": 0.9}}}
        
        return {"intent": "out_of_domain", "intent_confidence": 0.5, "slots": {}}
        
    mock.predict = override_predict
    return mock

@pytest.fixture
def dst(mock_joint_nlu):
    d = DialogueStateTracker()
    # Force joint NLU enabled for tests
    cfg.nlu.joint_nlu_enabled = True
    return d, mock_joint_nlu

def test_joint_book_appointment(dst):
    tracker, mock_nlu = dst
    result = tracker.process_turn("I want to book cardiology", system_context="How can I help?", joint_nlu_engine=mock_nlu)
    assert result["intent"] == "book_appointment"
    assert result["slots"].get("service_type") == "cardiology"

def test_joint_slot_answer_service(dst):
    tracker, mock_nlu = dst
    tracker.intent = "book_appointment"
    tracker.state = State.COLLECTING
    result = tracker.process_turn("Cardiology", system_context="What service?", joint_nlu_engine=mock_nlu)
    
    # Assert intent is continuation and active intent is preserved
    assert result["intent"] == "continuation"
    assert result["_active_intent"] == "book_appointment"
    assert result["slots"].get("service_type") == "cardiology"

def test_joint_slot_answer_name(dst):
    tracker, mock_nlu = dst
    tracker.intent = "book_appointment"
    tracker.state = State.COLLECTING
    result = tracker.process_turn("John Smith", system_context="Your name?", joint_nlu_engine=mock_nlu)
    
    assert result["intent"] == "continuation"
    assert result["_active_intent"] == "book_appointment"
    assert result["slots"].get("name") == "John Smith"

def test_joint_confirmation_okay(dst):
    tracker, mock_nlu = dst
    result = tracker.process_turn("Okay", system_context="Is Thursday at 3pm fine?", joint_nlu_engine=mock_nlu)
    assert result["intent"] == "confirmation"

def test_joint_denial_with_slot(dst):
    tracker, mock_nlu = dst
    tracker.intent = "book_appointment"
    tracker.state = State.CONFIRMING
    result = tracker.process_turn("No, make it Friday", system_context="Is Thursday fine?", joint_nlu_engine=mock_nlu)
    assert getattr(tracker, "_last_classified_intent", None) is None  # Check it clears
    
    # We expose denial as intent for the router, while active intent remains
    assert result["intent"] == "correction"
    assert result["slots"].get("date") == "Friday"
    assert result["_active_intent"] == "book_appointment"

def test_joint_negation_trap(dst):
    tracker, mock_nlu = dst
    result = tracker.process_turn("I do NOT want to cancel", system_context="Cancel your appointment?", joint_nlu_engine=mock_nlu)
    assert result["intent"] == "denial"

def test_joint_greeting_no_context(dst):
    tracker, mock_nlu = dst
    result = tracker.process_turn("Hello", system_context="", joint_nlu_engine=mock_nlu)
    assert result["intent"] == "greeting"

def test_joint_emergency(dst):
    tracker, mock_nlu = dst
    result = tracker.process_turn("My father is having a heart attack", system_context="How can I help?", joint_nlu_engine=mock_nlu)
    assert result["intent"] == "emergency_request"

def test_joint_phone_asr(dst):
    tracker, mock_nlu = dst
    tracker.intent = "register_new_patient"
    tracker.state = State.COLLECTING
    result = tracker.process_turn("nine eight seven six five four three two one zero", system_context="Your phone?", joint_nlu_engine=mock_nlu)
    assert result["intent"] == "continuation"
    assert result["_active_intent"] == "register_new_patient"
    assert result["slots"].get("phone") == "9876543210"
