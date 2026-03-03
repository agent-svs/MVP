"""
Tests for Decision Engine — Router and Response Templates.
"""


# ── Response Template Tests ──────────────────────────────────

def test_template_greeting():
    from core.response_templates import get_greeting
    greeting = get_greeting()
    assert isinstance(greeting, str)
    assert len(greeting) > 5


def test_template_goodbye():
    from core.response_templates import get_goodbye
    goodbye = get_goodbye()
    assert isinstance(goodbye, str)
    assert len(goodbye) > 5


def test_template_filler():
    from core.response_templates import get_filler
    filler = get_filler()
    assert isinstance(filler, str)


def test_template_booking_confirmed():
    from core.response_templates import get_booking_confirmed
    text = get_booking_confirmed({
        "date": "2026-03-10",
        "time": "15:00",
        "name": "John"
    })
    assert "2026-03-10" in text
    assert "15:00" in text
    assert "John" in text


def test_template_slot_prompt():
    from core.response_templates import get_slot_prompt
    date_prompt = get_slot_prompt("date")
    assert "date" in date_prompt.lower() or "day" in date_prompt.lower() or "when" in date_prompt.lower()
    time_prompt = get_slot_prompt("time")
    assert "time" in time_prompt.lower()


def test_template_action_response():
    from core.response_templates import get_action_response
    r = get_action_response("greeting", {})
    assert isinstance(r, str) and len(r) > 3
    r2 = get_action_response("goodbye", {})
    assert isinstance(r2, str) and len(r2) > 3


# ── Router Tests ─────────────────────────────────────────────

def test_route_greeting():
    from core.router import route, Path
    result = route({
        "intent": "greeting", "confidence": 0.9,
        "slots": {}, "state": "greeting",
        "missing_slots": [], "turn_count": 1, "failed_turns": 0,
    })
    assert result.path == Path.A_DIRECT
    assert result.needs_llm is False
    assert len(result.response) > 0


def test_route_goodbye():
    from core.router import route, Path
    result = route({
        "intent": "goodbye", "confidence": 0.9,
        "slots": {}, "state": "done",
        "missing_slots": [], "turn_count": 1, "failed_turns": 0,
    })
    assert result.path == Path.A_DIRECT


def test_route_path_a_all_slots():
    from core.router import route, Path
    result = route({
        "intent": "book_appointment", "confidence": 0.9,
        "slots": {"date": "2026-03-10", "time": "15:00", "name": "John"},
        "state": "confirming", "missing_slots": [],
        "turn_count": 3, "failed_turns": 0,
    })
    assert result.path == Path.A_DIRECT
    assert "John" in result.response
    assert result.needs_llm is False


def test_route_path_b_missing_slots():
    from core.router import route, Path
    result = route({
        "intent": "book_appointment", "confidence": 0.9,
        "slots": {"date": "2026-03-10"},
        "state": "collecting", "missing_slots": ["time", "name"],
        "turn_count": 2, "failed_turns": 0,
    })
    assert result.path == Path.B_CLARIFY
    assert result.needs_llm is False
    # Should ask for the first missing slot (time)
    assert "time" in result.response.lower()


def test_route_path_c_low_confidence():
    from core.router import route, Path
    result = route({
        "intent": "book_appointment", "confidence": 0.2,
        "slots": {}, "state": "collecting",
        "missing_slots": ["date", "time", "name"],
        "turn_count": 1, "failed_turns": 0,
    })
    assert result.path == Path.C_LLM
    assert result.needs_llm is True
    assert result.filler is not None


def test_route_path_c_out_of_domain():
    from core.router import route, Path
    result = route({
        "intent": "out_of_domain", "confidence": 0.3,
        "slots": {}, "state": "collecting",
        "missing_slots": [], "turn_count": 1, "failed_turns": 0,
    })
    assert result.path == Path.C_LLM
    assert result.needs_llm is True


def test_route_path_d_faq():
    from core.router import route, Path
    result = route({
        "intent": "faq", "confidence": 0.8,
        "slots": {}, "state": "collecting",
        "missing_slots": [], "turn_count": 1, "failed_turns": 0,
    })
    assert result.path == Path.D_RAG_LLM
    assert result.needs_llm is True


def test_route_path_e_escalation():
    from core.router import route, Path
    result = route({
        "intent": "book_appointment", "confidence": 0.5,
        "slots": {}, "state": "collecting",
        "missing_slots": ["date"], "turn_count": 5, "failed_turns": 3,
    })
    assert result.path == Path.E_ESCALATE
    assert "human" in result.response.lower() or "transfer" in result.response.lower()
