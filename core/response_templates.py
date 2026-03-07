"""
Response Templates — Path A & B responses

Pre-built parameterized responses for deterministic paths:
- Path A: Direct action responses (all slots filled)
- Path B: Clarification prompts (missing slots)
- Greetings and goodbyes
"""

import random

# ── Path A: Direct Action Templates ─────────────────────────

BOOKING_CONFIRMED = [
    "Your appointment is booked for {date} at {time}. We'll see you then, {name}!",
    "All set! {name}, you're booked for {date} at {time}. Is there anything else?",
    "Done! Your appointment is confirmed for {date} at {time}. See you soon, {name}.",
]

CANCEL_CONFIRMED = [
    "Your appointment has been cancelled, {name}. Would you like to reschedule?",
    "All done, {name}. Your appointment has been cancelled. Is there anything else?",
]

RESCHEDULE_CONFIRMED = [
    "Your appointment has been rescheduled to {date} at {time}. All good?",
    "Done! You're now booked for {date} at {time} instead. Anything else?",
]

STATUS_RESPONSE = [
    "Let me check on that for you, {name}. One moment please.",
    "Checking your appointment status, {name}.",
]

# ── Path B: Clarification Prompts ───────────────────────────

SLOT_PROMPTS = {
    "date": [
        "What date would you like your appointment?",
        "Which day works best for you?",
        "When would you like to come in?",
    ],
    "time": [
        "What time would you prefer?",
        "What time works best for you?",
        "And what time should I book that for?",
    ],
    "name": [
        "May I have your name, please?",
        "Could you tell me your name?",
        "And your name is?",
    ],
    "phone": [
        "Could I get a phone number to reach you at?",
        "What's the best number to contact you?",
    ],
    "service_type": [
        "What type of service are you looking for?",
        "Which service would you like to book?",
    ],
}

# ── Greetings & Goodbyes ────────────────────────────────────

GREETINGS = [
    "Hello! Welcome. How can I help you today?",
    "Hi there! What can I do for you?",
    "Good day! How may I assist you?",
    "Welcome! I'm here to help. What do you need?",
]

GOODBYES = [
    "Goodbye! Have a wonderful day.",
    "Thanks for calling. Take care!",
    "Bye now! Don't hesitate to call again.",
]

# ── Fillers (for LLM latency masking) ───────────────────────

FILLERS = [
    "Let me check that for you.",
    "One moment please.",
    "Sure, let me look into that.",
    "Just a second.",
]

# ── Out of Domain ───────────────────────────────────────────

OUT_OF_DOMAIN = [
    "I'm not sure I can help with that. I can assist with booking, cancelling, or rescheduling appointments. Would you like help with any of those?",
    "That's outside what I can help with. I'm best at managing appointments. How can I assist you with that?",
]

# ── Escalation ──────────────────────────────────────────────

ESCALATION = [
    "I'm sorry I'm having trouble helping you. Let me transfer you to a human representative who can assist you better.",
    "It seems like this needs a human touch. Let me connect you with someone who can help.",
]

# ── Help ────────────────────────────────────────────────────

HELP_RESPONSES = [
    "I can help you book, cancel, or reschedule appointments. I can also check the status of an existing appointment. What would you like to do?",
    "Here's what I can do: book a new appointment, cancel an existing one, reschedule, or check your appointment status. How can I help?",
]

# ── Repeat ──────────────────────────────────────────────────

REPEAT_RESPONSES = [
    "Sure, let me say that again.",
    "Of course. Let me repeat that for you.",
]


# ── Template Functions ──────────────────────────────────────

def get_greeting() -> str:
    return random.choice(GREETINGS)

def get_goodbye() -> str:
    return random.choice(GOODBYES)

def get_filler() -> str:
    return random.choice(FILLERS)

def get_out_of_domain() -> str:
    return random.choice(OUT_OF_DOMAIN)

def get_escalation() -> str:
    return random.choice(ESCALATION)

def get_help() -> str:
    return random.choice(HELP_RESPONSES)

def get_repeat() -> str:
    return random.choice(REPEAT_RESPONSES)

def get_booking_confirmed(slots: dict) -> str:
    template = random.choice(BOOKING_CONFIRMED)
    return template.format(
        date=slots.get("date", "the scheduled date"),
        time=slots.get("time", "the scheduled time"),
        name=slots.get("name", ""),
    )

def get_cancel_confirmed(slots: dict) -> str:
    template = random.choice(CANCEL_CONFIRMED)
    return template.format(name=slots.get("name", ""))

def get_reschedule_confirmed(slots: dict) -> str:
    template = random.choice(RESCHEDULE_CONFIRMED)
    return template.format(
        date=slots.get("date", "the new date"),
        time=slots.get("time", "the new time"),
    )

def get_slot_prompt(slot_name: str) -> str:
    """Get a clarification prompt for a missing slot."""
    prompts = SLOT_PROMPTS.get(slot_name, [f"Could you provide the {slot_name}?"])
    return random.choice(prompts)

def get_action_response(intent: str, slots: dict) -> str:
    """Get a template response for a completed action."""
    if intent == "book_appointment":
        return get_booking_confirmed(slots)
    elif intent == "cancel_appointment":
        return get_cancel_confirmed(slots)
    elif intent == "reschedule":
        return get_reschedule_confirmed(slots)
    elif intent == "check_status":
        return random.choice(STATUS_RESPONSE).format(name=slots.get("name", ""))
    elif intent == "greeting":
        return get_greeting()
    elif intent == "goodbye":
        return get_goodbye()
    else:
        return get_out_of_domain()
