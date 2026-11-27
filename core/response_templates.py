"""
Response Templates — deterministic Path A & B responses.

Clinic facts (name, doctors, hours) come from config/entity_store rather than
being hardcoded here, so they cannot drift from the guardrails or the LLM
system prompt.
"""

from __future__ import annotations

import random
from datetime import datetime

from config import cfg
from core.nlu.entities import entity_store


# ── Human-friendly rendering ────────────────────────────────
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def say_date(value) -> str:
    """Render 2026-08-24 as 'Monday the 24th of August' for speech."""
    if not value:
        return "the scheduled date"
    try:
        d = datetime.strptime(str(value), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return str(value)
    today = datetime.now().date()
    delta = (d - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    return f"{_WEEKDAYS[d.weekday()]} the {_ordinal(d.day)} of {_MONTHS[d.month - 1]}"


def say_date_phrase(value) -> str:
    """Date rendered as a natural phrase: "tomorrow", not "on tomorrow"."""
    rendered = say_date(value)
    if rendered in ("today", "tomorrow") or rendered.startswith("the scheduled"):
        return rendered
    return f"on {rendered}"


def say_time(value) -> str:
    """Render 15:00 as '3 PM' and 15:30 as '3:30 PM' for speech."""
    if not value:
        return "the scheduled time"
    try:
        hour, minute = (int(p) for p in str(value).split(":")[:2])
    except (ValueError, IndexError, TypeError):
        return str(value)
    suffix = "AM" if hour < 12 else "PM"
    h12 = hour % 12 or 12
    return f"{h12} {suffix}" if minute == 0 else f"{h12}:{minute:02d} {suffix}"


def say_phone(value) -> str:
    digits = "".join(c for c in str(value or "") if c.isdigit())
    if len(digits) == 10:
        return f"{digits[:3]} {digits[3:6]} {digits[6:]}"
    return str(value or "")


def _name_suffix(slots: dict) -> str:
    name = slots.get("patient_name")
    return f", {name}" if name else ""


# ── Greetings, fillers, escalations ─────────────────────────
GREETINGS = [
    f"Hello! Welcome to {cfg.business.name}. How can I help you today?",
    f"Hi there! You've reached {cfg.business.name}. What can I do for you?",
    f"Good day! Welcome to {cfg.business.name}. How may I assist you?",
]

GOODBYES = [
    "Goodbye! Have a wonderful day.",
    "Thanks for calling. Take care!",
    "Bye now! Don't hesitate to call again.",
]

FILLERS = [
    "Let me check that for you.",
    "One moment please.",
    "Sure, let me look into that.",
    "Just a second.",
]

OUT_OF_DOMAIN = [
    f"I'm not sure I can help with that. Here at {cfg.business.name} I can book, "
    "cancel or reschedule appointments, and answer questions about the clinic.",
    f"That's outside what I can help with. I'm best at managing appointments for "
    f"{cfg.business.name}.",
]

ESCALATION = [
    "I'm sorry I'm having trouble helping you. Let me transfer you to a colleague "
    "who can assist you better.",
    "It seems like this needs a human touch. Let me connect you with someone who can help.",
]

HELP_RESPONSES = [
    "I can book, cancel or reschedule appointments, check an existing booking, and "
    "answer questions about our hours, location and doctors. What would you like to do?",
    "Here's what I can do: book a new appointment, cancel or reschedule one, check "
    "your appointment, or tell you about the clinic. How can I help?",
]

REPEAT_RESPONSES = [
    "Sure, let me say that again.",
    "Of course. Let me repeat that for you.",
]

SLOT_PROMPTS = {
    "date": [
        "What date would you like to come in?",
        "Which day works best for you?",
        "When would you like your appointment?",
    ],
    "time": [
        "What time would you prefer?",
        "What time works best for you?",
        "And what time should I book that for?",
    ],
    "patient_name": [
        "May I have the patient's name, please?",
        "Could you tell me the patient's name?",
        "And what name is the appointment under?",
    ],
    "phone": [
        "What's the best phone number to reach you on?",
        "Could I get a contact number for you?",
    ],
    "service_type": [
        "What type of appointment do you need? For example a general checkup, a "
        "consultation, or a specialty visit like cardiology.",
        "Which service are you booking? We offer general checkups, consultations, "
        "follow-ups and specialty visits.",
    ],
    "doctor_name": [
        f"Which doctor would you like to see? We have {entity_store.doctor_roster_text()}.",
        "Do you have a preferred doctor? "
        + ", ".join(entity_store.doctor_names()[:-1])
        + f" or {entity_store.doctor_names()[-1]} are available.",
    ],
}


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


def get_slot_prompt(slot_name: str) -> str:
    prompts = SLOT_PROMPTS.get(slot_name, [f"Could you tell me the {slot_name.replace('_', ' ')}?"])
    return random.choice(prompts)


# ── Confirmation prompts (spoken BEFORE committing an action) ──
def get_confirmation_prompt(intent: str, slots: dict) -> str:
    """Read the pending action back and ask for an explicit yes.

    This step did not exist before: the agent executed the moment the last
    slot was filled, so the caller never got a chance to catch a mistake.
    """
    date_s, time_s = say_date(slots.get("date")), say_time(slots.get("time"))
    service = slots.get("service_type", "")
    doctor = slots.get("doctor_name", "")
    name = slots.get("patient_name", "")

    if intent == "book_appointment":
        parts = [p for p in (service, say_date_phrase(slots.get("date")),
                             f"at {time_s}") if p]
        who = f" for {name}" if name else ""
        with_dr = f" with {doctor}" if doctor else ""
        return f"Just to confirm — {', '.join(parts)}{with_dr}{who}. Shall I book that?"

    if intent == "cancel_appointment":
        return (
            f"Just to confirm, you'd like me to cancel the appointment for "
            f"{name or 'you'}. Is that right?"
        )

    if intent == "reschedule_appointment":
        return f"So that's moving your appointment to {date_s} at {time_s}. Shall I do that?"

    if intent == "change_date":
        return f"I'll change your appointment to {date_s}. Is that correct?"

    if intent == "change_time":
        return f"I'll move your appointment to {time_s}. Is that correct?"

    if intent == "change_doctor":
        return f"I'll switch your appointment to {doctor}. Shall I go ahead?"

    if intent == "change_service":
        return f"I'll change your appointment to {service}. Is that right?"

    if intent == "register_new_patient":
        return (
            f"I'll register {name} with the number {say_phone(slots.get('phone'))}. "
            "Shall I go ahead?"
        )

    if intent == "update_contact_info":
        return (
            f"I'll update the number for {name} to {say_phone(slots.get('phone'))}. "
            "Is that correct?"
        )

    return "Shall I go ahead with that?"


# ── Action responses (spoken AFTER the caller confirms) ─────
def get_action_response(intent: str, slots: dict) -> str:
    """Confirmation of a completed action."""
    date_s, time_s = say_date(slots.get("date")), say_time(slots.get("time"))
    suffix = _name_suffix(slots)

    if intent == "book_appointment":
        return (
            f"Done. You're booked for {date_s} at {time_s}{suffix}. "
            "Is there anything else I can help with?"
        )
    if intent == "cancel_appointment":
        return f"That's cancelled{suffix}. Would you like to book a different time?"
    if intent in ("reschedule", "reschedule_appointment"):
        return f"All set — your appointment is now {date_s} at {time_s}. Anything else?"
    if intent in ("check_status", "check_appointment_status"):
        return f"Let me pull up that appointment{suffix}. One moment please."
    if intent == "check_my_appointments":
        return f"Let me look up your appointments{suffix}. One moment please."
    if intent == "change_date":
        return f"Updated — your appointment is now on {date_s}. Anything else?"
    if intent == "change_time":
        return f"Updated — your appointment is now at {time_s}. Anything else?"
    if intent == "change_doctor":
        return (
            f"Updated — you're now booked with {slots.get('doctor_name', 'the new doctor')}. "
            "Anything else?"
        )
    if intent == "change_service":
        return (
            f"Updated — your appointment is now for "
            f"{slots.get('service_type', 'the new service')}. Anything else?"
        )
    if intent == "register_new_patient":
        return f"Welcome{suffix}! You're registered. Is there anything else you need?"
    if intent == "update_contact_info":
        return f"Your contact details are updated{suffix}. Anything else?"
    if intent == "verify_patient_identity":
        return f"Thank you{suffix}. Your identity is verified."
    if intent == "doctor_availability":
        return "Let me check who's available. One moment please."
    if intent == "greeting":
        return get_greeting()
    if intent == "goodbye":
        return get_goodbye()
    return get_out_of_domain()
