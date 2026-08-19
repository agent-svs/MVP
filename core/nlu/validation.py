"""
Business-rule slot validation.

Deliberately separate from guardrails. The distinction matters:

  * validation (here)  — "this date is a Saturday". Produces a REPAIR
    instruction: which slot to clear, and what to say to the caller. The
    dialogue keeps moving.
  * guardrails         — "this outgoing text leaks an SSN". Produces a
    sanitised response.

Previously guardrails did both, so an invalid slot replaced the agent's next
clarification question with a refusal. The caller was then never asked
anything again and could not correct the bad value — a permanent deadlock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

from config import cfg
from core.nlu.entities import entity_store

logger = logging.getLogger(__name__)

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"]


@dataclass
class SlotProblem:
    """One invalid slot, with everything needed to repair it in dialogue."""

    slot: str
    reason: str            # machine-readable: past_date, closed_day, ...
    message: str           # spoken explanation + what to do next
    clear_slot: bool = True

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"SlotProblem({self.slot}, {self.reason})"


@dataclass
class ValidationResult:
    problems: list[SlotProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def slots_to_clear(self) -> list[str]:
        return [p.slot for p in self.problems if p.clear_slot]

    def first_message(self) -> Optional[str]:
        return self.problems[0].message if self.problems else None


def _value_of(v: Any) -> Any:
    """Slots may be raw values or {"value": ...} dicts."""
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def _next_open_day(from_date: date) -> date:
    d = from_date
    for _ in range(14):
        if d.weekday() in cfg.business.open_days:
            return d
        d = date.fromordinal(d.toordinal() + 1)
    return from_date


def validate_date(date_str: str) -> list[SlotProblem]:
    b = cfg.business
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return [SlotProblem(
            "date", "unparseable",
            "I didn't catch that date. What day would you like to come in?",
        )]

    today = datetime.now().date()
    if d < today:
        return [SlotProblem(
            "date", "past_date",
            "That date has already passed. What day would you like instead?",
        )]

    if (d - today).days > b.max_booking_days_ahead:
        return [SlotProblem(
            "date", "too_far_ahead",
            f"We only book up to {b.max_booking_days_ahead // 30} months ahead. "
            "Could you pick a nearer date?",
        )]

    if d.weekday() not in b.open_days:
        nxt = _next_open_day(d)
        return [SlotProblem(
            "date", "closed_day",
            f"We're closed on {_DAY_NAMES[d.weekday()]}s. "
            f"{_DAY_NAMES[nxt.weekday()]} the {nxt.day}th works — shall I use that, "
            "or would you prefer another day?",
        )]
    return []


def validate_time(time_str: str) -> list[SlotProblem]:
    b = cfg.business
    try:
        parts = str(time_str).split(":")
        hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError
    except (ValueError, IndexError, TypeError):
        return [SlotProblem(
            "time", "unparseable",
            "I didn't catch that time. What time works for you?",
        )]

    if hour < b.hours_start or hour >= b.hours_end:
        return [SlotProblem(
            "time", "closed_hours",
            f"We're open {b.hours_human}. What time in that window suits you?",
        )]
    return []


def validate_doctor(name: str) -> list[SlotProblem]:
    if not name:
        return []
    if entity_store.resolve_doctor(str(name)):
        return []
    return [SlotProblem(
        "doctor_name", "unknown_doctor",
        f"I don't have a {name} here. Our doctors are "
        f"{entity_store.doctor_roster_text()}. Who would you like to see?",
    )]


def validate_service(service: str) -> list[SlotProblem]:
    if not service:
        return []
    if entity_store.resolve_service(str(service)):
        return []
    return [SlotProblem(
        "service_type", "unknown_service",
        f"We don't offer {service}. We do general checkups, consultations, "
        "follow-ups and specialty visits. What do you need?",
    )]


def validate_phone(phone: str) -> list[SlotProblem]:
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if len(digits) != 10:
        return [SlotProblem(
            "phone", "bad_phone",
            "That doesn't look like a complete number. "
            "Could you give me your ten digit phone number?",
        )]
    return []


_VALIDATORS = {
    "date": validate_date,
    "time": validate_time,
    "doctor_name": validate_doctor,
    "service_type": validate_service,
    "phone": validate_phone,
}


def validate_slots(slots: dict[str, Any]) -> ValidationResult:
    """Validate every known slot against business rules."""
    problems: list[SlotProblem] = []
    for key, validator in _VALIDATORS.items():
        raw = slots.get(key)
        value = _value_of(raw)
        if value in (None, ""):
            continue
        problems.extend(validator(value))
    if problems:
        logger.info(f"Slot validation problems: {problems}")
    return ValidationResult(problems=problems)
