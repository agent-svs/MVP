"""
Decision Engine — 5-Path Router.

Routes a turn to the cheapest path that can answer it correctly.

  A: Direct    — deterministic template response (greeting, confirmed action)
  B: Clarify   — a slot is missing or invalid; ask for it
  C: LLM       — open-ended / low confidence → Gemini
  D: RAG+LLM   — FAQ / knowledge question → Gemini with clinic knowledge
  E: Escalate  — loop, complaint, emergency, or explicit transfer request

The router is a pure function of the DST result. It never performs the action
itself; it returns `execute_action` so the pipeline can commit the action and
tell the DST to move to DONE exactly once.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from core.nlu.confidence import compute_confidence, confidence_level
from core.nlu.dst import ACTION_INTENTS, READONLY_INTENTS
from core.response_templates import (
    get_action_response,
    get_confirmation_prompt,
    get_escalation,
    get_filler,
    get_goodbye,
    get_greeting,
    get_help,
    get_repeat,
    get_slot_prompt,
)

logger = logging.getLogger(__name__)


class Path(str, Enum):
    """Routing paths."""

    A_DIRECT = "direct"
    B_CLARIFY = "clarify"
    C_LLM = "llm"
    D_RAG_LLM = "rag_llm"
    E_ESCALATE = "escalate"


class RoutingResult:
    """Result of a routing decision."""

    def __init__(
        self,
        path: Path,
        response: str,
        needs_llm: bool = False,
        needs_tts: bool = True,
        filler: Optional[str] = None,
        metadata: Optional[dict] = None,
        execute_action: Optional[str] = None,
        end_conversation: bool = False,
    ) -> None:
        self.path = path
        self.response = response
        self.needs_llm = needs_llm
        self.needs_tts = needs_tts
        self.filler = filler
        self.metadata = metadata or {}
        # Set when this turn commits a real action; the pipeline calls
        # dst.mark_action_complete() so it can never fire twice.
        self.execute_action = execute_action
        self.end_conversation = end_conversation

    def __repr__(self) -> str:
        return f"RoutingResult(path={self.path.value}, response='{self.response[:50]}...')"


MAX_FAILED_TURNS = 3

FAQ_INTENTS = frozenset({
    "clinic_hours", "clinic_location", "consultation_fee",
    "doctor_specialization", "services_available", "doctor_availability",
})


def _clarify(missing: list[str], prefix: str = "") -> RoutingResult:
    slot = missing[0]
    text = get_slot_prompt(slot)
    return RoutingResult(
        path=Path.B_CLARIFY,
        response=f"{prefix} {text}".strip(),
        metadata={"missing_slot": slot, "all_missing": missing},
    )


def route(nlu_result: dict, last_response: str = "") -> RoutingResult:
    """Route a conversation turn based on DST output."""
    intent = nlu_result.get("intent", "out_of_domain")
    active_intent = nlu_result.get("_active_intent") or intent
    raw_confidence = nlu_result.get("confidence", 0.0)
    slots = nlu_result.get("slots", {})
    missing = nlu_result.get("missing_slots", [])
    state = nlu_result.get("state", "idle")
    failed_turns = nlu_result.get("failed_turns", 0)
    repair_message = nlu_result.get("repair_message")
    pending_slot = nlu_result.get("pending_slot")

    overall_conf = compute_confidence(intent, raw_confidence, slots, pending_slot=pending_slot)
    level = confidence_level(overall_conf)

    # ── Path E: escalation ──────────────────────────────────
    if failed_turns >= MAX_FAILED_TURNS:
        logger.info(f"Route → E (escalate): {failed_turns} failed turns")
        return RoutingResult(
            Path.E_ESCALATE, get_escalation(),
            metadata={"reason": "loop_detected", "failed_turns": failed_turns},
        )

    if intent == "emergency_request":
        logger.info("Route → E (emergency)")
        return RoutingResult(
            Path.E_ESCALATE,
            "This sounds like an emergency. Please hang up and dial 911 right now. "
            "If you'd rather I connect you to our on-call team, stay on the line.",
            metadata={"reason": "emergency"}, end_conversation=True,
        )

    if intent == "complaint":
        logger.info("Route → E (complaint)")
        return RoutingResult(
            Path.E_ESCALATE,
            "I'm sorry you've had a bad experience. Let me connect you with a "
            "manager who can help resolve this.",
            metadata={"reason": "complaint"}, end_conversation=True,
        )

    if intent == "transfer_to_human":
        logger.info("Route → E (transfer)")
        return RoutingResult(
            Path.E_ESCALATE, get_escalation(),
            metadata={"reason": "user_requested_transfer"}, end_conversation=True,
        )

    # ── Invalid slot → speak the repair and re-ask ──────────
    # This used to be a guardrail block that DISCARDED the agent's question,
    # leaving the caller unable to fix the value. Now it is a normal clarify.
    if repair_message:
        logger.info("Route → B (slot repair)")
        return RoutingResult(
            Path.B_CLARIFY, repair_message,
            metadata={"reason": "slot_repair", "missing_slot": missing[0] if missing else None},
        )

    # ── Terminal / social intents ───────────────────────────
    if intent == "greeting":
        return RoutingResult(Path.A_DIRECT, get_greeting())

    if intent == "goodbye":
        return RoutingResult(Path.A_DIRECT, get_goodbye(), end_conversation=True)

    if intent == "help":
        return RoutingResult(Path.A_DIRECT, get_help())

    if intent == "repeat":
        return RoutingResult(Path.A_DIRECT, last_response or get_repeat())

    if intent == "clarification_needed":
        if missing:
            return _clarify(missing, "Sorry, I didn't quite catch that.")
        return RoutingResult(
            Path.B_CLARIFY,
            "Sorry, I didn't quite catch that. Could you say it another way?",
            metadata={"reason": "clarification_needed"},
        )

    # ── EXECUTING: the caller confirmed; commit the action ──
    if state == "executing":
        logger.info(f"Route → A (execute {active_intent})")
        return RoutingResult(
            Path.A_DIRECT,
            get_action_response(active_intent, slots),
            metadata={"action": active_intent, "slots": slots, "confirmed": True},
            execute_action=active_intent,
        )

    # ── CONFIRMING: read it back, wait for a yes ────────────
    if state == "confirming":
        logger.info(f"Route → A (confirm {active_intent})")
        return RoutingResult(
            Path.A_DIRECT,
            get_confirmation_prompt(active_intent, slots),
            metadata={"awaiting_confirmation": True, "action": active_intent},
        )

    # ── Denial outside a confirmation ───────────────────────
    if intent == "denial":
        # After a completed action "no" means "nothing else, thanks".
        if state == "done" or nlu_result.get("completed_action"):
            return RoutingResult(
                Path.A_DIRECT, get_goodbye(),
                metadata={"reason": "declined_further_help"}, end_conversation=True,
            )
        # No action in flight — "no thanks" means we're done, not "change what?".
        from core.nlu.dst import ACTION_INTENTS as _ACTIONS

        if active_intent not in _ACTIONS and not nlu_result.get("declined_confirmation"):
            return RoutingResult(
                Path.A_DIRECT, get_goodbye(),
                metadata={"reason": "declined_no_task"}, end_conversation=True,
            )
        if nlu_result.get("declined_confirmation"):
            return RoutingResult(
                Path.B_CLARIFY,
                "No problem — what would you like to change? "
                "I can update the date, the time, the service or the name.",
                metadata={"reason": "confirmation_declined"},
            )
        if missing:
            return _clarify(missing, "No problem.")
        return RoutingResult(
            Path.B_CLARIFY, "No problem — what would you like to change?",
            metadata={"reason": "denial_correction"},
        )

    # ── Bare confirmation with nothing pending ──────────────
    if intent in ("confirmation", "confirm_appointment"):
        if missing:
            return _clarify(missing)
        return RoutingResult(
            Path.A_DIRECT, "Great. Is there anything else I can help you with?"
        )

    # ── FAQ / knowledge → LLM with clinic knowledge ─────────
    if intent in FAQ_INTENTS:
        logger.info(f"Route → D (FAQ: {intent})")
        return RoutingResult(
            Path.D_RAG_LLM, "", needs_llm=True, filler=get_filler(),
            metadata={"reason": "faq", "faq_type": intent},
        )

    # ── Task intents (action + read-only) ───────────────────
    task_intent = active_intent if intent in (
        "correction", "continuation", "information_provision"
    ) else intent

    if task_intent in ACTION_INTENTS or task_intent in READONLY_INTENTS:
        if missing:
            logger.info(f"Route → B (clarify '{missing[0]}' for {task_intent})")
            return _clarify(missing)
        if task_intent in READONLY_INTENTS:
            logger.info(f"Route → A (read-only {task_intent})")
            return RoutingResult(
                Path.A_DIRECT, get_action_response(task_intent, slots),
                metadata={"action": task_intent, "slots": slots},
                execute_action=task_intent,
            )
        # Action with everything filled but state not yet CONFIRMING.
        logger.info(f"Route → A (confirm {task_intent})")
        return RoutingResult(
            Path.A_DIRECT, get_confirmation_prompt(task_intent, slots),
            metadata={"awaiting_confirmation": True, "action": task_intent},
        )

    # ── Out of domain / everything else → LLM ───────────────
    logger.info(f"Route → C (LLM fallback): intent={intent} conf={overall_conf:.2f} [{level}]")
    return RoutingResult(
        Path.C_LLM, "", needs_llm=True, filler=get_filler(),
        metadata={
            "reason": "out_of_domain" if intent == "out_of_domain" else "low_confidence",
            "confidence": overall_conf,
            "confidence_level": level,
        },
    )
