"""
Decision Engine — 5-Path Router

Routes each conversation turn to the optimal execution path
based on NLU output (intent, confidence, slots).

Paths:
  A: Direct    — confidence HIGH + all slots → template response
  B: Clarify   — confidence HIGH + missing slots → ask for slot
  C: LLM       — confidence MEDIUM or complex → send to Gemini
  D: RAG+LLM   — FAQ / knowledge question → LLM (v2: + retrieval)
  E: Escalate  — loop detected or frustration → human handoff
"""

import logging
from enum import Enum
from typing import Any, Optional

from config import cfg
from core.nlu.confidence import compute_confidence, confidence_level
from core.response_templates import (
    get_action_response,
    get_escalation,
    get_filler,
    get_greeting,
    get_goodbye,
    get_help,
    get_out_of_domain,
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
    ):
        self.path = path
        self.response = response
        self.needs_llm = needs_llm
        self.needs_tts = needs_tts
        self.filler = filler
        self.metadata = metadata or {}

    def __repr__(self):
        return f"RoutingResult(path={self.path.value}, response='{self.response[:50]}...')"


# Max failed turns before escalation
MAX_FAILED_TURNS = 3


def route(nlu_result: dict, last_response: str = "") -> RoutingResult:
    """
    Route a conversation turn based on NLU output.

    Args:
        nlu_result: Dict from DST.process_turn() with keys:
            intent, confidence, slots, state, missing_slots,
            turn_count, failed_turns
        last_response: The previous AI response to be used for repeat intents
        
    Returns:
        RoutingResult with path, response text, and metadata
    """
    intent = nlu_result.get("intent", "out_of_domain")
    raw_confidence = nlu_result.get("confidence", 0.0)
    slots = nlu_result.get("slots", {})
    missing = nlu_result.get("missing_slots", [])
    state = nlu_result.get("state", "idle")
    failed_turns = nlu_result.get("failed_turns", 0)
    turn_count = nlu_result.get("turn_count", 0)

    # Compute overall confidence
    overall_conf = compute_confidence(intent, raw_confidence, slots)
    level = confidence_level(overall_conf)

    # ── Path E: Escalation check ────────────────────────────
    if failed_turns >= MAX_FAILED_TURNS:
        logger.info(f"Route → Path E (escalate): {failed_turns} failed turns")
        return RoutingResult(
            path=Path.E_ESCALATE,
            response=get_escalation(),
            metadata={"reason": "loop_detected", "failed_turns": failed_turns},
        )

    # ── Greeting / Goodbye (always direct) ──────────────────
    if intent == "greeting":
        logger.info("Route → Path A (greeting)")
        return RoutingResult(
            path=Path.A_DIRECT,
            response=get_greeting(),
        )

    if intent == "goodbye":
        logger.info("Route → Path A (goodbye)")
        return RoutingResult(
            path=Path.A_DIRECT,
            response=get_goodbye(),
        )

    # ── Complaint / Transfer → Escalation ───────────────────
    if intent == "complaint":
        logger.info("Route → Path E (complaint → escalate)")
        return RoutingResult(
            path=Path.E_ESCALATE,
            response="I'm sorry you're having a bad experience. Let me connect you with a manager who can help resolve this.",
            metadata={"reason": "complaint"},
        )

    if intent == "transfer_to_human":
        logger.info("Route → Path E (transfer → human)")
        return RoutingResult(
            path=Path.E_ESCALATE,
            response=get_escalation(),
            metadata={"reason": "user_requested_transfer"},
        )

    # ── Clarification needed (dual low-confidence) ──────────
    if intent == "clarification_needed":
        active_intent = nlu_result.get("_active_intent", None)
        if missing:
            next_slot = missing[0]
            logger.info(f"Route → Path B (clarification needed, re-asking '{next_slot}')")
            return RoutingResult(
                path=Path.B_CLARIFY,
                response=f"I didn't quite catch that. {get_slot_prompt(next_slot)}",
                metadata={"missing_slot": next_slot, "all_missing": missing},
            )
        else:
            logger.info("Route → Path B (clarification needed, generic)")
            return RoutingResult(
                path=Path.B_CLARIFY,
                response="I'm sorry, I didn't understand that. Could you please rephrase?",
                metadata={"reason": "clarification_needed"},
            )

    # ── Emergency → immediate escalation ────────────────────
    if intent == "emergency_request":
        logger.info("Route → Path E (emergency → escalate)")
        return RoutingResult(
            path=Path.E_ESCALATE,
            response="This sounds urgent. Let me connect you with our emergency team right away. Please stay on the line.",
            metadata={"reason": "emergency"},
        )

    # ── Out of domain → LLM ────────────────────────────────
    if intent == "out_of_domain":
        logger.info(f"Route → Path C (out of domain)")
        return RoutingResult(
            path=Path.C_LLM,
            response="",
            needs_llm=True,
            filler=get_filler(),
            metadata={"reason": "out_of_domain"},
        )

    # ── FAQ-type intents → LLM ─────────────────────────────
    FAQ_INTENTS = {
        "clinic_hours", "clinic_location", "consultation_fee",
        "doctor_specialization", "services_available",
    }
    if intent in FAQ_INTENTS:
        logger.info(f"Route → Path D/C (FAQ: {intent} → LLM)")
        return RoutingResult(
            path=Path.D_RAG_LLM,
            response="",
            needs_llm=True,
            filler=get_filler(),
            metadata={"reason": "faq", "faq_type": intent},
        )

    # ── Conversational intents ──────────────────────────────

    # Confirmation — user said "yes" / "that's right"
    if intent in ("confirmation", "confirm_appointment"):
        active_intent = nlu_result.get("_active_intent", intent)
        if not missing and state == "confirming":
            logger.info(f"Route → Path A (confirmed): {active_intent}")
            return RoutingResult(
                path=Path.A_DIRECT,
                response=get_action_response(active_intent, slots),
                metadata={"action": active_intent, "slots": slots, "confirmed": True},
            )
        elif missing:
            next_slot = missing[0]
            logger.info(f"Route → Path B (confirmation but still missing '{next_slot}')")
            return RoutingResult(
                path=Path.B_CLARIFY,
                response=get_slot_prompt(next_slot),
                metadata={"missing_slot": next_slot, "all_missing": missing},
            )

    # Denial — user said "no" / "that's wrong"
    if intent == "denial":
        if missing:
            next_slot = missing[0]
            logger.info(f"Route → Path B (denial, re-asking '{next_slot}')")
            return RoutingResult(
                path=Path.B_CLARIFY,
                response=f"No problem. {get_slot_prompt(next_slot)}",
                metadata={"missing_slot": next_slot, "all_missing": missing},
            )
        else:
            logger.info("Route → Path B (denial, asking what to change)")
            return RoutingResult(
                path=Path.B_CLARIFY,
                response="I understand. What would you like to change?",
                metadata={"reason": "denial_correction"},
            )

    # Help
    if intent == "help":
        logger.info("Route → Path A (help)")
        return RoutingResult(
            path=Path.A_DIRECT,
            response=get_help(),
        )

    # Repeat
    if intent == "repeat":
        logger.info("Route → Path A (repeat)")
        return RoutingResult(
            path=Path.A_DIRECT,
            response=last_response if last_response else get_repeat(),
        )

    # Correction / Continuation — user providing slot data
    if intent in ("correction", "continuation", "information_provision"):
        active_intent = nlu_result.get("_active_intent", "book_appointment")
        if not missing:
            logger.info(f"Route → Path A (slot data provided, all filled): {active_intent}")
            return RoutingResult(
                path=Path.A_DIRECT,
                response=get_action_response(active_intent, slots),
                metadata={"action": active_intent, "slots": slots},
            )
        else:
            next_slot = missing[0]
            logger.info(f"Route → Path B (slot data, still need '{next_slot}')")
            return RoutingResult(
                path=Path.B_CLARIFY,
                response=get_slot_prompt(next_slot),
                metadata={"missing_slot": next_slot, "all_missing": missing},
            )

    # ── Action intents ──────────────────────────────────────
    ACTION_INTENTS = {
        "book_appointment", "cancel_appointment",
        "reschedule_appointment", "check_appointment_status",
        "check_my_appointments",
        "change_date", "change_doctor", "change_service",
        "register_new_patient", "update_contact_info",
        "verify_patient_identity",
    }

    if intent in ACTION_INTENTS:
        if not missing:
            logger.info(f"Route → Path A (direct): {intent} with {slots}")
            return RoutingResult(
                path=Path.A_DIRECT,
                response=get_action_response(intent, slots),
                metadata={"action": intent, "slots": slots},
            )
        else:
            next_slot = missing[0]
            logger.info(f"Route → Path B (clarify): asking for '{next_slot}'")
            return RoutingResult(
                path=Path.B_CLARIFY,
                response=get_slot_prompt(next_slot),
                metadata={"missing_slot": next_slot, "all_missing": missing},
            )

    # ── Low confidence → LLM fallback ──────────────────────
    logger.info(f"Route → Path C (LLM fallback): conf={overall_conf:.2f}")
    return RoutingResult(
        path=Path.C_LLM,
        response="",
        needs_llm=True,
        filler=get_filler(),
        metadata={"reason": "low_confidence", "confidence": overall_conf},
    )


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from config import cfg as _cfg
    _cfg.setup_logging()
    from core.nlu.dst import DialogueStateTracker

    print("🔀 Router — Interactive Test")
    print("   Type utterances to see routing decisions")
    print("   Type 'reset' to clear, 'quit' to exit")
    print()

    dst = DialogueStateTracker()

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        if text.lower() == "reset":
            dst.reset()
            print("  [State reset]\n")
            continue

        nlu_result = dst.process_turn(text)
        result = route(nlu_result)

        print(f"  Path:     {result.path.value}")
        print(f"  Response: \"{result.response}\"")
        if result.filler:
            print(f"  Filler:   \"{result.filler}\"")
        if result.needs_llm:
            print(f"  → Needs LLM call")
        if result.metadata:
            print(f"  Meta:     {result.metadata}")
        print()
