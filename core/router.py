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
    get_out_of_domain,
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


def route(nlu_result: dict) -> RoutingResult:
    """
    Route a conversation turn based on NLU output.

    Args:
        nlu_result: Dict from DST.process_turn() with keys:
            intent, confidence, slots, state, missing_slots,
            turn_count, failed_turns

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

    # ── Out of domain → LLM ────────────────────────────────
    if intent == "out_of_domain":
        logger.info(f"Route → Path C (out of domain)")
        return RoutingResult(
            path=Path.C_LLM,
            response="",  # LLM will generate
            needs_llm=True,
            filler=get_filler(),
            metadata={"reason": "out_of_domain"},
        )

    # ── FAQ → LLM (Path D in v2, C for now) ────────────────
    if intent == "faq":
        logger.info("Route → Path D/C (FAQ → LLM)")
        return RoutingResult(
            path=Path.D_RAG_LLM,
            response="",
            needs_llm=True,
            filler=get_filler(),
            metadata={"reason": "faq"},
        )

    # ── Action intents (book, cancel, reschedule, status) ───
    if level == "high" and not missing:
        # Path A: All slots filled, high confidence → direct action
        logger.info(f"Route → Path A (direct): {intent} with {slots}")
        return RoutingResult(
            path=Path.A_DIRECT,
            response=get_action_response(intent, slots),
            metadata={"action": intent, "slots": slots},
        )

    if level in ("high", "medium") and missing:
        # Path B: Missing slots → clarify
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
