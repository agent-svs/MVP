"""
Dialogue State Tracker — FSM

Tracks the conversation state across turns:
- Current intent and filled/missing slots
- State transitions: GREETING → COLLECTING → CONFIRMING → EXECUTING → DONE
- Slot corrections ("actually, make it Tuesday")
- Turn counting for loop detection
"""

import logging
from enum import Enum
from typing import Any, Optional

from core.nlu.intent import classify_intent
from core.nlu.slots import extract_slots

logger = logging.getLogger(__name__)


class State(str, Enum):
    """Dialogue states."""
    IDLE = "idle"
    GREETING = "greeting"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    DONE = "done"


# Required slots per intent
REQUIRED_SLOTS = {
    "book_appointment": ["date", "time", "name"],
    "cancel_appointment": ["name"],
    "reschedule": ["date", "time"],
    "check_status": ["name"],
}


class DialogueStateTracker:
    """
    Tracks conversation state across multiple turns.

    Manages intent tracking, slot filling, and state transitions.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset to initial state."""
        self.state: State = State.IDLE
        self.intent: Optional[str] = None
        self.intent_confidence: float = 0.0
        self.slots: dict[str, Any] = {}
        self.turn_count: int = 0
        self.failed_turns: int = 0
        self._history: list[dict] = []

    def process_turn(self, text: str) -> dict:
        """
        Process a user utterance and update state.

        Args:
            text: User utterance

        Returns:
            Dict with: intent, confidence, slots, state, missing_slots, action
        """
        self.turn_count += 1

        # Classify intent
        new_intent, confidence = classify_intent(text)

        # Extract slots from this utterance
        new_slots = extract_slots(text)

        # Determine if this is a new intent or continuation
        if self._is_new_conversation(new_intent, confidence):
            # New intent — reset slots
            self.intent = new_intent
            self.intent_confidence = confidence
            self.slots = new_slots
            self.failed_turns = 0
        else:
            # Continuation — merge slots (new values override old)
            self.slots.update(new_slots)
            # Update confidence if higher
            if confidence > self.intent_confidence:
                self.intent_confidence = confidence

        # Update state
        self._update_state()

        # Build result
        missing = self.missing_slots
        result = {
            "intent": self.intent,
            "confidence": self.intent_confidence,
            "slots": dict(self.slots),
            "state": self.state.value,
            "missing_slots": missing,
            "turn_count": self.turn_count,
            "failed_turns": self.failed_turns,
        }

        # Save to history
        self._history.append({
            "turn": self.turn_count,
            "text": text,
            "result": result,
        })

        logger.info(
            f"DST turn {self.turn_count}: intent={self.intent} "
            f"state={self.state.value} slots={self.slots} "
            f"missing={missing}"
        )

        return result

    def _is_new_conversation(self, new_intent: str, confidence: float) -> bool:
        """Check if this is a new conversation intent vs continuation."""
        # First turn or explicit new intent with high confidence
        if self.intent is None:
            return True
        if new_intent in ("greeting", "goodbye"):
            return True
        # Always allow transition FROM greeting/out_of_domain to real intents
        if self.intent in ("greeting", "out_of_domain") and new_intent not in ("greeting", "out_of_domain"):
            return True
        if new_intent != self.intent and confidence > 0.3:
            return True
        # If we're in DONE state, any intent is new
        if self.state == State.DONE:
            return True
        return False

    def _update_state(self) -> None:
        """Update the dialogue state based on current intent and slots."""
        if self.intent in ("greeting",):
            self.state = State.GREETING
        elif self.intent in ("goodbye",):
            self.state = State.DONE
        elif self.intent in ("out_of_domain", "faq"):
            self.state = State.COLLECTING
        elif self.intent in REQUIRED_SLOTS:
            missing = self.missing_slots
            if not missing:
                self.state = State.CONFIRMING
            else:
                self.state = State.COLLECTING
        else:
            self.state = State.COLLECTING

    @property
    def missing_slots(self) -> list[str]:
        """Get list of required slots that are not yet filled."""
        if self.intent not in REQUIRED_SLOTS:
            return []
        required = REQUIRED_SLOTS[self.intent]
        return [s for s in required if s not in self.slots]

    @property
    def all_slots_filled(self) -> bool:
        """Check if all required slots are filled."""
        return len(self.missing_slots) == 0

    @property
    def history(self) -> list[dict]:
        return list(self._history)


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from config import cfg
    cfg.setup_logging()

    print("🧠 NLU Brain — Interactive Test")
    print("   Type utterances to see intent, slots, and state")
    print("   Type 'reset' to clear state, 'quit' to exit")
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

        result = dst.process_turn(text)
        print(f"  Intent:   {result['intent']} ({result['confidence']:.2f})")
        print(f"  State:    {result['state']}")
        print(f"  Slots:    {result['slots']}")
        if result['missing_slots']:
            print(f"  Missing:  {result['missing_slots']}")
        print()
