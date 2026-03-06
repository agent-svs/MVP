"""
Dialogue State Tracker — FSM (v2 — ML-calibrated)

Tracks the conversation state across turns:
- Current intent and filled/missing slots
- State transitions: GREETING → COLLECTING → CONFIRMING → EXECUTING → DONE
- Slot corrections ("actually, make it Tuesday")
- Turn counting for loop detection
- Handles new intents: complaint, transfer_to_human

Updated for ML-based NLU (DeBERTa intent + BERT NER slots).
"""

import logging
import re
from enum import Enum
from typing import Any, Optional

from core.nlu.intent import classify_intent
from core.nlu.slots import extract_slots

logger = logging.getLogger(__name__)

# Phrases that indicate the user is correcting a slot, not changing intent
CORRECTION_TRIGGERS = re.compile(
    r'\b(actually|no|change|not|instead|wrong|correct|make it|switch to)\b',
    re.IGNORECASE
)


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

        # Detect slot corrections ("actually, make it Tuesday")
        is_correction = self._is_slot_correction(text, new_slots)

        # Determine if this is a new intent or continuation
        if is_correction:
            # Slot correction — keep current intent, update slots
            logger.info(f"Slot correction detected: {new_slots}")
            self.slots.update(new_slots)
        elif self._is_new_conversation(new_intent, confidence):
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
        # First turn
        if self.intent is None:
            return True
        # Explicit conversational intents always start fresh
        if new_intent in ("greeting", "goodbye"):
            return True
        # Always allow transition FROM greeting/out_of_domain to real intents
        if self.intent in ("greeting", "out_of_domain") and new_intent not in ("greeting", "out_of_domain"):
            return True
        # Complaint and transfer always override
        if new_intent in ("complaint", "transfer_to_human"):
            return True
        # If currently collecting slots and the ML sees a different intent
        # but with low confidence, treat it as continuation (user is giving slots)
        if self.state == State.COLLECTING and new_intent != self.intent:
            if confidence < 0.4:
                return False  # Stay on current intent
        # Different intent with sufficient confidence → new conversation
        if new_intent != self.intent and confidence > 0.4:
            return True
        # If we're in DONE state, any intent is new
        if self.state == State.DONE:
            return True
        return False

    def _is_slot_correction(self, text: str, new_slots: dict) -> bool:
        """Detect if the user is correcting a previously provided slot."""
        if not self.intent or not new_slots:
            return False
        if self.state not in (State.COLLECTING, State.CONFIRMING):
            return False
        # Check for correction trigger words
        if CORRECTION_TRIGGERS.search(text):
            # Verify the new slots overlap with existing ones (same keys)
            has_overlap = any(k in self.slots for k in new_slots)
            if has_overlap:
                return True
        return False

    def _update_state(self) -> None:
        """Update the dialogue state based on current intent and slots."""
        if self.intent in ("greeting",):
            self.state = State.GREETING
        elif self.intent in ("goodbye",):
            self.state = State.DONE
        elif self.intent in ("complaint", "transfer_to_human"):
            # These escalation-type intents go to DONE (will be routed to Path E)
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
