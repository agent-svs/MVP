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
    "book_appointment": ["service_type", "date", "time", "name"],
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
        # Expected-slot tracking: what question did the AI just ask?
        self.pending_slot: Optional[str] = None
        self.last_question: Optional[str] = None

    def set_expectation(self, slot_name: str, question: str = "") -> None:
        """Record what slot the AI just asked for.

        Called by the pipeline after the router returns a clarify response.
        On the next turn, process_turn() will prioritize extracting this slot
        before running intent classification.

        Args:
            slot_name: The slot being asked for (e.g., 'date', 'time', 'name')
            question: The question text the AI asked
        """
        self.pending_slot = slot_name
        self.last_question = question
        logger.debug(f"Expectation set: pending_slot='{slot_name}'")

    def process_turn(self, text: str) -> dict:
        """
        Process a user utterance and update state.

        If the AI just asked for a specific slot (pending_slot is set),
        we first try to extract that slot from the utterance. If found,
        we fill it and skip intent classification — the user was just
        answering our question.

        Args:
            text: User utterance

        Returns:
            Dict with: intent, confidence, slots, state, missing_slots, action
        """
        self.turn_count += 1

        # Extract slots from this utterance (always do this)
        new_slots = extract_slots(text)

        # ── Expected-slot shortcut ──────────────────────────────
        # If we asked for a specific slot and the user provided it,
        # skip intent classification — this is just an answer.
        if self._try_fill_expected_slot(text, new_slots):
            # Merge any other extracted slots too
            self.slots.update(new_slots)
            self._update_state()
            return self._build_result(text)

        # ── Normal flow: classify intent ────────────────────────
        new_intent, confidence = classify_intent(text)
        self._last_classified_intent = new_intent

        # Detect slot corrections ("actually, make it Tuesday")
        is_correction = self._is_slot_correction(text, new_slots, new_intent)

        # Determine if this is a new intent or continuation
        if is_correction:
            # Slot correction — keep current intent, update slots
            logger.info(f"Slot correction detected: {new_slots}")
            self.slots.update(new_slots)
            
            # Since we're keeping the current intent but the user said "no" or "make it",
            # we need to make sure _build_result doesn't treat it as the original intent.
            # We want to treat this turn as a slot correction / continuation.
            self._last_classified_intent = "correction"
        elif self._is_new_conversation(new_intent, confidence):
            # New intent — preserve compatible slots if transferring between action intents
            preserved_slots = {}
            if self.intent in REQUIRED_SLOTS and new_intent in REQUIRED_SLOTS:
                for key, val in self.slots.items():
                    if key in REQUIRED_SLOTS[new_intent]:
                        preserved_slots[key] = val
                        
            self.intent = new_intent
            self.intent_confidence = confidence
            self.slots = preserved_slots
            self.slots.update(new_slots)
            self.failed_turns = 0
        else:
            # Continuation — merge slots (new values override old)
            self.slots.update(new_slots)
            # Update confidence if higher
            if confidence > self.intent_confidence:
                self.intent_confidence = confidence
            
            # Since this is a continuation (not a new conversation), we want to
            # keep the active intent intact. Track this so _build_result knows.
            # The only exception is if it was already marked as something else
            # (like 'continuation' from _is_new_conversation).
            if not getattr(self, '_last_classified_intent', None) == "continuation":
                 self._last_classified_intent = "continuation"

        # Clear pending expectation (we've processed this turn)
        self.pending_slot = None
        self.last_question = None

        # Update state
        self._update_state()

        return self._build_result(text)

    def _try_fill_expected_slot(self, text: str, new_slots: dict) -> bool:
        """Try to fill the pending expected slot.

        If the AI asked 'What date?' and the user said 'Tomorrow at 3 PM',
        we find 'date' in new_slots and fill it without re-classifying intent.

        Returns True if the expected slot was filled (skip intent classification).
        """
        if not self.pending_slot:
            return False

        expected = self.pending_slot

        # Check if the expected slot was extracted
        if expected in new_slots:
            logger.info(
                f"Expected slot '{expected}' filled: {new_slots[expected]} "
                f"(skipping intent classification)"
            )
            self.pending_slot = None
            self.last_question = None
            return True

        # If user gave other slots (provided info but not what we asked),
        # still treat as continuation if we're collecting
        if new_slots and self.state == State.COLLECTING:
            logger.info(
                f"Expected '{expected}' not found, but got {list(new_slots.keys())} "
                f"— treating as continuation"
            )
            self.pending_slot = None
            self.last_question = None
            return True

        # Nothing relevant extracted — fall through to normal intent flow
        # (user might have said something unrelated like 'goodbye')
        return False

    def _build_result(self, text: str) -> dict:
        """Build and log the NLU result dict."""
        missing = self.missing_slots

        # For conversational intents (confirmation/denial/etc.), expose
        # the classified intent so the router can differentiate, while
        # keeping _active_intent as the real task intent.
        CONVERSATIONAL_INTENTS = {
            "confirmation", "denial", "information_provision", "repeat", "help", "correction", "continuation",
        }
        classified = getattr(self, '_last_classified_intent', None)
        if classified and classified in CONVERSATIONAL_INTENTS:
            output_intent = classified
            active_intent = self.intent  # real task (e.g., book_appointment)
        else:
            output_intent = self.intent
            active_intent = self.intent

        result = {
            "intent": output_intent,
            "_active_intent": active_intent,
            "confidence": self.intent_confidence,
            "slots": {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in self.slots.items()},
            "slot_confidences": {k: v["confidence"] if isinstance(v, dict) and "confidence" in v else 1.0 for k, v in self.slots.items()},
            "state": self.state.value,
            "missing_slots": missing,
            "turn_count": self.turn_count,
            "failed_turns": self.failed_turns,
        }

        self._history.append({
            "turn": self.turn_count,
            "text": text,
            "result": result,
        })

        logger.info(
            f"DST turn {self.turn_count}: intent={output_intent} "
            f"active={active_intent} state={self.state.value} "
            f"slots={self.slots} missing={missing}"
        )

        # Reset classified intent tracker
        self._last_classified_intent = None

        return result

    def _is_new_conversation(self, new_intent: str, confidence: float) -> bool:
        """Check if this is a new conversation intent vs continuation."""
        # First turn
        if self.intent is None:
            return True

        # Conversational intents are NEVER new conversations — they're
        # responses to what the AI said ("yes", "no", "5 PM", "say again")
        CONTINUATION_INTENTS = {
            "confirmation", "denial", "information_provision", "repeat",
        }
        if new_intent in CONTINUATION_INTENTS:
            return False

        # Explicit conversational intents always start fresh
        if new_intent in ("greeting", "goodbye"):
            return True
        # Always allow transition FROM greeting/out_of_domain to real intents
        if self.intent in ("greeting", "out_of_domain") and new_intent not in ("greeting", "out_of_domain"):
            return True
        # Complaint and transfer always override
        if new_intent in ("complaint", "transfer_to_human"):
            return True
        # "help" only starts new if we're idle or done
        if new_intent == "help":
            return self.state in (State.IDLE, State.DONE, State.GREETING)
            
        # If currently collecting slots and the ML sees a different intent
        # but with low confidence, treat it as a continuation of the active intent
        if self.state == State.COLLECTING and new_intent != self.intent:
            if confidence < 0.4:
                self._last_classified_intent = "continuation"
                return False  # Stay on current intent
                
        # If the user gives a greeting or out_of_domain response but we are actively collecting slots,
        # treat it as a continuation to prevent dropping context. User might have just been confused or said "Oh hello".
        if self.state in (State.COLLECTING, State.CONFIRMING) and new_intent in ("out_of_domain", "greeting"):
            if self.missing_slots:
                 self._last_classified_intent = "continuation"
                 return False

        # Different intent with sufficient confidence → new conversation
        if new_intent != self.intent and confidence > 0.4:
            return True
        # If we're in DONE state, any intent is new
        if self.state == State.DONE:
            return True
        return False

    def _is_slot_correction(self, text: str, new_slots: dict, new_intent: str) -> bool:
        """Detect if the user is correcting a previously provided slot."""
        if not self.intent:
            return False
            
        # If the user explicitly denies and provides slots, it's a correction
        if new_intent == "denial" and new_slots:
            return True
            
        if not new_slots:
            return False
            
        if self.state not in (State.COLLECTING, State.CONFIRMING):
            return False
            
        # Check for correction trigger words
        if CORRECTION_TRIGGERS.search(text):
            # Verify the new slots overlap with existing ones (same keys), OR it's just a general correction
            has_overlap = any(k in self.slots for k in new_slots)
            if has_overlap or len(new_slots) > 0:
                return True
        return False

    def _update_state(self) -> None:
        """Update the dialogue state based on current intent and slots."""
        # Conversational intents preserve current state
        # (confirmation/denial/info/repeat don't change what we're doing)
        if self.intent in ("confirmation", "denial", "information_provision", "repeat", "help"):
            # If we're collecting and all slots filled, advance to confirming
            if self.state == State.COLLECTING and self.intent in REQUIRED_SLOTS:
                if not self.missing_slots:
                    self.state = State.CONFIRMING
            return

        if self.intent in ("greeting",):
            self.state = State.GREETING
        elif self.intent in ("goodbye",):
            self.state = State.DONE
        elif self.intent in ("complaint", "transfer_to_human"):
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
