"""
Dialogue State Tracker — FSM (v3 — NER-first pipeline)

Tracks the conversation state across turns:
- Current intent and filled/missing slots
- State transitions: GREETING → COLLECTING → CONFIRMING → EXECUTING → DONE
- Slot corrections ("actually, make it Tuesday")
- Turn counting for loop detection
- 30-intent support with pending_slot expectation logic

Pipeline order (every turn):
  NER (unconditional) → pending_slot check → conditional intent → DST update

Updated for ML-based NLU (DeBERTa intent + BERT NER slots).
"""

import logging
import re
from enum import Enum
from typing import Any, Optional

# Confidence threshold for accepting NER slot fills
SLOT_CONFIDENCE_THRESHOLD = 0.75

from config import cfg
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


# Required slots per intent (29-label set)
REQUIRED_SLOTS = {
    "book_appointment": ["service_type", "date", "time", "name"],
    "cancel_appointment": ["name"],
    "reschedule_appointment": ["date", "time"],
    "check_appointment_status": ["name"],
    "check_my_appointments": ["name"],
    "change_date": ["date"],
    "change_time": ["time"],
    "change_doctor": ["doctor_name"],
    "change_service": ["service_type"],
    "register_new_patient": ["name", "phone"],
    "update_contact_info": ["name", "phone"],
    "verify_patient_identity": ["name"],
    "confirm_appointment": [],
    "doctor_availability": [],
    # These intents require no slots:
    # greeting, goodbye, complaint, transfer_to_human, confirmation, denial,
    # help, repeat, clinic_hours, clinic_location, consultation_fee,
    # doctor_specialization, services_available, emergency_request
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

    def process_turn(
        self, 
        text: str, 
        system_context: str = "", 
        joint_nlu_engine: Optional[Any] = None
    ) -> dict:
        """
        Process a user utterance and update state.

        Pipeline order (strict, every turn):
          1. NER runs unconditionally on raw text
          2. Check pending_slot flag in DST
          3. If pending_slot filled with conf >= 0.75 → skip intent classifier
          4. Otherwise → run intent classifier
          5. Dual low-confidence check → trigger clarification if both fail
          6. DST update → build result

        Args:
            text: User utterance

        Returns:
            Dict with: intent, confidence, slots, state, missing_slots, action
        """
        if cfg.nlu.joint_nlu_enabled:
            return self._process_turn_joint(text, system_context, joint_nlu_engine)
            
        self.turn_count += 1

        # ── Step 1: NER runs unconditionally ────────────────────
        new_slots = extract_slots(text)

        # ── Step 2+3: Expected-slot check with confidence gating ─
        # If we asked for a specific slot and the user provided it,
        # skip intent classification — this is just an answer.
        if self._try_fill_expected_slot(text, new_slots):
            # Merge any bonus slots found by NER beyond the pending one
            self.slots.update(new_slots)
            self._update_state()
            return self._build_result(text)

        # ── Step 4: Intent classification (conditional path) ────
        new_intent, confidence = classify_intent(text)
        self._last_classified_intent = new_intent

        # ── Step 5: Dual low-confidence check ───────────────────
        # If NER found nothing useful AND intent classifier has low
        # confidence while we're actively collecting, don't guess.
        if (not new_slots and confidence < cfg.nlu.confidence_low
                and self.state == State.COLLECTING and self.intent is not None):
            self.failed_turns += 1
            logger.info(
                f"Dual low-confidence: no slots extracted + "
                f"intent conf={confidence:.2f} < cfg '{cfg.nlu.confidence_low}' → clarification"
            )
            self._last_classified_intent = "clarification_needed"
            self.pending_slot = None
            self.last_question = None
            self._update_state()
            return self._build_result(text)

        # ── Step 6: Normal intent processing ────────────────────
        # Detect slot corrections ("actually, make it Tuesday")
        is_correction = self._is_slot_correction(text, new_slots, new_intent)

        # Determine if this is a new intent or continuation
        if is_correction:
            # Slot correction — keep current intent, update slots
            logger.info(f"Slot correction detected: {new_slots}")
            self.slots.update(new_slots)
            self._last_classified_intent = "correction"
        elif self._is_new_conversation(new_intent, confidence):
            # New intent — preserve compatible slots if transferring
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
            if confidence > self.intent_confidence:
                self.intent_confidence = confidence

            if not getattr(self, '_last_classified_intent', None) == "continuation":
                self._last_classified_intent = "continuation"

        # Clear pending expectation (we've processed this turn)
        self.pending_slot = None
        self.last_question = None

        # Update state
        self._update_state()

        return self._build_result(text)

    def _process_turn_joint(self, text: str, system_context: str, joint_nlu_engine) -> dict:
        """
        Process a turn using the unified Joint NLU Model.
        Replaces the separate NER + intent pipeline.
        """
        self.turn_count += 1
        
        if not joint_nlu_engine:
            raise RuntimeError("Joint NLU enabled but engine not provided to DST.")
            
        prediction = joint_nlu_engine.predict(system_context, text)
        new_intent = prediction["intent"]
        confidence = prediction["intent_confidence"]
        new_slots = prediction["slots"]
        
        self._last_classified_intent = new_intent
        
        # State transitions
        if new_intent is None or new_intent == "slot_answer":
            new_intent = None
            # Slot answer (continuation)
            self._last_classified_intent = "continuation"
            self.slots.update(new_slots)
        else:
            is_correction = self._is_slot_correction(text, new_slots, new_intent)
            
            if is_correction:
                logger.info(f"Slot correction detected: {new_slots}")
                self.slots.update(new_slots)
                self._last_classified_intent = "correction"
            elif self._is_new_conversation(new_intent, confidence):
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
                self.slots.update(new_slots)
                if confidence > self.intent_confidence:
                    self.intent_confidence = confidence
                    
                if not getattr(self, '_last_classified_intent', None) == "continuation":
                    self._last_classified_intent = "continuation"
        
        # Clear pending (not really used in joint model, but good to clean up)
        self.pending_slot = None
        self.last_question = None
        
        # Update state
        self._update_state()
        
        return self._build_result(text)

    def _try_fill_expected_slot(self, text: str, new_slots: dict) -> bool:
        """Try to fill the pending expected slot.

        Three-path resolution:
          Path 1: NER extracted the expected slot with conf >= threshold → fill it
          Path 2: NER missed it, but raw text looks like a direct answer → fill via fallback
          Path 3: NER extracted other (bonus) slots while collecting → treat as continuation

        Any additional slots found by NER beyond the pending one are treated
        as bonus fills and also written to DST (handled by caller).

        Returns True if the expected slot was filled (skip intent classification).
        """
        if not self.pending_slot:
            return False

        expected = self.pending_slot

        # ── Path 1: NER extracted the expected slot directly ────
        if expected in new_slots:
            slot_data = new_slots[expected]
            slot_conf = (
                slot_data.get("confidence", 0.0)
                if isinstance(slot_data, dict) else 1.0
            )
            if slot_conf >= SLOT_CONFIDENCE_THRESHOLD:
                logger.info(
                    f"Expected slot '{expected}' filled via NER "
                    f"(conf={slot_conf:.2f}): {slot_data} "
                    f"→ skipping intent classification"
                )
                self.pending_slot = None
                self.last_question = None
                return True
            else:
                logger.debug(
                    f"NER found '{expected}' but conf={slot_conf:.2f} "
                    f"< {SLOT_CONFIDENCE_THRESHOLD} — trying fallback"
                )

        # ── Path 2: Raw-text fallback ───────────────────────────
        # User gave a bare answer to our direct question (e.g., "Cardiology",
        # "John Smith"). NER couldn't tag it as the right slot type because
        # it only knows PER/ORG/LOC/MISC, not domain slots.
        # Only activate for short utterances likely to be direct answers.
        stripped = text.strip()
        if len(stripped) < 50 and self.state == State.COLLECTING:
            fallback_value = self._raw_text_to_slot(expected, stripped)
            if fallback_value:
                new_slots[expected] = {
                    "value": fallback_value,
                    "confidence": 0.70,
                }
                logger.info(
                    f"Expected slot '{expected}' filled via raw-text "
                    f"fallback: '{fallback_value}' → skipping intent"
                )
                self.pending_slot = None
                self.last_question = None
                return True

        # ── Path 3: Got other (bonus) slots while collecting ────
        if new_slots and self.state == State.COLLECTING:
            logger.info(
                f"Expected '{expected}' not found, but got "
                f"{list(new_slots.keys())} — treating as continuation"
            )
            self.pending_slot = None
            self.last_question = None
            return True

        # Nothing relevant — fall through to intent classification
        # (user might have said something unrelated like 'goodbye')
        return False

    def _raw_text_to_slot(
        self, slot_name: str, text: str
    ) -> Optional[str]:
        """Map raw text to a slot value when the user gives a bare answer.

        This is a stopgap for when NER cannot recognize domain-specific
        entity types (e.g., SERVICE_TYPE, DEPARTMENT). Once the NER model
        is fine-tuned on domain entities, this fallback becomes redundant.

        Args:
            slot_name: The slot we're expecting (e.g., 'service_type')
            text: The raw user utterance, stripped

        Returns:
            The slot value string, or None if text doesn't look like a valid answer.
        """
        text = text.strip().rstrip('.!?,;')
        if not text:
            return None

        # Skip obvious non-answers
        _SKIP_WORDS = {
            "yes", "no", "ok", "okay", "sure", "hi", "hello",
            "hey", "bye", "goodbye", "thanks", "thank you",
            "um", "uh", "hmm",
        }
        if text.lower() in _SKIP_WORDS:
            return None

        # service_type: accept any non-trivial text as the service
        if slot_name == "service_type":
            if len(text) >= 2:
                return text.lower()

        # name: accept if it looks like a proper name (capitalized words)
        if slot_name == "name":
            words = text.split()
            if 1 <= len(words) <= 4 and all(
                w[0].isupper() for w in words if w
            ):
                return text

        # doctor_name: accept with or without Dr. prefix
        if slot_name == "doctor_name":
            if len(text) >= 2:
                return text

        # date / time: these are already well-handled by NER extractors
        # (dateutil, regex patterns). If NER missed them, the text
        # probably isn't a date/time.

        # phone: very structured, shouldn't need fallback

        return None

    def _build_result(self, text: str) -> dict:
        """Build and log the NLU result dict."""
        missing = self.missing_slots

        # For conversational intents (confirmation/denial/etc.), expose
        # the classified intent so the router can differentiate, while
        # keeping _active_intent as the real task intent.
        CONVERSATIONAL_INTENTS = {
            "confirmation", "denial", "information_provision", "repeat",
            "help", "correction", "continuation", "clarification_needed",
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
            "confirm_appointment",
        }
        if new_intent in CONTINUATION_INTENTS:
            return False

        # Change-type intents are modifications to the current flow,
        # not entirely new conversations — they modify existing slots
        MODIFICATION_INTENTS = {
            "change_date", "change_doctor", "change_service",
        }
        if new_intent in MODIFICATION_INTENTS and self.state in (
            State.COLLECTING, State.CONFIRMING
        ):
            return False

        # Explicit conversational intents always start fresh
        if new_intent in ("greeting", "goodbye"):
            return True
        # Always allow transition FROM greeting/out_of_domain to real intents
        if self.intent in ("greeting", "out_of_domain") and new_intent not in ("greeting", "out_of_domain"):
            return True
        # Complaint, transfer, and emergency always override
        if new_intent in ("complaint", "transfer_to_human", "emergency_request"):
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
        PRESERVE_STATE_INTENTS = {
            "confirmation", "denial", "information_provision", "repeat",
            "help", "confirm_appointment",
        }
        if self.intent in PRESERVE_STATE_INTENTS:
            # If we're collecting and all slots filled, advance to confirming
            if self.state == State.COLLECTING and self.intent in REQUIRED_SLOTS:
                if not self.missing_slots:
                    self.state = State.CONFIRMING
            return

        if self.intent in ("greeting",):
            self.state = State.GREETING
        elif self.intent in ("goodbye",):
            self.state = State.DONE
        elif self.intent in ("complaint", "transfer_to_human", "emergency_request"):
            self.state = State.DONE
        elif self.intent in (
            "out_of_domain", "clinic_hours", "clinic_location",
            "consultation_fee", "doctor_specialization", "services_available",
        ):
            # FAQ-type intents: don't override state if already collecting
            if self.state not in (State.COLLECTING, State.CONFIRMING):
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
