"""
Dialogue State Tracker — FSM over the NLU output.

State machine:

    IDLE ──► GREETING ──► COLLECTING ──► CONFIRMING ──► EXECUTING ──► DONE
                              ▲              │
                              └──────────────┘  (denial / correction)

Contract with the router:
  * COLLECTING  → ask for the next missing slot
  * CONFIRMING  → read the booking back and wait for an explicit yes
  * EXECUTING   → the caller said yes; perform the action exactly once
  * DONE        → action complete; a new task intent starts a fresh turn

Two behaviours that were previously missing and made the agent unusable:
  1. An action was executed the instant the last slot was filled — there was
     no confirmation step, and because no terminal state was ever reached the
     confirmation sentence was re-emitted on every subsequent turn forever.
  2. `confirmation` / `denial` were overwritten with "continuation" before the
     router saw them, so the caller literally could not say no.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Optional

from config import cfg
from core.nlu.validation import validate_slots

logger = logging.getLogger(__name__)


class State(str, Enum):
    """Dialogue states."""

    IDLE = "idle"
    GREETING = "greeting"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    DONE = "done"


# Required slots per task intent.
REQUIRED_SLOTS: dict[str, list[str]] = {
    "book_appointment": ["service_type", "date", "time", "patient_name"],
    "cancel_appointment": ["patient_name"],
    "reschedule_appointment": ["date", "time"],
    "check_appointment_status": ["patient_name"],
    "check_my_appointments": ["patient_name"],
    "change_date": ["date"],
    "change_time": ["time"],
    "change_doctor": ["doctor_name"],
    "change_service": ["service_type"],
    "register_new_patient": ["patient_name", "phone"],
    "update_contact_info": ["patient_name", "phone"],
    "verify_patient_identity": ["patient_name"],
    "confirm_appointment": [],
    "doctor_availability": [],
}

# Task intents that perform a real action and therefore need confirmation.
ACTION_INTENTS = frozenset({
    "book_appointment", "cancel_appointment", "reschedule_appointment",
    "change_date", "change_time", "change_doctor", "change_service",
    "register_new_patient", "update_contact_info",
})

# Read-only lookups — safe to answer without a confirmation step.
READONLY_INTENTS = frozenset({
    "check_appointment_status", "check_my_appointments",
    "verify_patient_identity", "doctor_availability",
})

# Intents the DST synthesises rather than the model predicting them.
CONVERSATIONAL_INTENTS = frozenset({
    "confirmation", "denial", "information_provision", "repeat",
    "help", "correction", "continuation", "clarification_needed",
})

CORRECTION_TRIGGERS = re.compile(
    r'\b(actually|no|change|not|instead|wrong|correct|make it|switch to|'
    r'rather|different|sorry i meant|i meant)\b',
    re.IGNORECASE,
)

# The joint model classifies bare "yes"/"no" as slot_answer or goodbye when a
# system prompt is in context, so these are resolved lexically first. Short,
# unambiguous affirmations/denials only — anything longer goes to the model.
_YES_RE = re.compile(
    r'^(yes|yeah|yep|yup|yes please|sure|ok|okay|correct|right|that\'?s right|'
    r'confirm|confirmed|sounds good|go ahead|please do|do it|perfect|great|'
    r'that works|book it|absolutely|of course)[.!]?$',
    re.IGNORECASE,
)
# A real question must never be swallowed as a bare yes/no. After an action
# completes the system says "anything else?", which primes the model towards
# confirmation — so "what are your opening hours" came back as
# confirm_appointment and got answered with "Great, anything else?".
_QUESTION_RE = re.compile(
    r'^\s*(what|where|when|why|who|which|how|do|does|did|can|could|would|will|'
    r'is|are|am|should|shall|may|have|has|tell me|i need to know)\b',
    re.IGNORECASE,
)

# Safety net: on a medical line an emergency must never be missed because
# the model was primed by the previous system turn. These override everything.
_EMERGENCY_RE = re.compile(
    r'\b(heart attack|cardiac arrest|stroke|chest pain|can\'?t breathe|'
    r'cannot breathe|not breathing|unconscious|passed out|choking|'
    r'severe bleeding|bleeding badly|overdose|suicidal|kill myself|'
    r'anaphyla\w*|seizure|911|emergency)\b',
    re.IGNORECASE,
)

_NO_RE = re.compile(
    r'^(no|nope|nah|no thanks|no thank you|negative|not really|don\'?t|do not|'
    r'that\'?s wrong|wrong|incorrect|not right|cancel that|nevermind|'
    r'never mind|wait no|no wait)[.!]?$',
    re.IGNORECASE,
)


def _slot_value(v: Any) -> Any:
    """Slots may be {"value": x, ...} dicts or bare values."""
    if isinstance(v, dict):
        return v.get("value")
    return v


def _slot_confidence(v: Any) -> float:
    if isinstance(v, dict):
        return float(v.get("confidence", 1.0))
    return 1.0


def _is_filled(v: Any) -> bool:
    val = _slot_value(v)
    return val is not None and str(val).strip() != ""


class DialogueStateTracker:
    """Tracks intent, slots and dialogue state across turns."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state: State = State.IDLE
        self.intent: Optional[str] = None
        self.intent_confidence: float = 0.0
        self.slots: dict[str, Any] = {}
        self.turn_count: int = 0
        self.failed_turns: int = 0
        self._history: list[dict] = []
        self.pending_slot: Optional[str] = None
        self.last_question: Optional[str] = None
        self._last_classified_intent: Optional[str] = None
        # Set when validation rejects a slot; the router speaks it and the DST
        # clears the offending slot so the caller can supply a new value.
        self.repair_message: Optional[str] = None
        # Guards against re-executing (and re-announcing) a completed action.
        self.completed_action: Optional[str] = None
        # Set when the caller rejects a read-back; suppresses re-confirming
        # until they actually change something.
        self.declined_confirmation: bool = False

    # ── expectation ─────────────────────────────────────────
    def set_expectation(self, slot_name: str, question: str = "") -> None:
        """Record the slot the agent just asked for."""
        self.pending_slot = slot_name
        self.last_question = question
        logger.debug(f"Expectation set: pending_slot='{slot_name}'")

    # ── main entry point ────────────────────────────────────
    def process_turn(
        self,
        text: str,
        system_context: str = "",
        joint_nlu_engine: Optional[Any] = None,
    ) -> dict:
        """Process a user utterance and update state."""
        self.turn_count += 1
        self.repair_message = None

        if cfg.nlu.joint_nlu_enabled:
            if joint_nlu_engine is None:
                raise RuntimeError(
                    "JOINT_NLU_ENABLED=true but no joint_nlu_engine was passed to "
                    "DialogueStateTracker.process_turn()."
                )
            new_intent, confidence, new_slots = self._predict_joint(
                joint_nlu_engine, system_context, text
            )
        else:
            new_intent, confidence, new_slots = self._predict_legacy(text)

        return self._advance(text, new_intent, confidence, new_slots)

    # ── prediction backends ─────────────────────────────────
    def _predict_joint(self, engine, system_context: str, text: str):
        pred = engine.predict(system_context, text)
        return pred["intent"], pred["intent_confidence"], dict(pred["slots"])

    def _predict_legacy(self, text: str):
        from core.nlu.intent import classify_intent
        from core.nlu.slots import extract_slots

        new_slots = extract_slots(text)

        # Expected-slot fast path: if the agent asked for a specific slot and
        # we got it, skip intent classification entirely.
        if self._try_fill_expected_slot(text, new_slots):
            return None, 1.0, new_slots

        intent, confidence = classify_intent(text)

        # Dual low confidence while collecting → ask rather than guess.
        if (
            not new_slots
            and confidence < cfg.nlu.confidence_low
            and self.state == State.COLLECTING
            and self.intent is not None
        ):
            self.failed_turns += 1
            logger.info(
                f"Dual low-confidence (no slots, intent conf={confidence:.2f}) → clarify"
            )
            return "clarification_needed", confidence, {}

        return intent, confidence, new_slots

    # ── the state machine ───────────────────────────────────
    def _advance(self, text: str, new_intent, confidence: float, new_slots: dict) -> dict:
        stripped = (text or "").strip()

        # 0. Emergency safety net — overrides the model unconditionally.
        if _EMERGENCY_RE.search(stripped):
            logger.warning(f"Emergency phrase detected in {stripped!r}")
            self.intent = "emergency_request"
            self.intent_confidence = 1.0
            self._last_classified_intent = None
            self.state = State.DONE
            return self._build_result(text)

        # 1. Lexical yes/no wins over the model for bare affirmations.
        lexical = None
        if _YES_RE.match(stripped):
            lexical = "confirmation"
        elif _NO_RE.match(stripped):
            lexical = "denial"
        if lexical and not new_slots:
            new_intent, confidence = lexical, 0.99
            logger.debug(f"Lexical override: {stripped!r} → {lexical}")
        elif (
            new_intent in ("confirmation", "denial", "confirm_appointment")
            and not new_slots
            and _QUESTION_RE.match(stripped)
        ):
            # The model read a question as a yes/no. Send it to the LLM, which
            # is grounded, rather than replying "Great, anything else?".
            logger.info(
                f"Demoting {new_intent} → out_of_domain: {stripped!r} is a question"
            )
            new_intent, confidence = "out_of_domain", min(confidence, 0.4)

        # 2. A None/slot_answer intent means "this answers my question".
        if new_intent is None or new_intent == "slot_answer":
            self._merge_slots(new_slots, overwrite=False)
            self._last_classified_intent = "continuation"
            self.failed_turns = 0
            self._validate_and_repair()
            self._update_state()
            return self._build_result(text)

        # 3. Confirmation while waiting for one → execute the action.
        if new_intent in ("confirmation", "confirm_appointment") and self.state == State.CONFIRMING:
            self._merge_slots(new_slots, overwrite=False)
            self._validate_and_repair()
            if self.repair_message:
                self.state = State.COLLECTING
                self._last_classified_intent = "correction"
                return self._build_result(text)
            self.state = State.EXECUTING
            self.declined_confirmation = False
            self._last_classified_intent = "confirmation"
            self.failed_turns = 0
            return self._build_result(text)

        # 4. Denial. What it means depends on where we are:
        #      CONFIRMING → reject the read-back and wait for a change
        #      DONE       → "no thanks", i.e. nothing else needed
        if new_intent == "denial":
            self._last_classified_intent = "denial"
            self.failed_turns = 0
            if self.state == State.DONE:
                return self._build_result(text)
            if self.state in (State.CONFIRMING, State.COLLECTING):
                # Only a denial of an actual read-back counts as declining a
                # confirmation. A bare "no thanks" with nothing pending is the
                # caller closing the call, not rejecting a booking.
                was_confirming = self.state == State.CONFIRMING
                if new_slots:
                    # "No, make it Tuesday" — a denial carrying a correction.
                    self._merge_slots(new_slots, overwrite=True)
                    self._last_classified_intent = "correction"
                    self.declined_confirmation = False
                elif was_confirming:
                    self.declined_confirmation = True
                self.state = State.COLLECTING
                self._validate_and_repair()
                self._update_state()
                return self._build_result(text)

        # 5. Explicit slot correction ("actually, make it 5 pm").
        if self._is_slot_correction(text, new_slots, new_intent):
            logger.info(f"Slot correction: {list(new_slots)}")
            self._merge_slots(new_slots, overwrite=True)
            self._last_classified_intent = "correction"
            # Only adopt a different task intent when nothing has been
            # collected yet. Mid-booking, "actually make it Tuesday" is a slot
            # correction — not a request to reschedule an existing booking.
            if (
                new_intent in REQUIRED_SLOTS
                and new_intent != self.intent
                and not any(_is_filled(v) for v in self.slots.values())
            ):
                self.intent = new_intent
                self.intent_confidence = confidence
            self.failed_turns = 0
            self._validate_and_repair()
            self._update_state()
            return self._build_result(text)

        # 6. New conversation / topic switch.
        if self._is_new_conversation(new_intent, confidence):
            preserved: dict[str, Any] = {}
            if self.intent in REQUIRED_SLOTS and new_intent in REQUIRED_SLOTS:
                for key, val in self.slots.items():
                    if key in REQUIRED_SLOTS[new_intent]:
                        preserved[key] = val
            elif self.intent in ("greeting", None):
                preserved = dict(self.slots)

            self.intent = new_intent
            self.intent_confidence = confidence
            self.slots = preserved
            self.completed_action = None
            self._merge_slots(new_slots, overwrite=True)
            self.failed_turns = 0
            self._last_classified_intent = (
                new_intent if new_intent in CONVERSATIONAL_INTENTS else None
            )
        else:
            # 7. Continuation of the active task.
            self._merge_slots(new_slots, overwrite=False)
            if confidence > self.intent_confidence:
                self.intent_confidence = confidence
            # Preserve a real conversational intent. The old code force-set
            # this to "continuation" unconditionally, which is exactly why
            # confirmation and denial never reached the router.
            self._last_classified_intent = (
                new_intent if new_intent in CONVERSATIONAL_INTENTS else "continuation"
            )

        self._validate_and_repair()
        self._update_state()
        return self._build_result(text)

    # ── slot handling ───────────────────────────────────────
    def _merge_slots(self, new_slots: dict, overwrite: bool) -> None:
        """Merge newly extracted slots into state."""
        if not new_slots:
            return
        force = self.intent in (
            "change_date", "change_time", "change_doctor",
            "change_service", "update_contact_info",
        )
        # After the caller rejects a read-back, or while we are holding a
        # complete booking for confirmation, any value they supply is a
        # deliberate change and must replace what we already have.
        if self.declined_confirmation or self.state == State.CONFIRMING:
            force = True
        for key, val in new_slots.items():
            if not _is_filled(val):
                continue
            # A canonically resolved slot is trusted regardless of the model's
            # token confidence: the entity store matched it exactly, and the
            # tagger routinely over-extends spans ("to book a checkup"), which
            # drags the mean below the threshold for a slot we know precisely.
            resolved = isinstance(val, dict) and val.get("canonical_id")
            if not resolved and _slot_confidence(val) < cfg.nlu.slot_confidence_threshold:
                logger.debug(f"Dropping low-confidence slot {key}={_slot_value(val)!r}")
                continue
            if overwrite or force or not _is_filled(self.slots.get(key)):
                self.slots[key] = val
                self.declined_confirmation = False
        # Any new information means we must re-confirm.
        if self.state == State.CONFIRMING:
            self.state = State.COLLECTING

    def _validate_and_repair(self) -> None:
        """Reject business-rule-invalid slots and stage a spoken repair."""
        result = validate_slots(self.slots)
        if result.ok:
            return
        problem = result.problems[0]
        self.repair_message = problem.message
        for slot in result.slots_to_clear:
            self.slots.pop(slot, None)
        if self.state in (State.CONFIRMING, State.EXECUTING, State.DONE):
            self.state = State.COLLECTING
        logger.info(f"Slot repair staged: {problem.reason} (cleared {result.slots_to_clear})")

    def _try_fill_expected_slot(self, text: str, new_slots: dict) -> bool:
        """Legacy path: did this utterance answer the question we just asked?"""
        if not self.pending_slot:
            return False
        expected = self.pending_slot

        if expected in new_slots:
            if _slot_confidence(new_slots[expected]) >= cfg.nlu.slot_confidence_threshold:
                logger.info(f"Expected slot '{expected}' filled via NER")
                self.pending_slot = self.last_question = None
                return True

        stripped = text.strip()
        if len(stripped) < 50 and self.state == State.COLLECTING:
            fallback = self._raw_text_to_slot(expected, stripped)
            if fallback:
                new_slots[expected] = {"value": fallback, "confidence": 0.80}
                logger.info(f"Expected slot '{expected}' filled via raw-text fallback")
                self.pending_slot = self.last_question = None
                return True

        if new_slots and self.state == State.COLLECTING:
            self.pending_slot = self.last_question = None
            return True
        return False

    def _raw_text_to_slot(self, slot_name: str, text: str) -> Optional[str]:
        """Map a bare answer to a slot value (legacy path only)."""
        text = text.strip().rstrip('.!?,;')
        if not text:
            return None
        skip = {
            "yes", "no", "ok", "okay", "sure", "hi", "hello", "hey",
            "bye", "goodbye", "thanks", "thank you", "um", "uh", "hmm",
        }
        if text.lower() in skip:
            return None

        if slot_name == "service_type" and len(text) >= 2:
            return text.lower()
        if slot_name == "doctor_name" and len(text) >= 2:
            return text
        if slot_name == "patient_name":
            words = text.split()
            if 1 <= len(words) <= 4 and all(w[:1].isupper() for w in words if w):
                return text
        return None

    # ── transition helpers ──────────────────────────────────
    def _is_new_conversation(self, new_intent: str, confidence: float) -> bool:
        if self.intent is None:
            return True
        if new_intent in CONVERSATIONAL_INTENTS:
            return False
        # Escalations always win.
        if new_intent in ("complaint", "transfer_to_human", "emergency_request"):
            return True
        # After a completed action, anything new starts a fresh task.
        if self.state == State.DONE:
            return True
        if new_intent == "help":
            return self.state in (State.IDLE, State.DONE, State.GREETING)
        # Mid-task noise ("Oh, hello?") must not drop the task.
        if (
            self.state in (State.COLLECTING, State.CONFIRMING)
            and new_intent in ("out_of_domain", "greeting", "goodbye")
            and self.missing_slots
        ):
            return False
        # A modification intent refines the active task rather than replacing it.
        if (
            new_intent in ("change_date", "change_time", "change_doctor", "change_service")
            and self.state in (State.COLLECTING, State.CONFIRMING)
        ):
            return False
        if self.state == State.COLLECTING and new_intent != self.intent and confidence < 0.4:
            return False
        return new_intent != self.intent and confidence > 0.4

    def _is_slot_correction(self, text: str, new_slots: dict, new_intent: str) -> bool:
        if not self.intent or not new_slots:
            return False
        if self.state not in (State.COLLECTING, State.CONFIRMING):
            return False
        if new_intent == "denial":
            return True
        if not CORRECTION_TRIGGERS.search(text or ""):
            return False
        # Only a correction if it overwrites something already known.
        return any(_is_filled(self.slots.get(k)) for k in new_slots)

    def _update_state(self) -> None:
        """Derive the next state from the active intent and slot completeness."""
        if self.state == State.EXECUTING:
            return  # the router consumes EXECUTING and calls mark_action_complete()

        if self.repair_message:
            self.state = State.COLLECTING
            return

        if self.intent == "greeting":
            self.state = State.GREETING
            return
        if self.intent in ("goodbye", "complaint", "transfer_to_human", "emergency_request"):
            self.state = State.DONE
            return

        if self.intent in ACTION_INTENTS:
            if self.missing_slots:
                self.state = State.COLLECTING
            elif self.completed_action == self.intent:
                self.state = State.DONE
            elif self.declined_confirmation:
                # They said no and haven't changed anything yet — keep asking
                # what to change instead of repeating the same read-back.
                self.state = State.COLLECTING
            else:
                # All slots present → read it back and wait for a yes.
                self.state = State.CONFIRMING
            return

        if self.intent in READONLY_INTENTS:
            self.state = State.COLLECTING if self.missing_slots else State.EXECUTING
            return

        # FAQ / out-of-domain: don't clobber an in-flight task.
        if self.state not in (State.COLLECTING, State.CONFIRMING):
            self.state = State.COLLECTING

    def mark_action_complete(self, intent: Optional[str] = None) -> None:
        """Called by the pipeline once an action response has been delivered."""
        self.completed_action = intent or self.intent
        self.state = State.DONE
        self.pending_slot = self.last_question = None
        logger.info(f"Action complete: {self.completed_action} → state=done")

    # ── result ──────────────────────────────────────────────
    def _build_result(self, text: str) -> dict:
        missing = self.missing_slots
        classified = self._last_classified_intent

        if classified and classified in CONVERSATIONAL_INTENTS:
            output_intent = classified
            active_intent = self.intent
        else:
            output_intent = self.intent
            active_intent = self.intent

        if output_intent is None:
            output_intent = "continuation"

        result = {
            "intent": output_intent,
            "_active_intent": active_intent,
            "confidence": self.intent_confidence,
            "slots": {k: _slot_value(v) for k, v in self.slots.items()},
            "slot_confidences": {k: _slot_confidence(v) for k, v in self.slots.items()},
            "slot_meta": {
                k: v for k, v in self.slots.items() if isinstance(v, dict)
            },
            "state": self.state.value,
            "missing_slots": missing,
            "turn_count": self.turn_count,
            "failed_turns": self.failed_turns,
            "repair_message": self.repair_message,
            "pending_slot": self.pending_slot,
            "completed_action": self.completed_action,
            "declined_confirmation": self.declined_confirmation,
        }

        self._history.append({"turn": self.turn_count, "text": text, "result": result})
        logger.info(
            f"DST turn {self.turn_count}: intent={output_intent} active={active_intent} "
            f"state={self.state.value} slots={list(self.slots)} missing={missing}"
        )
        self._last_classified_intent = None
        return result

    # ── properties ──────────────────────────────────────────
    @property
    def missing_slots(self) -> list[str]:
        if self.intent not in REQUIRED_SLOTS:
            return []
        return [s for s in REQUIRED_SLOTS[self.intent] if not _is_filled(self.slots.get(s))]

    @property
    def all_slots_filled(self) -> bool:
        return not self.missing_slots

    @property
    def history(self) -> list[dict]:
        return list(self._history)
