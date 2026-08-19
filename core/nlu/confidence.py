"""
Confidence Scorer.

Combines the intent classifier's probability with slot completeness into one
routing score.

Two fixes over the previous version:
  * `pending_slot` is now actually threaded through from the router, so the
    context boost is live rather than dead code.
  * Intents with no required slots no longer receive the full slot weight for
    free. That gave `out_of_domain` at 0.0 intent confidence a floor of 0.30
    and made the score meaningless for exactly the case it should flag.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config import cfg
from core.nlu.dst import REQUIRED_SLOTS

logger = logging.getLogger(__name__)

INTENT_WEIGHT = 0.70
SLOT_WEIGHT = 0.30

# Intents that legitimately carry no slots — their confidence is entirely the
# classifier's. Anything else with no required slots gets its slot component
# scaled by the intent confidence rather than granted outright.
_SLOTLESS_INTENTS = frozenset({
    "greeting", "goodbye", "help", "repeat", "confirmation", "denial",
    "complaint", "transfer_to_human", "emergency_request",
    "clinic_hours", "clinic_location", "consultation_fee",
    "doctor_specialization", "services_available", "doctor_availability",
})


def _is_filled(v: Any) -> bool:
    if isinstance(v, dict):
        v = v.get("value")
    return v is not None and str(v).strip() != ""


def compute_confidence(
    intent: str,
    intent_confidence: float,
    slots: dict[str, Any],
    pending_slot: Optional[str] = None,
) -> float:
    """Compute the overall NLU confidence for a turn (0.0-1.0)."""
    intent_confidence = max(0.0, min(1.0, float(intent_confidence or 0.0)))
    intent_score = intent_confidence * INTENT_WEIGHT

    required = REQUIRED_SLOTS.get(intent)
    if required:
        filled = sum(1 for s in required if _is_filled(slots.get(s)))
        slot_score = (filled / len(required)) * SLOT_WEIGHT
    elif intent in _SLOTLESS_INTENTS:
        slot_score = SLOT_WEIGHT
    else:
        # Unknown / out_of_domain: no free floor.
        slot_score = intent_confidence * SLOT_WEIGHT

    overall = intent_score + slot_score

    if pending_slot and _is_filled(slots.get(pending_slot)):
        overall = min(1.0, overall + 0.15)
        logger.debug(f"Confidence boost: pending_slot '{pending_slot}' filled (+0.15)")

    logger.debug(
        f"Confidence: intent={intent_confidence:.2f}*{INTENT_WEIGHT}={intent_score:.2f} "
        f"+ slots={slot_score:.2f} = {overall:.2f}"
    )
    return overall


def confidence_level(score: float) -> str:
    """Categorise a score into high / medium / low."""
    if score >= cfg.nlu.confidence_high:
        return "high"
    if score >= cfg.nlu.confidence_low:
        return "medium"
    return "low"
