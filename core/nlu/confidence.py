"""
Confidence Scorer (v2 — ML-calibrated)

Combines DeBERTa intent confidence with slot fill percentage
to produce an overall NLU confidence score.

Since DeBERTa v3 outputs well-calibrated probabilities (unlike regex),
intent confidence gets higher weight (70%) and slot fill gets lower (30%).

Thresholds adjusted for ML:
  - NLU_CONFIDENCE_HIGH (0.80): Path A (direct action)
  - NLU_CONFIDENCE_LOW (0.50): below this → Path C (LLM fallback)
"""

import logging
from typing import Any

from config import cfg

logger = logging.getLogger(__name__)

# Required slots per intent (mirrors DST)
REQUIRED_SLOTS = {
    "book_appointment": ["service_type", "date", "time", "name"],
    "cancel_appointment": ["name"],
    "reschedule": ["date", "time"],
    "check_status": ["name"],
}

# Weight split — DeBERTa scores are well-calibrated, give them more weight
INTENT_WEIGHT = 0.70
SLOT_WEIGHT = 0.30


def compute_confidence(
    intent: str,
    intent_confidence: float,
    slots: dict[str, Any],
    pending_slot: str | None = None,
) -> float:
    """
    Compute overall NLU confidence score.

    Combines:
    - Intent classifier confidence (70% weight) — DeBERTa is well-calibrated
    - Slot fill percentage (30% weight)
    - Context-weighting: Boost if pending_slot is filled

    Args:
        intent: Classified intent name
        intent_confidence: Raw DeBERTa confidence (0.0-1.0)
        slots: Currently filled slots
        pending_slot: The slot the system is actively waiting for

    Returns:
        Overall confidence score (0.0-1.0)
    """
    # Intent weight: 70%
    intent_score = intent_confidence * INTENT_WEIGHT

    # Slot fill weight: 30%
    if intent in REQUIRED_SLOTS:
        required = REQUIRED_SLOTS[intent]
        if required:
            filled = sum(1 for s in required if s in slots)
            slot_score = (filled / len(required)) * SLOT_WEIGHT
        else:
            slot_score = SLOT_WEIGHT  # No slots required = full slot score
    else:
        # Intents without required slots (greeting, goodbye, faq, complaint)
        slot_score = SLOT_WEIGHT

    overall = intent_score + slot_score
    
    # Context-weighted boost
    if pending_slot and pending_slot in slots:
        boost = 0.15
        overall = min(1.0, overall + boost)
        logger.debug(f"Confidence boost! pending_slot '{pending_slot}' filled (+{boost})")

    logger.debug(
        f"Confidence: intent={intent_confidence:.2f}×{INTENT_WEIGHT}={intent_score:.2f} "
        f"+ slots={slot_score:.2f} = {overall:.2f}"
    )
    return overall


def confidence_level(score: float) -> str:
    """
    Categorize confidence score into a routing level.

    Thresholds (ML-calibrated):
    - high:   >= 0.80 (was 0.85 for regex)
    - medium: >= 0.50 (was 0.60 for regex)
    - low:    < 0.50

    Returns: "high", "medium", or "low"
    """
    if score >= cfg.nlu.confidence_high:
        return "high"
    elif score >= cfg.nlu.confidence_low:
        return "medium"
    else:
        return "low"
