"""
Confidence Scorer

Combines intent classifier confidence with slot fill percentage
to produce an overall NLU confidence score.

Thresholds from env:
  - NLU_CONFIDENCE_HIGH (0.85): confident enough for Path A (direct action)
  - NLU_CONFIDENCE_LOW (0.6): below this → Path C (LLM fallback)
"""

import logging
from typing import Any

from config import cfg

logger = logging.getLogger(__name__)

# Required slots per intent (mirrors DST)
REQUIRED_SLOTS = {
    "book_appointment": ["date", "time", "name"],
    "cancel_appointment": ["name"],
    "reschedule": ["date", "time"],
    "check_status": ["name"],
}


def compute_confidence(
    intent: str,
    intent_confidence: float,
    slots: dict[str, Any],
) -> float:
    """
    Compute overall NLU confidence score.

    Combines:
    - Intent classifier confidence (60% weight)
    - Slot fill percentage (40% weight)

    Args:
        intent: Classified intent name
        intent_confidence: Raw intent classifier confidence
        slots: Currently filled slots

    Returns:
        Overall confidence score (0.0-1.0)
    """
    # Intent weight: 60%
    intent_score = intent_confidence * 0.6

    # Slot fill weight: 40%
    if intent in REQUIRED_SLOTS:
        required = REQUIRED_SLOTS[intent]
        if required:
            filled = sum(1 for s in required if s in slots)
            slot_score = (filled / len(required)) * 0.4
        else:
            slot_score = 0.4  # No slots required = full slot score
    else:
        # Intents without required slots (greeting, goodbye, faq)
        slot_score = 0.4

    overall = intent_score + slot_score

    logger.debug(
        f"Confidence: intent={intent_confidence:.2f}×0.6={intent_score:.2f} "
        f"+ slots={slot_score:.2f} = {overall:.2f}"
    )
    return overall


def confidence_level(score: float) -> str:
    """
    Categorize confidence score into a routing level.

    Returns: "high", "medium", or "low"
    """
    if score >= cfg.nlu.confidence_high:
        return "high"
    elif score >= cfg.nlu.confidence_low:
        return "medium"
    else:
        return "low"
