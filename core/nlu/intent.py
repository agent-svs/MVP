"""
Intent Classifier — Rule-Based v1

Classifies user utterances into intents using keyword matching + pattern scoring.
Returns (intent, confidence) tuple.

Supported intents:
- book_appointment, cancel_appointment, reschedule
- check_status, faq, greeting, goodbye, out_of_domain
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Intent Definitions ──────────────────────────────────────
INTENTS = {
    "greeting": {
        "keywords": ["hello", "hi", "hey", "good morning", "good afternoon",
                     "good evening", "howdy", "what's up", "sup"],
        "patterns": [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgood\s+(morning|afternoon|evening)\b"],
        "weight": 0.9,
    },
    "goodbye": {
        "keywords": ["bye", "goodbye", "see you", "take care", "later",
                     "thanks bye", "that's all", "nothing else"],
        "patterns": [r"\bbye\b", r"\bgoodbye\b", r"\bsee\s+you\b", r"\bthat'?s\s+all\b"],
        "weight": 0.9,
    },
    "book_appointment": {
        "keywords": ["book", "schedule", "appointment", "reserve", "make an appointment",
                     "set up", "arrange", "slot", "visit", "book a time"],
        "patterns": [
            r"\b(book|schedule|make|set\s+up|arrange)\b.*\b(appointment|meeting|visit|slot|time)\b",
            r"\b(appointment|meeting|visit)\b.*\b(book|schedule|for)\b",
            r"\bwant\s+to\s+(book|schedule|come\s+in)\b",
            r"\b(can|could)\s+i\s+(book|schedule|get)\b",
        ],
        "weight": 1.0,
    },
    "cancel_appointment": {
        "keywords": ["cancel", "delete", "remove", "call off", "don't want",
                     "cancel appointment", "cancel my"],
        "patterns": [
            r"\bcancel\b.*\b(appointment|meeting|booking|visit)\b",
            r"\b(don'?t|do\s+not)\s+want\b.*\b(appointment|meeting)\b",
            r"\bcancel\s+(it|my|the)\b",
        ],
        "weight": 1.0,
    },
    "reschedule": {
        "keywords": ["reschedule", "change", "move", "postpone", "different time",
                     "another day", "change date", "change time"],
        "patterns": [
            r"\b(reschedule|change|move|postpone)\b.*\b(appointment|meeting|time|date)\b",
            r"\b(different|another|new)\s+(time|day|date|slot)\b",
        ],
        "weight": 1.0,
    },
    "check_status": {
        "keywords": ["status", "check", "when is", "my appointment", "upcoming",
                     "next appointment", "do i have", "confirm"],
        "patterns": [
            r"\b(check|when|what)\b.*\b(appointment|booking|status)\b",
            r"\b(do\s+i\s+have|is\s+my)\b.*\b(appointment|booking)\b",
            r"\b(upcoming|next)\s+(appointment|visit)\b",
        ],
        "weight": 0.95,
    },
    "faq": {
        "keywords": ["hours", "services", "location", "address", "phone",
                     "price", "cost", "how much", "what do you", "offer",
                     "open", "close", "where are you", "contact"],
        "patterns": [
            r"\b(what|which)\s+(services?|do\s+you)\b",
            r"\b(how\s+much|price|cost|fee)\b",
            r"\b(hours?|open|close|location|address)\b",
            r"\b(where|how)\s+(are|do|can)\s+(you|i)\b",
        ],
        "weight": 0.85,
    },
}


def classify_intent(text: str) -> tuple[str, float]:
    """
    Classify user text into an intent with confidence score.

    Args:
        text: User utterance (lowercase recommended)

    Returns:
        Tuple of (intent_name, confidence_score 0.0-1.0)
    """
    if not text or not text.strip():
        return "out_of_domain", 0.0

    text_lower = text.lower().strip()
    scores: dict[str, float] = {}

    for intent_name, intent_def in INTENTS.items():
        score = 0.0

        # Keyword matching (0-0.5 range) — use word boundaries to avoid substring false positives
        keywords = intent_def["keywords"]
        keyword_hits = sum(
            1 for kw in keywords
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
        )
        if keyword_hits > 0:
            keyword_score = min(keyword_hits / max(len(keywords) * 0.3, 1), 1.0) * 0.5
            score += keyword_score

        # Pattern matching (0-0.5 range)
        patterns = intent_def["patterns"]
        pattern_hits = sum(1 for p in patterns if re.search(p, text_lower))
        if pattern_hits > 0:
            pattern_score = min(pattern_hits / max(len(patterns) * 0.5, 1), 1.0) * 0.5
            score += pattern_score

        # Apply intent weight
        score *= intent_def["weight"]

        if score > 0:
            scores[intent_name] = min(score, 1.0)

    if not scores:
        return "out_of_domain", 0.3

    # Return highest scoring intent
    best_intent = max(scores, key=scores.get)
    confidence = scores[best_intent]

    logger.debug(f"Intent: {best_intent} ({confidence:.2f}) | scores={scores}")
    return best_intent, confidence
