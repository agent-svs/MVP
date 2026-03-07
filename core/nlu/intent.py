"""
Intent Classifier — DeBERTa v3 Zero-Shot (v2)

Classifies user utterances into intents using a DeBERTa v3 xsmall
model fine-tuned for Natural Language Inference (zero-shot classification).

No training data required — works by testing entailment between the
user's utterance and hypothesis templates like "The caller wants to {intent}".

Supported intents:
- book_appointment, cancel_appointment, reschedule
- check_status, faq, greeting, goodbye, complaint, transfer_to_human
- confirmation, denial, information_provision, help, repeat
- out_of_domain (fallback when top score < threshold)

Model: MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33 (22M params)
"""

import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Greeting prefix pattern ─────────────────────────────────
# Strips polite greeting prefixes before classification so
# "Hello, I want to book..." classifies as book_appointment, not greeting
_GREETING_PREFIX_RE = re.compile(
    r'^\s*(hi|hello|hey|good\s+(?:morning|afternoon|evening))'
    r'[,!.;\s]+',
    re.IGNORECASE
)

# ── Intent Definitions ──────────────────────────────────────
# Each intent has a label used for zero-shot classification
# and a natural-language hypothesis template

INTENT_LABELS = {
    "greeting": "greeting",
    "goodbye": "goodbye",
    "book_appointment": "booking a new appointment",
    "cancel_appointment": "canceling an appointment",
    "reschedule": "rescheduling an existing appointment",
    "check_status": "checking appointment status",
    "faq": "asking about business information",
    "complaint": "complaining",
    "transfer_to_human": "requesting a human agent",
    # v3: conversational intents
    "confirmation": "confirming or agreeing",
    "denial": "denying or disagreeing",
    "information_provision": "providing requested information",
    "help": "asking what services or options are available",
    "repeat": "asking to repeat or say something again",
}

# Candidate labels for the zero-shot pipeline
CANDIDATE_LABELS = list(INTENT_LABELS.values())
LABEL_TO_INTENT = {v: k for k, v in INTENT_LABELS.items()}

# Minimum confidence threshold — below this → out_of_domain
MIN_CONFIDENCE = 0.35

# Hypothesis template — optimized for DeBERTa v3 xsmall NLI
HYPOTHESIS_TEMPLATE = "This text is about {}."

# ── Lazy-loaded model ───────────────────────────────────────
_classifier = None
_model_name = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"


def _load_model():
    """Load the zero-shot classification pipeline (once)."""
    global _classifier
    if _classifier is not None:
        return

    from transformers import pipeline

    logger.info(f"Loading intent classifier: {_model_name}")
    t0 = time.time()

    _classifier = pipeline(
        "zero-shot-classification",
        model=_model_name,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        device="cpu",  # DeBERTa v3 xsmall is fast enough on CPU (~50ms)
    )

    elapsed = time.time() - t0
    logger.info(f"Intent classifier loaded in {elapsed:.1f}s")


def classify_intent(text: str) -> tuple[str, float]:
    """
    Classify user text into an intent with confidence score.

    Uses DeBERTa v3 zero-shot classification — tests entailment between
    the utterance and each intent hypothesis.

    Args:
        text: User utterance

    Returns:
        Tuple of (intent_name, confidence_score 0.0-1.0)
    """
    if not text or not text.strip():
        return "out_of_domain", 0.0

    # Ensure model is loaded
    _load_model()

    text = text.strip()

    # Strip greeting prefixes so "Hello, I want to book..." → classify the real intent
    stripped = _GREETING_PREFIX_RE.sub('', text).strip()
    had_greeting_prefix = len(stripped) < len(text)

    # If stripping leaves nothing (pure greeting like "Hello!"), keep original
    classify_text = stripped if stripped else text

    t0 = time.time()

    # Run zero-shot classification
    result = _classifier(
        classify_text,
        candidate_labels=CANDIDATE_LABELS,
        multi_label=False,  # Single best intent
    )

    elapsed = time.time() - t0

    # Extract top result
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    # Map label back to intent name
    intent = LABEL_TO_INTENT.get(top_label, "out_of_domain")

    # If we stripped a greeting prefix and the remaining text classified
    # as out_of_domain with low confidence, the greeting prefix was the
    # entire meaningful content → return greeting
    if had_greeting_prefix and intent == "out_of_domain":
        intent = "greeting"
        top_score = 0.90  # High confidence for detected greeting

    # Apply minimum confidence threshold
    if top_score < MIN_CONFIDENCE:
        logger.debug(
            f"Intent below threshold: {intent}={top_score:.3f} < {MIN_CONFIDENCE} "
            f"→ out_of_domain ({elapsed:.3f}s)"
        )
        return "out_of_domain", top_score

    # Log detailed scores for debugging
    if logger.isEnabledFor(logging.DEBUG):
        all_scores = {
            LABEL_TO_INTENT.get(label, label): f"{score:.3f}"
            for label, score in zip(result["labels"][:5], result["scores"][:5])
        }
        logger.debug(f"Intent: {intent} ({top_score:.3f}) top5={all_scores} ({elapsed:.3f}s)")
    else:
        logger.info(f"Intent: {intent} ({top_score:.3f}) ({elapsed:.3f}s)")

    return intent, top_score


# ── Standalone test ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from config import cfg
    cfg.setup_logging()

    print("🧠 DeBERTa v3 Intent Classifier — Interactive Test")
    print(f"   Model: {_model_name}")
    print("   Type utterances to classify. Type 'quit' to exit.\n")

    # Pre-load model
    print("Loading model...")
    _load_model()
    print("Model ready!\n")

    test_phrases = [
        "Hello, how are you?",
        "I'd like to book an appointment for tomorrow",
        "Can you cancel my appointment?",
        "What are your hours?",
        "I want to speak to a real person",
        "The weather is nice today",
        "I need to reschedule my visit to next Monday",
        "This is terrible service, I'm very unhappy",
        "bye, thanks",
        "Which one is hooked up?",
        "I am pretty good",
        "Can I come in at 3 PM?",
    ]

    print("── Batch test ──────────────────────────────\n")
    for phrase in test_phrases:
        intent, conf = classify_intent(phrase)
        print(f"  \"{phrase}\"")
        print(f"    → {intent} ({conf:.3f})\n")

    print("── Interactive mode ────────────────────────\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        intent, conf = classify_intent(text)
        print(f"  → {intent} ({conf:.3f})\n")
