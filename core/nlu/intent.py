"""
Intent Classifier — DeBERTa v3 Zero-Shot / Fine-Tuned (v3)

Classifies user utterances into 30 intents.

Two modes:
  1. Zero-shot (default): Uses DeBERTa v3 xsmall NLI model, tests entailment
     between the utterance and hypothesis templates. No training data needed.
  2. Fine-tuned: When INTENT_FINE_TUNED=true, loads a standard text-classification
     model trained on domain-specific data for higher accuracy.

Model: Default = MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33 (22M params)
"""

import logging
import os
import re
import time
from typing import Optional

from config import cfg

logger = logging.getLogger(__name__)

# ── Greeting prefix pattern ─────────────────────────────────
# Strips polite greeting prefixes before classification so
# "Hello, I want to book..." classifies as book_appointment, not greeting
_GREETING_PREFIX_RE = re.compile(
    r'^\s*(hi|hello|hey|good\s+(?:morning|afternoon|evening))'
    r'[,!.;\s]+',
    re.IGNORECASE
)

# ── Intent Definitions (30-label set) ────────────────────────
# Each intent has a label used for zero-shot classification
# and a natural-language hypothesis template.
# When fine-tuned, the hypothesis is not used — the model
# directly predicts the intent label.

INTENT_LABELS = {
    # Core conversational
    "greeting": "greeting",
    "goodbye": "goodbye",
    "confirmation": "confirming or agreeing",
    "denial": "denying or disagreeing",
    "help": "asking what services or options are available",
    "repeat": "asking to repeat or say something again",
    # Appointment actions
    "book_appointment": "booking a new appointment",
    "cancel_appointment": "canceling an appointment",
    "reschedule_appointment": "rescheduling an existing appointment",
    "confirm_appointment": "confirming an appointment booking",
    "check_appointment_status": "checking appointment status",
    "check_my_appointments": "checking or listing my appointments",
    # Modification actions
    "change_date": "changing the appointment date",
    "change_time": "changing the appointment time",
    "change_doctor": "changing the assigned doctor",
    "change_service": "changing the type of service",
    # Patient management
    "register_new_patient": "registering as a new patient",
    "update_contact_info": "updating contact information",
    "verify_patient_identity": "verifying patient identity",
    # FAQ / Informational
    "clinic_hours": "asking about clinic operating hours",
    "clinic_location": "asking about clinic location or address",
    "consultation_fee": "asking about consultation fees or costs",
    "doctor_specialization": "asking about doctor specialization or expertise",
    "doctor_availability": "asking about doctor availability or schedule",
    "services_available": "asking about available services",
    # Escalation
    "complaint": "complaining about service",
    "transfer_to_human": "requesting a human agent",
    "emergency_request": "reporting an emergency or urgent situation",
    # Fallback
    "out_of_domain": "off-topic or unrelated to clinic services",
}

# Candidate labels for the zero-shot pipeline
CANDIDATE_LABELS = list(INTENT_LABELS.values())
LABEL_TO_INTENT = {v: k for k, v in INTENT_LABELS.items()}

# For fine-tuned model: direct label list (keys)
INTENT_LABEL_LIST = list(INTENT_LABELS.keys())

# Minimum confidence threshold — below this → out_of_domain
# Zero-shot naturally has lower text-entailment scores. 
# Fine-tuned models use the .env threshold.
# Evaluated dynamically during inference
MIN_CONFIDENCE = 0.20

# Hypothesis template — optimized for DeBERTa v3 xsmall NLI
HYPOTHESIS_TEMPLATE = "This text is about {}."

# ── Lazy-loaded model ───────────────────────────────────────
_classifier = None
_model_name = None
_is_fine_tuned = False

def _load_model():
    """Load the classification pipeline (once).

    Supports two modes:
      - Zero-shot: zero-shot-classification pipeline with hypothesis template
      - Fine-tuned: standard text-classification pipeline
    """
    global _classifier, _model_name, _is_fine_tuned, MIN_CONFIDENCE
    if _classifier is not None:
        return

    # Read configuration now that config.py has loaded .env
    _model_name = os.getenv(
        "INTENT_MODEL",
        "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
    )
    _is_fine_tuned = os.getenv("INTENT_FINE_TUNED", "false").lower() == "true"
    MIN_CONFIDENCE = cfg.nlu.confidence_low if _is_fine_tuned else 0.20

    from transformers import pipeline

    if _is_fine_tuned:
        logger.info(f"Loading fine-tuned intent classifier: {_model_name}")
        t0 = time.time()
        _classifier = pipeline(
            "text-classification",
            model=_model_name,
            top_k=1,
            device="cpu",
        )
        elapsed = time.time() - t0
        logger.info(f"Fine-tuned intent classifier loaded in {elapsed:.1f}s")
    else:
        logger.info(f"Loading zero-shot intent classifier: {_model_name}")
        t0 = time.time()
        _classifier = pipeline(
            "zero-shot-classification",
            model=_model_name,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            device="cpu",
        )
        elapsed = time.time() - t0
        logger.info(f"Zero-shot intent classifier loaded in {elapsed:.1f}s")


def classify_intent(text: str) -> tuple[str, float]:
    """
    Classify user text into an intent with confidence score.

    Supports two modes:
      - Zero-shot: DeBERTa NLI entailment against hypothesis templates
      - Fine-tuned: Direct text-classification prediction

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

    if _is_fine_tuned:
        # ── Fine-tuned path ──────────────────────────────
        result = _classifier(classify_text)
        # Result format: [[{"label": "book_appointment", "score": 0.95}]]
        top = result[0] if isinstance(result[0], dict) else result[0][0]
        top_label = top["label"]
        top_score = top["score"]
        intent = top_label if top_label in INTENT_LABELS else "out_of_domain"
    else:
        # ── Zero-shot path ───────────────────────────────
        result = _classifier(
            classify_text,
            candidate_labels=CANDIDATE_LABELS,
            multi_label=False,
        )
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        intent = LABEL_TO_INTENT.get(top_label, "out_of_domain")

    elapsed = time.time() - t0

    # If we stripped a greeting prefix and the remaining text classified
    # as out_of_domain with low confidence, the greeting prefix was the
    # entire meaningful content → return greeting
    if had_greeting_prefix and intent == "out_of_domain":
        intent = "greeting"
        top_score = 0.90

    # Apply minimum confidence threshold
    if top_score < MIN_CONFIDENCE:
        logger.debug(
            f"Intent below threshold: {intent}={top_score:.3f} < {MIN_CONFIDENCE} "
            f"→ out_of_domain ({elapsed:.3f}s)"
        )
        return "out_of_domain", top_score

    # Log detailed scores for debugging
    if _is_fine_tuned:
        logger.info(f"Intent: {intent} ({top_score:.3f}) [fine-tuned] ({elapsed:.3f}s)")
    elif logger.isEnabledFor(logging.DEBUG):
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
