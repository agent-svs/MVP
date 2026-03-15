#!/usr/bin/env python3
"""
Push Fine-Tuned Intent Classifier to HuggingFace Hub

Pushes the locally trained model from models/intent_classifier/
to your HuggingFace Hub account so other users can download it.

Prerequisites:
    1. pip install huggingface_hub
    2. huggingface-cli login  (or set HF_TOKEN env var)

Usage:
    python scripts/push_to_hub.py                                    # uses default repo name
    python scripts/push_to_hub.py --repo your-username/model-name    # custom repo name
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models" / "intent_classifier"


def main():
    parser = argparse.ArgumentParser(description="Push fine-tuned model to HuggingFace Hub")
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="HuggingFace repo name (e.g., 'your-username/clinic-intent-classifier'). "
             "If not specified, you will be prompted.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repo private on HuggingFace Hub",
    )
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        print(f"❌ Model directory not found: {MODEL_DIR}")
        print("   Run 'python scripts/train_intent.py' first to train the model.")
        sys.exit(1)

    repo_name = args.repo
    if not repo_name:
        repo_name = input("Enter HuggingFace repo name (e.g., your-username/clinic-intent-classifier): ").strip()
        if not repo_name:
            print("❌ Repo name is required.")
            sys.exit(1)

    print(f"\n📤 Pushing model to HuggingFace Hub: {repo_name}")
    print(f"   Source: {MODEL_DIR}")
    print(f"   Private: {args.private}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("\n📂 Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"   Labels: {model.config.num_labels}")
    print(f"   Pushing model...")

    model.push_to_hub(repo_name, private=args.private)
    print(f"   Pushing tokenizer...")
    tokenizer.push_to_hub(repo_name, private=args.private)

    print(f"\n✅ Model pushed to: https://huggingface.co/{repo_name}")
    print(f"\n   Other users can now set in their .env:")
    print(f"     INTENT_FINE_TUNED=true")
    print(f"     INTENT_MODEL={repo_name}")


if __name__ == "__main__":
    main()
