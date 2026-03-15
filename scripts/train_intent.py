#!/usr/bin/env python3
"""
Fine-Tune Intent Classifier — DeBERTa v3 xsmall

Trains a text-classification model on clinic_intent_dataset.jsonl
and saves it to models/intent_classifier/.

Usage:
    python scripts/train_intent.py

Hardware:
    Runs on Apple Silicon (MPS), CUDA GPU, or CPU.
    M4 Air 16GB RAM: ~5-10 minutes, ~2-3GB RAM usage.

After training:
    Set INTENT_FINE_TUNED=true in .env to use the fine-tuned model.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaV2Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ── Configuration ───────────────────────────────────────────
BASE_MODEL = "microsoft/deberta-v3-xsmall"   # 22M params
DATASET_PATH = PROJECT_ROOT / "clinic_intent_dataset.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "intent_classifier"
LOGGING_DIR = PROJECT_ROOT / "models" / "intent_classifier" / "logs"

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_LENGTH = 128          # Max token length (utterances are short)
VAL_SPLIT = 0.1           # 10% validation
SEED = 42


# ── Dataset ─────────────────────────────────────────────────
class IntentDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset for intent classification."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Load JSONL dataset and return texts and labels."""
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(obj["label"])
    return texts, labels


def compute_metrics(eval_pred):
    """Compute accuracy for the Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def main():
    print("=" * 60)
    print("  🧠 Fine-Tuning Intent Classifier")
    print("=" * 60)
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Dataset:     {DATASET_PATH}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  LR:          {LEARNING_RATE}")
    print(f"  Max length:  {MAX_LENGTH}")
    print("=" * 60)

    # ── Detect device ───────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n📱 Device: {device}")

    # Note: MPS has some issues with certain operations in transformers.
    # If training fails on MPS, fall back to CPU.
    use_mps = device == "mps"
    if use_mps:
        print("   Using MPS (Apple Silicon). If training errors occur, set")
        print("   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 or use CPU fallback.")

    # ── Load dataset ────────────────────────────────────────
    print("\n📂 Loading dataset...")
    texts, labels = load_dataset(DATASET_PATH)
    print(f"   Total examples: {len(texts)}")

    # Build label mapping
    unique_labels = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(unique_labels)
    print(f"   Intent labels: {num_labels}")

    # Convert string labels to integer IDs
    label_ids = [label2id[label] for label in labels]

    # Split train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, label_ids, test_size=VAL_SPLIT, random_state=SEED, stratify=label_ids
    )
    print(f"   Train: {len(train_texts)} | Val: {len(val_texts)}")

    # ── Tokenize ────────────────────────────────────────────
    print(f"\n🔤 Tokenizing with {BASE_MODEL}...")
    # Note: AutoTokenizer + use_fast=False crashes on DeBERTa v3 in
    # transformers 4.56 due to a fast tokenizer conversion bug.
    # Use DebertaV2Tokenizer directly to bypass the issue.
    tokenizer = DebertaV2Tokenizer.from_pretrained(BASE_MODEL)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)

    # ── Load model ──────────────────────────────────────────
    print(f"\n🏗️  Loading {BASE_MODEL} for sequence classification ({num_labels} labels)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ── Training arguments ──────────────────────────────────
    # Use CPU for eval if on MPS (avoids some MPS-specific issues)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=str(LOGGING_DIR),
        logging_steps=50,
        report_to="none",       # Disable wandb/tensorboard
        seed=SEED,
        fp16=False,             # Not supported on MPS
        bf16=False,
        use_mps_device=use_mps,
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Mac
    )

    # ── Trainer ─────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ───────────────────────────────────────────────
    print("\n🚀 Starting training...")
    print("-" * 60)
    train_result = trainer.train()
    print("-" * 60)
    print(f"\n✅ Training complete!")
    print(f"   Total steps: {train_result.global_step}")
    print(f"   Training loss: {train_result.training_loss:.4f}")

    # ── Evaluate ────────────────────────────────────────────
    print("\n📊 Evaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"   Loss:     {eval_results['eval_loss']:.4f}")

    # Detailed classification report
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    print("\n📋 Classification Report:")
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=[id2label[i] for i in range(num_labels)],
        digits=3,
    ))

    # ── Save model ──────────────────────────────────────────
    print(f"\n💾 Saving model to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save model and tokenizer
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Save label mapping for reference
    label_map_path = OUTPUT_DIR / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump({
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
            "num_labels": num_labels,
        }, f, indent=2)

    print(f"   Model saved: {OUTPUT_DIR}")
    print(f"   Label map:   {label_map_path}")

    # ── Summary ─────────────────────────────────────────────
    model_size_mb = sum(
        f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("  ✅ Fine-tuning complete!")
    print("=" * 60)
    print(f"  Model size:    {model_size_mb:.1f} MB")
    print(f"  Val accuracy:  {eval_results['eval_accuracy']:.4f}")
    print(f"  Labels:        {num_labels}")
    print(f"  Saved to:      {OUTPUT_DIR}")
    print()
    print("  To use the fine-tuned model, set in your .env:")
    print("    INTENT_FINE_TUNED=true")
    print(f"    INTENT_MODEL={OUTPUT_DIR}")
    print()
    print("  To push to HuggingFace Hub:")
    print("    huggingface-cli login")
    print("    python scripts/push_to_hub.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
