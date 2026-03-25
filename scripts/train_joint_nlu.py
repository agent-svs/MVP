"""
Joint NLU Model — Production Training Script

Trains a JointDeBERTa model for simultaneous intent classification and
slot tagging on the clinic voice-receptionist dataset.

Usage:
    python scripts/train_joint_nlu.py

    # Custom args:
    python scripts/train_joint_nlu.py --epochs 20 --lr 2e-5 --batch_size 8

The script expects pre-processed data in data/ directory:
    data/joint_nlu_train.jsonl
    data/joint_nlu_val.jsonl
    data/joint_nlu_test.jsonl

Each line is JSON with fields: context, utterance, intent, slots, tokens, slot_labels
"""

import os
import sys
import json
import argparse
import logging
from collections import Counter

import numpy as np
import torch
import evaluate

from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.nlu.label_maps import (
    NUM_INTENTS, NUM_SLOT_LABELS,
    ID_TO_INTENT, ID_TO_TAG, INTENT_TO_ID,
)
from core.nlu.joint_model import JointDeBERTa
from scripts.joint_dataset_loader import JointNLUDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Lazy-loaded seqeval metric ──────────────────────────────
_seqeval = None

def _get_seqeval():
    global _seqeval
    if _seqeval is None:
        _seqeval = evaluate.load("seqeval")
    return _seqeval


def compute_intent_class_weights(data_path: str) -> torch.Tensor:
    """Compute inverse-frequency class weights for intent labels."""
    counts = Counter()
    total = 0
    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            counts[INTENT_TO_ID.get(d["intent"], 0)] += 1
            total += 1

    weights = torch.ones(NUM_INTENTS)
    for class_id, count in counts.items():
        # Inverse frequency, clamped to avoid extreme values
        weights[class_id] = min(total / (count * NUM_INTENTS), 5.0)

    # Normalize so mean = 1.0
    weights = weights / weights.mean()
    logger.info(f"Intent class weights (min={weights.min():.2f}, max={weights.max():.2f})")
    return weights


def compute_metrics(eval_preds):
    """
    Compute intent accuracy and slot F1 for evaluation.

    HF Trainer passes:
      - eval_preds.predictions: tuple of (intent_logits, slot_logits)
      - eval_preds.label_ids: tuple of (intent_labels, slot_labels)
    because we set label_names=["intent_labels", "slot_labels"].
    """
    predictions, label_ids = eval_preds

    # Unpack — Trainer collects these as tuples because of label_names
    intent_logits = predictions[0]
    slot_logits = predictions[1]
    intent_labels = label_ids[0]
    slot_labels = label_ids[1]

    # ── Intent accuracy ─────────────────────────────────────
    intent_preds = np.argmax(intent_logits, axis=-1)
    intent_acc = (intent_preds == intent_labels).mean()

    # ── Slot F1 (seqeval) ───────────────────────────────────
    slot_preds = np.argmax(slot_logits, axis=-1)

    true_predictions = []
    true_labels = []
    for pred_seq, label_seq in zip(slot_preds, slot_labels):
        pred_tags = []
        label_tags = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            pred_tags.append(ID_TO_TAG.get(int(p), "O"))
            label_tags.append(ID_TO_TAG.get(int(l), "O"))
        if pred_tags:  # skip empty sequences
            true_predictions.append(pred_tags)
            true_labels.append(label_tags)

    seqeval = _get_seqeval()
    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    slot_f1 = results["overall_f1"]
    joint_f1 = (intent_acc + slot_f1) / 2.0

    metrics = {
        "intent_accuracy": float(intent_acc),
        "slot_precision": results["overall_precision"],
        "slot_recall": results["overall_recall"],
        "slot_f1": slot_f1,
        "joint_f1": joint_f1,
    }

    # Per-slot-type F1
    for key, value in results.items():
        if isinstance(value, dict) and "f1" in value:
            metrics[f"slot_{key}_f1"] = value["f1"]

    return metrics


class WeightedJointTrainer(Trainer):
    """Custom Trainer that applies class-weighted intent loss."""

    def __init__(self, intent_class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.intent_class_weights = intent_class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override loss computation to use class-weighted intent CE."""
        intent_labels = inputs.pop("intent_labels", None)
        slot_labels = inputs.pop("slot_labels", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            intent_labels=intent_labels,
            slot_labels=slot_labels,
        )

        if intent_labels is not None and slot_labels is not None and self.intent_class_weights is not None:
            # Recompute intent loss with class weights
            from torch.nn import CrossEntropyLoss

            weight = self.intent_class_weights.to(outputs["intent_logits"].device)
            loss_fct = CrossEntropyLoss(weight=weight)
            intent_loss = loss_fct(
                outputs["intent_logits"].view(-1, model.num_intents),
                intent_labels.view(-1),
            )

            # Slot loss from the model's own computation
            slot_loss = outputs["slot_loss"] if outputs["slot_loss"] is not None else torch.tensor(0.0)

            # Combine with the model's configured weight split
            intent_w = model.intent_loss_weight
            loss = intent_w * intent_loss + (1.0 - intent_w) * slot_loss
        else:
            loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Train Joint NLU Model")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base",
                        help="Base model from HuggingFace Hub")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory with joint_nlu_{train,val,test}.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/joint_nlu/",
                        help="Where to save the trained model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Max training epochs (early stopping may stop sooner)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Fraction of total steps used for LR warmup")
    parser.add_argument("--intent_weight", type=float, default=0.6,
                        help="Intent loss weight (slot weight = 1 - this)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="Max token sequence length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_class_weights", action="store_true",
                        help="Disable class-weighted intent loss")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Joint NLU Training — Production Run")
    logger.info("=" * 60)
    logger.info(f"  Base model:       {args.model_name}")
    logger.info(f"  Data dir:         {args.data_dir}")
    logger.info(f"  Output dir:       {args.output_dir}")
    logger.info(f"  Epochs:           {args.epochs} (patience={args.patience})")
    logger.info(f"  Batch:            {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    logger.info(f"  Learning rate:    {args.lr}")
    logger.info(f"  Warmup:           {args.warmup_ratio:.0%} of steps")
    logger.info(f"  Weight decay:     {args.weight_decay}")
    logger.info(f"  Label smoothing:  {args.label_smoothing}")
    logger.info(f"  Intent weight:    {args.intent_weight}")
    logger.info(f"  Max seq len:      {args.max_seq_len}")
    logger.info(f"  Seed:             {args.seed}")

    # ── Device detection ────────────────────────────────────
    if torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    logger.info(f"  Device:           {device_str}")
    logger.info("=" * 60)

    # ── 1. Tokenizer ────────────────────────────────────────
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    special_tokens_dict = {"additional_special_tokens": ["[SYS]", "[USR]"]}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added} special tokens: [SYS], [USR]")

    # ── 2. Model ────────────────────────────────────────────
    logger.info("Loading base model...")
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_intents = NUM_INTENTS
    config.num_slot_labels = NUM_SLOT_LABELS
    config.intent_loss_weight = args.intent_weight

    model = JointDeBERTa.from_pretrained(args.model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── 3. Datasets ─────────────────────────────────────────
    train_path = os.path.join(args.data_dir, "joint_nlu_train.jsonl")
    val_path = os.path.join(args.data_dir, "joint_nlu_val.jsonl")
    test_path = os.path.join(args.data_dir, "joint_nlu_test.jsonl")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            logger.error(f"Missing data file: {p}")
            sys.exit(1)

    logger.info("Loading datasets...")
    train_dataset = JointNLUDataset(train_path, tokenizer, max_seq_len=args.max_seq_len)
    val_dataset = JointNLUDataset(val_path, tokenizer, max_seq_len=args.max_seq_len)
    test_dataset = JointNLUDataset(test_path, tokenizer, max_seq_len=args.max_seq_len)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ── 4. Class weights ────────────────────────────────────
    intent_weights = None
    if not args.no_class_weights:
        intent_weights = compute_intent_class_weights(train_path)

    # ── 5. Training arguments ───────────────────────────────
    # NOTE: Do NOT set fp16=True on MPS — it causes numerical garbage.
    # bf16 is also not supported on MPS.
    use_fp16 = device_str == "cuda"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        label_smoothing_factor=args.label_smoothing,
        fp16=use_fp16,
        # Evaluation and saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="joint_f1",
        greater_is_better=True,
        # Logging
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=20,
        logging_first_step=True,
        report_to="none",
        # Important for custom model
        remove_unused_columns=False,
        label_names=["intent_labels", "slot_labels"],
        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
    )

    # ── 6. Trainer ──────────────────────────────────────────
    trainer = WeightedJointTrainer(
        intent_class_weights=intent_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # ── 7. Train! ───────────────────────────────────────────
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log training summary
    logger.info(f"Training complete!")
    logger.info(f"  Total steps:    {train_result.global_step}")
    logger.info(f"  Training loss:  {train_result.training_loss:.4f}")

    # ── 8. Evaluate on test set ─────────────────────────────
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")

    logger.info("=" * 60)
    logger.info("  TEST RESULTS")
    logger.info("=" * 60)
    for key, val in sorted(test_results.items()):
        if isinstance(val, float):
            logger.info(f"  {key:30s} {val:.4f}")
    logger.info("=" * 60)

    # Sanity check — flag if model collapsed
    test_intent_acc = test_results.get("test_intent_accuracy", 0)
    test_slot_f1 = test_results.get("test_slot_f1", 0)
    test_joint_f1 = test_results.get("test_joint_f1", 0)

    if test_intent_acc < 0.50:
        logger.warning(f"⚠️  Intent accuracy is very low ({test_intent_acc:.2%}). Model may have collapsed!")
    if test_slot_f1 < 0.50:
        logger.warning(f"⚠️  Slot F1 is very low ({test_slot_f1:.2%}). Model may have collapsed!")
    if test_joint_f1 >= 0.85:
        logger.info("✅ Model quality looks good! (joint_f1 >= 0.85)")
    elif test_joint_f1 >= 0.75:
        logger.info("⚠️  Model quality is acceptable but could be better.")
    else:
        logger.warning("❌ Model quality is poor. Consider retraining with different hyperparameters.")

    # ── 9. Save final model ─────────────────────────────────
    logger.info(f"Saving best model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── 10. Quick sanity inference ──────────────────────────
    logger.info("Running quick sanity check...")
    model.eval()
    device = next(model.parameters()).device

    sanity_tests = [
        ("How can I help you?", "I want to book an appointment", "book_appointment"),
        ("Which day works best?", "Tomorrow", "slot_answer"),
        ("", "Hello", "greeting"),
        ("", "I want to cancel my appointment", "cancel_appointment"),
    ]

    all_pass = True
    for ctx, utt, expected in sanity_tests:
        ctx_enc = tokenizer(ctx, add_special_tokens=False)
        utt_enc = tokenizer(utt, add_special_tokens=False)
        sys_id = tokenizer.convert_tokens_to_ids("[SYS]")
        usr_id = tokenizer.convert_tokens_to_ids("[USR]")

        input_ids = (
            [tokenizer.cls_token_id, sys_id] + ctx_enc["input_ids"] +
            [tokenizer.sep_token_id, usr_id] + utt_enc["input_ids"] +
            [tokenizer.sep_token_id]
        )
        attn = [1] * len(input_ids)

        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([input_ids]).to(device),
                attention_mask=torch.tensor([attn]).to(device),
            )
            probs = torch.softmax(out["intent_logits"][0], dim=-1)
            pred_id = torch.argmax(probs).item()
            pred_intent = ID_TO_INTENT.get(pred_id, "???")
            conf = probs[pred_id].item()

        status = "✅" if pred_intent == expected else "❌"
        if pred_intent != expected:
            all_pass = False
        logger.info(f"  {status} \"{utt}\" → {pred_intent} ({conf:.3f})  [expected: {expected}]")

    if all_pass:
        logger.info("✅ All sanity checks passed!")
    else:
        logger.warning("⚠️  Some sanity checks failed — review model quality.")

    logger.info("Done! 🎉")


if __name__ == "__main__":
    main()
