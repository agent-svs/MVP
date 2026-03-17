import os
import argparse
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.nlu.label_maps import NUM_INTENTS, NUM_SLOT_LABELS, ID_TO_TAG
from core.nlu.joint_model import JointDeBERTa
from scripts.joint_dataset_loader import JointNLUDataset

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    
    # logits might include the loss or other values when return_dict=True
    # in our custom model we return a dict, but HF Trainer might pass it as a tuple
    intent_logits = logits[0] if isinstance(logits, tuple) else logits
    slot_logits = logits[1] if isinstance(logits, tuple) and len(logits) > 1 else None
    
    # Same for labels, if trainer passes them as a tuple
    intent_labels = labels[0] if isinstance(labels, tuple) else labels
    slot_labels = labels[1] if isinstance(labels, tuple) and len(labels) > 1 else None
    
    intent_preds = np.argmax(intent_logits, axis=-1)
    intent_acc = (intent_preds == intent_labels).mean()
    
    slot_preds = np.argmax(slot_logits, axis=-1)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [ID_TO_TAG[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_preds, slot_labels)
    ]
    true_labels = [
        [ID_TO_TAG[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(slot_preds, slot_labels)
    ]
    
    # Load seqeval
    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    slot_f1 = results["overall_f1"]
    
    # Combined metric
    joint_f1 = (intent_acc + slot_f1) / 2
    
    metrics = {
        "intent_accuracy": sum(intent_preds == intent_labels) / len(intent_labels),
        "slot_precision": results["overall_precision"],
        "slot_recall": results["overall_recall"],
        "slot_f1": slot_f1,
        "joint_f1": joint_f1
    }
    
    # Add individual slot detailed metrics if needed
    for key, value in results.items():
        if isinstance(value, dict):
            metrics[f"slot_{key}_f1"] = value["f1"]
            
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="models/joint_nlu/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--intent_weight", type=float, default=0.6)
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    special_tokens_dict = {'additional_special_tokens': ['[SYS]', '[USR]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens to tokenizer.")
    
    # 2. Load config and model
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_intents = NUM_INTENTS
    config.num_slot_labels = NUM_SLOT_LABELS
    config.intent_loss_weight = args.intent_weight
    
    model = JointDeBERTa.from_pretrained(args.model_name, config=config)
    
    # Important: resize token embeddings since we added [SYS] and [USR]
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. Load datasets
    print("Loading datasets...")
    train_dataset = JointNLUDataset(os.path.join(args.data_dir, "joint_nlu_train.jsonl"), tokenizer)
    val_dataset = JointNLUDataset(os.path.join(args.data_dir, "joint_nlu_val.jsonl"), tokenizer)
    test_dataset = JointNLUDataset(os.path.join(args.data_dir, "joint_nlu_test.jsonl"), tokenizer)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # 4. Setup TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="joint_f1",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=10,
        remove_unused_columns=False, # Important: our forward pass takes dict outputs custom
        label_names=["intent_labels", "slot_labels"]
    )
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 6. Train!
    print("Starting training...")
    trainer.train()
    
    # 7. Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")
    
    # 8. Save final best model & tokenizer
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
