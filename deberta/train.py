"""
DeBERTa Structural Specialist — Training Script
TECPE Project (Member 2)

Usage:
    python train.py --processed_dir ./NLP_Project/processed_data \
                    --output_dir ./NLP_Project/models/member2/final_model/deberta \
                    --checkpoint_dir ./NLP_Project/models/member2/checkpoints/deberta

Loads the same HF dataset as RoBERTa (Member 1).
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

# ── Label registry (must match RoBERTa) ────────────────────────────────────
EMOTION_LIST = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EMOTION2ID   = {e: i for i, e in enumerate(EMOTION_LIST)}
ID2EMOTION   = {i: e for i, e in enumerate(EMOTION_LIST)}
NUM_LABELS   = len(EMOTION_LIST)

# ── Hyper-parameters ──────────────────────────────────────────────────────
class Config:
    MAX_LENGTH                  = 256
    BATCH_SIZE                  = 4
    GRADIENT_ACCUMULATION_STEPS = 8
    RANDOM_SEED                 = 42
    LEARNING_RATE_DEBERTA       = 2e-5
    NUM_EPOCHS_DEBERTA          = 10
    WARMUP_STEPS                = 500
    WEIGHT_DECAY                = 0.01
    SAVE_STEPS                  = 500
    EVAL_STEPS                  = 500
    LOGGING_STEPS               = 100
    SAVE_TOTAL_LIMIT            = 3
    DEBERTA_MODEL_NAME          = "microsoft/deberta-v3-base"


# ── Reproducibility ───────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── WeightedTrainer ───────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss    = torch.nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Metrics ───────────────────────────────────────────────────────────────
def compute_emotion_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0,
                            labels=list(range(NUM_LABELS)))
    metrics = {
        "f1_weighted":        float(f1),
        "precision_weighted": float(precision),
        "recall_weighted":    float(recall),
    }
    for i, emotion in enumerate(EMOTION_LIST):
        metrics[f"f1_{emotion}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0
    return metrics


# ── Tokenization ──────────────────────────────────────────────────────────
def tokenize_for_deberta(examples):
    inputs   = []
    contexts = examples.get("context", [""] * len(examples["text"]))
    for ctx, txt in zip(contexts, examples["text"]):
        ctx = ctx or ""
        inputs.append(f"{ctx} [SEP] {txt}" if ctx else txt)

    tokenized = deberta_tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_LENGTH,
        return_tensors=None,
    )
    tokenized["labels"] = examples["emotion_label"]
    return tokenized


# ── Main ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train DeBERTa Structural Specialist")
    p.add_argument("--processed_dir",  default="./NLP_Project/processed_data",
                   help="Directory with HF dataset (from RoBERTa)")
    p.add_argument("--output_dir",     default="./NLP_Project/models/member2/final_model/deberta",
                   help="Where to save the final model")
    p.add_argument("--checkpoint_dir", default="./NLP_Project/models/member2/checkpoints/deberta",
                   help="Where to save training checkpoints")
    p.add_argument("--epochs",         type=int,   default=Config.NUM_EPOCHS_DEBERTA)
    p.add_argument("--batch_size",     type=int,   default=Config.BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=Config.LEARNING_RATE_DEBERTA)
    p.add_argument("--max_length",     type=int,   default=Config.MAX_LENGTH)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(Config.RANDOM_SEED)

    print("=" * 60)
    print("DEBERTA STRUCTURAL SPECIALIST — TRAINING")
    print("=" * 60)
    print(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU only'}\n")

    # Load HF dataset (from RoBERTa)
    hf_dataset_path = Path(args.processed_dir) / "hf_dataset"
    print(f"Loading HF dataset from {hf_dataset_path}...")
    raw_dataset = load_from_disk(str(hf_dataset_path))
    print(f"✅ Train: {len(raw_dataset['train']):,}  Val: {len(raw_dataset['validation']):,}")

    # Global tokenizer
    global deberta_tokenizer
    deberta_tokenizer = AutoTokenizer.from_pretrained(Config.DEBERTA_MODEL_NAME)
    print("Tokenising for DeBERTa...")
    deberta_dataset = raw_dataset.map(
        tokenize_for_deberta,
        batched=True,
        remove_columns=[c for c in raw_dataset["train"].column_names
                        if c != "emotion_label"],
        desc="DeBERTa tokenisation",
    )
    deberta_dataset.set_format("torch")
    print("✅ Tokenisation complete")

    # Class weights
    train_labels_list = raw_dataset["train"]["emotion_label"]
    unique_labels_in_y = np.unique(train_labels_list)
    present_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_labels_in_y,
        y=train_labels_list,
    )
    class_weights_full = np.ones(NUM_LABELS, dtype=np.float32)
    for i, label in enumerate(unique_labels_in_y):
        class_weights_full[label] = present_weights[i]
    class_weights_tensor = torch.tensor(class_weights_full, dtype=torch.float32)
    print("✅ Class weights computed")

    # Model
    deberta_model = AutoModelForSequenceClassification.from_pretrained(
        Config.DEBERTA_MODEL_NAME, num_labels=NUM_LABELS,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=str(Path(args.checkpoint_dir)),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.lr,
        warmup_steps=Config.WARMUP_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        logging_steps=Config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=Config.SAVE_STEPS,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=Config.RANDOM_SEED,
        data_seed=Config.RANDOM_SEED,
        dataloader_drop_last=False,
    )

    # Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=deberta_model,
        args=training_args,
        train_dataset=deberta_dataset["train"],
        eval_dataset=deberta_dataset["validation"],
        compute_metrics=compute_emotion_metrics,
    )

    print("🚀 Training started...")
    result = trainer.train()

    final_path = Path(args.output_dir)
    final_path.mkdir(exist_ok=True, parents=True)
    trainer.save_model(str(final_path))
    deberta_tokenizer.save_pretrained(str(final_path))
    print(f"\n✅ Training complete. Loss: {result.training_loss:.4f}")
    print(f"   Model saved → {final_path}")

    # Evaluate
    print("\nEvaluating...")
    eval_results = trainer.evaluate(deberta_dataset["validation"])
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    eval_path = final_path / "eval_results_val.json"
    with open(eval_path, "w") as fh:
        json.dump(eval_results, fh, indent=2)
    print(f"✅ Eval results → {eval_path}")


if __name__ == "__main__":
    main()
