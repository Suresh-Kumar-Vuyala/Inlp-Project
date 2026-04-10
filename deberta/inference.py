"""
DeBERTa Structural Specialist — Inference & Results Script
TECPE Project (Member 2)

Usage:
    python inference.py

Outputs (in --output_dir):
    deberta_logits_val.csv   ← primary deliverable for Member 3
    classification_report_val.txt
    eval_results_val.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
LOCAL_ROOT    = "./NLP_Project"
LOCAL_PROC    = f"{LOCAL_ROOT}/processed_data"
LOCAL_MODEL   = f"{LOCAL_ROOT}/models/member2/final_model/deberta"
LOCAL_OUTPUT  = f"{LOCAL_ROOT}/models/member2"

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ── Label registry ────────────────────────────────────────────────────────
EMOTION_LIST = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
NUM_LABELS   = 7

EMOTION2ID   = {e: i for i, e in enumerate(EMOTION_LIST)}
ID2EMOTION   = {i: e for i, e in enumerate(EMOTION_LIST)}


# ── Inference helper ──────────────────────────────────────────────────────
class DeBERTaInference:
    def __init__(self, model_dir: str, max_length: int = 256):
        self.model_dir  = Path(model_dir)
        self.max_length = max_length
        print(f"Loading tokenizer + model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        print(f"✅ Model ready on {device.upper()}")

    def _tokenize(self, examples):
        inputs   = []
        contexts = examples.get("context", [""] * len(examples["text"]))
        for ctx, txt in zip(contexts, examples["text"]):
            ctx = ctx or ""
            inputs.append(f"{ctx} [SEP] {txt}" if ctx else txt)
        tok = self.tokenizer(inputs, padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors=None)
        tok["labels"] = examples["emotion_label"]
        return tok

    def prepare_dataset(self, dataset):
        remove_cols = [c for c in dataset.column_names if c != "emotion_label"]
        dataset = dataset.map(self._tokenize, batched=True,
                              remove_columns=remove_cols,
                              desc="Tokenising for inference")
        dataset.set_format("torch")
        return dataset

    def run(self, dataset, output_dir: str, split: str = "validation"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\nRunning inference on '{split}' split ({len(dataset):,} samples)...")

        # Use Trainer just for its predict()
        dummy_args = TrainingArguments(
            output_dir=str(output_dir / ".tmp_trainer"),
            per_device_eval_batch_size=8,
            report_to="none",
            dataloader_drop_last=False,
        )
        trainer = Trainer(model=self.model, args=dummy_args)

        predictions = trainer.predict(dataset)
        logits      = predictions.predictions          # (N, 7)
        true_labels = predictions.label_ids            # (N,)
        pred_labels = np.argmax(logits, axis=-1)
        probs       = torch.nn.functional.softmax(
            torch.tensor(logits, dtype=torch.float32), dim=-1
        ).numpy()

        # ── 1. Logit CSV (primary deliverable for ensemble) ───────────────
        df = pd.DataFrame({
            "index":        range(len(true_labels)),
            "true_label":   true_labels,
            "true_emotion": [ID2EMOTION[l] for l in true_labels],
        })
        for i, emotion in enumerate(EMOTION_LIST):
            df[f"logit_{emotion}"] = logits[:, i]
        for i, emotion in enumerate(EMOTION_LIST):
            df[f"prob_{emotion}"]  = probs[:, i]

        csv_path = output_dir / "deberta_logits_val.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Logit CSV → {csv_path}")
        print(f"   Rows    : {len(df):,}")
        print(f"   Columns : {list(df.columns)}")

        # ── 2. Accuracy & weighted F1 ─────────────────────────────────────
        accuracy = (pred_labels == true_labels).mean()
        print(f"\n{'='*50}")
        print(f"  Accuracy (validation) : {accuracy:.4f}")

        # ── 3. Classification report ──────────────────────────────────────
        report = classification_report(
            true_labels, pred_labels,
            labels=list(range(NUM_LABELS)),
            target_names=EMOTION_LIST,
            digits=4, zero_division=0
        )
        print(f"\nClassification Report:\n{report}")

        report_path = output_dir / "classification_report_val.txt"
        with open(report_path, "w") as fh:
            fh.write(report)
        print(f"✅ Classification report → {report_path}")

        # ── 4. Per-class summary ──────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"  Per-class prediction counts:")
        unique, counts = np.unique(pred_labels, return_counts=True)
        for lbl, cnt in zip(unique, counts):
            print(f"    {EMOTION_LIST[lbl]:10s} (label {lbl}): {cnt:,} predicted")

        # ── 5. Eval JSON ──────────────────────────────────────────────────
        from sklearn.metrics import precision_recall_fscore_support, f1_score
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="weighted", zero_division=0
        )
        per_class_f1 = f1_score(true_labels, pred_labels, average=None,
                                zero_division=0, labels=list(range(NUM_LABELS)))
        eval_results = {
            "accuracy":           float(accuracy),
            "f1_weighted":        float(f1),
            "precision_weighted": float(precision),
            "recall_weighted":    float(recall),
        }
        for i, emotion in enumerate(EMOTION_LIST):
            eval_results[f"f1_{emotion}"] = float(per_class_f1[i])

        json_path = output_dir / "eval_results_val.json"
        with open(json_path, "w") as fh:
            json.dump(eval_results, fh, indent=2)
        print(f"\n✅ Eval JSON → {json_path}")
        print(f"{'='*50}")
        print(f"  f1_weighted : {f1:.4f}")
        print(f"  accuracy    : {accuracy:.4f}")
        print(f"{'='*50}")

        return df, eval_results


# ── Main ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DeBERTa inference + results")
    p.add_argument("--model_dir",     default=LOCAL_MODEL,
                   help=f"Path to saved model  (default: {LOCAL_MODEL})")
    p.add_argument("--processed_dir", default=LOCAL_PROC,
                   help=f"Preprocessed HF dataset dir  (default: {LOCAL_PROC})")
    p.add_argument("--output_dir",    default=LOCAL_OUTPUT,
                   help=f"Where to write outputs  (default: {LOCAL_OUTPUT})")
    p.add_argument("--split",         default="validation",
                   help="Dataset split: train / validation")
    p.add_argument("--max_length",    type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("DEBERTA STRUCTURAL SPECIALIST — INFERENCE & RESULTS")
    print("=" * 60)
    print(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU only'}")
    print(f"Model dir    : {args.model_dir}")
    print(f"Processed dir: {args.processed_dir}")
    print(f"Output dir   : {args.output_dir}\n")

    # Validate paths exist
    if not Path(args.model_dir).exists():
        sys.exit(f"❌ model_dir not found: {args.model_dir}\n"
                 f"   Check your path or pass --model_dir explicitly.")
    dataset_path = Path(args.processed_dir) / "hf_dataset"
    if not dataset_path.exists():
        sys.exit(f"❌ HF dataset not found at: {dataset_path}\n"
                 f"   Run RoBERTa train.py first, or pass --processed_dir explicitly.")

    # Load pre-processed dataset
    print(f"Loading dataset from {dataset_path}...")
    full_dataset = load_from_disk(str(dataset_path))
    split_dataset = full_dataset[args.split]
    print(f"✅ '{args.split}' split: {len(split_dataset):,} samples")

    # Inference
    inferer  = DeBERTaInference(args.model_dir, max_length=args.max_length)
    tokenized = inferer.prepare_dataset(split_dataset)
    df, results = inferer.run(tokenized, args.output_dir, split=args.split)

    print("\n✅ INFERENCE COMPLETE")
    print(f"   Primary deliverable → {args.output_dir}/deberta_logits_val.csv")
    print(f"   Rows: {len(df):,}  |  Weighted F1: {results['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
