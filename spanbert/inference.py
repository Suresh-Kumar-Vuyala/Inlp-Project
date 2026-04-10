"""
SpanBERT Cause QA — Inference & Results Script
TECPE Project (Member ?)

Usage:
    python inference.py

Outputs (in --output_dir):
    eval_results_validation.json
    classification_report_validation.txt (adapted for QA)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)


# ── Paths ─────────────────────────────────────────────────────────────────
LOCAL_ROOT    = "./NLP_Project"
LOCAL_PROC    = f"{LOCAL_ROOT}/processed_data"
LOCAL_MODEL   = f"{LOCAL_ROOT}/models/spanbert_cause_qa/final_model"
LOCAL_OUTPUT  = f"{LOCAL_ROOT}/models/spanbert_cause_qa"

# ── Config ───────────────────────────────────────────────────────────────
class Config:
    SPANBERT_MODEL = "SpanBERT/spanbert-base-cased"
    MAX_LENGTH = 384


# ── Inference helper ──────────────────────────────────────────────────────
class SpanBERTInference:
    def __init__(self, model_dir: str, max_length: int = 384):
        self.model_dir  = Path(model_dir)
        self.max_length = max_length
        print(f"Loading tokenizer + model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        print(f"✅ Model ready on {device.upper()}")

    def run(self, dataset, raw_dataset, output_dir: str, split: str = "validation"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\nRunning inference on '{split}' split ({len(dataset)} samples)...")

        f1_scores = []
        em_scores = []

        for i in range(len(dataset)):
            item = dataset[i]
            raw_item = raw_dataset[split][i]

            input_ids = torch.tensor(item['input_ids']).unsqueeze(0).to(self.model.device)
            attention_mask = torch.tensor(item['attention_mask']).unsqueeze(0).to(self.model.device)

            gold_start = item['start_positions'].item()
            gold_end = item['end_positions'].item()

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            pred_start = torch.argmax(outputs.start_logits, dim=-1).item()
            pred_end = torch.argmax(outputs.end_logits, dim=-1).item()

            pred_tokens = input_ids[0][pred_start: pred_end + 1]
            gold_tokens = input_ids[0][gold_start: gold_end + 1]

            prediction_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True).strip().lower()
            ground_truth_text = self.tokenizer.decode(gold_tokens, skip_special_tokens=True).strip().lower()

            # EM
            em = 1 if prediction_text == ground_truth_text else 0
            em_scores.append(em)

            # F1
            pred_words = prediction_text.split()
            gold_words = ground_truth_text.split()

            if len(pred_words) == 0 or len(gold_words) == 0:
                f1 = 1 if len(pred_words) == len(gold_words) else 0
            else:
                common = Counter(pred_words) & Counter(gold_words)
                num_same = sum(common.values())
                precision = 1.0 * num_same / len(pred_words)
                recall = 1.0 * num_same / len(gold_words)
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)

        # Metrics
        em_mean = np.mean(em_scores)
        f1_mean = np.mean(f1_scores)

        print(f"\n{'='*50}")
        print(f"  Exact Match (EM): {em_mean:.4f}")
        print(f"  F1 Score:        {f1_mean:.4f}")

        # Eval JSON
        eval_results = {
            "eval_em": float(em_mean),
            "eval_f1": float(f1_mean),
        }

        json_path = output_dir / "eval_results_validation.json"
        with open(json_path, "w") as fh:
            json.dump(eval_results, fh, indent=2)
        print(f"\n✅ Eval JSON → {json_path}")

        # Classification report (adapted)
        report = f"QA Evaluation Results:\nEM: {em_mean:.4f}\nF1: {f1_mean:.4f}\n"

        report_path = output_dir / "classification_report_validation.txt"
        with open(report_path, "w") as fh:
            fh.write(report)
        print(f"✅ Report → {report_path}")

        return eval_results


# ── Main ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SpanBERT Cause QA inference")
    p.add_argument("--model_dir",     default=LOCAL_MODEL,
                   help=f"Path to saved model  (default: {LOCAL_MODEL})")
    p.add_argument("--processed_dir", default=LOCAL_PROC,
                   help=f"Processed data dir  (default: {LOCAL_PROC})")
    p.add_argument("--output_dir",    default=LOCAL_OUTPUT,
                   help=f"Where to write outputs  (default: {LOCAL_OUTPUT})")
    p.add_argument("--split",         default="validation",
                   help="Dataset split: train / validation")
    p.add_argument("--max_length",    type=int, default=384)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SPANBERT CAUSE QA — INFERENCE & RESULTS")
    print("=" * 60)
    print(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU only'}")
    print(f"Model dir    : {args.model_dir}")
    print(f"Processed dir: {args.processed_dir}")
    print(f"Output dir   : {args.output_dir}\n")

    # Validate paths
    if not Path(args.model_dir).exists():
        sys.exit(f"❌ model_dir not found: {args.model_dir}")
    tok_path = Path(args.processed_dir) / "cause_qa_tokenized"
    raw_path = Path(args.processed_dir) / "cause_qa_dataset"
    if not tok_path.exists() or not raw_path.exists():
        sys.exit(f"❌ Datasets not found in {args.processed_dir}")

    # Load datasets
    tokenized_dataset = DatasetDict.load_from_disk(str(tok_path))
    raw_dataset = DatasetDict.load_from_disk(str(raw_path))
    split_dataset = tokenized_dataset[args.split]
    print(f"✅ '{args.split}' split: {len(split_dataset)} samples")

    # Inference
    inferer = SpanBERTInference(args.model_dir, max_length=args.max_length)
    results = inferer.run(split_dataset, raw_dataset, args.output_dir, split=args.split)

    print("\n✅ INFERENCE COMPLETE")
    print(f"   EM: {results['eval_em']:.4f} | F1: {results['eval_f1']:.4f}")


if __name__ == "__main__":
    main()
