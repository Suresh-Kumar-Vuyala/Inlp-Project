"""
LLM-Lite Social Reasoning Expert — Inference & Results Script
TECPE Project (Member 3)

Usage:
    python inference.py

Outputs (in --output_dir):
    llm_lite_logits_val.csv   ← primary deliverable for Member 3
    classification_report_val.txt
    eval_results_val.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ── Label registry ────────────────────────────────────────────────────────
EMOTION_LIST = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
NUM_LABELS   = 7

EMOTION2ID   = {e: i for i, e in enumerate(EMOTION_LIST)}
ID2EMOTION   = {i: e for i, e in enumerate(EMOTION_LIST)}
EMOTION_OPTIONS = ", ".join(EMOTION_LIST)


# ── Paths ─────────────────────────────────────────────────────────────────
LOCAL_ROOT    = "./NLP_Project"
LOCAL_DATA    = f"{LOCAL_ROOT}/data"
LOCAL_PROC    = f"{LOCAL_ROOT}/processed_data"
LOCAL_MODEL   = f"{LOCAL_ROOT}/models/lite_llm/lora_adapters"
LOCAL_OUTPUT  = f"{LOCAL_ROOT}/models/lite_llm"

# ── Config ───────────────────────────────────────────────────────────────
class Config:
    LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
    MAX_LENGTH = 256


# ── Prompt building ──────────────────────────────────────────────────────
def format_context(context: str) -> str:
    if not context or context.strip() == "":
        return "(none)"
    return context.replace(" | ", "\n")


def build_prompt(row: dict, include_label: bool = False) -> str:
    context = format_context(row.get('context', ''))
    speaker = row.get('speaker', 'Speaker')
    text    = row.get('text', '').strip()

    system = (
        f"You are an expert in conversational emotion analysis.\n"
        f"Choose exactly one from: {EMOTION_OPTIONS}.\n"
        f"Output ONLY the emotion label, nothing else."
    )

    user = (
        f"Context:\n{context}\n"
        f"Utterance: {speaker}: \"{text}\"\n"
        f"Emotion:"
    )

    prompt = (
        f"<|system|>\n{system}<|end|>\n"
        f"<|user|>\n{user}<|end|>\n"
        f"<|assistant|>\n"
    )
    return prompt


# ── Inference helper ──────────────────────────────────────────────────────
class LLMInference:
    def __init__(self, model_dir: str, max_length: int = 256):
        self.model_dir  = Path(model_dir)
        self.max_length = max_length
        print(f"Loading tokenizer + model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
            device_map={"": 0},
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, model_dir)
        self.model.eval()

        # Emotion token IDs
        PREFIX = "x"
        prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        n_prefix = len(prefix_ids)
        self.emotion_token_ids = {}
        for emo in EMOTION_LIST:
            full_ids = self.tokenizer.encode(PREFIX + " " + emo, add_special_tokens=False)
            self.emotion_token_ids[emo] = full_ids[n_prefix:]

        print(f"✅ Model ready on GPU")

    def prepare_dataset(self, df: pd.DataFrame):
        records = df.to_dict('records')
        prompts = [build_prompt(r, include_label=False) for r in records]
        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None,
        )
        return records, tokenized

    def run(self, df: pd.DataFrame, output_dir: str, split: str = "validation"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\nRunning inference on '{split}' split ({len(df)} samples)...")

        records, _ = self.prepare_dataset(df)
        results = []

        for row in records:
            context = format_context(row.get('context', ''))
            utterance = row['text']

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in conversational emotion analysis.\n"
                        "Choose exactly one from: anger, disgust, fear, joy, sadness, surprise, neutral.\n"
                        "Output ONLY the emotion label, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\nUtterance: {utterance}\nEmotion:"
                }
            ]

            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                outputs = self.model(input_ids=prompt_ids)
                last_logits = outputs.logits[:, -1, :]

                emotion_scores = []
                for emo in EMOTION_LIST:
                    tids = self.emotion_token_ids[emo]
                    score = last_logits[0, tids[0]]
                    emotion_scores.append(score)

                logits = torch.stack(emotion_scores).cpu().numpy()
                probs = F.softmax(torch.stack(emotion_scores), dim=0).cpu().numpy()

            results.append((logits, probs))

        # Build unified schema DataFrame
        rows = []
        for i, (logits, probs) in enumerate(results):
            true_label = df.iloc[i]['emotion_label']
            true_emotion = df.iloc[i]['emotion']
            row = {
                'index': i,
                'true_label': true_label,
                'true_emotion': true_emotion,
            }
            for j, emo in enumerate(EMOTION_LIST):
                row[f'logit_{emo}'] = logits[j]
                row[f'prob_{emo}'] = probs[j]
            rows.append(row)

        df_out = pd.DataFrame(rows)
        csv_path = output_dir / "llm_lite_logits_val.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"\n✅ Logit CSV → {csv_path}")
        print(f"   Rows    : {len(df_out)}")
        print(f"   Columns : {list(df_out.columns)}")

        # Predictions
        pred_labels = df_out[[f'prob_{e}' for e in EMOTION_LIST]].idxmax(axis=1).str.replace('prob_', '')
        pred_emotions = [e for e in pred_labels]
        true_labels = df['emotion'].tolist()

        # Accuracy
        accuracy = (np.array(pred_labels) == np.array(true_labels)).mean()
        print(f"\n{'='*50}")
        print(f"  Accuracy (validation) : {accuracy:.4f}")

        # Classification report
        report = classification_report(
            true_labels, pred_labels,
            labels=EMOTION_LIST,
            target_names=EMOTION_LIST,
            digits=4, zero_division=0
        )
        print(f"\nClassification Report:\n{report}")

        report_path = output_dir / "classification_report_val.txt"
        with open(report_path, "w") as fh:
            fh.write(report)
        print(f"✅ Classification report → {report_path}")

        # Eval JSON
        from sklearn.metrics import precision_recall_fscore_support, f1_score
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="weighted", zero_division=0
        )
        per_class_f1 = f1_score(true_labels, pred_labels, average=None,
                                zero_division=0, labels=EMOTION_LIST)
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

        return df_out, eval_results


# ── Main ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LLM-Lite inference + results")
    p.add_argument("--model_dir",     default=LOCAL_MODEL,
                   help=f"Path to saved model  (default: {LOCAL_MODEL})")
    p.add_argument("--data_path",     default=f"{LOCAL_DATA}/Subtask_1_train.json",
                   help=f"Path to data  (default: {LOCAL_DATA}/Subtask_1_train.json)")
    p.add_argument("--processed_dir", default=LOCAL_PROC,
                   help=f"Preprocessed dir  (default: {LOCAL_PROC})")
    p.add_argument("--output_dir",    default=LOCAL_OUTPUT,
                   help=f"Where to write outputs  (default: {LOCAL_OUTPUT})")
    p.add_argument("--split",         default="validation",
                   help="Dataset split: train / validation")
    p.add_argument("--max_length",    type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LLM-LITE SOCIAL REASONING EXPERT — INFERENCE & RESULTS")
    print("=" * 60)
    print(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU only'}")
    print(f"Model dir    : {args.model_dir}")
    print(f"Data path    : {args.data_path}")
    print(f"Output dir   : {args.output_dir}\n")

    # Validate paths
    if not Path(args.model_dir).exists():
        sys.exit(f"❌ model_dir not found: {args.model_dir}")
    if not Path(args.data_path).exists():
        sys.exit(f"❌ data_path not found: {args.data_path}")

    # Load pre-processed dataset
    hf_dataset_path = Path(args.processed_dir) / "hf_dataset"
    if not hf_dataset_path.exists():
        sys.exit(f"❌ HF dataset not found at: {hf_dataset_path}")
    from datasets import load_from_disk
    full_dataset = load_from_disk(str(hf_dataset_path))
    val_dataset = full_dataset[args.split]
    print(f"✅ '{args.split}' split: {len(val_dataset)} samples")

    # Convert to df
    val_df = pd.DataFrame(val_dataset)

    # Inference
    inferer = LLMInference(args.model_dir, max_length=args.max_length)
    df, results = inferer.run(val_df, args.output_dir, split=args.split)

    print("\n✅ INFERENCE COMPLETE")
    print(f"   Primary deliverable → {args.output_dir}/llm_lite_logits_val.csv")
    print(f"   Rows: {len(df)}  |  Weighted F1: {results['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
