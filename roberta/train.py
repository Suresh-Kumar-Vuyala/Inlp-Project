"""
RoBERTa Semantic Expert — Training Script
TECPE Project (Member 1)

Usage:
    python train.py --data_path data/Subtask_1_train.json \
                    --output_dir models/roberta_semantic \
                    --checkpoint_dir checkpoints

All paths default to local directories (no Google Drive dependency).
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import datasets
from datasets import load_from_disk
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    EvalPrediction,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)

# ── Label registry (frozen — changing breaks the ensemble) ────────────────
EMOTION_LABELS = {
    "anger": 0, "disgust": 1, "fear": 2, "joy": 3,
    "sadness": 4, "surprise": 5, "neutral": 6,
}
LABEL_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}
EMOTION_LIST = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
NUM_LABELS = 7

# ── Shared hyper-parameters ───────────────────────────────────────────────
class Config:
    RANDOM_SEED                 = 42
    CONTEXT_WINDOW              = 3
    VALIDATION_SPLIT            = 0.1
    MAX_LENGTH                  = 256
    BATCH_SIZE                  = 4
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE               = 2e-5
    NUM_EPOCHS                  = 10
    WARMUP_STEPS                = 500
    WEIGHT_DECAY                = 0.01
    ROBERTA_MODEL               = "roberta-base"


# ── Reproducibility ───────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data preprocessing ────────────────────────────────────────────────────
class DataPreprocessor:
    def __init__(self, data_path: str, output_dir: str = "./processed_data"):
        self.data_path  = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.raw_data       = None
        self.flattened_data = []

    def load_raw_data(self):
        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        print(f"✅ Loaded {len(self.raw_data)} conversations")
        return self.raw_data

    @staticmethod
    def _add_context(utterances, window_size=3):
        for i, utt in enumerate(utterances):
            ctx_utts = utterances[max(0, i - window_size):i]
            utt["context"] = " | ".join(
                f"{u.get('speaker','')}: {u.get('text','')}" for u in ctx_utts
            )
            utt["context_length"] = len(ctx_utts)
        return utterances

    def flatten_conversations(self, add_context=True, context_window=3):
        if self.raw_data is None:
            self.load_raw_data()
        print("Flattening conversations...")
        flattened = []
        for conv_idx, conv in enumerate(self.raw_data):
            conv_id    = conv.get("conversation_ID", f"conv_{conv_idx}")
            utterances = conv.get("conversation", [])
            if add_context:
                utterances = self._add_context(utterances, context_window)
            for utt_idx, utt in enumerate(utterances):
                emotion_str   = utt.get("emotion", "neutral").lower()
                emotion_label = EMOTION_LABELS.get(emotion_str, 6)
                pairs = utt.get("emotion-cause_pairs", [])
                flattened.append({
                    "conversation_id": conv_id,
                    "utterance_id":    f"{conv_id}_utt_{utt_idx}",
                    "utterance_idx":   utt_idx,
                    "speaker":         utt.get("speaker", ""),
                    "text":            utt.get("text", ""),
                    "emotion":         emotion_str,
                    "emotion_label":   emotion_label,
                    "context":         utt.get("context", "") if add_context else "",
                    "context_length":  utt.get("context_length", 0) if add_context else 0,
                    "cause_span":      pairs[0].get("cause_span", "") if pairs else "",
                    "has_cause":       1 if pairs else 0,
                })
        self.flattened_data = flattened
        neutral_count = sum(1 for r in flattened if r["emotion_label"] == 6)
        print(f"✅ {len(flattened)} utterances  |  Neutral samples: {neutral_count}")
        assert neutral_count > 0, "BUG: no Neutral samples"
        return flattened

    def create_train_val_split(self, test_size=0.1, random_state=42):
        if not self.flattened_data:
            self.flatten_conversations()
        conv_groups = {}
        for rec in self.flattened_data:
            conv_groups.setdefault(rec["conversation_id"], []).append(rec)
        conv_ids = list(conv_groups.keys())
        train_ids, val_ids = train_test_split(
            conv_ids, test_size=test_size, random_state=random_state, shuffle=True
        )
        train_data = [r for cid in train_ids for r in conv_groups[cid]]
        val_data   = [r for cid in val_ids   for r in conv_groups[cid]]
        print(f"✅ Train: {len(train_data):,} utterances from {len(train_ids)} convs")
        print(f"✅ Val:   {len(val_data):,}   utterances from {len(val_ids)} convs")
        return train_data, val_data

    def save_to_hf_dataset(self, train_data, val_data):
        ds = datasets.DatasetDict({
            "train":      datasets.Dataset.from_pandas(pd.DataFrame(train_data)),
            "validation": datasets.Dataset.from_pandas(pd.DataFrame(val_data)),
        })
        save_path = self.output_dir / "hf_dataset"
        ds.save_to_disk(str(save_path))
        print(f"✅ HF dataset saved → {save_path}")
        return ds


# ── Model + Trainer wrapper ───────────────────────────────────────────────
class RoBERTaSemanticExpert:
    def __init__(self, model_name="roberta-base", output_dir="./models/roberta_semantic",
                 checkpoint_dir="./checkpoints", max_length=256):
        self.model_name     = model_name
        self.output_dir     = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.max_length = max_length
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model   = None
        self.dataset = None
        self.trainer = None

    def load_dataset(self, dataset_path: str):
        print(f"Loading dataset from {dataset_path}...")
        self.dataset = load_from_disk(dataset_path)
        print(f"✅ Train: {len(self.dataset['train']):,}  Val: {len(self.dataset['validation']):,}")
        return self.dataset

    def _tokenize(self, examples):
        contexts = examples.get("context", [""] * len(examples["text"]))
        inputs   = [
            f"{ctx} {self.tokenizer.sep_token} {txt}" if ctx else txt
            for ctx, txt in zip(contexts, examples["text"])
        ]
        tok = self.tokenizer(inputs, padding="max_length", truncation=True,
                             max_length=self.max_length, return_tensors=None)
        tok["labels"] = examples["emotion_label"]
        return tok

    def prepare_dataset(self):
        print("Tokenising...")
        remove_cols = [c for c in self.dataset["train"].column_names
                       if c != "emotion_label"]
        self.dataset = self.dataset.map(
            self._tokenize, batched=True, remove_columns=remove_cols,
            desc="RoBERTa tokenisation"
        )
        self.dataset.set_format("torch")
        print("✅ Tokenisation complete")
        return self.dataset

    def _compute_metrics(self, eval_pred: EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        per_class_f1 = f1_score(labels, preds, average=None,
                                zero_division=0, labels=list(range(NUM_LABELS)))
        metrics = {"f1_weighted": f1, "precision_weighted": precision, "recall_weighted": recall}
        for i, emotion in enumerate(EMOTION_LIST):
            metrics[f"f1_{emotion}"] = float(per_class_f1[i]) if i < len(per_class_f1) else 0.0
        return metrics

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name, num_labels=NUM_LABELS,
            problem_type="single_label_classification"
        )
        print(f"✅ Model loaded ({NUM_LABELS} labels)")
        return self.model

    def train(self, num_epochs=10, batch_size=4, learning_rate=2e-5):
        training_args = TrainingArguments(
            output_dir=str(self.checkpoint_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=Config.WARMUP_STEPS,
            weight_decay=Config.WEIGHT_DECAY,
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
            logging_dir=str(self.checkpoint_dir / "logs"),
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=Config.RANDOM_SEED,
            data_seed=Config.RANDOM_SEED,
            dataloader_drop_last=False,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            compute_metrics=self._compute_metrics,
        )
        print("🚀 Training started...")
        result = self.trainer.train()

        final_path = self.output_dir / "final_model"
        self.trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        print(f"\n✅ Training complete. Loss: {result.training_loss:.4f}")
        print(f"   Model saved → {final_path}")
        return result

    def evaluate(self, split="validation"):
        print(f"\nEvaluating on {split}...")
        results = self.trainer.evaluate(self.dataset[split])
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        out_path = self.output_dir / f"eval_results_{split}.json"
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"✅ Eval results → {out_path}")
        return results


# ── Main ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train RoBERTa Semantic Expert")
    p.add_argument("--data_path",      default="data/Subtask_1_train.json",
                   help="Path to Subtask_1_train.json")
    p.add_argument("--processed_dir",  default="./processed_data",
                   help="Where to save the preprocessed HF dataset")
    p.add_argument("--output_dir",     default="./models/roberta_semantic",
                   help="Where to save the final model")
    p.add_argument("--checkpoint_dir", default="./checkpoints",
                   help="Where to save training checkpoints")
    p.add_argument("--epochs",         type=int,   default=Config.NUM_EPOCHS)
    p.add_argument("--batch_size",     type=int,   default=Config.BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=Config.LEARNING_RATE)
    p.add_argument("--max_length",     type=int,   default=Config.MAX_LENGTH)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(Config.RANDOM_SEED)

    print("=" * 60)
    print("ROBERTA SEMANTIC EXPERT — TRAINING")
    print("=" * 60)
    print(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU only'}\n")

    # 1. Preprocess
    preprocessor = DataPreprocessor(args.data_path, args.processed_dir)
    preprocessor.load_raw_data()
    preprocessor.flatten_conversations(add_context=True,
                                       context_window=Config.CONTEXT_WINDOW)
    train_data, val_data = preprocessor.create_train_val_split(
        test_size=Config.VALIDATION_SPLIT, random_state=Config.RANDOM_SEED
    )
    preprocessor.save_to_hf_dataset(train_data, val_data)

    # 2. Train
    expert = RoBERTaSemanticExpert(
        model_name=Config.ROBERTA_MODEL,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        max_length=args.max_length,
    )
    expert.load_dataset(str(Path(args.processed_dir) / "hf_dataset"))
    expert.prepare_dataset()
    expert.load_model()
    expert.train(num_epochs=args.epochs, batch_size=args.batch_size,
                 learning_rate=args.lr)
    expert.evaluate("validation")

    print("\n✅ TRAINING COMPLETE")
    print(f"   Model → {args.output_dir}/final_model")
    print(f"   Run inference with: python inference.py --model_dir {args.output_dir}/final_model")


if __name__ == "__main__":
    main()