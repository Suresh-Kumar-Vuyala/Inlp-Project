"""
LLM-Lite Social Reasoning Expert — Training Script
TECPE Project (Member 3)

Usage:
    python train.py --data_path ./NLP_Project/data/Subtask_1_train.json \
                    --output_dir ./NLP_Project/models/lite_llm \
                    --checkpoint_dir ./NLP_Project/checkpoints/llm_lite

Fine-tunes Phi-3-mini-4k-instruct with QLoRA for emotion classification.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ── Label registry (must match RoBERTa) ────────────────────────────────────
EMOTION_LABELS = {
    'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
    'sadness': 4, 'surprise': 5, 'neutral': 6
}
LABEL_TO_EMOTION = {v: k for k, v in EMOTION_LABELS.items()}
EMOTION_LIST = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
EMOTION_OPTIONS = ", ".join(EMOTION_LIST)
NUM_LABELS = 7

# ── Hyper-parameters ──────────────────────────────────────────────────────
class Config:
    RANDOM_SEED             = 42
    CONTEXT_WINDOW          = 3
    VALIDATION_SPLIT        = 0.1
    LLM_MODEL               = "microsoft/Phi-3-mini-4k-instruct"
    MAX_LENGTH              = 256
    BATCH_SIZE              = 4
    GRADIENT_ACCUMULATION   = 4
    LEARNING_RATE           = 3e-4
    NUM_EPOCHS              = 1
    WARMUP_STEPS            = 50
    WEIGHT_DECAY            = 0.01
    LORA_R                  = 32
    LORA_ALPHA              = 64
    LORA_DROPOUT            = 0.05


# ── Reproducibility ───────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data preprocessing ────────────────────────────────────────────────────
class DataPreprocessor:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.raw_data = None
        self.flattened_data = []

    def load_raw_data(self):
        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        print(f"✅ Loaded {len(self.raw_data)} conversations")

    def add_context_window(self, utterances, window_size: int = 3):
        for i, utt in enumerate(utterances):
            start_idx = max(0, i - window_size)
            context_utts = utterances[start_idx:i]
            context = []
            for ctx_utt in context_utts:
                speaker = ctx_utt.get('speaker', 'Unknown')
                text = ctx_utt.get('text', '')
                context.append(f"{speaker}: {text}")
            utt['context'] = " | ".join(context) if context else ""
            utt['context_length'] = len(context)
        return utterances

    def flatten_conversations(self, context_window: int = 3):
        if self.raw_data is None:
            self.load_raw_data()
        print("Flattening conversations...")
        flattened = []
        for conv_idx, conversation in enumerate(self.raw_data):
            conv_id = conversation.get('conversation_ID', f'conv_{conv_idx}')
            utterances = conversation.get('conversation', [])
            utterances = self.add_context_window(utterances, context_window)
            for utt_idx, utterance in enumerate(utterances):
                record = {
                    'conversation_id': conv_id,
                    'utterance_id': f"{conv_id}_utt_{utt_idx}",
                    'utterance_idx': utt_idx,
                    'speaker': utterance.get('speaker', ''),
                    'text': utterance.get('text', ''),
                    'emotion': utterance.get('emotion', 'neutral').lower(),
                    'emotion_label': EMOTION_LABELS.get(
                        utterance.get('emotion', 'neutral').lower(), 6
                    ),
                    'context': utterance.get('context', ''),
                    'context_length': utterance.get('context_length', 0),
                }
                emotion_cause_pairs = utterance.get('emotion-cause_pairs', [])
                record['cause_span'] = emotion_cause_pairs[0].get('cause_span', '') if emotion_cause_pairs else ''
                record['has_cause'] = 1 if emotion_cause_pairs else 0
                flattened.append(record)
        self.flattened_data = flattened
        print(f"✅ Created {len(flattened)} utterance records")
        return flattened

    def create_train_val_split(self, test_size=0.1, random_state=42):
        if not self.flattened_data:
            self.flatten_conversations()
        print(f"Creating {int((1-test_size)*100)}-{int(test_size*100)} split...")
        conv_groups = {}
        for record in self.flattened_data:
            conv_id = record['conversation_id']
            conv_groups.setdefault(conv_id, []).append(record)
        conv_ids = list(conv_groups.keys())
        train_conv_ids, val_conv_ids = train_test_split(
            conv_ids, test_size=test_size, random_state=random_state
        )
        train_data = [r for cid in train_conv_ids for r in conv_groups[cid]]
        val_data   = [r for cid in val_conv_ids   for r in conv_groups[cid]]
        print(f"✅ Train: {len(train_data)} samples ({len(train_conv_ids)} conversations)")
        print(f"✅ Val:   {len(val_data)} samples ({len(val_conv_ids)} conversations)")
        return train_data, val_data


# ── Prompt building ──────────────────────────────────────────────────────
def format_context(context: str) -> str:
    if not context or context.strip() == "":
        return "(none)"
    return context.replace(" | ", "\n")


def build_prompt(row: dict, include_label: bool = True) -> str:
    context = format_context(row.get('context', ''))
    speaker = row.get('speaker', 'Speaker')
    text    = row.get('text', '').strip()
    emotion = row.get('emotion', '').strip()

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

    if include_label:
        prompt = (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{user}<|end|>\n"
            f"<|assistant|>\n{emotion}<|end|>"
        )
    else:
        prompt = (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{user}<|end|>\n"
            f"<|assistant|>\n"
        )
    return prompt


# ── Main ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train LLM-Lite Social Reasoning Expert")
    p.add_argument("--data_path",      default="./NLP_Project/data/Subtask_1_train.json",
                   help="Path to Subtask_1_train.json")
    p.add_argument("--output_dir",     default="./NLP_Project/models/lite_llm",
                   help="Where to save the final model")
    p.add_argument("--checkpoint_dir", default="./NLP_Project/checkpoints/llm_lite",
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
    print("LLM-LITE SOCIAL REASONING EXPERT — TRAINING")
    print("=" * 60)
    print(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU only'}\n")

    # Preprocess
    preprocessor = DataPreprocessor(args.data_path)
    preprocessor.load_raw_data()
    preprocessor.flatten_conversations(context_window=Config.CONTEXT_WINDOW)
    train_data, val_data = preprocessor.create_train_val_split(
        test_size=Config.VALIDATION_SPLIT, random_state=Config.RANDOM_SEED
    )
    train_df = pd.DataFrame(train_data)
    val_df   = pd.DataFrame(val_data)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.LLM_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.LLM_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    def build_hf_dataset(df: pd.DataFrame, include_label: bool = True) -> Dataset:
        records = df.to_dict('records')
        prompts = [build_prompt(r, include_label=include_label) for r in records]
        tokenized = tokenizer(
            prompts,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors=None,
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        ds = Dataset.from_dict(tokenized)
        ds.set_format('torch')
        return ds

    train_dataset = build_hf_dataset(train_df, include_label=True)
    val_dataset = build_hf_dataset(val_df, include_label=True)

    # Training
    training_args = TrainingArguments(
        output_dir=str(Path(args.checkpoint_dir)),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION,
        fp16=True,
        bf16=False,
        learning_rate=args.lr,
        warmup_steps=Config.WARMUP_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        logging_dir=str(Path(args.checkpoint_dir) / "logs"),
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        group_by_length=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("🚀 Training started...")
    train_result = trainer.train()

    # Save
    adapter_path = Path(args.output_dir) / "lora_adapters"
    adapter_path.mkdir(exist_ok=True, parents=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n✅ Training complete. Loss: {train_result.training_loss:.4f}")
    print(f"   Model saved → {adapter_path}")


if __name__ == "__main__":
    main()
