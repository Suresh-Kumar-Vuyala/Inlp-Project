"""
SpanBERT Cause QA — Training Script
TECPE Project (Member ?)

Usage:
    python train.py --data_path ./NLP_Project/data/Subtask_1_train.json \
                    --output_dir ./NLP_Project/models/spanbert_cause_qa \
                    --checkpoint_dir ./NLP_Project/checkpoints/spanbert_qa

Trains SpanBERT for question answering on emotion-cause pairs.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

# ── Config ────────────────────────────────────────────────────────────
class Config:
    RANDOM_SEED                 = 42
    VALIDATION_SPLIT            = 0.1
    MAX_LENGTH                  = 384
    DOC_STRIDE                  = 64
    BATCH_SIZE                  = 8
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE               = 2e-5
    NUM_EPOCHS                  = 8
    WARMUP_STEPS                = 300
    WEIGHT_DECAY                = 0.01
    SPANBERT_MODEL              = "SpanBERT/spanbert-base-cased"


# ── Reproducibility ───────────────────────────────────────────────────
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Data preprocessing ────────────────────────────────────────────────
class DataPreprocessor:
    def __init__(self, data_path: str, output_dir: str = "./processed_data"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.raw_data = None
        self.qa_data = []

    def load_raw_data(self):
        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        print(f"✅ Loaded {len(self.raw_data)} conversations")

    def preprocess_qa(self):
        if self.raw_data is None:
            self.load_raw_data()

        stats = {
            'total_pairs': 0,
            'found': 0,
            'not_found': 0,
            'multi_cause': 0,
        }

        for conv_idx, conversation in enumerate(self.raw_data):
            utterances = conversation.get('conversation', [])
            emotion_cause_pairs = conversation.get('emotion-cause_pairs', [])

            if not utterances or not emotion_cause_pairs:
                continue

            utt_map = {u['utterance_ID']: u for u in utterances}

            # Build full conversation context
            full_context = ' | '.join(
                f"[{u['utterance_ID']}] {u.get('speaker','?')}: {u.get('text','').strip()}"
                for u in utterances
            )

            # Group causes by (emotion_utt_id, emotion_label)
            emotion_groups = {}
            for pair in emotion_cause_pairs:
                stats['total_pairs'] += 1

                e_parts = pair[0].split('_', 1)
                c_parts = pair[1].split('_', 1)
                if len(e_parts) != 2 or len(c_parts) != 2:
                    continue

                try:
                    emotion_utt_id = int(e_parts[0])
                    emotion = e_parts[1].lower()
                    cause_text = c_parts[1].strip()
                except:
                    continue

                if not cause_text or emotion_utt_id not in utt_map:
                    continue

                key = (emotion_utt_id, emotion)
                emotion_groups.setdefault(key, []).append(cause_text)

            # Build QA examples
            for (emotion_utt_id, emotion), cause_list in emotion_groups.items():
                if len(cause_list) > 1:
                    stats['multi_cause'] += 1

                cause_text = cause_list[0]

                answer_start = full_context.find(cause_text)
                if answer_start == -1:
                    stats['not_found'] += 1
                    continue

                if full_context[answer_start: answer_start + len(cause_text)] != cause_text:
                    stats['not_found'] += 1
                    continue

                stats['found'] += 1

                question = (
                    f"What caused the {emotion} expressed by "
                    f"{utt_map[emotion_utt_id].get('speaker','speaker')} "
                    f"in utterance {emotion_utt_id}?"
                )

                qa_id = f"conv_{conv_idx}_e{emotion_utt_id}_{emotion}"

                self.qa_data.append({
                    'id': qa_id,
                    'title': f'emotion_{emotion}',
                    'context': full_context,
                    'question': question,
                    'answers': {
                        'text': [cause_text],
                        'answer_start': [answer_start],
                    },
                })

        print(f"Stats: total_pairs={stats['total_pairs']}, found={stats['found']}, not_found={stats['not_found']}, multi_cause={stats['multi_cause']}")
        return self.qa_data

    def create_train_val_split(self):
        if not self.qa_data:
            self.preprocess_qa()

        all_ids = [d['id'] for d in self.qa_data]
        train_ids, val_ids = train_test_split(
            all_ids,
            test_size=Config.VALIDATION_SPLIT,
            random_state=Config.RANDOM_SEED,
        )
        train_data = [d for d in self.qa_data if d['id'] in set(train_ids)]
        val_data = [d for d in self.qa_data if d['id'] in set(val_ids)]
        print(f"✅ Train: {len(train_data)} | Val: {len(val_data)}")
        return train_data, val_data

    def save_to_hf_dataset(self, train_data, val_data, output_dir):
        def to_hf(data):
            return Dataset.from_dict({
                'id': [d['id'] for d in data],
                'title': [d['title'] for d in data],
                'context': [d['context'] for d in data],
                'question': [d['question'] for d in data],
                'answers': [d['answers'] for d in data],
            })

        raw_dataset = DatasetDict({
            'train': to_hf(train_data),
            'validation': to_hf(val_data),
        })

        raw_path = output_dir / "cause_qa_dataset"
        if os.path.exists(raw_path):
            shutil.rmtree(raw_path)
        raw_dataset.save_to_disk(str(raw_path))
        print(f"✅ Saved raw dataset to {raw_path}")
        return raw_dataset


# ── Tokenization ──────────────────────────────────────────────────────
def tokenize_qa(examples, tokenizer, max_length=384, doc_stride=64):
    questions = [q.strip() for q in examples['question']]
    contexts = examples['context']

    tokenized = tokenizer(
        questions,
        contexts,
        truncation='only_second',
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length',
    )

    sample_mapping = tokenized.pop('overflow_to_sample_mapping')
    offset_mapping = tokenized.pop('offset_mapping')

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answers = examples['answers'][sample_idx]

        if len(answers['answer_start']) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        answer_start_char = answers['answer_start'][0]
        answer_text = answers['text'][0]
        answer_end_char = answer_start_char + len(answer_text)

        sequence_ids = tokenized.sequence_ids(i)
        context_start = None
        context_end = None
        for idx, sid in enumerate(sequence_ids):
            if sid == 1:
                if context_start is None:
                    context_start = idx
                context_end = idx

        if context_start is None:
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_start_index = context_start
        found_start = False
        while token_start_index <= context_end:
            if offsets[token_start_index][0] <= answer_start_char < offsets[token_start_index][1]:
                found_start = True
                break
            token_start_index += 1

        if not found_start:
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_end_index = context_end
        found_end = False
        while token_end_index >= context_start:
            if offsets[token_end_index][0] <= answer_end_char - 1 < offsets[token_end_index][1]:
                found_end = True
                break
            token_end_index -= 1

        if not found_end or token_start_index > token_end_index:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)

    tokenized['start_positions'] = start_positions
    tokenized['end_positions'] = end_positions
    return tokenized


# ── Main ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train SpanBERT Cause QA")
    p.add_argument("--data_path",      default="./NLP_Project/data/Subtask_1_train.json",
                   help="Path to Subtask_1_train.json")
    p.add_argument("--processed_dir",  default="./NLP_Project/processed_data",
                   help="Where to save processed datasets")
    p.add_argument("--output_dir",     default="./NLP_Project/models/spanbert_cause_qa",
                   help="Where to save the final model")
    p.add_argument("--checkpoint_dir", default="./NLP_Project/checkpoints/spanbert_qa",
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
    print("SPANBERT CAUSE QA — TRAINING")
    print("=" * 60)
    print(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU only'}\n")

    # Preprocess
    preprocessor = DataPreprocessor(args.data_path, args.processed_dir)
    preprocessor.load_raw_data()
    preprocessor.preprocess_qa()
    train_data, val_data = preprocessor.create_train_val_split()
    raw_dataset = preprocessor.save_to_hf_dataset(train_data, val_data, Path(args.processed_dir))

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(Config.SPANBERT_MODEL)
    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenize_qa(examples, tokenizer, Config.MAX_LENGTH, Config.DOC_STRIDE),
        batched=True,
        remove_columns=['id', 'title', 'context', 'question', 'answers'],
    )
    tokenized_dataset.set_format('torch')

    tok_path = Path(args.processed_dir) / "cause_qa_tokenized"
    if os.path.exists(tok_path):
        shutil.rmtree(tok_path)
    tokenized_dataset.save_to_disk(str(tok_path))
    print(f"✅ Saved tokenized dataset to {tok_path}")

    # Model
    model = AutoModelForQuestionAnswering.from_pretrained(
        Config.SPANBERT_MODEL,
        ignore_mismatched_sizes=True,
    )
    nn.init.normal_(model.qa_outputs.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.qa_outputs.bias)
    model = model.float()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Training
    training_args = TrainingArguments(
        output_dir=str(Path(args.checkpoint_dir)),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.lr,
        warmup_steps=Config.WARMUP_STEPS,
        weight_decay=Config.WEIGHT_DECAY,
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=50,
        save_strategy='steps',
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        fp16=False,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=default_data_collator,
    )

    print("🚀 Training started...")
    train_result = trainer.train()

    # Save
    final_path = Path(args.output_dir)
    final_path.mkdir(exist_ok=True, parents=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n✅ Training complete. Loss: {train_result.training_loss:.4f}")
    print(f"   Model saved → {final_path}")

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"   Eval loss: {eval_results['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
