# parseforce_pipeline.py
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # This makes the GPU ops deterministic (may slow down execution)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ── STEP 0: Install dependencies ─────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q",
                           "--break-system-packages"], stderr=subprocess.DEVNULL)

for pkg in ["xgboost", "scikit-learn", "pandas", "numpy", "transformers", "torch","peft"]:
    install(pkg)

print("✅ All dependencies ready.")

# ── STEP 2: Configuration & file paths ──────────────────────────────────────
import os

PROJECT_ROOT = "NLP_Project"  # Adjusted for local

# ── XGBoost judge selector ────────────────────────────────────────────────────
# Multiple judge files found in models/. Pick the one you want to use:
#   judge.json                  ← default (used in meta_learner.ipynb save cell)
#   meta_learner_FINAL_v3.json  ← likely your best tuned version
#   meta_learner_xgb_final.json
#   meta_learner_xgb_balanced.json
XGB_JUDGE_FILE = "judge.json"   # ← change this to try a different judge

PATHS = {
    "roberta":   f"{PROJECT_ROOT}/models/roberta/final_model",
    "deberta":   f"{PROJECT_ROOT}/models/deberta/final_model/deberta",
    "llm_lite":  f"{PROJECT_ROOT}/models/llm_lite/final_model",
    "spanbert":  f"{PROJECT_ROOT}/models/spanberta/best_model",
    "xgb_judge": f"{PROJECT_ROOT}/models/{XGB_JUDGE_FILE}",
    "test_json":  f"{PROJECT_ROOT}/data/test.json",
}

print(f"Using XGB judge: {XGB_JUDGE_FILE}")
print("\nChecking paths...")
all_ok = True
for name, path in PATHS.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "❌ MISSING"
    if not exists:
        all_ok = False
    print(f"  {status}  {name}: {path}")

if all_ok:
    print("\n✅ All paths verified — ready to load models.")
else:
    print("\n⚠️  Fix missing paths above before continuing.")

# ── STEP 3: Load all models ──────────────────────────────────────────────────
import torch
import numpy as np
import xgboost as xgb
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
)
from peft import PeftModel

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_idx = 0 if torch.cuda.is_available() else -1
print(f"Device: {device}")

# ── Level-0: RoBERTa ─────────────────────────────────────────────────────────
print("\nLoading RoBERTa...")
roberta_pipe = pipeline("text-classification",
                         model=PATHS["roberta"],
                         return_all_scores=True,
                         device=device_idx)

# ── Level-0: DeBERTa ─────────────────────────────────────────────────────────
print("Loading DeBERTa...")
deberta_pipe = pipeline("text-classification",
                         model=PATHS["deberta"],
                         return_all_scores=True,
                         device=device_idx)

# ── Level-0: LLM-Lite (Phi-2 + LoRA) ────────────────────────────────────────
print("Loading Phi-2 base model (this takes ~2 min first time)...")
LLM_BASE = "microsoft/phi-2"
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_BASE, trust_remote_code=True)
llm_tokenizer.pad_token = llm_tokenizer.eos_token

llm_config = AutoConfig.from_pretrained(LLM_BASE, trust_remote_code=True)
llm_config.pad_token_id = llm_tokenizer.eos_token_id  # patch missing attribute

llm_base = AutoModelForCausalLM.from_pretrained(
    LLM_BASE,
    config=llm_config,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print("Loading LoRA adapter...")
llm_model = PeftModel.from_pretrained(llm_base, PATHS["llm_lite"])
llm_model.eval()
print("✅ LLM-Lite loaded")

# Pre-compute token IDs for the 7 emotion words
EMOTION_COLS_LLM = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_TOKEN_IDS = {}
for emo in EMOTION_COLS_LLM:
    tids = llm_tokenizer.encode(" " + emo, add_special_tokens=False)
    EMOTION_TOKEN_IDS[emo] = tids[0]
print("Emotion token IDs:", EMOTION_TOKEN_IDS)

# ── Level-1: XGBoost Meta-Judge ──────────────────────────────────────────────
print("\nLoading XGBoost Meta-Judge...")
xgb_judge = xgb.XGBClassifier()
xgb_judge.load_model(PATHS["xgb_judge"])

# ── Level-2: SpanBERT ────────────────────────────────────────────────────────
print("Loading SpanBERT (best_model)...")
spanbert_tokenizer = AutoTokenizer.from_pretrained(PATHS["spanbert"])
spanbert_model     = AutoModelForQuestionAnswering.from_pretrained(PATHS["spanbert"])
spanbert_model     = spanbert_model.to(device)
spanbert_model.eval()

print("\n✅ All 5 models loaded successfully!")

# ── STEP 4: Label mappings (must match training order exactly) ───────────────

# Canonical emotion order used during XGBoost training (from meta_learner.ipynb)
EMOTION_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EMOTION_MAP  = {i: e for i, e in enumerate(EMOTION_COLS)}

# Per-model label orders (check your model configs if predictions look wrong)
# RoBERTa: LABEL_0=anger, LABEL_1=disgust ... LABEL_6=neutral (canonical)
ROBERTA_ORDER = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
# DeBERTa: different order — check config.json id2label if needed
DEBERTA_ORDER = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]
# LLM-Lite: check config.json id2label to confirm
LLM_ORDER     = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Confidence threshold τ* from meta_learner training (update if you have it saved)
OPTIMAL_TAU = 0.0   # ← replace with the τ* value printed in meta_learner.ipynb

print("Label config:")
print(f"  Canonical order  : {EMOTION_COLS}")
print(f"  RoBERTa order    : {ROBERTA_ORDER}")
print(f"  DeBERTa order    : {DEBERTA_ORDER}")
print(f"  LLM-Lite order   : {LLM_ORDER}")
print(f"  Confidence gate τ*: {OPTIMAL_TAU}")

# ── STEP 5: Feature extraction helpers + SpanBERT constrained prediction ─────
import torch.nn.functional as F
from collections import Counter

def get_probs_canonical(pipe_output, model_order, canonical_order=EMOTION_COLS):
    """Convert return_all_scores output → canonical-order prob vector (length 7)."""
    raw = {item['label']: item['score'] for item in pipe_output}
    score_by_emotion = {}
    for i, emotion in enumerate(model_order):
        score_by_emotion[emotion] = raw.get(f'LABEL_{i}', 0.0)
    return np.array([score_by_emotion[e] for e in canonical_order], dtype=np.float32)


def get_llm_probs(text, canonical_order=EMOTION_COLS):
    """
    Get 7-dim emotion probability vector from Phi-2 by reading next-token
    logits over the 7 emotion vocabulary tokens.
    LLM CSV column order was: anger disgust fear joy neutral sadness surprise
    We reorder to canonical: anger disgust fear joy sadness surprise neutral
    """
    prompt = (
        'Classify the emotion in this utterance.\n'
        'Options: anger, disgust, fear, joy, neutral, sadness, surprise\n'
        f'Utterance: {text}\n'
        'Emotion:'
    )
    inputs = llm_tokenizer(prompt, return_tensors='pt').to(llm_model.device)

    with torch.no_grad():
        outputs = llm_model(**inputs)
        last_logits = outputs.logits[0, -1, :]   # (vocab_size,)

    emotion_logits = torch.tensor(
        [last_logits[EMOTION_TOKEN_IDS[e]].item() for e in EMOTION_COLS_LLM],
        dtype=torch.float32
    )
    llm_order_probs = F.softmax(emotion_logits, dim=0).numpy()
    llm_to_canonical = {e: llm_order_probs[i] for i, e in enumerate(EMOTION_COLS_LLM)}
    return np.array([llm_to_canonical[e] for e in canonical_order], dtype=np.float32)


def extract_dialogue_metafeatures(utterances, utt_idx):
    """speaker_shift, utterance_position, conversation_length"""
    conv_len = len(utterances)
    if utt_idx == 0:
        shift = 0
    else:
        prev_speaker = utterances[utt_idx - 1].get('speaker', '')
        curr_speaker = utterances[utt_idx].get('speaker', '')
        shift = int(curr_speaker != prev_speaker)
    return np.array([shift, utt_idx, conv_len], dtype=np.float32)


def build_feature_vector(text, utterances, utt_idx):
    """Build the full 24-dim feature vector: rob(7) + deb(7) + llm(7) + meta(3)"""
    rob_out = roberta_pipe(text)
    deb_out = deberta_pipe(text)

    # pipeline with return_all_scores=True returns [[{label, score}, ...]]
    # unwrap outer list if needed
    rob_scores = rob_out[0] if isinstance(rob_out[0], list) else rob_out
    deb_scores = deb_out[0] if isinstance(deb_out[0], list) else deb_out

    rob_probs = get_probs_canonical(rob_scores, ROBERTA_ORDER)
    deb_probs = get_probs_canonical(deb_scores, DEBERTA_ORDER)
    llm_probs = get_llm_probs(text)
    meta      = extract_dialogue_metafeatures(utterances, utt_idx)
    return np.concatenate([rob_probs, deb_probs, llm_probs, meta]).reshape(1, -1)


def build_spanbert_context(utterances):
    """
    Build context in the exact format used during SpanBERT training:
    '[1] Speaker: text | [2] Speaker: text | ...'
    """
    return ' | '.join(
        f"[{u['utterance_ID']}] {u.get('speaker','?')}: {u.get('text','').strip()}"
        for u in utterances
    )


def get_constrained_prediction(context, question, max_answer_len=35, null_threshold=-8.0):
    """
    Block-aware SpanBERT inference (ported from SpanBERT training notebook).
    Respects utterance boundaries (| separators) and applies null threshold.
    """
    inputs = spanbert_tokenizer(
        question, context,
        return_tensors='pt',
        truncation='only_second',
        max_length=384,
        stride=64,
        return_overflowing_tokens=False,
        padding=True,
    )
    input_ids      = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = spanbert_model(input_ids=input_ids, attention_mask=attention_mask)

    start_logits = outputs.start_logits[0]
    end_logits   = outputs.end_logits[0]

    tokens = spanbert_tokenizer.convert_ids_to_tokens(input_ids[0])
    block_ids = []
    current_block = 0
    for t in tokens:
        if '|' in t:
            current_block += 1
        block_ids.append(current_block)

    null_score = (start_logits[0] + end_logits[0]).item()
    best_score = -float('inf')
    best_span  = (0, 0)

    start_indexes = torch.argsort(start_logits, descending=True)[:20]
    end_indexes   = torch.argsort(end_logits,   descending=True)[:20]

    for start_i in start_indexes:
        s_idx = start_i.item()
        if s_idx == 0:
            continue
        for end_i in end_indexes:
            e_idx = end_i.item()
            if s_idx >= e_idx or (e_idx - s_idx) > max_answer_len:
                continue
            if block_ids[s_idx] != block_ids[e_idx]:
                continue
            score = (start_logits[s_idx] + end_logits[e_idx]).item()
            if score > best_score:
                best_score = score
                best_span  = (s_idx, e_idx)

    if (best_score - null_score) < null_threshold or best_score == -float('inf'):
        return '', 0.0

    raw_text = spanbert_tokenizer.decode(
        input_ids[0][best_span[0]: best_span[1] + 1],
        skip_special_tokens=True
    )
    answer = raw_text.strip().strip('|').strip()
    for filler in ['anyway', 'anyway.', 'anyway,']:
        if answer.lower().endswith(filler):
            answer = answer[:-len(filler)].strip().strip(',').strip()

    confidence = float(torch.sigmoid(torch.tensor(best_score - null_score)))
    return answer, round(confidence, 4)


print('✅ All feature helpers defined.')

def predict_full_pipeline(text, utterances, utt_idx, context_history=None):

    # ── Build individual model probs ─────────────────────────────────────
    rob_out    = roberta_pipe(text)
    deb_out    = deberta_pipe(text)
    rob_scores = rob_out[0] if isinstance(rob_out[0], list) else rob_out
    deb_scores = deb_out[0] if isinstance(deb_out[0], list) else deb_out

    rob_probs  = get_probs_canonical(rob_scores, ROBERTA_ORDER)
    deb_probs  = get_probs_canonical(deb_scores, DEBERTA_ORDER)
    llm_probs  = get_llm_probs(text)

    # ── Weighted ensemble (tune weights if needed) ───────────────────────
    ensemble   = 0.4 * rob_probs + 0.4 * deb_probs + 0.2 * llm_probs

    pred_idx   = int(np.argmax(ensemble))
    confidence = float(ensemble[pred_idx])
    emotion    = EMOTION_MAP[pred_idx]
    trigger    = emotion != 'neutral'

    result = {
        "emotion":          emotion,
        "confidence":       round(confidence, 4),
        "trigger_level2":   trigger,
        "cause":            "",
        "cause_confidence": None,
    }

    # ── Level 2: SpanBERT ────────────────────────────────────────────────
    if trigger:
        window_start  = max(0, utt_idx - 4)
        window_end    = min(len(utterances), utt_idx + 2)
        windowed_utts = utterances[window_start:window_end]
        full_context  = build_spanbert_context(windowed_utts)
        try:
            cause, cause_conf = get_constrained_prediction(
                context=full_context,
                question=f"What caused the {emotion}?"
            )
            result["cause"]            = cause
            result["cause_confidence"] = cause_conf
        except Exception as e:
            result["cause"] = ""

    return result

# ── STEP 8: Load test data & run full evaluation ─────────────────────────────
import json, re
from collections import defaultdict

# ── Token-level overlap (Proportional F1) ────────────────────────────────────
def token_overlap(pred_span, gold_span):
    pred_tokens = set(re.findall(r'\w+', pred_span.lower()))
    gold_tokens = set(re.findall(r'\w+', gold_span.lower()))
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    return len(pred_tokens & gold_tokens) / len(gold_tokens)

# ── Accumulators ─────────────────────────────────────────────────────────────
EMOTIONS     = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
ALL_EMOTIONS = EMOTIONS + ['neutral']

strict = {e: {'tp': 0, 'fp': 0, 'fn': 0} for e in ALL_EMOTIONS}
prop   = {e: {'overlap_sum': 0.0, 'pred_count': 0, 'gold_count': 0} for e in ALL_EMOTIONS}

emo_correct = 0
emo_total   = 0
per_emo_acc = defaultdict(lambda: {'correct': 0, 'total': 0})
errors      = []

correct_samples = []

# ── Load test data ────────────────────────────────────────────────────────────
with open(PATHS["test_json"], 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
print(f"Loaded {len(raw_data)} conversations from test.json")

# ── STEP 9: Main evaluation loop ─────────────────────────────────────────────
import re

def normalize_span(s):
    s = re.sub(r'\s+', ' ', s).strip().lower()
    s = re.sub(r'\s([?.!,;:])', r'\1', s)
    return s

for conv_idx, conversation in enumerate(raw_data):
    utterances          = conversation.get('conversation', [])
    emotion_cause_pairs = conversation.get('emotion-cause_pairs', [])

    utt_map = {u['utterance_ID']: u for u in utterances}

    for i, utt in enumerate(utterances):
        utt['_context'] = [
            f"[{utterances[j].get('speaker','')}]: {utterances[j].get('text','')}"
            for j in range(max(0, i - 3), i)
        ]

    gold_tuples = []
    for pair in emotion_cause_pairs:
        emotion_part, cause_part = pair[0], pair[1]
        ep = emotion_part.split('_', 1)
        cp = cause_part.split('_', 1)
        if len(ep) != 2 or len(cp) != 2:
            continue
        try:
            emotion_utt_id = int(ep[0])
            cause_utt_id   = int(cp[0])
        except ValueError:
            continue
        gold_emotion = ep[1].lower().strip()
        gold_span    = cp[1].strip()
        if not gold_span:
            continue
        gold_tuples.append({
            'emotion_utt_id': emotion_utt_id,
            'emotion':        gold_emotion,
            'cause_utt_id':   cause_utt_id,
            'span':           gold_span,
        })

    by_emo_utt = defaultdict(list)
    for gt in gold_tuples:
        by_emo_utt[gt['emotion_utt_id']].append(gt)

    for emotion_utt_id, gold_list in by_emo_utt.items():
        if emotion_utt_id not in utt_map:
            continue

        emotion_utt  = utt_map[emotion_utt_id]
        utt_idx_in_conv = next(
            (i for i, u in enumerate(utterances) if u['utterance_ID'] == emotion_utt_id),
            0
        )
        emotion_text    = f"[{emotion_utt.get('speaker','')}]: {emotion_utt.get('text','')}"
        context_history = emotion_utt.get('_context', [])
        gold_emotion    = gold_list[0]['emotion']

        emo_total += 1
        per_emo_acc[gold_emotion]['total'] += 1

        for gt in gold_list:
            prop[gt['emotion']]['gold_count']  += 1
            strict[gt['emotion']]['fn']        += 1

        try:
            pred = predict_full_pipeline(
                emotion_text, utterances, utt_idx_in_conv, context_history
            )
        except Exception as e:
            print(f"  ⚠️  Error on conv {conv_idx}, utt {emotion_utt_id}: {e}")
            continue

        pred_emotion = pred['emotion']
        pred_span    = pred.get('cause', '').strip()
        if pred_span == 'N/A':
            pred_span = ''

        if pred_emotion == gold_emotion:
            emo_correct += 1
            per_emo_acc[gold_emotion]['correct'] += 1
        elif len(errors) < 10:
            errors.append({
                'text':      emotion_text,
                'gold':      gold_emotion,
                'pred':      pred_emotion,
                'conf':      pred['confidence'],
                'triggered': pred['trigger_level2'],
                'gold_spans':[g['span'] for g in gold_list],
                'pred_span': pred_span,
            })

        matched_gi = None
        if pred_emotion != 'neutral' and pred_span:
            for gi, gt in enumerate(gold_list):
                if (pred_emotion == gt['emotion'] and
                        normalize_span(pred_span) == normalize_span(gt['span'])):
                    matched_gi = gi
                    break

        if matched_gi is not None:
            strict[pred_emotion]['tp'] += 1
            strict[pred_emotion]['fn'] -= 1
            if len(correct_samples) < 10:
                correct_samples.append({
                    'text': emotion_text,
                    'emotion': pred_emotion,
                    'span': pred_span,
                    'conf': pred['confidence']
                })

        elif pred_emotion != 'neutral' and pred_span:
            strict[pred_emotion]['fp'] += 1

        if pred_emotion != 'neutral' and pred_span:
            prop[pred_emotion]['pred_count'] += 1
            best_overlap = 0.0
            for gt in gold_list:
                if pred_emotion == gt['emotion']:
                    ov = token_overlap(pred_span, gt['span'])
                    best_overlap = max(best_overlap, ov)
            prop[pred_emotion]['overlap_sum'] += best_overlap

    if (conv_idx + 1) % 100 == 0:
        print(f"  ✅ Processed {conv_idx+1}/{len(raw_data)} conversations...")

print(f"\n✅ Evaluation complete. {emo_total} utterances processed.")

# ── STEP 10: Compute & print official metrics ────────────────────────────────

def get_strict_f1(e):
    tp = strict[e]['tp']
    fp = strict[e]['fp']
    fn = strict[e]['fn']
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def get_prop_f1(e):
    pc   = prop[e]['pred_count']
    gc   = prop[e]['gold_count']
    osum = prop[e]['overlap_sum']
    p    = osum / pc if pc > 0 else 0.0
    r    = osum / gc if gc > 0 else 0.0
    f1   = 2*p*r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

total_gold      = sum(prop[e]['gold_count'] for e in EMOTIONS)
wavg_strict_f1  = sum(get_strict_f1(e)[2] * prop[e]['gold_count'] for e in EMOTIONS) / total_gold if total_gold else 0.0
wavg_prop_f1    = sum(get_prop_f1(e)[2]   * prop[e]['gold_count'] for e in EMOTIONS) / total_gold if total_gold else 0.0

SEP = "=" * 70

print(SEP)
print("  Parse-Force — Full Pipeline  |  SemEval-2024 Task 3 Results")
print(SEP)

# 1. Emotion Detection
print(f"\n{'─'*70}")
print("EMOTION DETECTION ACCURACY")
print(f"  Overall : {emo_correct}/{emo_total} = {100*emo_correct/max(emo_total,1):.2f}%")
print(f"\n  {'Emotion':<12} {'Correct':>8} {'Total':>8} {'Acc%':>8}")
print(f"  {'─'*42}")
for emo in ALL_EMOTIONS:
    s = per_emo_acc[emo]
    if s['total'] > 0:
        acc = 100 * s['correct'] / s['total']
        bar = '█' * int(acc / 5)
        print(f"  {emo:<12} {s['correct']:>8} {s['total']:>8} {acc:>7.2f}%  {bar}")

# 2. Strict F1
print(f"\n{'─'*70}")
print("STRICT EVALUATION  (exact span + exact emotion match)")
print(f"  w-avg. Strict F1 = {wavg_strict_f1:.4f}  ← PRIMARY RANKING METRIC")
print(f"\n  {'Emotion':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'#Gold':>8}")
print(f"  {'─'*50}")
for emo in EMOTIONS:
    p, r, f1 = get_strict_f1(emo)
    gc = prop[emo]['gold_count']
    print(f"  {emo:<12} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {gc:>8}")
print(f"\n  {'w-avg':<12} {'':>8} {'':>8} {wavg_strict_f1:>8.4f} {total_gold:>8}")

# 3. Proportional F1
print(f"\n{'─'*70}")
print("PROPORTIONAL EVALUATION  (token-overlap span credit)")
print(f"  w-avg. Proportional F1 = {wavg_prop_f1:.4f}")
print(f"\n  {'Emotion':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'#Gold':>8}")
print(f"  {'─'*50}")
for emo in EMOTIONS:
    p, r, f1 = get_prop_f1(emo)
    gc = prop[emo]['gold_count']
    print(f"  {emo:<12} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {gc:>8}")
print(f"\n  {'w-avg':<12} {'':>8} {'':>8} {wavg_prop_f1:>8.4f} {total_gold:>8}")

# 4. Summary
print(f"\n{'─'*70}")
print("SUMMARY")
print(f"  {'Metric':<42} {'Score':>10}")
print(f"  {'─'*54}")
print(f"  {'Emotion Accuracy':<42} {100*emo_correct/max(emo_total,1):>9.2f}%")
print(f"  {'w-avg Strict F1  (PRIMARY METRIC)':<42} {wavg_strict_f1:>10.4f}")
print(f"  {'w-avg Proportional F1':<42} {wavg_prop_f1:>10.4f}")
print(f"  {'Total gold tuples evaluated':<42} {total_gold:>10}")
print(f"  {'Level-2 gate τ*':<42} {OPTIMAL_TAU:>10.3f}")
print(SEP)

# 5. Sample errors
if errors:
    print(f"\n{'─'*70}")
    print("SAMPLE ERRORS (first 10)")
    for err in errors:
        print(f"\n  Text         : '{err['text']}'")
        print(f"  Gold emotion : {err['gold']}  →  Predicted: {err['pred']}  (conf={err['conf']}, triggered={err['triggered']})")
        print(f"  Gold spans   : {err['gold_spans']}")
        print(f"  Pred span    : '{err['pred_span']}'")

# ── STEP 12: Print Correct Prediction Samples ────────────────────────────────
print("======================================================================")
print("  PARSE-FORCE SUCCESS SAMPLES (Exact Emotion + Exact Span Match)")
print("======================================================================")

if not correct_samples:
    print("No correct samples captured yet. Ensure Step 9 has finished running.")
else:
    for i, success in enumerate(correct_samples):
        print(f"\n✅ Sample {i+1}")
        print(f"  Text      : '{success['text']}'")
        print(f"  Emotion   : {success['emotion']} (conf={success['conf']})")
        print(f"  Pred Span : '{success['span']}'")

print("\n" + "="*70)

# ── STEP 13: 5 Custom Dialogue Prediction Tests ──────────────────────────────

test_scenarios = [
    {
        "name": "Scenario 1: Surprise (Cross-turn Cause)",
        "history": [
            {"utterance_ID": 1, "speaker": "Rachel", "text": "I'm not going to the party because I'm moving to Paris tonight."},
            {"utterance_ID": 2, "speaker": "Monica", "text": "Wait, what did you just say?"}
        ],
        "target": "Paris? You're moving to Paris and you didn't tell me?!",
        "target_speaker": "Monica"
    },
    {
        "name": "Scenario 2: Sadness (Direct Statement)",
        "history": [
            {"utterance_ID": 1, "speaker": "Joey", "text": "The director called. They cut my entire scene from the movie."},
        ],
        "target": "I worked so hard on that role, and now it is just gone.",
        "target_speaker": "Joey"
    },
    {
        "name": "Scenario 3: Anger (Conflict/Metadata Check)",
        "history": [
            {"utterance_ID": 1, "speaker": "Chandler", "text": "I accidentally deleted your high score on the game."},
            {"utterance_ID": 2, "speaker": "Joey", "text": "You did what?"}
        ],
        "target": "I spent three weeks on that! I can't believe you were so careless!",
        "target_speaker": "Joey"
    },
    {
        "name": "Scenario 4: Joy (Social Validation)",
        "history": [
            {"utterance_ID": 1, "speaker": "Phoebe", "text": "Central Perk wants me to play a full set every Friday night!"},
        ],
        "target": "Oh my god, Phoebe! That is amazing news, I am so happy for you!",
        "target_speaker": "Rachel"
    },
    {
        "name": "Scenario 5: Fear (Threat/Safety)",
        "history": [
            {"utterance_ID": 1, "speaker": "Ross", "text": "I think there is someone lurking in the hallway outside the apartment."},
        ],
        "target": "Did you lock the door? Ross, go check the door right now!",
        "target_speaker": "Rachel"
    }
]

print(f"{'='*80}\n  PARSE-FORCE MANUAL TEST SUITE\n{'='*80}")

for scenario in test_scenarios:
    # Build the conversation format expected by Step 9
    conv_list = scenario["history"] + [
        {"utterance_ID": len(scenario["history"])+1,
         "speaker": scenario["target_speaker"],
         "text": scenario["target"]}
    ]
    current_idx = len(conv_list) - 1
    formatted_input = f"[{scenario['target_speaker']}]: {scenario['target']}"

    # Run the full pipeline
    try:
        res = predict_full_pipeline(formatted_input, conv_list, current_idx)

        print(f"\n▶ TEST: {scenario['name']}")
        print(f"  Input Text : '{formatted_input}'")
        print(f"  Emotion    : {res['emotion']} (Conf: {res['confidence']})")

        if res['trigger_level2']:
            print(f"  Pred Span  : '{res['cause']}'")
            if res['cause_confidence']:
                print(f"  Span Conf  : {res['cause_confidence']}")
        else:
            print(f"  Pred Span  : [SKIPPED - NEUTRAL]")

    except Exception as e:
        print(f"   Error in {scenario['name']}: {e}")

print(f"\n{'='*80}")
