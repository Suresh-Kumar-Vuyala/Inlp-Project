# meta_learner/train.py
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import argparse

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Emotion label map (0-6 → name) ─────────────────────────────────────────
EMOTION_MAP = {0: "anger", 1: "disgust", 2: "fear",
               3: "joy",   4: "sadness", 5: "surprise", 6: "neutral"}

# ── File paths — update DATA_DIR for your environment ──────────────────────
DATA_DIR = "."          # ← change to e.g. "/content/drive/MyDrive/parse_force"

ROBERTA_CSV = os.path.join(DATA_DIR, "roberta_logits_validation.csv")
DEBERTA_CSV = os.path.join(DATA_DIR, "deberta_logits_validation.csv")
LLM_CSV     = os.path.join(DATA_DIR, "llm_lite_logits_validation.csv")
VAL_JSON    = os.path.join(DATA_DIR, "val.json")
TRAIN_JSON  = os.path.join(DATA_DIR, "train.json")  # optional; used for meta-feature stats

# Standard emotion order (0-indexed):  anger disgust fear joy sadness surprise neutral
EMOTION_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

def load_roberta(path: str) -> pd.DataFrame:
    """Load RoBERTa CSV and rename prob columns with rob_ prefix."""
    df = pd.read_csv(path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    rename = {c: f"rob_{c.replace('prob_', '')}" for c in prob_cols}
    df = df.rename(columns=rename)[list(rename.values()) + ["true_label"]]
    # Reorder to canonical emotion order
    rob_cols = [f"rob_{e}" for e in EMOTION_COLS]
    return df[rob_cols + ["true_label"]]


def load_deberta(path: str) -> pd.DataFrame:
    """Load DeBERTa CSV and rename prob columns with deb_ prefix."""
    df = pd.read_csv(path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    rename = {c: f"deb_{c.replace('prob_', '')}" for c in prob_cols}
    df = df.rename(columns=rename)[list(rename.values())]
    deb_cols = [f"deb_{e}" for e in EMOTION_COLS]
    return df[deb_cols]


def load_llm(path: str) -> pd.DataFrame:
    """Load LLM-Lite CSV and rename columns with llm_ prefix.

    LLM CSV column order: anger disgust fear joy neutral sadness surprise
    We reorder to canonical order before prefixing.
    """
    df = pd.read_csv(path)
    # Map original columns → canonical order
    rename = {e: f"llm_{e}" for e in EMOTION_COLS if e in df.columns}
    df = df.rename(columns=rename)
    llm_cols = [f"llm_{e}" for e in EMOTION_COLS]
    return df[llm_cols]


def fuse_experts(rob_path: str, deb_path: str, llm_path: str) -> pd.DataFrame:
    """Merge three expert DataFrames into a single 21-dim feature frame."""
    rob_df = load_roberta(rob_path)
    deb_df = load_deberta(deb_path)
    llm_df = load_llm(llm_path)

    fused = pd.concat([
        rob_df.drop(columns=["true_label"]),
        deb_df,
        llm_df
    ], axis=1)
    fused["true_label"] = rob_df["true_label"].values

    assert fused.shape[1] == 22, f"Expected 22 cols (21 features + label), got {fused.shape[1]}"
    return fused


def build_dialogue_metafeatures(json_path: str) -> pd.DataFrame:
    """
    Parse val.json and extract:
      - speaker_shift    : bool, current speaker != previous speaker
      - utterance_position : int, 0-indexed position within conversation
      - conversation_length: int, total utterances in conversation
    Returns a DataFrame aligned row-for-row with the CSV files.
    """
    with open(json_path) as f:
        records = json.load(f)

    # Group by conversation_id to compute per-conversation stats
    from collections import defaultdict
    conv_groups = defaultdict(list)
    for r in records:
        conv_groups[r["conversation_id"]].append(r)

    # Sort each conversation by utterance_idx (just in case)
    for cid in conv_groups:
        conv_groups[cid].sort(key=lambda x: x["utterance_idx"])

    rows = []
    for r in records:
        cid   = r["conversation_id"]
        u_idx = r["utterance_idx"]
        conv  = conv_groups[cid]
        conv_len = len(conv)

        # Speaker shift: compare to the utterance immediately before in this conv
        if u_idx == 0:
            shift = 0
        else:
            prev_utt = conv[u_idx - 1]          # sorted by utterance_idx
            shift = int(r["speaker"] != prev_utt["speaker"])

        rows.append({
            "speaker_shift"       : shift,
            "utterance_position"  : u_idx,
            "conversation_length" : conv_len,
        })

    meta_df = pd.DataFrame(rows)
    return meta_df


def build_xgb_classifier(params: dict = None) -> xgb.XGBClassifier:
    """Instantiate XGBClassifier with sensible defaults for small validation sets."""
    defaults = dict(
        objective      = "multi:softprob",
        num_class      = 7,
        max_depth      = 4,
        learning_rate  = 0.1,
        n_estimators   = 200,
        subsample      = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,       # regularisation against overfitting
        gamma          = 0.1,
        reg_alpha      = 0.1,       # L1
        reg_lambda     = 1.0,       # L2
        use_label_encoder = False,
        eval_metric    = "mlogloss",
        random_state   = 42,
        n_jobs         = -1,
    )
    if params:
        defaults.update(params)
    return xgb.XGBClassifier(**defaults)


def grid_search_xgb(X, y, cv: int = 5) -> xgb.XGBClassifier:
    """Run a focused grid search and return the best estimator."""
    param_grid = {
        "max_depth"       : [3, 4, 5],
        "learning_rate"   : [0.05, 0.1, 0.15],
        "n_estimators"    : [150, 200, 300],
        "min_child_weight": [2, 3, 5],
    }
    base = build_xgb_classifier()
    skf  = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs   = GridSearchCV(
        base, param_grid,
        scoring="f1_macro",
        cv=skf, n_jobs=-1,
        verbose=0, refit=True
    )
    gs.fit(X, y)
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV Macro-F1 : {gs.best_score_:.4f}")
    return gs.best_estimator_


def cross_val_predict_proba(clf, X, y, cv=5):
    """Generate OOF probability predictions via StratifiedKFold."""
    n_classes = len(np.unique(y))
    oof_proba = np.zeros((len(y), n_classes), dtype=np.float32)
    oof_preds = np.zeros(len(y), dtype=int)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        clone_clf = build_xgb_classifier(clf.get_params())
        clone_clf.fit(X[tr_idx], y[tr_idx])
        proba = clone_clf.predict_proba(X[val_idx])
        oof_proba[val_idx] = proba
        oof_preds[val_idx] = proba.argmax(axis=1)
        macro_f = f1_score(y[val_idx], oof_preds[val_idx], average="macro", zero_division=0)
        print(f"  Fold {fold} — Macro-F1: {macro_f:.4f}")

    return oof_proba, oof_preds


def main():
    parser = argparse.ArgumentParser(description="Train Meta-Learner XGBoost model")
    parser.add_argument("--data_dir", default=".", help="Directory containing the CSV and JSON files")
    parser.add_argument("--output_model", default="models/meta_learner.json", help="Path to save the trained model")
    args = parser.parse_args()

    global DATA_DIR, ROBERTA_CSV, DEBERTA_CSV, LLM_CSV, VAL_JSON
    DATA_DIR = args.data_dir
    ROBERTA_CSV = os.path.join(DATA_DIR, "roberta_logits_validation.csv")
    DEBERTA_CSV = os.path.join(DATA_DIR, "deberta_logits_validation.csv")
    LLM_CSV = os.path.join(DATA_DIR, "llm_lite_logits_validation.csv")
    VAL_JSON = os.path.join(DATA_DIR, "val.json")

    # Verify files exist
    for path in [ROBERTA_CSV, DEBERTA_CSV, LLM_CSV, VAL_JSON]:
        status = "✅" if os.path.exists(path) else "❌ MISSING"
        print(f"{status}  {path}")

    # Fuse experts
    fused_df = fuse_experts(ROBERTA_CSV, DEBERTA_CSV, LLM_CSV)
    FEATURE_COLS_21 = [c for c in fused_df.columns if c != "true_label"]

    # Build meta-features
    meta_df = build_dialogue_metafeatures(VAL_JSON)

    # Final feature matrix
    full_df = pd.concat([fused_df.reset_index(drop=True),
                         meta_df.reset_index(drop=True)], axis=1)
    ALL_FEATURE_COLS = FEATURE_COLS_21 + ["speaker_shift", "utterance_position", "conversation_length"]
    X = full_df[ALL_FEATURE_COLS].values.astype(np.float32)
    y = full_df["true_label"].values.astype(int)

    print(f"✅ Final feature matrix X: {X.shape}  |  Labels y: {y.shape}")

    # Grid Search
    print("⏳ Running grid search (5-fold StratifiedKFold) ...")
    best_clf = grid_search_xgb(X, y, cv=5)
    print("✅ Grid search complete.")

    # OOF predictions
    print("⏳ Generating out-of-fold predictions ...")
    oof_proba, oof_preds = cross_val_predict_proba(best_clf, X, y, cv=5)

    # Classification report
    target_names = [EMOTION_MAP[i] for i in range(7)]
    print(classification_report(y, oof_preds,
                                 target_names=target_names,
                                 digits=4, zero_division=0))

    macro_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
    macro_prec = precision_score(y, oof_preds, average="macro", zero_division=0)
    macro_rec = recall_score(y, oof_preds, average="macro", zero_division=0)

    print(f"Macro-F1: {macro_f1:.4f}, Macro-Precision: {macro_prec:.4f}, Macro-Recall: {macro_rec:.4f}")

    # Confidence scores
    confidence_scores = oof_proba.max(axis=1)
    predicted_labels = oof_proba.argmax(axis=1)

    # Threshold sweep
    thresholds = np.linspace(0.20, 0.95, 76)
    macro_f1s = []
    coverages = []

    for tau in thresholds:
        mask = confidence_scores >= tau
        n_triggered = mask.sum()
        if n_triggered < 10:
            macro_f1s.append(np.nan)
            coverages.append(n_triggered / len(y))
            continue
        mf = f1_score(y[mask], predicted_labels[mask],
                      average="macro", zero_division=0)
        macro_f1s.append(mf)
        coverages.append(n_triggered / len(y))

    macro_f1s = np.array(macro_f1s)
    coverages = np.array(coverages)
    valid = ~np.isnan(macro_f1s)
    best_idx = np.nanargmax(macro_f1s)
    OPTIMAL_TAU = thresholds[best_idx]

    print(f"Optimal threshold τ* = {OPTIMAL_TAU:.3f}, Macro-F1 at τ* = {macro_f1s[best_idx]:.4f}, Coverage = {coverages[best_idx]*100:.1f}%")

    # Refit final model
    final_clf = build_xgb_classifier(best_clf.get_params())
    final_clf.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    final_clf.save_model(args.output_model)
    print(f"✅ Model saved to {args.output_model}")

    # Optionally, save OPTIMAL_TAU somewhere, but for now, print it
    print(f"Optimal threshold: {OPTIMAL_TAU}")


if __name__ == "__main__":
    main()
