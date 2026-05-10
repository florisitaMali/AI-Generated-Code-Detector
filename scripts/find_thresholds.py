"""
Threshold calibration script.

Runs the statistical and CodeBERT modules over the test split (no LLM API calls),
collects score distributions per true label, and reports:
  - Mean / std per class per module
  - Optimal Youden threshold (maximises sensitivity + specificity)
  - Optimal F1 threshold
  - Logs every sample score to debug-a9537f.log for analysis
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "debug-a9537f.log"

def _log(msg, data, hyp="ALL"):
    entry = {"sessionId":"a9537f","hypothesisId":hyp,
             "location":"find_thresholds.py","message":msg,
             "data":data,"timestamp":int(time.time()*1000)}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def youden_threshold(y_true, scores):
    """Return threshold that maximises Youden's J = sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_idx = np.argmax(j)
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(fpr[best_idx])


def f1_threshold(y_true, scores):
    """Return threshold that maximises F1 on the positive (AI) class."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        preds = (np.array(scores) >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def main():
    splits_dir = ROOT / "data" / "splits"
    test_path = splits_dir / "test.parquet"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. Run build_dataset.py first.")
        sys.exit(1)

    df = pd.read_parquet(test_path)
    print(f"Test split: {len(df)} samples  (human={sum(df.label==0)}, AI={sum(df.label==1)})")

    # --- Statistical module ---
    try:
        from src.models.statistical_baseline import predict as stat_predict
        print("\nRunning statistical module...")
        stat_scores, stat_labels = [], []
        for i, row in enumerate(df.itertuples(), 1):
            try:
                s = stat_predict(row.code, row.language)
            except Exception:
                s = None
            if s is not None:
                stat_scores.append(s)
                stat_labels.append(row.label)
                _log("stat_score", {"score": round(s, 4), "label": row.label, "language": row.language}, "B")
            if i % 50 == 0:
                print(f"  stat: {i}/{len(df)}")

        stat_labels = np.array(stat_labels)
        stat_scores = np.array(stat_scores)
        print(f"\n  STATISTICAL MODULE  ({len(stat_scores)} scored)")
        for lbl, name in [(0,"Human"),(1,"AI")]:
            mask = stat_labels == lbl
            s = stat_scores[mask]
            print(f"    {name:6s}: mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}")
        st_youden, tpr, fpr = youden_threshold(stat_labels, stat_scores)
        st_f1t, st_f1 = f1_threshold(stat_labels, stat_scores)
        print(f"    Youden threshold = {st_youden:.3f}  (TPR={tpr:.3f}, FPR={fpr:.3f})")
        print(f"    F1    threshold  = {st_f1t:.3f}  (F1={st_f1:.3f})")
        _log("stat thresholds", {"youden": st_youden, "f1_threshold": st_f1t, "f1": st_f1}, "B-D")
    except Exception as e:
        print(f"  Statistical module failed: {e}")

    # --- CodeBERT module ---
    try:
        from src.models.codebert_classifier import (
            predict_zero_shot, warm_up_zero_shot, _get_zero_shot_model
        )
        from src.models.codebert_classifier import _zs_loading as _check_loading, _zs_ready as _check_ready
        print("\nRunning CodeBERT zero-shot module (this takes a few minutes)...")
        print("  Waiting for model to load...", end="", flush=True)
        warm_up_zero_shot()
        # Wait until loaded OR load thread finishes (failed)
        deadline = time.time() + 120
        while True:
            if _get_zero_shot_model() is not None:
                print(" ready.")
                break
            # Re-read module globals to detect load failure
            import src.models.codebert_classifier as _cb_mod
            if not _cb_mod._zs_loading and not _cb_mod._zs_ready:
                print("\n  ERROR: CodeBERT model failed to load.")
                print("  FIX:  Activate your venv and upgrade torch:")
                print("        .venv\\Scripts\\Activate.ps1")
                print("        pip install --upgrade torch")
                raise RuntimeError("CodeBERT model load failed — upgrade torch to >= 2.6")
            if time.time() > deadline:
                raise RuntimeError("Timed out waiting for CodeBERT model")
            time.sleep(0.5)
            print(".", end="", flush=True)

        cb_scores, cb_labels = [], []
        # Use a subset for speed (max 200 samples, balanced)
        human_rows = df[df.label == 0].head(100)
        ai_rows    = df[df.label == 1].head(100)
        subset = pd.concat([human_rows, ai_rows]).sample(frac=1, random_state=42)
        for i, row in enumerate(subset.itertuples(), 1):
            try:
                s = predict_zero_shot(row.code)
            except Exception as e:
                print(f"  skip row {i}: {e}")
                continue
            cb_scores.append(s)
            cb_labels.append(row.label)
            _log("codebert_score", {"score": round(s, 4), "label": row.label, "language": row.language}, "A")
            if i % 20 == 0:
                print(f"  codebert: {i}/{len(subset)}")

        cb_labels = np.array(cb_labels)
        cb_scores = np.array(cb_scores)
        print(f"\n  CODEBERT MODULE  ({len(cb_scores)} scored)")
        for lbl, name in [(0,"Human"),(1,"AI")]:
            mask = cb_labels == lbl
            s = cb_scores[mask]
            if len(s) == 0:
                continue
            print(f"    {name:6s}: mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}")
        if len(np.unique(cb_labels)) > 1:
            cb_youden, tpr, fpr = youden_threshold(cb_labels, cb_scores)
            cb_f1t, cb_f1 = f1_threshold(cb_labels, cb_scores)
            print(f"    Youden threshold = {cb_youden:.3f}  (TPR={tpr:.3f}, FPR={fpr:.3f})")
            print(f"    F1    threshold  = {cb_f1t:.3f}  (F1={cb_f1:.3f})")
            _log("codebert thresholds", {"youden": cb_youden, "f1_threshold": cb_f1t, "f1": cb_f1}, "A-D")
    except Exception as e:
        print(f"  CodeBERT module failed: {e}")
        import traceback; traceback.print_exc()

    print("\nDone. Scores logged to debug-a9537f.log")


if __name__ == "__main__":
    main()
