"""
Fit sigmoid temperature T (prob = sigmoid(logit / T)) on the validation split.

Fine-tuned CodeBERT logits are often extreme, so raw sigmoid outputs cluster near
0 and 1. Choosing T > 1 lowers cross-entropy on held-out labels and yields smoother
probabilities for calibration plots — ROC-AUC is unchanged (monotonic transform).

Writes models/codebert_final/inference_calibration.json (override with --output).

Usage:
  python scripts/fit_codebert_temperature.py
  python scripts/fit_codebert_temperature.py --max-rows 800
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import MODELS_DIR, SPLITS_DIR  # noqa: E402


def binary_nll(y: np.ndarray, logits: np.ndarray, T: float) -> float:
    T = max(float(T), 1e-8)
    z = np.clip(logits / T, -60.0, 60.0)
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-12
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit CodeBERT logit temperature on val.parquet.")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = full validation split")
    ap.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "codebert_final" / "inference_calibration.json",
        help="Where to write inference_calibration.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    val_path = SPLITS_DIR / "val.parquet"
    if not val_path.exists():
        print(f"Missing {val_path}")
        sys.exit(1)

    ckpt = MODELS_DIR / "codebert_final"
    if not ckpt.exists():
        print(f"Missing fine-tuned checkpoint directory: {ckpt}")
        sys.exit(1)

    df = pd.read_parquet(val_path)
    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

    from src.models.codebert_classifier import predict_raw_logit  # noqa: E402

    logits: list[float] = []
    ys: list[int] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="val logits"):
        code = getattr(row, "code")
        lab = int(getattr(row, "label"))
        lg = predict_raw_logit(code)
        if lg is None:
            print("Fine-tuned CodeBERT weights not loaded (predict_raw_logit returned None).")
            sys.exit(1)
        logits.append(float(lg))
        ys.append(lab)

    y = np.asarray(ys, dtype=int)
    L = np.asarray(logits, dtype=np.float64)

    Ts = np.unique(np.concatenate([np.linspace(0.25, 8.0, 120), np.linspace(8.0, 24.0, 35)]))
    nlls = np.array([binary_nll(y, L, t) for t in Ts])
    best_i = int(np.argmin(nlls))
    best_T = float(Ts[best_i])

    out = {
        "temperature": best_T,
        "val_binary_nll": float(nlls[best_i]),
        "val_binary_nll_T_eq_1": binary_nll(y, L, 1.0),
        "n_samples": int(len(y)),
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Restart API / Streamlit after updating so cached temperature reloads.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {args.output.resolve()}")

    try:
        import src.models.codebert_classifier as cb_mod

        cb_mod.reload_inference_temperature()
    except Exception:
        pass


if __name__ == "__main__":
    main()
