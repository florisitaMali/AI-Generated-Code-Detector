"""
Performance evaluation restricted to two modalities:

  1) XGBoost / stylometric statistical baseline (`statistical_baseline.predict`)
  2) LLM-as-judge (`llm_judge.judge`) with optional human reference solutions

Reads `data/splits/test.parquet`. XGBoost is scored on the full split.

LLM calls are quota-expensive — use --llm-max-samples to cap (default 60).
Caches LLM outputs under `DATA_DIR/llm_judge_cache/` (reuse on re-run).

Usage:
  python scripts/evaluate_xgb_llm_judge.py
  python scripts/evaluate_xgb_llm_judge.py --llm-max-samples 0     # XGBoost only
  python scripts/evaluate_xgb_llm_judge.py --llm-max-samples 100 --llm-concurrency 2
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import RAW_DIR


def load_human_examples(problem_id: str, language: str, max_examples: int = 3) -> list[str]:
    lang_ext = {"cpp": ".cpp", "python": ".py", "java": ".java"}
    ext = lang_ext.get(language, "")
    problem_dir = RAW_DIR / str(problem_id)
    if not problem_dir.exists():
        return []
    examples = []
    for f in sorted(problem_dir.iterdir()):
        if f.is_file() and f.suffix == ext:
            try:
                examples.append(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
            if len(examples) >= max_examples:
                break
    return examples


def evaluate_xgboost(df: pd.DataFrame) -> dict:
    from src.models.statistical_baseline import predict as stat_predict

    scores, labels, skipped = [], [], 0
    for row in df.itertuples(index=False):
        try:
            s = stat_predict(row.code, row.language)
        except Exception:
            s = None
        if s is None:
            skipped += 1
            continue
        scores.append(float(s))
        labels.append(int(row.label))

    y = np.array(labels)
    p = np.array(scores)
    pred = (p >= 0.5).astype(int)

    out: dict = {
        "n_scored": len(y),
        "n_skipped": skipped,
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "classification_report": classification_report(
            y, pred, target_names=["Human", "AI"], digits=4, zero_division=0
        ),
    }
    return out


async def evaluate_llm(
    df: pd.DataFrame,
    max_samples: int,
    concurrency: int,
    seed: int,
) -> dict | None:
    if max_samples <= 0:
        return None

    from src.models.llm_judge import judge

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)

    sem = asyncio.Semaphore(max(concurrency, 1))
    lock = asyncio.Lock()
    scores: list[float] = []
    labels: list[int] = []
    errors = 0

    async def one_row(_idx: int, row: pd.Series):
        nonlocal errors
        async with sem:
            pid = row.get("problem_id")
            pid = str(pid) if pd.notna(pid) else "unknown"
            lang = str(row.get("language", "python"))
            examples = load_human_examples(pid, lang) if pid != "unknown" else []
            try:
                r = await judge(str(row["code"]), problem_id=pid, human_examples=examples)
                sc = float(r.get("score", 0.5))
            except Exception:
                async with lock:
                    errors += 1
                sc = 0.5
            async with lock:
                scores.append(sc)
                labels.append(int(row["label"]))

    await asyncio.gather(*[one_row(i, row) for i, row in df.iterrows()])

    y = np.array(labels)
    p = np.array(scores)
    pred = (p >= 0.5).astype(int)

    return {
        "n_scored": len(y),
        "n_errors_fallback": errors,
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "classification_report": classification_report(
            y, pred, target_names=["Human", "AI"], digits=4, zero_division=0
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate XGBoost + LLM-judge on test.parquet.")
    ap.add_argument("--llm-max-samples", type=int, default=60, help="Max test rows for LLM (0=skip)")
    ap.add_argument("--llm-concurrency", type=int, default=2, help="Concurrent LLM requests")
    ap.add_argument("--seed", type=int, default=42, help="Subsample RNG for LLM cap")
    args = ap.parse_args()

    test_path = ROOT / "data" / "splits" / "test.parquet"
    if not test_path.exists():
        print(f"Missing {test_path}; run scripts/build_dataset.py first.")
        sys.exit(1)

    df = pd.read_parquet(test_path)
    print(f"Test split: N={len(df)}  human={(df.label == 0).sum()}  AI={(df.label == 1).sum()}")

    print("\n=== Stylometric baseline (XGBoost) — full test split ===\n")
    xgb_out = evaluate_xgboost(df)
    print(f"Scored={xgb_out['n_scored']}  skipped={xgb_out['n_skipped']}")
    print(f"Accuracy: {xgb_out['accuracy']:.4f}")
    if xgb_out["roc_auc"] is not None:
        print(f"ROC-AUC: {xgb_out['roc_auc']:.4f}")
    print(xgb_out["classification_report"])

    if args.llm_max_samples <= 0:
        print("\n(LLM-as-judge skipped: --llm-max-samples 0)")
        return

    print(
        f"\n=== LLM-as-judge — up to {args.llm_max_samples} samples "
        f"(concurrency={args.llm_concurrency}) ===\n"
    )
    print("Ensure API keys in .env; responses are cached under data/llm_judge_cache/\n")

    llm_out = asyncio.run(
        evaluate_llm(df, args.llm_max_samples, args.llm_concurrency, args.seed)
    )
    if llm_out:
        print(f"Scored={llm_out['n_scored']}  heuristic_errors={llm_out['n_errors_fallback']}")
        print(f"Accuracy: {llm_out['accuracy']:.4f}")
        if llm_out["roc_auc"] is not None:
            print(f"ROC-AUC: {llm_out['roc_auc']:.4f}")
        print(llm_out["classification_report"])

    print("\nDone.")


if __name__ == "__main__":
    main()
