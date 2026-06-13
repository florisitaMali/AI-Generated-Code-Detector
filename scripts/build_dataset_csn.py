"""
Build a training dataset from basakdemirok/AIGCodeSet.

Source  : https://huggingface.co/datasets/basakdemirok/AIGCodeSet
Paper   : IEEE — research-backed, clean labels
Language: Python only
Size    : ~7,583 samples (2,828 AI-generated + 4,755 human-written)
AI models: CodeLlama 34B, Codestral 22B, Gemini 1.5 Flash
Output  : data/splits_csn/{train,val,test}.parquet

No API key or generation step required — labels are already provided.

Usage:
    python scripts/build_dataset_csn.py

Options (env vars):
    AIGCODE_SPLITS_DIR   override output directory
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

HF_DATASET = "basakdemirok/AIGCodeSet"

DEFAULT_SPLITS_DIR = DATA_DIR / "splits_csn"
SPLITS_DIR = Path(os.getenv("AIGCODE_SPLITS_DIR", str(DEFAULT_SPLITS_DIR)))


def _find_code_col(cols: list[str]) -> str | None:
    for candidate in ("code", "source_code", "content", "text", "func_code_string", "Code"):
        if candidate in cols:
            return candidate
    return None


def _find_label_col(cols: list[str]) -> str | None:
    for candidate in ("label", "labels", "is_ai", "ai_generated", "generated",
                      "Label", "AI", "class"):
        if candidate in cols:
            return candidate
    return None


def main():
    logger.info("=== Building basakdemirok/AIGCodeSet dataset ===")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Run: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading {HF_DATASET} from HuggingFace …")
    try:
        ds = load_dataset(HF_DATASET)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    frames = []
    for split_name in ds.keys():
        df = ds[split_name].to_pandas()
        logger.info(f"  Split '{split_name}': {len(df)} rows | columns: {list(df.columns)}")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    cols = list(df.columns)
    logger.info(f"All columns: {cols}")

    # ── Find code column ──────────────────────────────────────────────────────
    code_col = _find_code_col(cols)
    if code_col is None:
        logger.error(f"Cannot find a code column. Available: {cols}")
        sys.exit(1)
    logger.info(f"Using code column: '{code_col}'")

    # ── Find label column ─────────────────────────────────────────────────────
    label_col = _find_label_col(cols)
    if label_col is None:
        logger.error(f"Cannot find a label column. Available: {cols}")
        sys.exit(1)
    logger.info(f"Using label column: '{label_col}'")

    df = df.rename(columns={code_col: "code", label_col: "label"})
    df["label"] = df["label"].astype(int)
    df["language"] = "python"   # AIGCodeSet is Python-only

    # Synthetic problem_id
    df["problem_id"] = [f"aigcode_{i}" for i in range(len(df))]

    # Drop empty code
    before = len(df)
    df = df[df["code"].str.strip().str.len() >= 10].reset_index(drop=True)
    logger.info(f"Dropped {before - len(df)} rows with too-short code")

    df = df[["code", "language", "label", "problem_id"]].copy()

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    # Stratified split
    train_df, tmp = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO), random_state=42, stratify=df["label"]
    )
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        tmp, test_size=relative_test, random_state=42, stratify=tmp["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    logger.info(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    val_df.to_parquet(SPLITS_DIR   / "val.parquet",   index=False)
    test_df.to_parquet(SPLITS_DIR  / "test.parquet",  index=False)

    stats = {
        "source": HF_DATASET,
        "ai_models_covered": ["CodeLlama-34B", "Codestral-22B", "Gemini-1.5-Flash"],
        "total": len(df),
        "human": int((df["label"] == 0).sum()),
        "ai": int((df["label"] == 1).sum()),
        "languages": df["language"].value_counts().to_dict(),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
    (SPLITS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(f"Splits saved to {SPLITS_DIR}")
    logger.info("Done. Run: python scripts/cross_dataset_comparison.py --dataset csn")


if __name__ == "__main__":
    main()
