"""
Build a training dataset from OSS-forge/HumanVsAICode.

Source  : https://huggingface.co/datasets/OSS-forge/HumanVsAICode
Format  : Wide format — each row has one human solution + 3 AI solutions
          (ChatGPT, DeepSeek-Coder, Qwen-Coder) for the same problem.
Languages: Python (~285k rows) and Java (~222k rows)
Output  : data/splits_llmgen/{train,val,test}.parquet

The wide format is melted to long format:
  - Human-code  → label=0
  - ChatGPT/DeepSeek-Coder/Qwen-Coder code → label=1

No API key or generation step required — labels are already provided.

Usage:
    python scripts/build_dataset_llmgen.py

Options (env vars):
    LLMGEN_MAX_SAMPLES   max rows to keep per language before melting (default: 10000)
    LLMGEN_SPLITS_DIR    override output directory
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

HF_DATASET = "OSS-forge/HumanVsAICode"

DEFAULT_SPLITS_DIR = DATA_DIR / "splits_llmgen"
SPLITS_DIR = Path(os.getenv("LLMGEN_SPLITS_DIR", str(DEFAULT_SPLITS_DIR)))

# Max rows to sample per language BEFORE melting (each row → 4 samples after melt)
MAX_ROWS_PER_LANG = int(os.getenv("LLMGEN_MAX_SAMPLES", "10000"))

# HumanVsAICode subset names → our internal language keys
LANG_SUBSETS = {
    "python": "python",
    "java":   "java",
}

# Wide columns that contain code (human + AI variants).
# The dataset uses snake_case column names: human_code, chatgpt_code, dsc_code, qwen_code.
HUMAN_COL = "human_code"
AI_COLS = {
    "chatgpt_code": "chatgpt",
    "dsc_code":     "deepseek",
    "qwen_code":    "qwen",
}


def load_subset(lang_key: str, our_lang: str, max_rows: int) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Run: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading {HF_DATASET} — '{lang_key}' subset …")
    try:
        ds = load_dataset(HF_DATASET, lang_key, split="train")
        df = ds.to_pandas()
    except Exception:
        # Try without subset name (some datasets expose all languages in one split)
        try:
            ds = load_dataset(HF_DATASET, split="train")
            df = ds.to_pandas()
            if "language" in df.columns:
                df = df[df["language"].str.lower() == lang_key].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not load '{lang_key}' subset: {e}")
            return pd.DataFrame()

    logger.info(f"  Raw rows: {len(df)} | columns: {list(df.columns)}")

    # Sample down if needed
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)
        logger.info(f"  Sampled down to {max_rows} rows")

    # Melt wide → long
    records = []
    for _, row in df.iterrows():
        problem_id = f"oss_{our_lang}_{row.get('index', len(records))}"

        # Human solution
        human_code = str(row.get(HUMAN_COL, "") or "").strip()
        if len(human_code) >= 10:
            records.append({
                "code":       human_code,
                "language":   our_lang,
                "label":      0,
                "problem_id": problem_id,
            })

        # AI solutions
        for col, source in AI_COLS.items():
            ai_code = str(row.get(col, "") or "").strip()
            if len(ai_code) >= 10:
                records.append({
                    "code":       ai_code,
                    "language":   our_lang,
                    "label":      1,
                    "problem_id": problem_id,
                })

    result = pd.DataFrame(records)
    logger.info(
        f"  After melt: {len(result)} samples "
        f"(human: {(result['label']==0).sum()}, AI: {(result['label']==1).sum()})"
    )
    return result


def main():
    logger.info("=== Building OSS-forge/HumanVsAICode dataset ===")

    frames = []
    for lang_key, our_lang in LANG_SUBSETS.items():
        df = load_subset(lang_key, our_lang, MAX_ROWS_PER_LANG)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.error("No data loaded. Check dataset name and internet connection.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["code"].str.strip().str.len() >= 10].reset_index(drop=True)

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(f"Language distribution:\n{df['language'].value_counts().to_string()}")

    # Stratified split by rows (problem_ids repeat across human/AI rows)
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
        "ai_models_covered": list(AI_COLS.values()),
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
    logger.info("Done. Run: python scripts/cross_dataset_comparison.py --dataset llmgen")


if __name__ == "__main__":
    main()
