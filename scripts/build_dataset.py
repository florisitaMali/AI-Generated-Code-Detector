"""
Merge human (CodeNet) and AI-generated solutions into a labeled Parquet dataset
with stratified train/val/test splits.

Output columns: problem_id, language, source, label, code, model_name
  - label 0 = human, label 1 = AI-generated
  - source = "codenet" or model identifier
"""

import re
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_DIR,
    GENERATED_DIR,
    LANGUAGE_EXTENSIONS,
    PROCESSED_DIR,
    RAW_DIR,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def detect_language(filepath: Path) -> str | None:
    ext = filepath.suffix.lower()
    for lang, exts in LANGUAGE_EXTENSIONS.items():
        if ext in exts:
            return lang
    return None


def load_human_solutions() -> list[dict]:
    """Load all human solutions from data/raw/."""
    records = []
    if not RAW_DIR.exists():
        logger.warning(f"Raw data directory not found: {RAW_DIR}")
        return records

    for problem_dir in sorted(RAW_DIR.iterdir()):
        if not problem_dir.is_dir():
            continue
        pid = problem_dir.name
        for solution_file in problem_dir.iterdir():
            if solution_file.is_file():
                lang = detect_language(solution_file)
                if lang is None:
                    continue
                try:
                    code = solution_file.read_text(encoding="utf-8", errors="replace")
                    if len(code.strip()) < 10:
                        continue
                    records.append({
                        "problem_id": pid,
                        "language": lang,
                        "source": "codenet",
                        "label": 0,
                        "code": code,
                        "model_name": None,
                    })
                except Exception as e:
                    logger.warning(f"Error reading {solution_file}: {e}")

    logger.info(f"Loaded {len(records)} human solutions")
    return records


def load_ai_solutions() -> list[dict]:
    """Load all AI-generated solutions from data/generated/."""
    records = []
    if not GENERATED_DIR.exists():
        logger.warning(f"Generated data directory not found: {GENERATED_DIR}")
        return records

    for problem_dir in sorted(GENERATED_DIR.iterdir()):
        if not problem_dir.is_dir():
            continue
        pid = problem_dir.name
        for solution_file in problem_dir.iterdir():
            if not solution_file.is_file():
                continue
            lang = detect_language(solution_file)
            if lang is None:
                continue

            model_name = extract_model_name(solution_file.stem)
            try:
                code = solution_file.read_text(encoding="utf-8", errors="replace")
                if len(code.strip()) < 10:
                    continue
                records.append({
                    "problem_id": pid,
                    "language": lang,
                    "source": model_name or "ai_unknown",
                    "label": 1,
                    "code": code,
                    "model_name": model_name,
                })
            except Exception as e:
                logger.warning(f"Error reading {solution_file}: {e}")

    logger.info(f"Loaded {len(records)} AI-generated solutions")
    return records


def extract_model_name(stem: str) -> str | None:
    """Extract model name from filename like 'gpt_4_turbo_python_0'."""
    match = re.match(r"^(.+?)_(cpp|python|java)_\d+$", stem)
    if match:
        return match.group(1)
    return stem


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by (problem_id, label) groups.
    Ensures no problem leaks across splits.
    """
    # Plain list — avoids sklearn + pyarrow-backed Index (TypeError on fancy indexing).
    problem_ids = df["problem_id"].astype(str).unique().tolist()

    train_pids, temp_pids = train_test_split(
        problem_ids,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=42,
    )
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_pids, test_pids = train_test_split(
        temp_pids,
        test_size=relative_test,
        random_state=42,
    )

    train_df = df[df["problem_id"].isin(train_pids)].reset_index(drop=True)
    val_df = df[df["problem_id"].isin(val_pids)].reset_index(drop=True)
    test_df = df[df["problem_id"].isin(test_pids)].reset_index(drop=True)

    return train_df, val_df, test_df


def main():
    logger.info("=== Building Labeled Dataset ===")

    human = load_human_solutions()
    ai = load_ai_solutions()

    if not human and not ai:
        logger.error(
            "No data found. Run scripts/download_hf_dataset.py (or download_codenet.py) "
            "and scripts/generate_ai_solutions.py first."
        )
        return

    df = pd.DataFrame(human + ai)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(f"Language distribution:\n{df['language'].value_counts().to_string()}")
    logger.info(f"Problems: {df['problem_id'].nunique()}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full_path = PROCESSED_DIR / "dataset_full.parquet"
    df.to_parquet(full_path, index=False)
    logger.info(f"Saved full dataset to {full_path}")

    train_df, val_df, test_df = stratified_split(df)
    logger.info(
        f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    val_df.to_parquet(SPLITS_DIR / "val.parquet", index=False)
    test_df.to_parquet(SPLITS_DIR / "test.parquet", index=False)

    logger.info(f"Saved train/val/test splits to {SPLITS_DIR}")

    stats = {
        "total": len(df),
        "human": int((df["label"] == 0).sum()),
        "ai": int((df["label"] == 1).sum()),
        "problems": int(df["problem_id"].nunique()),
        "languages": df["language"].unique().tolist(),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
    stats_path = PROCESSED_DIR / "dataset_stats.json"
    import json
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"Dataset stats saved to {stats_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
