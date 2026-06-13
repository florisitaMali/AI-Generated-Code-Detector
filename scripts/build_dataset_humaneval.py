"""
Build a labeled dataset from HumanEval-X + Gemini AI solutions.

Source  : https://huggingface.co/datasets/THUDM/humaneval-x
Human   : canonical_solution for each problem (164 tasks × chosen languages)
AI      : Gemini generates a solution given the function prompt
Output  : data/splits_humaneval/{train,val,test}.parquet

Note: HumanEval-X is small (~164 problems × 3 languages = ~492 human samples).
After AI generation the total is ~984 samples. The dataset can be used for both
training (to see how models perform on function-completion style code) and as a
cross-dataset generalisation benchmark.

Usage:
    python scripts/build_dataset_humaneval.py

Options (env vars):
    HUMANEVAL_LANGUAGES   comma-separated (default: python,java,cpp)
    HUMANEVAL_SPLITS_DIR  override output directory
    GOOGLE_API_KEY        required for AI generation
    HUMANEVAL_SKIP_GEN    set to "1" to skip AI generation (human-only)
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR, GOOGLE_API_KEY, GEMINI_MODEL, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

HF_DATASET = "THUDM/humaneval-x"

DEFAULT_SPLITS_DIR = DATA_DIR / "splits_humaneval"
SPLITS_DIR = Path(os.getenv("HUMANEVAL_SPLITS_DIR", str(DEFAULT_SPLITS_DIR)))

LANGUAGES_RAW = os.getenv("HUMANEVAL_LANGUAGES", "python,java,cpp")
LANGUAGES = [l.strip() for l in LANGUAGES_RAW.split(",") if l.strip()]
SKIP_GENERATION = os.getenv("HUMANEVAL_SKIP_GEN", "0") in ("1", "true", "yes")

CACHE_DIR = DATA_DIR / "humaneval_generation_cache"
GEMINI_RATE_SLEEP = 5.0
GEMINI_RETRY_SLEEPS = [30, 60, 120]

# HumanEval-X language names → our internal keys
LANG_MAP = {
    "python": "python",
    "java":   "java",
    "cpp":    "cpp",
    "go":     "go",
    "js":     "javascript",
    "rust":   "rust",
}


# ── Gemini generation ────────────────────────────────────────────────────────

def _cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8"), usedforsecurity=False).hexdigest()


def _load_cache(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _save_cache(key: str, code: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.txt").write_text(code, encoding="utf-8")


def _gemini_generate(prompt: str, api_key: str) -> str | None:
    """Call Gemini with exponential backoff on 429 rate-limit errors."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
    }
    for attempt, wait in enumerate([0] + GEMINI_RETRY_SLEEPS):
        if wait:
            logger.info(f"  Rate-limited (429) — waiting {wait}s before retry {attempt}/{len(GEMINI_RETRY_SLEEPS)} …")
            time.sleep(wait)
        try:
            r = httpx.post(url, json=payload, timeout=30)
            if r.status_code == 429:
                continue
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            lines = text.strip().splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                continue
            logger.warning(f"Gemini HTTP error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Gemini generation failed: {e}")
            return None
    logger.warning("Gemini: exhausted all retries (still rate-limited)")
    return None


def _make_prompt(task_prompt: str, language: str) -> str:
    lang_display = {"python": "Python", "java": "Java", "cpp": "C++",
                    "go": "Go", "javascript": "JavaScript", "rust": "Rust"}.get(language, language)
    return (
        f"Complete the following {lang_display} function. "
        f"Return ONLY the complete function code, no explanations, no markdown fences.\n\n"
        f"{task_prompt}"
    )


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_humaneval(languages: list[str]) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Run: pip install datasets")
        sys.exit(1)

    frames = []
    for lang in languages:
        logger.info(f"Loading HumanEval-X '{lang}' …")
        try:
            ds = load_dataset(HF_DATASET, lang, split="test")
            df = ds.to_pandas()
        except Exception as e:
            logger.warning(f"Could not load '{lang}': {e}")
            continue

        logger.info(f"  Columns: {list(df.columns)}")

        our_lang = LANG_MAP.get(lang, lang)

        # Human solutions: prompt + canonical_solution gives the full function
        df["code"] = df.apply(
            lambda r: (str(r.get("prompt", "")) + str(r.get("canonical_solution", ""))).strip(),
            axis=1,
        )
        df["language"]   = our_lang
        df["label"]      = 0
        df["problem_id"] = df.get("task_id", pd.Series([f"{lang}_{i}" for i in range(len(df))])).astype(str)

        df = df[df["code"].str.len() >= 10].reset_index(drop=True)
        frames.append(df[["code", "language", "label", "problem_id", "prompt"]])
        logger.info(f"  Kept {len(df)} '{lang}' problems")

    if not frames:
        logger.error("No data loaded from HumanEval-X")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def generate_ai(human_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    ai_records = []
    total = len(human_df)
    logger.info(f"Generating AI counterparts for {total} HumanEval-X problems …")

    for i, row in enumerate(human_df.itertuples(), 1):
        prompt_text = str(getattr(row, "prompt", "") or "")
        language    = str(getattr(row, "language", "python"))
        problem_id  = str(getattr(row, "problem_id", f"he_{i}"))

        if not prompt_text.strip():
            continue

        gen_prompt = _make_prompt(prompt_text, language)
        key = _cache_key(gen_prompt)

        code = _load_cache(key)
        if code is None:
            code = _gemini_generate(gen_prompt, api_key)
            if code:
                _save_cache(key, code)
            time.sleep(GEMINI_RATE_SLEEP)

        if code and len(code.strip()) >= 10:
            ai_records.append({
                "code":       code,
                "language":   language,
                "label":      1,
                "problem_id": problem_id,
            })

        if i % 50 == 0:
            logger.info(f"  {i}/{total} processed")

    logger.info(f"Generated {len(ai_records)} AI solutions")
    return pd.DataFrame(ai_records)


def main():
    logger.info("=== Building HumanEval-X evaluation dataset ===")
    logger.info(f"Languages: {LANGUAGES}")
    logger.info("Note: this dataset is for OUT-OF-DISTRIBUTION evaluation only (no training)")

    human_df = load_humaneval(LANGUAGES)
    logger.info(f"Human samples: {len(human_df)}")

    if SKIP_GENERATION:
        logger.warning("HUMANEVAL_SKIP_GEN=1 — skipping AI generation")
        ai_df = pd.DataFrame(columns=["code", "language", "label", "problem_id"])
    else:
        if not GOOGLE_API_KEY:
            logger.error(
                "GOOGLE_API_KEY is not set. Set it in .env or set HUMANEVAL_SKIP_GEN=1."
            )
            sys.exit(1)
        ai_df = generate_ai(human_df[["code", "language", "label", "problem_id", "prompt"]], GOOGLE_API_KEY)

    human_out = human_df[["code", "language", "label", "problem_id"]].copy()
    df = pd.concat([human_out, ai_df], ignore_index=True)

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(f"Language distribution:\n{df['language'].value_counts().to_string()}")

    # Stratified split on rows (problem_ids are unique per sample here)
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
    logger.info(
        "Note: HumanEval-X is small (~492 human + ~492 AI = ~984 total). "
        "Expect higher variance in results compared to larger datasets."
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(SPLITS_DIR / "train.parquet", index=False)
    val_df.to_parquet(SPLITS_DIR   / "val.parquet",   index=False)
    test_df.to_parquet(SPLITS_DIR  / "test.parquet",  index=False)

    stats = {
        "source": HF_DATASET,
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
    logger.info("Done. Run cross_dataset_comparison.py to train and compare all models.")


if __name__ == "__main__":
    main()
