"""
Downloads IBM Project CodeNet from HuggingFace (iNeil77/CodeNet).

This is the official IBM CodeNet dataset — the same one originally planned for
this project — now available on HuggingFace without the IBM CDN dependency.

Dataset: https://huggingface.co/datasets/iNeil77/CodeNet
Subsets : one per language (C++, Python, Java, …)
Columns : s_id, p_id, u_id, date, language, filename_ext,
          status, cpu_time, memory, code_size, code

Produces the same output layout the rest of the pipeline expects:
  data/raw/{problem_id}/{language}_{idx:03d}.{ext}  ← human solutions (label 0)
  data/downloads/metadata/problem_descriptions/{pid}.html  ← for AI prompts
  data/selected_problems.txt                        ← one problem ID per line

• Resume-safe: re-running skips problems already in data/raw/.
• Incremental: selected_problems.txt is updated after every problem collected.

Run this INSTEAD of download_codenet.py, then continue with:
  python scripts/generate_ai_solutions.py
  python scripts/build_dataset.py
"""

import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_DIR,
    MIN_ACCEPTED_SOLUTIONS,
    NUM_PROBLEMS,
    RAW_DIR,
)

# ── Config ─────────────────────────────────────────────────────────────────
HF_DATASET = "iNeil77/CodeNet"

# Map CodeNet language subset names → our internal keys and file extensions
LANG_CONFIG = {
    "C++":    ("cpp",    ".cpp"),
    "Python": ("python", ".py"),
    "Java":   ("java",   ".java"),
}

MAX_PER_LANG = 20          # max human solutions to keep per language per problem
ACCEPTED_STATUSES = {"Accepted"}

METADATA_DESC_DIR = DATA_DIR / "downloads" / "metadata" / "problem_descriptions"
SELECTED_PATH = DATA_DIR / "selected_problems.txt"


def save_description(pid: str, description: str) -> None:
    METADATA_DESC_DIR.mkdir(parents=True, exist_ok=True)
    dest = METADATA_DESC_DIR / f"{pid}.html"
    if not dest.exists():
        html = f"<html><body><pre>{description[:4000]}</pre></body></html>"
        dest.write_text(html, encoding="utf-8")


def load_existing() -> list[str]:
    if not RAW_DIR.exists():
        return []
    return [d.name for d in sorted(RAW_DIR.iterdir()) if d.is_dir() and any(d.iterdir())]


def write_selected(selected: list[str]) -> None:
    SELECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_PATH.write_text("\n".join(selected), encoding="utf-8")


def main() -> None:
    logger.info("=== IBM CodeNet Download (via HuggingFace: iNeil77/CodeNet) ===")
    logger.info(f"Target: {NUM_PROBLEMS} problems, >= {MIN_ACCEPTED_SOLUTIONS} accepted solutions/language")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Run: pip install datasets")
        sys.exit(1)

    # ── Resume: check existing data ────────────────────────────────────────
    existing = load_existing()
    if existing:
        logger.info(f"Found {len(existing)} problem(s) already in {RAW_DIR}")

    if len(existing) >= NUM_PROBLEMS:
        selected = existing[:NUM_PROBLEMS]
        write_selected(selected)
        logger.info(f"Already have enough problems. Saved {len(selected)} IDs → {SELECTED_PATH}")
        logger.info("Done. Run scripts/generate_ai_solutions.py next.")
        return

    # ── Collect accepted solutions per problem per language ────────────────
    # Structure: {p_id: {lang_key: [(code, s_id), ...]}}
    pool: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for subset_name, (lang_key, ext) in LANG_CONFIG.items():
        logger.info(f"Streaming {subset_name} subset …")
        try:
            ds = load_dataset(HF_DATASET, name=subset_name, split="train", streaming=True)
        except Exception as e:
            logger.warning(f"Could not load {subset_name}: {e}")
            continue

        scanned = 0
        for record in ds:
            status = (record.get("status") or "").strip()
            if status not in ACCEPTED_STATUSES:
                continue
            code = (record.get("code") or "").strip()
            if len(code) < 20:
                continue
            p_id = (record.get("p_id") or "").strip()
            if not p_id:
                continue

            pool[p_id][lang_key].append(code)
            scanned += 1

            # Stop early once we have enough data to fill NUM_PROBLEMS × MAX_PER_LANG
            # for this language (generous upper bound to ensure enough qualifying problems)
            if scanned >= NUM_PROBLEMS * MAX_PER_LANG * 10:
                break

        logger.info(f"  Collected accepted solutions from {len(pool)} problems so far")

    # ── Select problems that meet the minimum threshold in all 3 languages ─
    eligible = [
        p_id for p_id, langs in pool.items()
        if all(len(langs.get(lk, [])) >= MIN_ACCEPTED_SOLUTIONS for lk in ("cpp", "python", "java"))
    ]
    eligible.sort(key=lambda pid: sum(len(pool[pid].get(lk, [])) for lk in ("cpp", "python", "java")), reverse=True)

    logger.info(f"Found {len(eligible)} problems meeting the threshold")

    if not eligible:
        logger.warning(
            f"No problems meet threshold of {MIN_ACCEPTED_SOLUTIONS} per language. "
            "Lowering threshold: selecting problems with any accepted solutions."
        )
        eligible = sorted(pool.keys())

    selected: list[str] = list(existing)
    seen: set[str] = set(existing)
    write_selected(selected)

    bar = tqdm(total=NUM_PROBLEMS, initial=len(selected), desc="Problems saved", unit="prob")

    for p_id in eligible:
        if len(selected) >= NUM_PROBLEMS:
            break
        if p_id in seen:
            continue

        out_dir = RAW_DIR / p_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for lang_key, ext in [("cpp", ".cpp"), ("python", ".py"), ("java", ".java")]:
            codes = pool[p_id].get(lang_key, [])
            for idx, code in enumerate(codes[:MAX_PER_LANG]):
                fname = out_dir / f"{lang_key}_{idx:03d}{ext}"
                if not fname.exists():
                    fname.write_text(code, encoding="utf-8", errors="replace")

        # Placeholder description (CodeNet doesn't ship descriptions on HF)
        save_description(p_id, f"IBM CodeNet problem {p_id}. Solve the competitive programming problem.")

        seen.add(p_id)
        selected.append(p_id)
        write_selected(selected)
        bar.update(1)

    bar.close()

    if not selected:
        logger.error("No problems collected. Check your internet connection.")
        sys.exit(1)

    logger.info(f"Saved {len(selected)} problem IDs → {SELECTED_PATH}")
    logger.info(f"Human solutions     → {RAW_DIR}")
    logger.info(f"Problem descriptions → {METADATA_DESC_DIR}")
    logger.info("Done. Run scripts/generate_ai_solutions.py next.")


if __name__ == "__main__":
    main()
