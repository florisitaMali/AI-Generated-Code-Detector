"""
Download and filter the IBM CodeNet dataset.

Selects NUM_PROBLEMS problems across C++, Python, and Java that have at least
MIN_ACCEPTED_SOLUTIONS accepted human submissions each. Organises filtered
solutions into data/raw/{problem_id}/{solution_id}.{ext}.
"""

import csv
import io
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import httpx
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    CODENET_METADATA_URL,
    CODENET_URL,
    DATA_DIR,
    LANGUAGE_EXTENSIONS,
    MIN_ACCEPTED_SOLUTIONS,
    NUM_PROBLEMS,
    RAW_DIR,
)

DOWNLOAD_DIR = DATA_DIR / "downloads"


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Stream-download a large file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"Already downloaded: {dest}")
        return dest

    logger.info(f"Downloading {url} -> {dest}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_bytes(chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def extract_tar(tar_path: Path, dest: Path) -> Path:
    """Extract a tar.gz archive."""
    if dest.exists() and any(dest.iterdir()):
        logger.info(f"Already extracted: {dest}")
        return dest

    logger.info(f"Extracting {tar_path} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dest, filter="data")
    return dest


def load_problem_list(metadata_dir: Path) -> dict:
    """
    Parse the CodeNet problem_list.csv to get problem IDs and metadata.
    Returns {problem_id: {name, time_limit, ...}}.
    """
    csv_path = metadata_dir / "Project_CodeNet" / "metadata" / "problem_list.csv"
    if not csv_path.exists():
        for candidate in metadata_dir.rglob("problem_list.csv"):
            csv_path = candidate
            break

    problems = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("id", row.get("problem_id", "")).strip()
            if pid:
                problems[pid] = row
    logger.info(f"Loaded {len(problems)} problems from metadata")
    return problems


def count_accepted_by_language(metadata_dir: Path) -> dict:
    """
    Scan per-problem CSV files to count accepted solutions per (problem, language).
    Returns {problem_id: {language_key: count}}.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    problem_csv_dir = None
    for candidate in metadata_dir.rglob("*.csv"):
        if candidate.parent.name in ("metadata",) or "problem_list" in candidate.name:
            continue
        problem_csv_dir = candidate.parent
        break

    if problem_csv_dir is None:
        for candidate in metadata_dir.rglob("p*"):
            if candidate.is_dir():
                problem_csv_dir = candidate.parent
                break

    if problem_csv_dir is None:
        logger.error("Could not locate per-problem CSV directory in metadata")
        return counts

    csv_files = sorted(problem_csv_dir.glob("p*.csv"))
    logger.info(f"Scanning {len(csv_files)} problem metadata CSV files")

    ext_to_lang = {}
    for lang_key, exts in LANGUAGE_EXTENSIONS.items():
        for ext in exts:
            ext_to_lang[ext] = lang_key

    for csv_file in tqdm(csv_files, desc="Scanning problems"):
        pid = csv_file.stem
        try:
            with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = row.get("status", "").strip()
                    filename = row.get("filename", "").strip()
                    if status.lower() != "accepted":
                        continue
                    ext = Path(filename).suffix.lower()
                    lang = ext_to_lang.get(ext)
                    if lang:
                        counts[pid][lang] += 1
        except Exception as e:
            logger.warning(f"Error reading {csv_file}: {e}")

    return counts


def select_problems(
    counts: dict, num_problems: int, min_accepted: int
) -> list[str]:
    """
    Select problems that have at least min_accepted solutions in ALL three
    target languages. Returns a list of problem IDs sorted by total count.
    """
    eligible = []
    target_langs = set(LANGUAGE_EXTENSIONS.keys())

    for pid, lang_counts in counts.items():
        if all(lang_counts.get(lang, 0) >= min_accepted for lang in target_langs):
            total = sum(lang_counts[lang] for lang in target_langs)
            eligible.append((pid, total))

    eligible.sort(key=lambda x: -x[1])
    selected = [pid for pid, _ in eligible[:num_problems]]
    logger.info(
        f"Selected {len(selected)} problems (from {len(eligible)} eligible) "
        f"with >= {min_accepted} accepted solutions per language"
    )

    if len(selected) < num_problems:
        logger.warning(
            f"Only {len(selected)} problems meet the criteria. "
            f"Relaxing: selecting problems with >= {min_accepted} in ANY target language."
        )
        fallback = []
        for pid, lang_counts in counts.items():
            if pid in selected:
                continue
            if any(lang_counts.get(lang, 0) >= min_accepted for lang in target_langs):
                total = sum(lang_counts.get(lang, 0) for lang in target_langs)
                fallback.append((pid, total))
        fallback.sort(key=lambda x: -x[1])
        for pid, _ in fallback:
            if len(selected) >= num_problems:
                break
            selected.append(pid)

    return selected


def copy_solutions(
    codenet_dir: Path,
    selected_problems: list[str],
    metadata_dir: Path,
    max_per_language: int = 20,
) -> int:
    """
    Copy accepted solutions for selected problems into data/raw/.
    Returns total number of files copied.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total_copied = 0

    ext_to_lang = {}
    for lang_key, exts in LANGUAGE_EXTENSIONS.items():
        for ext in exts:
            ext_to_lang[ext] = lang_key

    data_root = None
    for candidate in codenet_dir.rglob("data"):
        if candidate.is_dir():
            data_root = candidate
            break
    if data_root is None:
        data_root = codenet_dir

    for pid in tqdm(selected_problems, desc="Copying solutions"):
        problem_out = RAW_DIR / pid
        problem_out.mkdir(parents=True, exist_ok=True)

        per_lang_count: dict[str, int] = defaultdict(int)

        for lang_key, exts in LANGUAGE_EXTENSIONS.items():
            lang_dir = data_root / pid / lang_key
            if not lang_dir.exists():
                for alt in [lang_key.capitalize(), lang_key.upper()]:
                    alt_dir = data_root / pid / alt
                    if alt_dir.exists():
                        lang_dir = alt_dir
                        break

            if not lang_dir.exists():
                continue

            for src_file in sorted(lang_dir.iterdir()):
                if per_lang_count[lang_key] >= max_per_language:
                    break
                if src_file.suffix.lower() in exts:
                    dst = problem_out / f"{src_file.stem}{src_file.suffix}"
                    if not dst.exists():
                        shutil.copy2(src_file, dst)
                    per_lang_count[lang_key] += 1
                    total_copied += 1

    return total_copied


def main():
    logger.info("=== CodeNet Dataset Download and Filter ===")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download metadata
    meta_tar = download_file(CODENET_METADATA_URL, DOWNLOAD_DIR / "Project_CodeNet_metadata.tar.gz")
    meta_dir = DOWNLOAD_DIR / "metadata"
    extract_tar(meta_tar, meta_dir)

    # Step 2: Count accepted solutions per problem per language
    counts = count_accepted_by_language(meta_dir)
    logger.info(f"Found {len(counts)} problems with accepted solutions")

    # Step 3: Select top problems
    selected = select_problems(counts, NUM_PROBLEMS, MIN_ACCEPTED_SOLUTIONS)
    if not selected:
        logger.error("No problems selected. Check metadata paths and criteria.")
        return

    selected_path = DATA_DIR / "selected_problems.txt"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text("\n".join(selected))
    logger.info(f"Saved selected problem IDs to {selected_path}")

    # Step 4: Download full dataset
    codenet_tar = download_file(CODENET_URL, DOWNLOAD_DIR / "Project_CodeNet.tar.gz")
    codenet_dir = DOWNLOAD_DIR / "codenet"
    extract_tar(codenet_tar, codenet_dir)

    # Step 5: Copy filtered solutions
    total = copy_solutions(codenet_dir, selected, meta_dir)
    logger.info(f"Copied {total} solution files to {RAW_DIR}")
    logger.info("Done. Run scripts/generate_ai_solutions.py next.")


if __name__ == "__main__":
    main()
