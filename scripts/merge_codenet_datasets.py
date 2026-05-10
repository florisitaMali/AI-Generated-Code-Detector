"""
Merge Project_CodeNet (human) with AI_CODENET_ROOT (AI) into MERGED_CODENET_ROOT.

Produces:
  merged_codenet/submissions.csv
  merged_codenet/problem_list.csv
  merged_codenet/split/{train,val,test}.csv

Problem-level split (no leakage): ratios from MERGE_TRAIN_RATIO / MERGE_VAL_RATIO / MERGE_TEST_RATIO.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    AI_CODENET_ROOT,
    MERGED_CODENET_ROOT,
    MERGE_TEST_RATIO,
    MERGE_TRAIN_RATIO,
    MERGE_VAL_RATIO,
)
from src.datasets.codenet_paths import get_codenet_root
from src.datasets.codenet_schema import AI_EXTENSION_FIELDS, MERGED_CSV_FIELDS


def iter_human_rows(metadata_dir: Path):
    for csv_file in sorted(metadata_dir.glob("p*.csv")):
        if csv_file.name == "problem_list.csv":
            continue
        stem = csv_file.stem
        if not stem.startswith("p") or not stem[1:].isdigit():
            continue
        with open(csv_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["source"] = "human"
                for field in AI_EXTENSION_FIELDS:
                    row.setdefault(field, "")
                yield row


def iter_ai_rows(metadata_dir: Path):
    for csv_file in sorted(metadata_dir.glob("p*.csv")):
        if csv_file.name == "problem_list.csv":
            continue
        stem = csv_file.stem
        if not stem.startswith("p") or not stem[1:].isdigit():
            continue
        with open(csv_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["source"] = "ai"
                for field in AI_EXTENSION_FIELDS:
                    row.setdefault(field, "")
                yield row


def get_source_code(row: dict[str, Any], codenet_root: Path, ai_root: Path) -> str:
    if row.get("source") == "human":
        root = codenet_root
    else:
        root = ai_root
    pid = row.get("problem_id", "")
    lang = row.get("language", "")
    sid = row.get("submission_id", "")
    ext = row.get("filename_ext", "")
    path = root / "data" / pid / lang / f"{sid}.{ext}"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def merge_and_split(
    include_code: bool = False,
    accepted_only: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    codenet_root = get_codenet_root()
    ai_root = AI_CODENET_ROOT
    out_root = MERGED_CODENET_ROOT

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "split").mkdir(parents=True, exist_ok=True)

    src_pl = codenet_root / "metadata" / "problem_list.csv"
    dst_pl = out_root / "problem_list.csv"
    if src_pl.exists():
        shutil.copy2(src_pl, dst_pl)

    fields = list(MERGED_CSV_FIELDS)
    if include_code:
        fields.append("code")

    all_rows: list[dict[str, Any]] = []
    meta_h = codenet_root / "metadata"
    meta_ai = ai_root / "metadata"

    if meta_h.exists():
        for row in iter_human_rows(meta_h):
            if accepted_only and row.get("status") != "Accepted":
                continue
            if include_code:
                row["code"] = get_source_code(row, codenet_root, ai_root)
            all_rows.append({f: row.get(f, "") for f in fields})

    if meta_ai.exists():
        for row in iter_ai_rows(meta_ai):
            if accepted_only and row.get("status") not in ("Accepted", "Unknown"):
                continue
            if include_code:
                row["code"] = get_source_code(row, codenet_root, ai_root)
            all_rows.append({f: row.get(f, "") for f in fields})

    merged_csv = out_root / "submissions.csv"
    with open(merged_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    ratio_sum = MERGE_TRAIN_RATIO + MERGE_VAL_RATIO + MERGE_TEST_RATIO
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"MERGE_* ratios must sum to 1.0, got {ratio_sum}")

    random.seed(seed)
    all_pids = sorted({str(r["problem_id"]) for r in all_rows})
    random.shuffle(all_pids)
    n = len(all_pids)
    i_train = int(n * MERGE_TRAIN_RATIO)
    i_val = int(n * (MERGE_TRAIN_RATIO + MERGE_VAL_RATIO))
    train_pids = set(all_pids[:i_train])
    val_pids = set(all_pids[i_train:i_val])
    test_pids = set(all_pids[i_val:])

    splits = [
        ("train", train_pids),
        ("val", val_pids),
        ("test", test_pids),
    ]
    for split_name, pid_set in splits:
        path = out_root / "split" / f"{split_name}.csv"
        rows = [r for r in all_rows if r["problem_id"] in pid_set]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    return all_rows


def push_to_huggingface(repo_id: str, token: str) -> None:
    from datasets import load_dataset
    from huggingface_hub import login

    login(token=token)
    root = MERGED_CODENET_ROOT
    ds = load_dataset(
        "csv",
        data_files={
            "train": str(root / "split" / "train.csv"),
            "validation": str(root / "split" / "val.csv"),
            "test": str(root / "split" / "test.csv"),
        },
    )
    ds.push_to_hub(repo_id)


def main():
    parser = argparse.ArgumentParser(description="Merge CodeNet + AI-CodeNet CSV submissions.")
    parser.add_argument("--include-code", action="store_true", help="Include inline source code column")
    parser.add_argument("--accepted-only", action="store_true", help="Filter by Accepted status")
    parser.add_argument("--push-to-hub", metavar="REPO_ID", help="HuggingFace dataset repo id")
    parser.add_argument("--hf-token", metavar="TOKEN", default="", help="HuggingFace token")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merge_and_split(
        include_code=args.include_code,
        accepted_only=args.accepted_only,
        seed=args.seed,
    )

    if args.push_to_hub:
        tok = args.hf_token or ""
        if not tok:
            raise SystemExit("--hf-token required with --push-to-hub")
        push_to_huggingface(args.push_to_hub, tok)


if __name__ == "__main__":
    main()
