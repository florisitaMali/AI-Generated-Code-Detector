"""Resolve Project_CodeNet root inside this repo (after download/extract)."""

from __future__ import annotations

import os
from pathlib import Path

from config.settings import DATA_DIR


def get_codenet_root() -> Path:
    explicit = os.getenv("CODENET_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    candidates = [
        DATA_DIR / "downloads" / "codenet" / "Project_CodeNet",
        DATA_DIR / "downloads" / "metadata" / "Project_CodeNet",
        DATA_DIR / "Project_CodeNet",
    ]
    for c in candidates:
        pl = c / "metadata" / "problem_list.csv"
        if pl.exists():
            return c
    return candidates[0]
