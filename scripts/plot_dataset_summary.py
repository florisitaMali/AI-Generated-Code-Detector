"""
Build a multi-panel PNG summarising the labelled dataset (full merge or splits).

Output: docs/figures/dataset_overview.png

Run after:
  python scripts/build_dataset.py

If no parquet is found, falls back to illustrative counts matching a typical build
(so the thesis graphic path still resolves — re-run locally for real proportions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "figures"
OUT_PATH = OUT_DIR / "dataset_overview.png"

# Fallback approximate split rows (matches one logged CodeBERT training run when
# full parquet was unavailable in CI); labels ~aligned with test-set imbalance ~80/20.
_FALLBACK_SPLITS = {
    "train": {"n": 13334, "human_share": 0.82},
    "val": {"n": 2838, "human_share": 0.81},
    "test": {"n": 2900, "human_share": 0.798},
}


def _load_frames() -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame] | None]:
    full_path = ROOT / "data" / "processed" / "dataset_full.parquet"
    split_dir = ROOT / "data" / "splits"
    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        p = split_dir / f"{name}.parquet"
        if p.exists():
            splits[name] = pd.read_parquet(p)
    if len(splits) == 3:
        return (pd.read_parquet(full_path) if full_path.exists() else None), splits
    if full_path.exists():
        return pd.read_parquet(full_path), None
    return None, None


def _plot(df: pd.DataFrame | None, splits: dict[str, pd.DataFrame] | None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        "Dataset overview — labelled human vs AI submissions",
        fontsize=14,
        fontweight="bold",
    )

    # ── Panel A: split sizes ───────────────────────────────────────────
    ax0 = axes[0, 0]
    if splits and len(splits) == 3:
        order = ["train", "val", "test"]
        sizes = [len(splits[k]) for k in order]
        colors = ["#1f4e79", "#2e75b6", "#9dc3e6"]
        x = np.arange(len(order))
        ax0.bar(x, sizes, color=colors, edgecolor="#333333", linewidth=0.6)
        ax0.set_xticks(x)
        ax0.set_xticklabels(order)
        for i, v in enumerate(sizes):
            ax0.text(i, v + max(sizes) * 0.01, str(v), ha="center", fontsize=10)
        ax0.set_title("Samples per split")
    elif df is not None:
        ax0.bar([0], [len(df)], color=["#1f4e79"], edgecolor="#333333", width=0.5)
        ax0.set_xticks([0])
        ax0.set_xticklabels(["full merge"])
        ax0.text(0, len(df) * 0.5, str(len(df)), ha="center", color="white", fontsize=11, fontweight="bold")
        ax0.set_title("Merged corpus (splits not found)")
    else:
        order = list(_FALLBACK_SPLITS.keys())
        sizes = [_FALLBACK_SPLITS[k]["n"] for k in order]
        x = np.arange(len(order))
        ax0.bar(x, sizes, color=["#c55a11", "#e69138", "#f4b183"], edgecolor="#333")
        ax0.set_xticks(x)
        ax0.set_xticklabels(order)
        ax0.set_title("Samples per split (illustrative)")
    ax0.set_ylabel("Number of samples")

    # ── Panel B: label balance (global or train merge) ─────────────────
    ax1 = axes[0, 1]
    if df is not None:
        vc = df["label"].value_counts().reindex([0, 1]).dropna().astype(int)
        labels = vc.index.map({0: "Human", 1: "AI"})
        colors_p = ["#2e7d32" if lbl == "Human" else "#c62828" for lbl in labels]
        ax1.pie(
            vc.values,
            labels=labels.tolist(),
            autopct="%1.1f%%",
            colors=colors_p,
            explode=[0.03 if lbl == "AI" else 0 for lbl in labels],
            startangle=90,
        )
        ax1.set_title("Class balance (full dataset)")
    elif splits and len(splits) == 3:
        cat = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
        vc = cat["label"].value_counts().reindex([0, 1]).dropna().astype(int)
        labels = vc.index.map({0: "Human", 1: "AI"})
        colors_p = ["#2e7d32" if lbl == "Human" else "#c62828" for lbl in labels]
        ax1.pie(
            vc.values,
            labels=labels.tolist(),
            autopct="%1.1f%%",
            colors=colors_p,
            explode=[0.03 if lbl == "AI" else 0 for lbl in labels],
            startangle=90,
        )
        ax1.set_title("Class balance (train+val+test)")
    else:
        tot_h = sum(d["n"] * d["human_share"] for d in _FALLBACK_SPLITS.values())
        tot_ai = sum(d["n"] * (1 - d["human_share"]) for d in _FALLBACK_SPLITS.values())
        ax1.pie(
            [tot_h, tot_ai],
            labels=["Human", "AI"],
            autopct="%1.1f%%",
            colors=["#2e7d32", "#c62828"],
            startangle=90,
        )
        ax1.set_title("Class balance (illustrative)")

    # ── Panel C: language mix ─────────────────────────────────────────
    ax2 = axes[1, 0]
    src = (
        df
        if df is not None
        else (
            pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
            if splits and len(splits) == 3
            else None
        )
    )
    if src is not None and "language" in src.columns:
        lc = src["language"].value_counts().reindex(["cpp", "python", "java"]).fillna(0)
        x = np.arange(len(lc))
        ax2.bar(x, lc.values, color=["#5c6bc0", "#26a69a", "#ffb74d"], edgecolor="#333")
        ax2.set_xticks(x)
        ax2.set_xticklabels([k.upper() for k in lc.index])
        ax2.set_ylabel("Samples")
        ax2.set_title("Samples per language")
    else:
        ax2.text(0.5, 0.5, "Language column unavailable", ha="center", va="center")
        ax2.set_axis_off()

    # ── Panel D: human vs AI per split (stacked) ──────────────────────
    ax3 = axes[1, 1]
    if splits and len(splits) == 3:
        order = ["train", "val", "test"]
        human = [int((splits[k]["label"] == 0).sum()) for k in order]
        ai = [int((splits[k]["label"] == 1).sum()) for k in order]
        ax3.bar(order, human, label="Human (0)", color="#2e7d32")
        ax3.bar(order, ai, bottom=human, label="AI (1)", color="#c62828")
        ax3.legend()
        ax3.set_ylabel("Samples")
        ax3.set_title("Label mix by split")
    elif df is not None:
        ax3.text(0.5, 0.5, "Provide splits for per-split mix", ha="center", va="center")
        ax3.set_axis_off()
    else:
        order = list(_FALLBACK_SPLITS.keys())
        human = [int(round(_FALLBACK_SPLITS[k]["n"] * _FALLBACK_SPLITS[k]["human_share"])) for k in order]
        ai = [_FALLBACK_SPLITS[k]["n"] - human[i] for i, k in enumerate(order)]
        ax3.bar(order, human, label="Human (0)", color="#388e3c")
        ax3.bar(order, ai, bottom=human, label="AI (1)", color="#e53935")
        ax3.legend()
        ax3.set_title("Label mix by split (illustrative)")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Wrote {OUT_PATH}")


def main() -> None:
    df, splits = _load_frames()
    if df is None and splits is None:
        print(
            "No parquet found — writing illustrative dataset_overview.png "
            "(re-run after: python scripts/build_dataset.py)."
        )
    _plot(df, splits)


if __name__ == "__main__":
    main()
