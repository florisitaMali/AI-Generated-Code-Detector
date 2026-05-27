"""
Generate performance figures for the AI code detector:

  • ROC curves (test)
  • Precision–recall curves
  • Calibration (reliability)
  • Confusion matrices @ threshold 0.5
  • Metric bar chart (accuracy / F1 / ROC-AUC)
  • Val vs test comparison (from component_comparison.json)
  • CodeBERT eval loss snapshot (mid-training vs final held-out test)

Scores source (pick one):
  • --scores-npz  Cached arrays from evaluate_components_and_ensemble.py --save-scores-npz
    (includes codebert_logits + full_test_parquet_n when produced by that script).
  • or omit to score test.parquet on the fly (--max-rows limits for smoke tests; ignored if NPZ is used).

Outputs PNGs under docs/figures/ by default.

Usage:
  python scripts/evaluate_components_and_ensemble.py --save-scores-npz results/test_detector_scores.npz
  python scripts/plot_evaluation_performance.py --scores-npz results/test_detector_scores.npz \\
      --comparison-json results/component_comparison.json

  python scripts/plot_evaluation_performance.py --max-rows 200
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ensemble.scorer import EnsembleScorer  # noqa: E402


def _load_eval_helpers():
    path = ROOT / "scripts" / "evaluate_components_and_ensemble.py"
    spec = importlib.util.spec_from_file_location("eval_comp", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_arrays_from_df(mod, df: pd.DataFrame, desc: str) -> tuple[np.ndarray, ...]:
    pack = mod.collect_scores(df, label=desc, skip_llm=True)
    y = np.asarray(pack["y"], dtype=int)
    stat = np.asarray(pack["stat"], dtype=float)
    cb = np.asarray(pack["cb"], dtype=float)
    cb_log = np.asarray(pack["cb_logit"], dtype=float)
    scorer = EnsembleScorer()
    ens = np.array(
        [
            scorer.score({"statistical": float(s), "codebert": float(c)})["risk_score"]
            for s, c in zip(stat, cb)
        ],
        dtype=float,
    )
    return y, stat, cb, ens, cb_log


def load_arrays(
    npz_path: Path | None,
    mod,
    test_df: pd.DataFrame,
    threshold: float,
    parquet_full_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray | None, int]:
    if npz_path is not None and npz_path.is_file():
        z = np.load(npz_path)
        thr = float(z["threshold"]) if "threshold" in z.files else threshold
        cb_log = z["codebert_logits"] if "codebert_logits" in z.files else None
        full_n = int(z["full_test_parquet_n"]) if "full_test_parquet_n" in z.files else parquet_full_n
        return z["y_te"], z["statistical"], z["codebert"], z["ensemble"], thr, cb_log, full_n
    y, stat, cb, ens, cb_log = build_arrays_from_df(mod, test_df, "test")
    return y, stat, cb, ens, threshold, cb_log, parquet_full_n


def _figure_footer(fig, note: str) -> None:
    if not note:
        return
    fig.subplots_adjust(bottom=0.14)
    fig.text(0.5, 0.02, note, ha="center", fontsize=8, color="#444")


def _calibration_bins_strategy(s: np.ndarray) -> tuple[int, str]:
    """Use quantile bins when predictions are nearly discrete."""
    s = np.asarray(s, dtype=float)
    uniq = len(np.unique(np.round(s, decimals=6)))
    if uniq <= 16:
        return min(10, max(3, uniq)), "quantile"
    return 10, "uniform"


def plot_roc(y: np.ndarray, scores: dict[str, np.ndarray], out: Path, *, footer_note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {
        "Stylometric (XGBoost)": "#3498db",
        "CodeBERT": "#e74c3c",
        "Ensemble (stat+cb)": "#9b59b6",
    }
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})", color=colors[name])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — held-out test split")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _figure_footer(fig, footer_note)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_pr(y: np.ndarray, scores: dict[str, np.ndarray], out: Path, *, footer_note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {
        "Stylometric (XGBoost)": "#3498db",
        "CodeBERT": "#e74c3c",
        "Ensemble (stat+cb)": "#9b59b6",
    }
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP = {ap:.4f})", color=colors[name])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curves — held-out test split")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _figure_footer(fig, footer_note)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_calibration(y: np.ndarray, scores: dict[str, np.ndarray], out: Path, *, footer_note: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {
        "Stylometric (XGBoost)": "#3498db",
        "CodeBERT": "#e74c3c",
        "Ensemble (stat+cb)": "#9b59b6",
    }
    for name, s in scores.items():
        n_bins, strat = _calibration_bins_strategy(s)
        prob_true, prob_pred = calibration_curve(y, s, n_bins=n_bins, strategy=strat)
        ax.plot(prob_pred, prob_true, marker="o", lw=1.5, label=f"{name} ({strat})", color=colors[name])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration (adaptive bins) — test split")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _figure_footer(fig, footer_note)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_confusion_matrices(
    y: np.ndarray, scores: dict[str, np.ndarray], thr: float, out: Path, *, footer_note: str = ""
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    names = list(scores.keys())
    for ax, name in zip(axes, names):
        s = scores[name]
        pred = (s >= thr).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Human", "AI"])
        ax.set_yticklabels(["Human", "AI"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}\n(threshold={thr:g})")
        for (j, i), v in np.ndenumerate(cm):
            ax.text(i, j, str(v), ha="center", va="center", color="black", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Confusion matrices — test split", y=1.02)
    fig.tight_layout()
    _figure_footer(fig, footer_note)
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def plot_metric_bars(
    y: np.ndarray, scores: dict[str, np.ndarray], thr: float, out: Path, *, footer_note: str = ""
) -> None:
    methods = list(scores.keys())
    accs, f1s, aucs = [], [], []
    for s in scores.values():
        pred = (s >= thr).astype(int)
        accs.append(accuracy_score(y, pred))
        f1s.append(f1_score(y, pred, zero_division=0))
        aucs.append(roc_auc_score(y, s))

    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, accs, width=w, label="Accuracy")
    ax.bar(x, f1s, width=w, label="F1")
    ax.bar(x + w, aucs, width=w, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.3)
    ax.set_ylabel("Score")
    ax.set_title(f"Test metrics @ threshold {thr:g}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _figure_footer(fig, footer_note)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_score_histograms(scores: dict[str, np.ndarray], out: Path, *, footer_note: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(np.asarray(scores["Stylometric (XGBoost)"], dtype=float), bins=40, color="#3498db", alpha=0.82)
    axes[0].set_title("Stylometric (XGBoost)")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")
    axes[1].hist(np.asarray(scores["CodeBERT"], dtype=float), bins=40, color="#e74c3c", alpha=0.82)
    axes[1].set_title("CodeBERT probability σ(logit/T)")
    axes[1].set_xlabel("Probability")
    axes[1].set_ylabel("Count")
    fig.suptitle("Score distributions — same rows as ROC/PR", y=1.02)
    fig.tight_layout()
    _figure_footer(fig, footer_note)
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def plot_codebert_logits_hist(cb_logits: np.ndarray, out: Path, *, footer_note: str = "") -> None:
    mask = np.isfinite(cb_logits)
    if not np.any(mask):
        return
    vals = np.asarray(cb_logits[mask], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(vals, bins=45, color="#c0392b", alpha=0.85)
    ax.set_title("CodeBERT raw logits (pre-scaling)")
    ax.set_xlabel("Logit")
    ax.set_ylabel("Count")
    fig.tight_layout()
    _figure_footer(fig, footer_note)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _footer_line(n_scored: int, thr: float, parquet_full_n: int, from_npz: bool) -> str:
    parts = [f"n={n_scored} scored rows", f"decision thr={thr:g}"]
    if from_npz:
        parts.append("NPZ cache")
    elif n_scored < parquet_full_n:
        parts.append(f"subset of test.parquet ({parquet_full_n} rows)")
    else:
        parts.append("full test.parquet")
    return " · ".join(parts)


def plot_val_vs_test(json_path: Path, out: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    val_block = data.get("generalization_and_overfitting", {}).get("val_metrics_threshold_0p5", {})
    test_block = data.get("methods", {})
    pairs = [
        ("Stylometric", val_block.get("stylometric_xgb"), test_block.get("stylometric_xgb")),
        ("CodeBERT", val_block.get("codebert_finetuned"), test_block.get("codebert_finetuned")),
    ]
    labels_ok: list[str] = []
    auc_val: list[float] = []
    auc_te: list[float] = []
    f1_val: list[float] = []
    f1_te: list[float] = []
    for lbl, v, t in pairs:
        if v is None or t is None:
            continue
        labels_ok.append(lbl)
        auc_val.append(float(v["roc_auc"]))
        auc_te.append(float(t["roc_auc"]))
        f1_val.append(float(v["f1"]))
        f1_te.append(float(t["f1"]))

    if not labels_ok:
        return

    x = np.arange(len(labels_ok))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - 1.5 * w, auc_val, width=w, label="Val ROC-AUC")
    ax.bar(x - 0.5 * w, auc_te, width=w, label="Test ROC-AUC")
    ax.bar(x + 0.5 * w, f1_val, width=w, label="Val F1")
    ax.bar(x + 1.5 * w, f1_te, width=w, label="Test F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_ok)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Validation vs held-out test (from component_comparison.json)")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_codebert_eval_loss_summary(meta_path: Path, json_path: Path | None, out: Path) -> None:
    """Scalar eval losses from checkpoint meta + optional comparison JSON notes."""
    eval_final = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        eval_final = meta.get("test_results", {}).get("eval_loss")

    mid_eval = 0.072
    if json_path and json_path.is_file():
        rep = json.loads(json_path.read_text(encoding="utf-8"))
        notes = rep.get("colab_training_notes", {}).get("observed_during_training_examples", {})
        mid = notes.get("mid_training_eval_example_epoch_3_6", {}).get("eval_loss")
        if isinstance(mid, (int, float)):
            mid_eval = float(mid)
        elif isinstance(mid, str):
            mid_clean = mid.replace("~", "").strip()
            try:
                mid_eval = float(mid_clean)
            except ValueError:
                pass

    labels = ["Mid-training\n(val snapshot)", "Held-out test\n(checkpoint meta)"]
    values = [mid_eval, eval_final if eval_final is not None else float("nan")]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#f39c12", "#27ae60"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Cross-entropy (eval)")
    ax.set_title("CodeBERT fine-tuning — evaluation loss")
    for b, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.text(
        0.02,
        0.98,
        "Mid ≈ epoch 3–4 checkpoint logs on Colab;\nfinal = Trainer test eval at save.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot detector performance figures.")
    ap.add_argument("--scores-npz", type=Path, default=None, help="Cached scores from evaluate script")
    ap.add_argument("--comparison-json", type=Path, default=None, help="component_comparison.json for val-vs-test")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = full test split when recomputing scores")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_path = ROOT / "data" / "splits" / "test.parquet"
    if not test_path.exists():
        print(f"Missing {test_path}")
        sys.exit(1)

    test_full_df = pd.read_parquet(test_path)
    parquet_full_n = len(test_full_df)

    from_npz = args.scores_npz is not None and args.scores_npz.is_file()
    if from_npz:
        test_df = test_full_df
    elif args.max_rows > 0:
        n = min(args.max_rows, parquet_full_n)
        test_df = test_full_df.sample(n=n, random_state=42).reset_index(drop=True)
    else:
        test_df = test_full_df

    mod = _load_eval_helpers()
    y, stat, cb, ens, thr, cb_logits_arr, ref_full_n = load_arrays(
        args.scores_npz if from_npz else None,
        mod,
        test_df,
        args.threshold,
        parquet_full_n,
    )
    parquet_full_n = max(parquet_full_n, ref_full_n)

    footer = _footer_line(len(y), thr, parquet_full_n, from_npz)

    if len(np.unique(y)) < 2:
        print(
            "Need both classes (human & AI) in the subset for ROC/PR curves. "
            "Increase --max-rows or use full test / --scores-npz from full evaluation."
        )
        sys.exit(1)

    scores = {
        "Stylometric (XGBoost)": stat,
        "CodeBERT": cb,
        "Ensemble (stat+cb)": ens,
    }

    out_dir = args.output_dir
    plot_roc(y, scores, out_dir / "detection_roc_test.png", footer_note=footer)
    plot_pr(y, scores, out_dir / "detection_pr_test.png", footer_note=footer)
    plot_calibration(y, scores, out_dir / "detection_calibration_test.png", footer_note=footer)
    plot_confusion_matrices(y, scores, thr, out_dir / "detection_confusion_test.png", footer_note=footer)
    plot_metric_bars(y, scores, thr, out_dir / "detection_metrics_bars_test.png", footer_note=footer)
    plot_score_histograms(scores, out_dir / "detection_score_hist_test.png", footer_note=footer)
    if cb_logits_arr is not None:
        plot_codebert_logits_hist(cb_logits_arr, out_dir / "detection_codebert_logits_hist.png", footer_note=footer)

    cmp_json = args.comparison_json
    if cmp_json is None:
        for cand in (ROOT / "results" / "component_comparison.json", ROOT / "component_comparison.json"):
            if cand.is_file():
                cmp_json = cand
                break

    if cmp_json is not None and cmp_json.is_file():
        plot_val_vs_test(cmp_json, out_dir / "detection_val_vs_test_metrics.png")
        print(f"Val vs test chart from {cmp_json}")

    meta_path = ROOT / "models" / "codebert_final" / "meta.json"
    plot_codebert_eval_loss_summary(
        meta_path,
        cmp_json if cmp_json and cmp_json.is_file() else None,
        out_dir / "codebert_eval_loss_summary.png",
    )

    print(f"Wrote figures under {out_dir.resolve()}:")
    for name in (
        "detection_roc_test.png",
        "detection_pr_test.png",
        "detection_calibration_test.png",
        "detection_confusion_test.png",
        "detection_metrics_bars_test.png",
        "detection_score_hist_test.png",
        "detection_codebert_logits_hist.png",
        "detection_val_vs_test_metrics.png",
        "codebert_eval_loss_summary.png",
    ):
        p = out_dir / name
        if p.is_file():
            print(f"  {name}")


if __name__ == "__main__":
    main()
