"""
Evaluation metrics for AI-generated code detection.

Computes precision, recall, F1-score, AUC-ROC, and confusion matrices
with per-language and per-problem breakdowns.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import MODELS_DIR


def compute_all_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """
    Compute all evaluation metrics for binary classification.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_prob: Predicted probabilities for class 1.
        threshold: Decision threshold.

    Returns:
        Dict with all metric values.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }

    if len(set(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        metrics["auc_pr"] = float(auc(rec_curve, prec_curve))
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def per_language_metrics(
    df: pd.DataFrame, prob_col: str = "prob", threshold: float = 0.5
) -> dict:
    """
    Compute metrics per language.

    Args:
        df: DataFrame with columns: label, language, and prob_col.

    Returns:
        {language: metrics_dict}
    """
    results = {}
    for lang, group in df.groupby("language"):
        y_true = group["label"].values
        y_prob = group[prob_col].values
        results[lang] = compute_all_metrics(y_true, y_prob, threshold)
        results[lang]["n_samples"] = len(group)
    return results


def per_problem_metrics(
    df: pd.DataFrame, prob_col: str = "prob", threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute metrics per problem.

    Returns a DataFrame indexed by problem_id with F1, precision, recall per problem.
    """
    records = []
    for pid, group in df.groupby("problem_id"):
        y_true = group["label"].values
        y_prob = group[prob_col].values
        y_pred = (y_prob >= threshold).astype(int)

        records.append({
            "problem_id": pid,
            "n_samples": len(group),
            "n_human": int((y_true == 0).sum()),
            "n_ai": int((y_true == 1).sum()),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        })

    return pd.DataFrame(records).sort_values("f1", ascending=True)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot ROC curve and return the figure."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    labels = ["Human", "AI"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        ylabel="True Label", xlabel="Predicted Label",
        title=f"Confusion Matrix — {model_name}",
    )

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot precision-recall curve."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, label=f"{model_name} (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compare_models(
    y_true: np.ndarray,
    model_probs: dict[str, np.ndarray],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot overlaid ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Model Comparison — ROC Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def print_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Print a formatted evaluation report."""
    metrics = compute_all_metrics(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)

    logger.info("=" * 50)
    logger.info("EVALUATION REPORT")
    logger.info("=" * 50)
    logger.info(f"Samples: {metrics['n_samples']} (pos={metrics['n_positive']}, neg={metrics['n_negative']})")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"AUC-PR:    {metrics['auc_pr']:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['Human', 'AI'])}")
