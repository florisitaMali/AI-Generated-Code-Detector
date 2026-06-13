"""
Cross-dataset model comparison.

Trains ALL 6 detection models on EACH available dataset, then evaluates each
on its own test split. Results are collected into a single JSON + Markdown table.

Datasets expected (build each with its corresponding script first):
    data/splits/            ← CodeNet + Gemini          (scripts/build_dataset.py)
    data/splits_llmgen/     ← dejanseo/llm-generated-code  (scripts/build_dataset_llmgen.py)
    data/splits_csn/        ← CodeSearchNet + Gemini        (scripts/build_dataset_csn.py)
    data/splits_humaneval/  ← HumanEval-X + Gemini          (scripts/build_dataset_humaneval.py)

Models trained on EACH dataset:
    xgboost, random_forest, logistic_regression   (classical ML — fast, CPU)
    codebert, graphcodebert, unixcoder            (transformer — slow, GPU recommended)

Usage:
    # Train and evaluate ALL models on ALL datasets:
    python scripts/cross_dataset_comparison.py

    # Classical models only (fast, CPU):
    python scripts/cross_dataset_comparison.py --models classical

    # Transformer models only (needs GPU):
    python scripts/cross_dataset_comparison.py --models transformer

    # One dataset only:
    python scripts/cross_dataset_comparison.py --dataset llmgen

    # Skip training, only evaluate existing checkpoints:
    python scripts/cross_dataset_comparison.py --eval-only

Options:
    --models      classical | transformer | all  (default: all)
    --eval-only   skip training, only run test inference on existing checkpoints
    --dataset     codenet | llmgen | csn | humaneval | all  (default: all)
    --output      path to results JSON  (default: results/cross_dataset_results.json)
    --no-optuna   reduce Optuna trials to 5 for classical models (faster)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR, MODELS_DIR

# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "codenet": {
        "label":               "CodeNet + Gemini (original)",
        "splits":              DATA_DIR / "splits",
        "use_general_features": False,   # competitive-programming; base features are sufficient
    },
    "llmgen": {
        "label":               "OSS-forge/HumanVsAICode",
        "splits":              DATA_DIR / "splits_llmgen",
        "use_general_features": True,    # open-source library code; needs extended feature set
    },
    "csn": {
        "label":               "basakdemirok/AIGCodeSet",
        "splits":              DATA_DIR / "splits_csn",
        "use_general_features": True,    # Python library functions; needs extended feature set
    },
}

CLASSICAL_MODELS = ["xgboost", "random_forest", "logistic_regression"]
TRANSFORMER_MODELS = ["codebert", "graphcodebert", "unixcoder"]
ALL_MODELS = CLASSICAL_MODELS + TRANSFORMER_MODELS

TRANSFORMER_CONFIG = {
    "codebert":      ("microsoft/codebert-base",        "codebert"),
    "graphcodebert": ("microsoft/graphcodebert-base",   "graphcodebert"),
    "unixcoder":     ("microsoft/unixcoder-base",       "unixcoder"),
}


# ── Classical ML helpers ──────────────────────────────────────────────────────

def _eval_classical_model(model_key: str, splits_dir: Path, models_dir: Path,
                           use_general_features: bool = False) -> dict | None:
    """
    Load a trained classical model from models_dir and evaluate on the test split.
    Returns a metrics dict or None if no model file found.

    ``use_general_features`` is read from the saved meta when available (so
    eval-only mode automatically uses the same feature set used at training).
    """
    import pickle

    pkl_name = {
        "xgboost":             "xgb_baseline.pkl",
        "random_forest":       "rf_baseline.pkl",
        "logistic_regression": "lr_baseline.pkl",
    }[model_key]
    meta_name = pkl_name.replace(".pkl", "_meta.json")

    model_path = models_dir / pkl_name
    meta_path  = models_dir / meta_name

    if not model_path.exists() or not meta_path.exists():
        return None

    test_df = pd.read_parquet(splits_dir / "test.parquet")
    meta    = json.loads(meta_path.read_text())
    feature_cols = meta["feature_columns"]
    # Prefer the flag recorded at training time; fall back to caller-supplied value
    use_gen = meta.get("use_general_features", use_general_features)

    if model_key == "random_forest":
        from src.models.rf_baseline import _build_rf_feature_matrix
        test_feat = _build_rf_feature_matrix(test_df, use_general_features=use_gen)
    else:
        from src.models.statistical_baseline import build_feature_matrix
        test_feat = build_feature_matrix(test_df, use_general_features=use_gen)

    X_test = test_feat[feature_cols].fillna(0).values
    y_test = test_feat["label"].values

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1":       float(f1_score(y_test, preds)),
        "auc":      float(roc_auc_score(y_test, probs)),
    }


def _train_and_eval_classical(model_key: str, splits_dir: Path, models_dir: Path,
                               n_trials: int = 30,
                               use_general_features: bool = False) -> dict:
    logger.info(f"Training {model_key} on {splits_dir.name} "
                f"(general_features={use_general_features}) …")
    if model_key == "xgboost":
        from src.models.statistical_baseline import train_and_evaluate
        _, _, meta = train_and_evaluate(n_trials=n_trials, splits_dir=splits_dir,
                                        models_dir=models_dir,
                                        use_general_features=use_general_features)
    elif model_key == "random_forest":
        from src.models.rf_baseline import train_and_evaluate
        _, _, meta = train_and_evaluate(n_trials=n_trials, splits_dir=splits_dir,
                                        models_dir=models_dir,
                                        use_general_features=use_general_features)
    else:
        from src.models.lr_baseline import train_and_evaluate
        _, _, meta = train_and_evaluate(n_trials=n_trials, splits_dir=splits_dir,
                                        models_dir=models_dir,
                                        use_general_features=use_general_features)

    return {
        "accuracy": meta["test_accuracy"],
        "f1":       meta["test_f1"],
        "auc":      meta["test_auc"],
    }


# ── Transformer helpers ───────────────────────────────────────────────────────

def _eval_transformer(model_key: str, splits_dir: Path, models_dir: Path) -> dict | None:
    """
    Evaluate a fine-tuned transformer on the test split of splits_dir.
    Loads weights from models_dir/{output_dir_name}_final.
    """
    import torch
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from transformers import AutoTokenizer
    from src.models.codebert_classifier import (
        CodeBERTClassifier, CodeDataset, CODEBERT_MAX_LENGTH, _prob_from_logit
    )

    hf_name, output_dir_name = TRANSFORMER_CONFIG[model_key]
    model_path = models_dir / f"{output_dir_name}_final"

    if not model_path.exists():
        logger.warning(f"No checkpoint at {model_path} — skipping {model_key}")
        return None

    test_df = pd.read_parquet(splits_dir / "test.parquet")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = CodeBERTClassifier(hf_name)

    safe_path = model_path / "model.safetensors"
    bin_path  = model_path / "pytorch_model.bin"
    if safe_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(safe_path))
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        logger.warning(f"No weights in {model_path}")
        return None

    model.load_state_dict(state)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    probs, labels = [], []
    batch_size = 32

    for start in range(0, len(test_df), batch_size):
        batch = test_df.iloc[start:start + batch_size]
        encodings = tokenizer(
            batch["code"].tolist(),
            truncation=True, max_length=CODEBERT_MAX_LENGTH,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out["logits"].cpu().numpy()

        batch_probs = [_prob_from_logit(float(l)) for l in logits]
        probs.extend(batch_probs)
        labels.extend(batch["label"].tolist())

    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1":       float(f1_score(labels, preds)),
        "auc":      float(roc_auc_score(labels, probs)),
    }


def _train_and_eval_transformer(model_key: str, splits_dir: Path, models_dir: Path) -> dict | None:
    hf_name, output_dir_name = TRANSFORMER_CONFIG[model_key]
    logger.info(f"Fine-tuning {model_key} ({hf_name}) on {splits_dir.name} …")
    from src.models.codebert_classifier import train
    _, test_results = train(
        model_name=hf_name,
        output_dir_name=output_dir_name,
        splits_dir=splits_dir,
        models_dir=models_dir,
    )
    return {
        "accuracy": test_results.get("eval_accuracy", 0.0),
        "f1":       test_results.get("eval_f1", 0.0),
        "auc":      test_results.get("eval_auc", 0.0),
    }


# ── Markdown report ───────────────────────────────────────────────────────────

def _write_markdown(results: dict, output_path: Path) -> None:
    lines = [
        "# Cross-Dataset Model Comparison",
        "",
        "All scores are on the **test split** of each dataset.  ",
        "OOD = out-of-distribution (model trained on a different dataset).",
        "",
    ]

    dataset_labels = {k: v["label"] for k, v in DATASETS.items() if k in results}

    # One table per model
    for model_key in ALL_MODELS:
        lines.append(f"## {model_key.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Dataset | Accuracy | F1 | AUC |")
        lines.append("|---------|:--------:|:--:|:---:|")
        for ds_key, ds_label in dataset_labels.items():
            metrics = results.get(ds_key, {}).get(model_key)
            if metrics is None:
                row = f"| {ds_label} | — | — | — |"
            else:
                row = (
                    f"| {ds_label} "
                    f"| {metrics['accuracy']*100:.2f} % "
                    f"| {metrics['f1']*100:.2f} % "
                    f"| {metrics['auc']*100:.2f} % |"
                )
            lines.append(row)
        lines.append("")

    # Summary table: best model per dataset
    lines.append("## Summary — Best Model per Dataset")
    lines.append("")
    lines.append("| Dataset | Best Model | Accuracy | F1 | AUC |")
    lines.append("|---------|-----------|:--------:|:--:|:---:|")
    for ds_key, ds_label in dataset_labels.items():
        ds_results = results.get(ds_key, {})
        best_model, best_auc, best_metrics = None, -1.0, {}
        for m, metrics in ds_results.items():
            if metrics and metrics.get("auc", 0) > best_auc:
                best_auc = metrics["auc"]
                best_model = m
                best_metrics = metrics
        if best_model:
            lines.append(
                f"| {ds_label} | **{best_model}** "
                f"| {best_metrics['accuracy']*100:.2f} % "
                f"| {best_metrics['f1']*100:.2f} % "
                f"| {best_metrics['auc']*100:.2f} % |"
            )
        else:
            lines.append(f"| {ds_label} | — | — | — | — |")
    lines.append("")

    md_path = output_path.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Markdown report written to {md_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-dataset model comparison")
    parser.add_argument("--models",    choices=["classical", "transformer", "all"], default="all")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; only evaluate existing checkpoints")
    parser.add_argument("--dataset",   default="all",
                        help="Which dataset(s) to use: codenet|llmgen|csn|humaneval|all")
    parser.add_argument("--output",    default="results/cross_dataset_results.json")
    parser.add_argument("--no-optuna", action="store_true",
                        help="Reduce Optuna trials to 5 for classical models (faster)")
    args = parser.parse_args()

    model_keys = (
        CLASSICAL_MODELS if args.models == "classical"
        else TRANSFORMER_MODELS if args.models == "transformer"
        else ALL_MODELS
    )

    ds_keys = (
        list(DATASETS.keys()) if args.dataset == "all"
        else [args.dataset]
    )

    n_trials = 5 if args.no_optuna else 30

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results so we can resume partial runs
    results: dict = {}
    if output_path.exists():
        results = json.loads(output_path.read_text())
        logger.info(f"Resuming from existing results: {output_path}")

    for ds_key in ds_keys:
        ds_config = DATASETS[ds_key]
        splits_dir: Path = ds_config["splits"]
        use_gen: bool = ds_config.get("use_general_features", False)

        if not (splits_dir / "test.parquet").exists():
            logger.warning(f"No test.parquet found for '{ds_key}' at {splits_dir} — skipping")
            logger.warning(f"  Run the corresponding build script first:")
            logger.warning(f"    python scripts/build_dataset_{ds_key}.py")
            continue

        has_train = (splits_dir / "train.parquet").exists()

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset          : {ds_config['label']}")
        logger.info(f"Splits           : {splits_dir}")
        logger.info(f"General features : {use_gen}")
        logger.info(f"{'='*60}")

        if ds_key not in results:
            results[ds_key] = {}

        # Per-dataset models directory keeps artifacts separate across datasets
        ds_models_dir = MODELS_DIR / ds_key
        ds_models_dir.mkdir(parents=True, exist_ok=True)

        for model_key in model_keys:
            if model_key in results[ds_key] and results[ds_key][model_key] is not None:
                logger.info(f"  {model_key}: already in results — skipping")
                continue

            logger.info(f"\n  ── {model_key} ──")
            metrics = None
            is_classical = model_key in CLASSICAL_MODELS

            if args.eval_only:
                # Load existing checkpoint from this dataset's models dir
                if is_classical:
                    metrics = _eval_classical_model(model_key, splits_dir, ds_models_dir,
                                                    use_general_features=use_gen)
                else:
                    metrics = _eval_transformer(model_key, splits_dir, ds_models_dir)
                if metrics:
                    logger.info(f"    eval — AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}")
                else:
                    logger.warning(f"    No checkpoint found in {ds_models_dir}")
            elif not has_train:
                logger.warning(
                    f"    No train.parquet in {splits_dir} — run build script first, skipping"
                )
            else:
                try:
                    if is_classical:
                        metrics = _train_and_eval_classical(
                            model_key, splits_dir, ds_models_dir, n_trials,
                            use_general_features=use_gen,
                        )
                    else:
                        metrics = _train_and_eval_transformer(model_key, splits_dir, ds_models_dir)
                    if metrics:
                        logger.info(
                            f"    AUC={metrics['auc']:.4f}  "
                            f"F1={metrics['f1']:.4f}  "
                            f"Acc={metrics['accuracy']:.4f}"
                        )
                except Exception as e:
                    logger.error(f"    Failed: {e}")

            results[ds_key][model_key] = metrics

            # Save incrementally — Colab disconnects won't lose progress
            output_path.write_text(json.dumps(results, indent=2))

    output_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\nResults saved to {output_path}")

    _write_markdown(results, output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
