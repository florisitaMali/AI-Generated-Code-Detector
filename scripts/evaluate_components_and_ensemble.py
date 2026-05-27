"""
Evaluate stylometric (XGBoost), fine-tuned CodeBERT, optional LLM-as-judge,
and fixed-weight ensembles on held-out splits.

Writes (default: project root):
  component_comparison.json
  component_comparison.md

Override output folder with `--output-dir`. Rebuild only the Markdown with
`--markdown-only path/to/component_comparison.json`.

Use `--save-scores-npz path.npz` during evaluation to cache per-sample scores for fast replots.
Responses are cached under `data/llm_judge_cache/` when using LLM.

LLM calls are expensive — default --llm-max-samples 0 (skip).

Usage:
  python scripts/evaluate_components_and_ensemble.py
  python scripts/evaluate_components_and_ensemble.py --max-test-rows 500 --llm-max-samples 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (  # noqa: E402
    LLM_GATE_THRESHOLD,
    MODELS_DIR,
    RAW_DIR,
    SPLITS_DIR,
)
from src.ensemble.scorer import EnsembleScorer, weighted_score  # noqa: E402


def load_human_examples(problem_id: str, language: str, max_examples: int = 3) -> list[str]:
    lang_ext = {"cpp": ".cpp", "python": ".py", "java": ".java", "c": ".c",
                "csharp": ".cs", "javascript": ".js"}
    ext = lang_ext.get(language, "")
    problem_dir = RAW_DIR / str(problem_id)
    if not problem_dir.exists():
        return []
    examples = []
    for f in sorted(problem_dir.iterdir()):
        if f.is_file() and f.suffix == ext:
            try:
                examples.append(f.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
            if len(examples) >= max_examples:
                break
    return examples


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {
        "threshold": threshold,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p))
        if len(np.unique(y_true)) > 1
        else None,
    }
    return out


def score_statistical(code: str, language: str) -> float | None:
    try:
        from src.models.statistical_baseline import predict as stat_predict

        return float(stat_predict(code, language))
    except Exception:
        return None


def score_codebert(code: str) -> float | None:
    try:
        from src.models.codebert_classifier import predict as cb_predict

        return float(cb_predict(code))
    except Exception:
        return None


def score_codebert_with_logit(code: str) -> tuple[float | None, float | None]:
    """Return (logit_or_none, probability). Matches a single fine-tuned forward pass."""
    try:
        from src.models.codebert_classifier import predict_with_details

        d = predict_with_details(code)
        return d.get("logit"), float(d["prob"]) if d.get("prob") is not None else None
    except Exception:
        return None, None


def _limit_split_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Random subset preserving both labels when possible (unlike naive `.head`)."""
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def collect_scores(df: pd.DataFrame, *, label: str, skip_llm: bool) -> dict[str, list]:
    stat, cb, cb_logit, y = [], [], [], []
    llm_raw: list[float | None] = []
    rows_meta: list[tuple] = []

    it = df.itertuples(index=False)
    it = tqdm(it, total=len(df), desc=label)

    for row in it:
        code = getattr(row, "code")
        language = str(getattr(row, "language"))
        lab = int(getattr(row, "label"))

        s = score_statistical(code, language)
        lg, c = score_codebert_with_logit(code)

        if s is None or c is None:
            continue

        stat.append(s)
        cb.append(c)
        cb_logit.append(float(lg) if lg is not None else float("nan"))
        y.append(lab)
        llm_raw.append(None)
        pid = getattr(row, "problem_id", None)
        rows_meta.append((code, str(pid) if pd.notna(pid) else "unknown", language))

    return {
        "stat": stat,
        "cb": cb,
        "cb_logit": cb_logit,
        "y": y,
        "llm_raw": llm_raw,
        "rows_meta": rows_meta,
        "skip_llm": skip_llm,
    }


async def collect_llm_scores(
    rows_meta: list[tuple],
    max_samples: int,
    concurrency: int,
    seed: int,
) -> list[float | None]:
    from src.models.llm_judge import judge

    n = len(rows_meta)
    indices = list(range(n))
    if max_samples > 0 and n > max_samples:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(indices, size=max_samples, replace=False).tolist())

    sem = asyncio.Semaphore(max(1, concurrency))
    out_map: dict[int, float] = {}

    async def one(k: int):
        code, pid, lang = rows_meta[k]
        examples = load_human_examples(pid, lang) if pid != "unknown" else []
        async with sem:
            try:
                r = await judge(code, problem_id=pid, human_examples=examples)
                sc = float(r.get("score", 0.5))
            except Exception:
                sc = 0.5
        out_map[k] = sc

    await asyncio.gather(*[one(k) for k in indices])
    return [out_map.get(i) for i in range(n)]


def ensemble_fixed(stat: float, cb: float, llm: float | None, gate: float) -> float:
    """Match production: LLM included only when max(stat, cb) >= gate and llm is not None."""
    comp: dict[str, float] = {"statistical": stat, "codebert": cb}
    if llm is not None and max(stat, cb) >= gate:
        comp["llm_judge"] = llm
    return float(weighted_score(comp))


def _path_for_report(p: Path, base: Path = ROOT) -> str:
    """Relative to repo when under ROOT; otherwise absolute (avoids relative_to crash)."""
    try:
        return str(p.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(p.resolve())


def load_codebert_training_meta() -> dict | None:
    p = MODELS_DIR / "codebert_final" / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_comparison_markdown(report: dict, json_path: Path, md_path: Path) -> None:
    """Write human-readable summary; paths may be outside repo."""
    thr = float(report.get("settings_snapshot", {}).get("ensemble_threshold_for_metrics", 0.5))

    lines = [
        "# Component vs ensemble comparison",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Test rows scored (both modalities): **{report['methods']['stylometric_xgb']['n']}**",
        "",
        "## Test split — metrics @ {:.2f} threshold".format(thr),
        "",
        "| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|--------|----------|-----------|--------|----|---------|",
    ]

    def row(name: str, m: dict):
        auc = m.get("roc_auc")
        auc_s = f"{auc:.4f}" if auc is not None else "—"
        return (
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {auc_s} |"
        )

    lines.append(row("Stylometric (XGBoost)", report["methods"]["stylometric_xgb"]))
    lines.append(row("CodeBERT (fine-tuned)", report["methods"]["codebert_finetuned"]))
    llm_m = report["methods"].get("llm_judge_subsample")
    if llm_m:
        lines.append(
            row(f"LLM-judge (n={llm_m['n_llm_calls']})", {k: v for k, v in llm_m.items() if k != "n_llm_calls"})
        )
    lines.append("")
    lines.append("### Ensembles")
    lines.append("")
    lines.append("| Ensemble | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    lines.append("|----------|----------|-----------|--------|----|---------|")
    lines.append(row("Weighted stat + CodeBERT", report["ensembles"]["weighted_stat_plus_codebert"]))
    if "weighted_with_llm_where_called" in report["ensembles"]:
        lines.append(
            row(
                "Weighted + LLM (where sampled)",
                report["ensembles"]["weighted_with_llm_where_called"],
            )
        )
    lines.extend(["", "## Overfitting / generalization checklist", ""])
    for bullet in report["generalization_and_overfitting"]["how_to_assess_overfitting"]:
        lines.append(f"- {bullet}")
    if "test_minus_val_gap" in report["generalization_and_overfitting"]:
        lines.extend(["", "### Val → test gap (same threshold)", "", "```json"])
        lines.append(json.dumps(report["generalization_and_overfitting"]["test_minus_val_gap"], indent=2))
        lines.append("```")

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{_path_for_report(json_path)}` (JSON)",
            f"- `{_path_for_report(md_path)}` (this summary)",
            "",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare detectors + ensemble on splits.")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for component_comparison.{json,md} (default: project root)",
    )
    ap.add_argument("--max-test-rows", type=int, default=0, help="0 = full test split")
    ap.add_argument("--max-val-rows", type=int, default=0, help="0 = full val split for gap analysis")
    ap.add_argument("--llm-max-samples", type=int, default=0, help="LLM rows (0=skip LLM entirely)")
    ap.add_argument("--llm-concurrency", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on risk scores")
    ap.add_argument(
        "--markdown-only",
        type=Path,
        metavar="JSON_PATH",
        default=None,
        help="Rebuild component_comparison.md from existing JSON (no model scoring).",
    )
    ap.add_argument(
        "--save-scores-npz",
        type=Path,
        default=None,
        help=(
            "Save test-split arrays for plotting (y_te, statistical, codebert, codebert_logits, ensemble, threshold, …)."
        ),
    )
    args = ap.parse_args()

    if args.markdown_only is not None:
        json_path = args.markdown_only.expanduser().resolve()
        if not json_path.is_file():
            print(f"Missing JSON file: {json_path}")
            sys.exit(1)
        report = json.loads(json_path.read_text(encoding="utf-8"))
        md_path = json_path.with_suffix(".md")
        write_comparison_markdown(report, json_path, md_path)
        print(f"Wrote: {md_path}")
        return

    test_path = SPLITS_DIR / "test.parquet"
    val_path = SPLITS_DIR / "val.parquet"
    if not test_path.exists():
        print(f"Missing {test_path}. Run scripts/build_dataset.py first.")
        sys.exit(1)

    test_full_df = pd.read_parquet(test_path)
    full_test_parquet_n = len(test_full_df)
    test_df = _limit_split_rows(test_full_df, args.max_test_rows, args.seed)

    val_df = None
    if val_path.exists():
        val_full_df = pd.read_parquet(val_path)
        val_df = _limit_split_rows(val_full_df, args.max_val_rows, args.seed)

    scorer = EnsembleScorer()

    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits": {
            "test_path": str(test_path),
            "test_n": len(test_df),
            "test_human": int((test_df.label == 0).sum()),
            "test_ai": int((test_df.label == 1).sum()),
            "val_n": len(val_df) if val_df is not None else None,
        },
        "settings_snapshot": {
            "LLM_GATE_THRESHOLD": LLM_GATE_THRESHOLD,
            "ensemble_threshold_for_metrics": args.threshold,
            "default_weights": scorer.weights,
        },
        "methods": {},
        "ensembles": {},
        "generalization_and_overfitting": {},
        "codebert_checkpoint_meta": load_codebert_training_meta(),
        "colab_training_notes": {
            "checkpoint_download": "model.safetensors ~499MB from fine-tuned CodeBERT run.",
            "trainer_warning": (
                "If resuming, eval_steps/save_steps in TrainingArguments (e.g. 250) may differ "
                "from trainer_state.json in an old checkpoint folder (e.g. 500). "
                "Final saved codebert_final reflects Trainer.save_model after training — "
                "use consistent args when resuming, or start from new output_dir."
            ),
            "observed_during_training_examples": {
                "mid_training_eval_example_epoch_3_6": {
                    "eval_loss": "~0.072",
                    "eval_accuracy": "~0.986",
                    "eval_f1": "~0.965",
                    "eval_auc": "~0.999",
                },
                "final_test_after_5_epochs": {
                    "eval_loss": 0.0597,
                    "eval_accuracy": 0.99,
                    "eval_f1": 0.976,
                    "eval_auc": 0.9996,
                },
                "train_vs_eval_gap": (
                    "Late training train_loss ~1e-3 vs eval_loss ~0.06–0.07 indicates "
                    "some train/eval gap (normal for classification); monitor val F1 "
                    "and use early stopping + held-out test."
                ),
            },
        },
    }

    # ── Test split scores ─────────────────────────────────────────────
    pack = collect_scores(test_df, label="test", skip_llm=args.llm_max_samples <= 0)
    y_te = np.array(pack["y"], dtype=int)
    stat_te = np.array(pack["stat"], dtype=float)
    cb_te = np.array(pack["cb"], dtype=float)

    report["methods"]["stylometric_xgb"] = binary_metrics(y_te, stat_te, args.threshold)
    report["methods"]["codebert_finetuned"] = binary_metrics(y_te, cb_te, args.threshold)

    # Ensemble: statistical + CodeBERT only (production-like when LLM not triggered)
    ens_sc = []
    for s, c in zip(stat_te, cb_te):
        r = scorer.score({"statistical": s, "codebert": c})["risk_score"]
        ens_sc.append(r)
    report["ensembles"]["weighted_stat_plus_codebert"] = binary_metrics(
        y_te, np.array(ens_sc), args.threshold
    )

    if args.save_scores_npz is not None:
        args.save_scores_npz.parent.mkdir(parents=True, exist_ok=True)
        cb_logits_te = np.asarray(pack["cb_logit"], dtype=np.float64)
        np.savez_compressed(
            args.save_scores_npz,
            y_te=y_te,
            statistical=stat_te,
            codebert=cb_te,
            codebert_logits=cb_logits_te,
            ensemble=np.asarray(ens_sc, dtype=np.float64),
            threshold=np.float64(args.threshold),
            full_test_parquet_n=np.int64(full_test_parquet_n),
        )
        print(f"Saved score cache: {args.save_scores_npz.resolve()}")

    # Optional LLM + gated full ensemble on same rows (subsampled LLM calls)
    llm_scores: list[float | None] = [None] * len(stat_te)
    if args.llm_max_samples > 0:
        rows_meta = pack["rows_meta"]
        print(f"\nLLM-as-judge on up to {args.llm_max_samples} rows (cached after first run)...")
        idx_subset = list(range(len(rows_meta)))
        if len(idx_subset) > args.llm_max_samples:
            rng = np.random.default_rng(args.seed)
            idx_subset = sorted(rng.choice(idx_subset, size=args.llm_max_samples, replace=False).tolist())

        async def run_llm():
            sem = asyncio.Semaphore(max(1, args.llm_concurrency))
            from src.models.llm_judge import judge

            mapping: dict[int, float] = {}

            async def one(global_idx: int):
                code, pid, lang = rows_meta[global_idx]
                ex = load_human_examples(pid, lang) if pid != "unknown" else []
                async with sem:
                    try:
                        r = await judge(code, problem_id=pid, human_examples=ex)
                        mapping[global_idx] = float(r.get("score", 0.5))
                    except Exception:
                        mapping[global_idx] = 0.5

            await asyncio.gather(*[one(i) for i in idx_subset])
            return mapping

        m = asyncio.run(run_llm())
        for i, sc in m.items():
            llm_scores[i] = sc

        mask_llm = np.array([x is not None for x in llm_scores])
        if mask_llm.any():
            y_sub = y_te[mask_llm]
            llm_arr = np.array([llm_scores[i] for i in range(len(llm_scores)) if llm_scores[i] is not None])
            report["methods"]["llm_judge_subsample"] = {
                "n_llm_calls": int(mask_llm.sum()),
                **binary_metrics(y_sub, llm_arr, args.threshold),
            }

        ens_full = []
        for i in range(len(stat_te)):
            s, c = stat_te[i], cb_te[i]
            ll = llm_scores[i]
            if ll is not None:
                risk = ensemble_fixed(s, c, ll, LLM_GATE_THRESHOLD)
            else:
                risk = float(scorer.score({"statistical": s, "codebert": c})["risk_score"])
            ens_full.append(risk)
        report["ensembles"]["weighted_with_llm_where_called"] = binary_metrics(
            y_te, np.array(ens_full), args.threshold
        )
        note = (
            "Rows without an LLM score use the stat+codebert ensemble only; "
            "rows with LLM use gated inclusion like the API when max(stat,cb)>=gate."
        )
        report["ensembles"]["weighted_with_llm_where_called"]["note"] = note
    else:
        report["methods"]["llm_judge_subsample"] = None

    # ── Val split (generalization gap vs test) ─────────────────────────
    if val_df is not None:
        pv = collect_scores(val_df, label="val", skip_llm=True)
        y_va = np.array(pv["y"], dtype=int)
        stat_va = np.array(pv["stat"], dtype=float)
        cb_va = np.array(pv["cb"], dtype=float)
        report["generalization_and_overfitting"]["val_metrics_threshold_0p5"] = {
            "stylometric_xgb": binary_metrics(y_va, stat_va, args.threshold),
            "codebert_finetuned": binary_metrics(y_va, cb_va, args.threshold),
        }
        ens_va = []
        for s, c in zip(stat_va, cb_va):
            ens_va.append(scorer.score({"statistical": s, "codebert": c})["risk_score"])
        report["generalization_and_overfitting"]["val_weighted_stat_plus_codebert"] = binary_metrics(
            y_va, np.array(ens_va), args.threshold
        )

        def gap(method_key_val: str, method_key_test: str):
            v = report["generalization_and_overfitting"]["val_metrics_threshold_0p5"][method_key_val]
            t = report["methods"][method_key_test]
            return {
                "accuracy_delta_test_minus_val": round(t["accuracy"] - v["accuracy"], 4),
                "f1_delta_test_minus_val": round(t["f1"] - v["f1"], 4),
                "roc_auc_delta_test_minus_val": None
                if v["roc_auc"] is None or t["roc_auc"] is None
                else round(t["roc_auc"] - v["roc_auc"], 4),
            }

        report["generalization_and_overfitting"]["test_minus_val_gap"] = {
            "stylometric_xgb": gap("stylometric_xgb", "stylometric_xgb"),
            "codebert_finetuned": gap("codebert_finetuned", "codebert_finetuned"),
        }

    report["generalization_and_overfitting"]["how_to_assess_overfitting"] = [
        "Train vs validation loss: if train loss keeps dropping while val loss rises, reduce epochs or increase regularization.",
        "Validation vs test: large drop from val to test suggests distribution shift or peeking at test during development — keep test untouched until final reporting.",
        "Use early stopping on val F1 (as in CodeBERT Trainer) so the saved checkpoint tracks validation, not training loss.",
        "Compare stylometric vs neural tracks: tree models often generalize differently than transformers on out-of-domain contests.",
        "Your Colab logs show train_loss ~1e-3 late vs eval_loss ~0.06–0.07 — expect some gap; key is stable val metrics and honest held-out test numbers.",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "component_comparison.json"
    md_path = args.output_dir / "component_comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    write_comparison_markdown(report, json_path, md_path)

    print(f"\nWrote:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
