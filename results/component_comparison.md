# Component vs ensemble comparison

- Generated: `2026-05-16T22:39:58.013939+00:00`
- Test rows scored (both modalities): **2900**

## Test split — metrics @ 0.50 threshold

| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Stylometric (XGBoost) | 0.9507 | 0.8850 | 0.8684 | 0.8766 | 0.9831 |
| CodeBERT (fine-tuned) | 0.9900 | 0.9542 | 0.9983 | 0.9758 | 0.9996 |

### Ensembles

| Ensemble | Accuracy | Precision | Recall | F1 | ROC-AUC |
|----------|----------|-----------|--------|----|---------|
| Weighted stat + CodeBERT | 0.9655 | 0.9203 | 0.9077 | 0.9139 | 0.9938 |

## Overfitting / generalization checklist

- Train vs validation loss: if train loss keeps dropping while val loss rises, reduce epochs or increase regularization.
- Validation vs test: large drop from val to test suggests distribution shift or peeking at test during development — keep test untouched until final reporting.
- Use early stopping on val F1 (as in CodeBERT Trainer) so the saved checkpoint tracks validation, not training loss.
- Compare stylometric vs neural tracks: tree models often generalize differently than transformers on out-of-domain contests.
- Your Colab logs show train_loss ~1e-3 late vs eval_loss ~0.06–0.07 — expect some gap; key is stable val metrics and honest held-out test numbers.

### Val → test gap (same threshold)

```json
{
  "stylometric_xgb": {
    "accuracy_delta_test_minus_val": 0.0081,
    "f1_delta_test_minus_val": 0.0231,
    "roc_auc_delta_test_minus_val": 0.005
  },
  "codebert_finetuned": {
    "accuracy_delta_test_minus_val": 0.003,
    "f1_delta_test_minus_val": 0.0076,
    "roc_auc_delta_test_minus_val": 0.0002
  }
}
```

## Files

- `results\component_comparison.json` (JSON)
- `results\component_comparison.md` (this summary)
