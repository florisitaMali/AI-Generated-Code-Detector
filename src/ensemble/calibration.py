"""
Probability calibration for the ensemble risk scorer.

Ensures that the output score is a well-calibrated probability (e.g., if
the model outputs 0.8, then ~80% of such submissions are truly AI-generated).
Supports Platt scaling and isotonic regression.
"""

import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_DIR


class PlattCalibrator:
    """Platt scaling: fit a logistic regression on raw scores vs. labels."""

    def __init__(self):
        self.model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10000)
        self._fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        X = scores.reshape(-1, 1)
        self.model.fit(X, labels)
        self._fitted = True
        logger.info("Platt calibrator fitted")

    def calibrate(self, score: float) -> float:
        if not self._fitted:
            return score
        X = np.array([[score]])
        return float(self.model.predict_proba(X)[0, 1])

    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return scores
        X = scores.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


class IsotonicCalibrator:
    """Isotonic regression: non-parametric monotonic calibration."""

    def __init__(self):
        self.model = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self._fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        self.model.fit(scores, labels)
        self._fitted = True
        logger.info("Isotonic calibrator fitted")

    def calibrate(self, score: float) -> float:
        if not self._fitted:
            return score
        return float(self.model.predict(np.array([score]))[0])

    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return scores
        return self.model.predict(scores)


def create_calibrator(method: str = "platt") -> PlattCalibrator | IsotonicCalibrator:
    """Factory function for calibrators."""
    if method == "isotonic":
        return IsotonicCalibrator()
    return PlattCalibrator()


def fit_and_save(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str = "platt",
    path: Path | None = None,
) -> PlattCalibrator | IsotonicCalibrator:
    """Fit a calibrator and save to disk."""
    calibrator = create_calibrator(method)
    calibrator.fit(scores, labels)

    if path is None:
        path = MODELS_DIR / f"calibrator_{method}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)
    logger.info(f"Calibrator saved to {path}")
    return calibrator


def load_calibrator(
    method: str = "platt", path: Path | None = None
) -> PlattCalibrator | IsotonicCalibrator:
    """Load a saved calibrator."""
    if path is None:
        path = MODELS_DIR / f"calibrator_{method}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_calibration(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict:
    """
    Compute calibration metrics: expected calibration error (ECE) and
    per-bin accuracy vs. confidence.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = scores[mask].mean()
        bin_accs.append(float(bin_acc))
        bin_confs.append(float(bin_conf))
        bin_counts.append(int(mask.sum()))

    total = sum(bin_counts)
    ece = sum(
        count / total * abs(acc - conf)
        for acc, conf, count in zip(bin_accs, bin_confs, bin_counts)
    ) if total > 0 else 0.0

    return {
        "ece": round(ece, 4),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }
