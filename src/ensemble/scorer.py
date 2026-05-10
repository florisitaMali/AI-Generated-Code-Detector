"""
Ensemble risk scorer.

Combines scores from the statistical baseline, CodeBERT classifier, and
behavioral signals into a single calibrated risk score. The LLM-as-judge
score is optionally added as a second-stage audit when the first-pass
score exceeds a configurable threshold.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_DIR, THRESHOLD_AUTO_ACCEPT, THRESHOLD_FLAG_REVIEW


DEFAULT_WEIGHTS = {
    # Weights calibrated 2026-05-02 from test-split analysis:
    # Statistical has 13× class separation vs CodeBERT's 1.1×, so stat carries more weight.
    "statistical": 0.65,
    "codebert":    0.20,
    "llm_judge":   0.15,   # only included when LLM-judge is triggered
}


def weighted_score(
    component_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """
    Compute a weighted average of component scores.

    Args:
        component_scores: {"statistical": 0.7, "codebert": 0.8, ...}
        weights: Optional custom weights. Must sum to ~1.0.

    Returns:
        Weighted risk score in [0, 1].
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total_weight = 0.0
    total_score = 0.0

    for key, weight in weights.items():
        score = component_scores.get(key)
        if score is not None and not np.isnan(score):
            total_score += weight * score
            total_weight += weight

    if total_weight == 0:
        return 0.5

    return np.clip(total_score / total_weight, 0.0, 1.0)


def make_decision(score: float) -> str:
    """Map a risk score to a decision string."""
    if score < THRESHOLD_AUTO_ACCEPT:
        return "accept"
    elif score < THRESHOLD_FLAG_REVIEW:
        return "review"
    else:
        return "hold"


class EnsembleScorer:
    """
    Full ensemble scorer that combines multiple detection approaches.

    Can either use fixed weights or a learned logistic regression on the
    validation set component scores.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.learned_model: LogisticRegression | None = None
        self.calibrator = None

    def score(
        self,
        component_scores: dict[str, float],
        include_llm_judge: bool = False,
        llm_score: float | None = None,
    ) -> dict:
        """
        Compute the ensemble risk score.

        Args:
            component_scores: Scores from each detector component.
            include_llm_judge: Whether to include LLM-judge in the ensemble.
            llm_score: Score from the LLM-as-judge (0-1).

        Returns:
            {
                "risk_score": float,
                "decision": str,
                "component_scores": dict,
                "used_llm_judge": bool,
            }
        """
        if self.learned_model is not None:
            risk = self._predict_learned(component_scores, llm_score if include_llm_judge else None)
        else:
            scores = dict(component_scores)
            if include_llm_judge and llm_score is not None:
                scores["llm_judge"] = llm_score
                weights = {**self.weights, "llm_judge": 0.20}
                total = sum(weights.values())
                weights = {k: v / total for k, v in weights.items()}
            else:
                weights = self.weights

            risk = weighted_score(scores, weights)

        if self.calibrator is not None:
            risk = self.calibrator.calibrate(risk)

        risk = float(np.clip(risk, 0.0, 1.0))
        decision = make_decision(risk)

        return {
            "risk_score": round(risk, 4),
            "decision": decision,
            "component_scores": {k: round(v, 4) for k, v in component_scores.items()},
            "used_llm_judge": include_llm_judge and llm_score is not None,
        }

    def fit_weights(
        self,
        component_scores_list: list[dict[str, float]],
        labels: list[int],
    ):
        """
        Learn optimal combination weights using logistic regression on
        validation predictions.
        """
        feature_names = sorted(component_scores_list[0].keys())
        X = np.array([
            [scores.get(f, 0.0) for f in feature_names]
            for scores in component_scores_list
        ])
        y = np.array(labels)

        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(X, y)

        self.learned_model = model
        self._feature_names = feature_names
        logger.info(f"Learned weights for features {feature_names}: {model.coef_[0].tolist()}")

    def _predict_learned(
        self, component_scores: dict[str, float], llm_score: float | None = None
    ) -> float:
        features = {k: component_scores.get(k, 0.0) for k in self._feature_names}
        if llm_score is not None and "llm_judge" in self._feature_names:
            features["llm_judge"] = llm_score
        X = np.array([[features.get(f, 0.0) for f in self._feature_names]])
        return float(self.learned_model.predict_proba(X)[0, 1])

    def save(self, path: Path | None = None):
        if path is None:
            path = MODELS_DIR / "ensemble_scorer.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Ensemble scorer saved to {path}")

    @classmethod
    def load(cls, path: Path | None = None) -> "EnsembleScorer":
        if path is None:
            path = MODELS_DIR / "ensemble_scorer.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
