"""Tests for model scoring and ensemble logic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ensemble.scorer import EnsembleScorer, make_decision, weighted_score
from src.ensemble.calibration import PlattCalibrator, IsotonicCalibrator, evaluate_calibration


def test_weighted_score_basic():
    scores = {"statistical": 0.8, "codebert": 0.6, "behavioral": 0.4}
    weights = {"statistical": 0.3, "codebert": 0.45, "behavioral": 0.25}
    result = weighted_score(scores, weights)
    expected = 0.3 * 0.8 + 0.45 * 0.6 + 0.25 * 0.4
    assert abs(result - expected) < 1e-6


def test_weighted_score_missing_component():
    scores = {"statistical": 0.8}
    weights = {"statistical": 0.3, "codebert": 0.45, "behavioral": 0.25}
    result = weighted_score(scores, weights)
    assert 0.0 <= result <= 1.0


def test_make_decision():
    assert make_decision(0.1) == "accept"
    assert make_decision(0.39) == "accept"
    assert make_decision(0.5) == "review"
    assert make_decision(0.69) == "review"
    assert make_decision(0.8) == "hold"
    assert make_decision(0.95) == "hold"


def test_ensemble_scorer():
    scorer = EnsembleScorer()
    result = scorer.score({"statistical": 0.7, "codebert": 0.8})
    assert "risk_score" in result
    assert "decision" in result
    assert 0.0 <= result["risk_score"] <= 1.0


def test_ensemble_with_llm_judge():
    scorer = EnsembleScorer()
    result = scorer.score(
        {"statistical": 0.5, "codebert": 0.6},
        include_llm_judge=True,
        llm_score=0.9,
    )
    assert result["used_llm_judge"] is True
    assert 0.0 <= result["risk_score"] <= 1.0


def test_platt_calibrator():
    cal = PlattCalibrator()
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    cal.fit(scores, labels)
    calibrated = cal.calibrate(0.5)
    assert 0.0 <= calibrated <= 1.0


def test_isotonic_calibrator():
    cal = IsotonicCalibrator()
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    cal.fit(scores, labels)
    calibrated = cal.calibrate(0.5)
    assert 0.0 <= calibrated <= 1.0


def test_calibration_evaluation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    result = evaluate_calibration(scores, labels, n_bins=5)
    assert "ece" in result
    assert result["ece"] >= 0.0
