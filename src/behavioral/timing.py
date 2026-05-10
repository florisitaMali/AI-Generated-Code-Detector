"""
Time-to-submit analysis for behavioral AI detection.

Flags submissions that arrive suspiciously fast relative to the expected solve
time for a given problem difficulty. AI-assisted users often submit within
seconds or with unnaturally consistent delays.
"""

import numpy as np
from loguru import logger


def compute_expected_time_percentiles(
    historical_times: list[float],
) -> dict[str, float]:
    """
    Compute time-to-submit percentiles from historical data.

    Args:
        historical_times: List of solve times (seconds) for a specific problem.

    Returns:
        Dict with p5, p10, p25, p50, p75, p90 percentiles.
    """
    if not historical_times:
        return {f"p{p}": 0.0 for p in [5, 10, 25, 50, 75, 90]}

    arr = np.array(historical_times)
    return {
        f"p{p}": float(np.percentile(arr, p))
        for p in [5, 10, 25, 50, 75, 90]
    }


def score_submission_time(
    submit_time_seconds: float,
    historical_times: list[float],
    fast_threshold_percentile: float = 5.0,
) -> dict:
    """
    Score how suspicious a submission's timing is.

    Args:
        submit_time_seconds: Time from contest/problem start to submission.
        historical_times: Past solve times for this problem.
        fast_threshold_percentile: Percentile below which timing is suspicious.

    Returns:
        {
            "timing_score": float (0-1, higher = more suspicious),
            "percentile_rank": float (where this time falls in the distribution),
            "is_fast_outlier": bool,
            "z_score": float,
        }
    """
    if not historical_times or submit_time_seconds <= 0:
        return {
            "timing_score": 0.0,
            "percentile_rank": 50.0,
            "is_fast_outlier": False,
            "z_score": 0.0,
        }

    arr = np.array(historical_times)
    mean_time = arr.mean()
    std_time = arr.std() if len(arr) > 1 else mean_time * 0.3

    percentile_rank = float(np.sum(arr <= submit_time_seconds) / len(arr) * 100)
    z_score = (submit_time_seconds - mean_time) / std_time if std_time > 0 else 0.0

    threshold = float(np.percentile(arr, fast_threshold_percentile))
    is_fast_outlier = submit_time_seconds < threshold

    if is_fast_outlier:
        ratio = submit_time_seconds / threshold if threshold > 0 else 0.0
        timing_score = max(0.0, 1.0 - ratio)
    else:
        timing_score = max(0.0, min(1.0, (50.0 - percentile_rank) / 50.0))
        timing_score = max(timing_score, 0.0)

    return {
        "timing_score": round(timing_score, 4),
        "percentile_rank": round(percentile_rank, 2),
        "is_fast_outlier": is_fast_outlier,
        "z_score": round(z_score, 4),
    }


def score_submission_velocity(
    problems_solved: int,
    elapsed_minutes: float,
    expected_rate: float = 0.5,
) -> dict:
    """
    Score how suspicious the overall solve velocity is.

    Args:
        problems_solved: Number of problems solved correctly.
        elapsed_minutes: Total contest time elapsed.
        expected_rate: Expected problems per minute for a strong human solver.

    Returns:
        {"velocity_score": float (0-1), "rate": float (problems/min)}
    """
    if elapsed_minutes <= 0:
        return {"velocity_score": 0.0, "rate": 0.0}

    rate = problems_solved / elapsed_minutes
    ratio = rate / expected_rate if expected_rate > 0 else 0.0

    velocity_score = max(0.0, min(1.0, (ratio - 1.0) / 2.0))

    return {
        "velocity_score": round(velocity_score, 4),
        "rate": round(rate, 4),
    }


def extract_features(metadata: dict) -> dict:
    """
    Public interface: extract timing features from submission metadata.

    Expected metadata keys:
        - submit_time_seconds: float
        - historical_times: list[float] (optional)
        - problems_solved: int (optional)
        - elapsed_minutes: float (optional)
    """
    features = {}

    submit_time = metadata.get("submit_time_seconds", 0)
    historical = metadata.get("historical_times", [])
    if submit_time and historical:
        timing = score_submission_time(submit_time, historical)
        features.update(timing)
    else:
        features.update({
            "timing_score": 0.0,
            "percentile_rank": 50.0,
            "is_fast_outlier": False,
            "z_score": 0.0,
        })

    solved = metadata.get("problems_solved")
    elapsed = metadata.get("elapsed_minutes")
    if solved is not None and elapsed is not None:
        velocity = score_submission_velocity(solved, elapsed)
        features.update(velocity)
    else:
        features.update({"velocity_score": 0.0, "rate": 0.0})

    return features
