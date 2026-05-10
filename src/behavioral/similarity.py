"""
Cross-user code similarity detection.

Detects suspiciously similar submissions to the same problem within a time
window. Combines normalised edit distance and TF-IDF cosine similarity on
tokenised code to flag potential collusion or shared AI-generated sources.
"""

import re
from difflib import SequenceMatcher

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalise_code(code: str) -> str:
    """Strip comments, normalise whitespace, and lowercase for comparison."""
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
    code = re.sub(r"\s+", " ", code).strip().lower()
    return code


def edit_distance_similarity(code_a: str, code_b: str) -> float:
    """
    Normalised sequence similarity using SequenceMatcher.
    Returns 0-1 where 1 means identical.
    """
    a = normalise_code(code_a)
    b = normalise_code(code_b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def tfidf_similarity(codes: list[str]) -> np.ndarray:
    """
    Compute pairwise TF-IDF cosine similarity for a list of code snippets.
    Returns an NxN similarity matrix.
    """
    normalised = [normalise_code(c) for c in codes]
    normalised = [c if c else " " for c in normalised]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5000,
    )
    tfidf_matrix = vectorizer.fit_transform(normalised)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix


def find_suspicious_pairs(
    submissions: list[dict],
    similarity_threshold: float = 0.85,
    time_window_seconds: float = 300,
) -> list[dict]:
    """
    Find pairs of submissions that are suspiciously similar within a time window.

    Args:
        submissions: List of dicts with keys:
            - user_id: str
            - code: str
            - timestamp: float (epoch seconds)
            - problem_id: str
        similarity_threshold: Minimum similarity to flag.
        time_window_seconds: Max time difference between submissions.

    Returns:
        List of flagged pairs with similarity scores.
    """
    if len(submissions) < 2:
        return []

    codes = [s["code"] for s in submissions]
    sim_matrix = tfidf_similarity(codes)
    flagged = []

    for i in range(len(submissions)):
        for j in range(i + 1, len(submissions)):
            if submissions[i].get("user_id") == submissions[j].get("user_id"):
                continue

            time_diff = abs(
                submissions[i].get("timestamp", 0) - submissions[j].get("timestamp", 0)
            )
            if time_diff > time_window_seconds:
                continue

            tfidf_sim = float(sim_matrix[i, j])
            edit_sim = edit_distance_similarity(
                submissions[i]["code"], submissions[j]["code"]
            )
            combined_sim = 0.6 * tfidf_sim + 0.4 * edit_sim

            if combined_sim >= similarity_threshold:
                flagged.append({
                    "user_a": submissions[i].get("user_id"),
                    "user_b": submissions[j].get("user_id"),
                    "problem_id": submissions[i].get("problem_id"),
                    "tfidf_similarity": round(tfidf_sim, 4),
                    "edit_similarity": round(edit_sim, 4),
                    "combined_similarity": round(combined_sim, 4),
                    "time_diff_seconds": round(time_diff, 1),
                })

    flagged.sort(key=lambda x: -x["combined_similarity"])
    return flagged


def extract_features(code: str, peer_codes: list[str] | None = None) -> dict:
    """
    Public interface: compute similarity features for a submission.

    Args:
        code: The submission to analyse.
        peer_codes: Other submissions to the same problem (optional).

    Returns:
        Feature dict with max/mean similarity to peers.
    """
    if not peer_codes:
        return {
            "sim_max_tfidf": 0.0,
            "sim_mean_tfidf": 0.0,
            "sim_max_edit": 0.0,
            "sim_mean_edit": 0.0,
        }

    all_codes = [code] + peer_codes
    sim_matrix = tfidf_similarity(all_codes)
    tfidf_scores = sim_matrix[0, 1:]

    edit_scores = [edit_distance_similarity(code, peer) for peer in peer_codes]

    return {
        "sim_max_tfidf": float(np.max(tfidf_scores)) if len(tfidf_scores) else 0.0,
        "sim_mean_tfidf": float(np.mean(tfidf_scores)) if len(tfidf_scores) else 0.0,
        "sim_max_edit": float(max(edit_scores)) if edit_scores else 0.0,
        "sim_mean_edit": float(np.mean(edit_scores)) if edit_scores else 0.0,
    }
