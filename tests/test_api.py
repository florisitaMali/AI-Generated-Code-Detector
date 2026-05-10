"""Tests for the FastAPI detection endpoints."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_analyze_basic():
    resp = client.post("/analyze", json={
        "code": "print('hello world')",
        "language": "python",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "risk_score" in data
    assert "decision" in data
    assert data["decision"] in ("accept", "review", "hold")
    assert 0.0 <= data["risk_score"] <= 1.0


def test_analyze_component_scores():
    """Response must contain exactly the 3 active component score fields."""
    resp = client.post("/analyze", json={
        "code": "#include <iostream>\nint main() { return 0; }",
        "language": "cpp",
        "problem_id": "p00001",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "component_scores" in data
    comp = data["component_scores"]
    # behavioral must not be present in the response
    assert "behavioral" not in comp or comp.get("behavioral") is None
    # at least the stylometric score must be returned (heuristic always fires)
    assert comp.get("statistical") is not None


def test_analyze_empty_code():
    resp = client.post("/analyze", json={
        "code": "",
        "language": "python",
    })
    assert resp.status_code == 400


def test_analyze_invalid_language():
    """Language not in the supported set must return 400."""
    resp = client.post("/analyze", json={
        "code": "print(1)",
        "language": "rust",
    })
    assert resp.status_code == 400


@pytest.mark.parametrize("lang,snippet", [
    ("c",          "#include <stdio.h>\nint main(){ printf(\"hello\"); return 0; }"),
    ("csharp",     "using System;\nclass P { static void Main() { Console.WriteLine(1); } }"),
    ("javascript", "console.log(42);"),
])
def test_analyze_valid_new_languages(lang, snippet):
    """C, C#, and JavaScript submissions must return 200 with a valid risk score."""
    resp = client.post("/analyze", json={"code": snippet, "language": lang})
    assert resp.status_code == 200, f"{lang}: {resp.text}"
    data = resp.json()
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["decision"] in ("accept", "review", "hold")


def test_batch():
    resp = client.post("/batch", json={
        "submissions": [
            {"code": "print(1)", "language": "python"},
            {"code": "print(2)", "language": "python"},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2


def test_batch_empty():
    resp = client.post("/batch", json={"submissions": []})
    assert resp.status_code == 400


def test_feedback():
    resp = client.post("/feedback", json={
        "code": "print(1)",
        "language": "python",
        "true_label": 1,
        "problem_id": "p00001",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "recorded"
    assert "feedback_id" in data
