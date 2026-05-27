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


def test_analyze_stylometric_mode_skips_other_detectors(monkeypatch):
    """Stylometric mode must not invoke CodeBERT or LLM."""
    cb_calls = []
    llm_calls = []

    def fake_codebert(code):
        cb_calls.append(code)
        return 0.99

    async def fake_llm(*args, **kwargs):
        llm_calls.append(1)
        return 0.99

    monkeypatch.setattr("src.api.routes._get_codebert_score", fake_codebert)
    monkeypatch.setattr("src.api.routes._get_llm_judge_score", fake_llm)

    resp = client.post("/analyze", json={
        "code": "print('hello')",
        "language": "python",
        "detection_mode": "stylometric",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["detection_mode"] == "stylometric"
    comp = data["component_scores"]
    assert comp.get("statistical") is not None
    assert comp.get("codebert") is None
    assert comp.get("llm_judge") is None
    assert cb_calls == []
    assert llm_calls == []


def test_analyze_codebert_mode_skips_statistical(monkeypatch):
    stat_calls = []

    def fake_stat(code, language):
        stat_calls.append((code, language))
        return 0.1

    monkeypatch.setattr("src.api.routes._get_statistical_score", fake_stat)

    resp = client.post("/analyze", json={
        "code": "print(1)",
        "language": "python",
        "detection_mode": "codebert",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["detection_mode"] == "codebert"
    assert stat_calls == []


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
