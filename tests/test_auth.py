"""Tests for TraceCoder authentication and scan history."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_auth_db(tmp_path, monkeypatch):
    db_path = tmp_path / "tracecoder_test.db"
    monkeypatch.setattr("config.settings.TRACECODER_DB_PATH", db_path)
    from src.auth.store import init_db

    init_db()


@pytest.fixture
def client():
    from src.api.main import app

    return TestClient(app)


def _token(client, username="alice", password="password123"):
    reg = client.post("/auth/register", json={"username": username, "password": password})
    if reg.status_code == 409:
        login = client.post("/auth/login", json={"username": username, "password": password})
        assert login.status_code == 200
        return login.json()["access_token"]
    assert reg.status_code == 200
    return reg.json()["access_token"]


def test_register_duplicate_username(client):
    assert _token(client) 
    dup = client.post("/auth/register", json={"username": "alice", "password": "password123"})
    assert dup.status_code == 409


def test_login_invalid_password(client):
    _token(client)
    bad = client.post("/auth/login", json={"username": "alice", "password": "wrong"})
    assert bad.status_code == 401


def test_me_requires_auth(client):
    assert client.get("/auth/me").status_code == 401
    token = _token(client)
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "alice"


def test_analyze_saves_history_when_authenticated(client):
    token = _token(client)
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post(
        "/analyze",
        json={"code": "print('hist')", "language": "python", "detection_mode": "stylometric"},
        headers=headers,
    )
    assert resp.status_code == 200

    hist = client.get("/auth/history", headers=headers)
    assert hist.status_code == 200
    entries = hist.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["language"] == "python"
    assert "print" in entries[0]["code_preview"]

    entry_id = entries[0]["id"]
    detail = client.get(f"/auth/history/{entry_id}", headers=headers)
    assert detail.status_code == 200
    assert "print('hist')" in detail.json()["code"]

    deleted = client.delete(f"/auth/history/{entry_id}", headers=headers)
    assert deleted.status_code == 200
    assert client.get("/auth/history", headers=headers).json()["entries"] == []
