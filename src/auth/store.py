"""Supabase persistence for users and scan history."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from loguru import logger

import config.settings as settings

MAX_HISTORY_CODE_CHARS = 100_000
DEFAULT_HISTORY_LIMIT = 50


@dataclass(frozen=True)
class User:
    id: str
    username: str
    created_at: str


@dataclass(frozen=True)
class HistoryEntry:
    id: str
    user_id: str
    code: str
    language: str
    problem_id: str | None
    detection_mode: str
    risk_score: float
    decision: str
    component_scores: dict
    signals: list[str]
    created_at: str


def _supabase_enabled() -> bool:
    return bool(settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY)


def _rest_url(path: str) -> str:
    return f"{settings.SUPABASE_URL}/rest/v1/{path.lstrip('/')}"


def _rest_headers() -> dict[str, str]:
    return {
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


def _request(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    json: dict | list | None = None,
    prefer: str | None = None,
) -> httpx.Response:
    if not _supabase_enabled():
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
    headers = _rest_headers()
    if prefer:
        headers["Prefer"] = prefer
    with httpx.Client(timeout=20) as client:
        resp = client.request(method, _rest_url(path), headers=headers, params=params, json=json)
    if resp.status_code >= 300:
        raise RuntimeError(f"Supabase request failed ({resp.status_code}): {resp.text}")
    return resp


def init_db() -> None:
    if not _supabase_enabled():
        logger.warning("Supabase not configured; auth/history endpoints will be unavailable.")


def ensure_user(user: User) -> None:
    """Upsert a local app user profile row in Supabase."""
    _request(
        "POST",
        settings.SUPABASE_USERS_TABLE,
        params={"on_conflict": "id"},
        prefer="resolution=merge-duplicates",
        json=[{
            "id": user.id,
            "email": user.username,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }],
    )


def add_history(
    user_id: str,
    *,
    code: str,
    language: str,
    problem_id: str | None,
    detection_mode: str,
    risk_score: float,
    decision: str,
    component_scores: dict,
    signals: list[str],
) -> HistoryEntry:
    if len(code) > MAX_HISTORY_CODE_CHARS:
        code = code[:MAX_HISTORY_CODE_CHARS]
    entry_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "id": entry_id,
        "user_id": user_id,
        "code": code,
        "language": language,
        "problem_id": problem_id,
        "detection_mode": detection_mode,
        "risk_score": risk_score,
        "decision": decision,
        "component_scores": component_scores,
        "signals": signals,
        "created_at": created_at,
    }
    _request("POST", settings.SUPABASE_HISTORY_TABLE, json=payload)
    return HistoryEntry(
        id=entry_id,
        user_id=user_id,
        code=code,
        language=language,
        problem_id=problem_id,
        detection_mode=detection_mode,
        risk_score=risk_score,
        decision=decision,
        component_scores=component_scores,
        signals=signals,
        created_at=created_at,
    )


def list_history(user_id: str, limit: int = DEFAULT_HISTORY_LIMIT) -> list[HistoryEntry]:
    limit = max(1, min(limit, 200))
    params = {
        "select": "id,user_id,code,language,problem_id,detection_mode,risk_score,decision,component_scores,signals,created_at",
        "user_id": f"eq.{user_id}",
        "order": "created_at.desc",
        "limit": str(limit),
    }
    rows = _request("GET", settings.SUPABASE_HISTORY_TABLE, params=params).json()
    return [_row_to_entry(r) for r in rows]


def get_history_entry(user_id: str, entry_id: str) -> HistoryEntry | None:
    params = {
        "select": "id,user_id,code,language,problem_id,detection_mode,risk_score,decision,component_scores,signals,created_at",
        "id": f"eq.{entry_id}",
        "user_id": f"eq.{user_id}",
        "limit": "1",
    }
    rows = _request("GET", settings.SUPABASE_HISTORY_TABLE, params=params).json()
    if not rows:
        return None
    return _row_to_entry(rows[0])


def delete_history_entry(user_id: str, entry_id: str) -> bool:
    existing = get_history_entry(user_id, entry_id)
    if existing is None:
        return False
    params = {"id": f"eq.{entry_id}", "user_id": f"eq.{user_id}"}
    _request("DELETE", settings.SUPABASE_HISTORY_TABLE, params=params)
    return get_history_entry(user_id, entry_id) is None


def _row_to_entry(row: dict) -> HistoryEntry:
    return HistoryEntry(
        id=row["id"],
        user_id=row["user_id"],
        code=row["code"],
        language=row["language"],
        problem_id=row["problem_id"],
        detection_mode=row["detection_mode"],
        risk_score=row["risk_score"],
        decision=row["decision"],
        component_scores=row["component_scores"] or {},
        signals=row["signals"] or [],
        created_at=row["created_at"],
    )
