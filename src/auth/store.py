"""Supabase persistence for users and scan history."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from loguru import logger

import config.settings as settings

MAX_HISTORY_CODE_CHARS = 100_000
DEFAULT_HISTORY_LIMIT = 50
_CODE_PREVIEW_LEN = 120
_USER_ENSURE_TTL = 300.0

_LIST_COLUMNS = (
    "id,user_id,language,problem_id,detection_mode,risk_score,decision,created_at,code_preview"
)
_DETAIL_COLUMNS = (
    "id,user_id,code,language,problem_id,detection_mode,risk_score,decision,"
    "component_scores,signals,created_at,code_preview"
)

_HTTP_CLIENT: httpx.Client | None = None
_USER_ENSURE_CACHE: dict[str, float] = {}


@dataclass(frozen=True)
class User:
    id: str
    username: str
    created_at: str


@dataclass(frozen=True)
class HistoryEntry:
    id: str
    user_id: str
    language: str
    problem_id: str | None
    detection_mode: str
    risk_score: float
    decision: str
    created_at: str
    code: str = ""
    code_preview: str = ""
    component_scores: dict | None = None
    signals: list[str] | None = None


def _preview(code: str) -> str:
    one_line = " ".join(code.split())
    if len(one_line) <= _CODE_PREVIEW_LEN:
        return one_line
    return one_line[: _CODE_PREVIEW_LEN - 1] + "…"


def _http_client() -> httpx.Client:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.Client(
            timeout=20.0,
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )
    return _HTTP_CLIENT


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
    resp = _http_client().request(
        method, _rest_url(path), headers=headers, params=params, json=json
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Supabase request failed ({resp.status_code}): {resp.text}")
    return resp


def init_db() -> None:
    if not _supabase_enabled():
        logger.warning("Supabase not configured; auth/history endpoints will be unavailable.")


def ensure_user(user: User) -> None:
    """Upsert a local app user profile row in Supabase (cached per session)."""
    now = time.monotonic()
    last = _USER_ENSURE_CACHE.get(user.id)
    if last is not None and now - last < _USER_ENSURE_TTL:
        return
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
    _USER_ENSURE_CACHE[user.id] = now


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
    code_preview = _preview(code)
    payload = {
        "id": entry_id,
        "user_id": user_id,
        "code": code,
        "code_preview": code_preview,
        "language": language,
        "problem_id": problem_id,
        "detection_mode": detection_mode,
        "risk_score": risk_score,
        "decision": decision,
        "component_scores": component_scores,
        "signals": signals,
        "created_at": created_at,
    }
    try:
        _request("POST", settings.SUPABASE_HISTORY_TABLE, json=payload)
    except RuntimeError as exc:
        # Backward-compatible insert before code_preview column migration.
        if "code_preview" not in str(exc):
            raise
        payload.pop("code_preview", None)
        _request("POST", settings.SUPABASE_HISTORY_TABLE, json=payload)
    return HistoryEntry(
        id=entry_id,
        user_id=user_id,
        code=code,
        code_preview=code_preview,
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
        "select": _LIST_COLUMNS,
        "user_id": f"eq.{user_id}",
        "order": "created_at.desc",
        "limit": str(limit),
    }
    try:
        rows = _request("GET", settings.SUPABASE_HISTORY_TABLE, params=params).json()
    except RuntimeError as exc:
        if "code_preview" not in str(exc):
            raise
        params["select"] = _LIST_COLUMNS.replace(",code_preview", "")
        rows = _request("GET", settings.SUPABASE_HISTORY_TABLE, params=params).json()
    return [_row_to_entry(r, include_body=False) for r in rows]


def get_history_entry(user_id: str, entry_id: str) -> HistoryEntry | None:
    params = {
        "select": _DETAIL_COLUMNS,
        "id": f"eq.{entry_id}",
        "user_id": f"eq.{user_id}",
        "limit": "1",
    }
    try:
        rows = _request("GET", settings.SUPABASE_HISTORY_TABLE, params=params).json()
    except RuntimeError as exc:
        if "code_preview" not in str(exc):
            raise
        params["select"] = _DETAIL_COLUMNS.replace(",code_preview", "")
        rows = _request("GET", settings.SUPABASE_HISTORY_TABLE, params=params).json()
    if not rows:
        return None
    return _row_to_entry(rows[0], include_body=True)


def delete_history_entry(user_id: str, entry_id: str) -> bool:
    resp = _request(
        "DELETE",
        settings.SUPABASE_HISTORY_TABLE,
        params={"id": f"eq.{entry_id}", "user_id": f"eq.{user_id}"},
        prefer="return=representation",
    )
    rows = resp.json() if resp.content else []
    return len(rows) > 0


def _row_to_entry(row: dict, *, include_body: bool) -> HistoryEntry:
    code = row.get("code") or ""
    preview = row.get("code_preview") or (_preview(code) if code else "")
    return HistoryEntry(
        id=row["id"],
        user_id=row["user_id"],
        code=code if include_body else "",
        code_preview=preview,
        language=row["language"],
        problem_id=row.get("problem_id"),
        detection_mode=row["detection_mode"],
        risk_score=row["risk_score"],
        decision=row["decision"],
        component_scores=row.get("component_scores") or {} if include_body else None,
        signals=row.get("signals") or [] if include_body else None,
        created_at=row["created_at"],
    )
