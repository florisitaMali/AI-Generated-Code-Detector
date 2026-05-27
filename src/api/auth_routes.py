"""Authentication and scan history endpoints (Supabase-based)."""

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.schemas import (
    HistoryDetailResponse,
    HistoryEntryResponse,
    HistoryListResponse,
    SupabaseConfigResponse,
    UserPublic,
)
from src.auth.deps import get_current_user
from src.auth.store import User, delete_history_entry, get_history_entry, list_history
from config.settings import SUPABASE_ANON_KEY, SUPABASE_URL

router = APIRouter(prefix="/auth", tags=["auth"])

_CODE_PREVIEW_LEN = 120


def _preview(code: str) -> str:
    one_line = " ".join(code.split())
    if len(one_line) <= _CODE_PREVIEW_LEN:
        return one_line
    return one_line[: _CODE_PREVIEW_LEN - 1] + "…"


def _entry_summary(entry) -> HistoryEntryResponse:
    return HistoryEntryResponse(
        id=entry.id,
        language=entry.language,
        problem_id=entry.problem_id,
        detection_mode=entry.detection_mode,
        risk_score=entry.risk_score,
        decision=entry.decision,
        code_preview=_preview(entry.code),
        created_at=entry.created_at,
    )


@router.get("/supabase/config", response_model=SupabaseConfigResponse)
async def supabase_config():
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return SupabaseConfigResponse(enabled=False, url=None, anon_key=None)
    return SupabaseConfigResponse(enabled=True, url=SUPABASE_URL, anon_key=SUPABASE_ANON_KEY)


@router.get("/me", response_model=UserPublic)
async def me(user: User = Depends(get_current_user)):
    return UserPublic(id=user.id, username=user.username)


@router.get("/history", response_model=HistoryListResponse)
async def history(
    user: User = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
):
    entries = list_history(user.id, limit=limit)
    return HistoryListResponse(entries=[_entry_summary(e) for e in entries])


@router.get("/history/{entry_id}", response_model=HistoryDetailResponse)
async def history_detail(
    entry_id: str,
    user: User = Depends(get_current_user),
):
    entry = get_history_entry(user.id, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="History entry not found")
    summary = _entry_summary(entry)
    return HistoryDetailResponse(
        **summary.model_dump(),
        code=entry.code,
        component_scores=entry.component_scores,
        signals=entry.signals,
    )


@router.delete("/history/{entry_id}")
async def history_delete(
    entry_id: str,
    user: User = Depends(get_current_user),
):
    if not delete_history_entry(user.id, entry_id):
        raise HTTPException(status_code=404, detail="History entry not found")
    return {"status": "deleted"}
