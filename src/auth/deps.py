"""FastAPI dependencies for optional/required authentication."""

from typing import Annotated

from fastapi import Header, HTTPException

from src.auth.security import decode_access_token
from src.auth.store import User, ensure_user
from src.debug_log import agent_log


def _user_from_bearer(authorization: str | None) -> User | None:
    has_header = bool(authorization and authorization.lower().startswith("bearer "))
    if not has_header:
        agent_log("deps.py:_user_from_bearer", "no bearer header", {}, "H4")
        return None
    token = authorization[7:].strip()
    if not token:
        agent_log("deps.py:_user_from_bearer", "empty bearer token", {}, "H4")
        return None
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        agent_log(
            "deps.py:_user_from_bearer",
            "jwt decode failed",
            {"token_len": len(token)},
            "H4",
        )
        return None
    user_id = str(payload.get("sub") or "").strip()
    if not user_id:
        return None
    email = (
        payload.get("email")
        or (payload.get("user_metadata") or {}).get("email")
        or "supabase-user"
    )
    user = User(
        id=user_id,
        username=str(email),
        created_at=str(payload.get("iat") or ""),
    )
    ensure_user(user)
    agent_log(
        "deps.py:_user_from_bearer",
        "jwt ok",
        {"user_id_prefix": user_id[:8], "has_email": "@" in user.username},
        "H4",
    )
    return user


async def get_optional_user(
    authorization: Annotated[str | None, Header()] = None,
) -> User | None:
    return _user_from_bearer(authorization)


async def get_current_user(
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    user = _user_from_bearer(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
