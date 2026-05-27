"""Supabase JWT verification helpers."""

from __future__ import annotations

import jwt
from jwt import PyJWKClient

from config.settings import SUPABASE_JWT_AUDIENCE, SUPABASE_JWT_SECRET, SUPABASE_URL

_jwk_client: PyJWKClient | None = None


def _get_jwk_client() -> PyJWKClient | None:
    global _jwk_client
    if not SUPABASE_URL:
        return None
    if _jwk_client is None:
        _jwk_client = PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    return _jwk_client


def _check_audience(payload: dict) -> bool:
    if not SUPABASE_JWT_AUDIENCE:
        return True
    aud = payload.get("aud")
    if isinstance(aud, str):
        return aud == SUPABASE_JWT_AUDIENCE
    if isinstance(aud, list):
        return SUPABASE_JWT_AUDIENCE in aud
    return False


def decode_access_token(token: str) -> dict | None:
    """Verify Supabase access token (HS256 or asymmetric via JWKS)."""
    issuer = f"{SUPABASE_URL}/auth/v1"
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        return None

    alg = header.get("alg", "")
    payload = None

    if alg == "HS256":
        if not SUPABASE_JWT_SECRET:
            from src.debug_log import agent_log
            agent_log(
                "security.py:decode_access_token",
                "HS256 token but SUPABASE_JWT_SECRET missing",
                {},
                "H4",
            )
            return None
        try:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                issuer=issuer,
                options={"verify_aud": False},
            )
        except jwt.PyJWTError as exc:
            from src.debug_log import agent_log
            agent_log(
                "security.py:decode_access_token",
                "HS256 decode error",
                {"error_type": type(exc).__name__},
                "H4",
            )
            return None
    elif alg in ("RS256", "ES256"):
        jwk_client = _get_jwk_client()
        if jwk_client is None:
            return None
        try:
            signing_key = jwk_client.get_signing_key_from_jwt(token).key
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=[alg],
                issuer=issuer,
                options={"verify_aud": False},
            )
        except jwt.PyJWTError as exc:
            from src.debug_log import agent_log
            agent_log(
                "security.py:decode_access_token",
                "jwks decode error",
                {"error_type": type(exc).__name__, "alg": alg},
                "H4",
            )
            return None
    else:
        from src.debug_log import agent_log
        agent_log(
            "security.py:decode_access_token",
            "unsupported jwt alg",
            {"alg": alg},
            "H4",
        )
        return None

    if payload and not _check_audience(payload):
        return None
    if payload:
        from src.debug_log import agent_log
        agent_log(
            "security.py:decode_access_token",
            "jwt ok",
            {"alg": alg},
            "H4",
        )
    return payload
