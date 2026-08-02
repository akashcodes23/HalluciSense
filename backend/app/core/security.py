"""
Security utilities: password hashing, JWT creation and verification.
All functions are pure — no database or network calls.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

# ---------------------------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------------------------

import bcrypt

def hash_password(plain_password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(plain_password.encode('utf-8'), salt)
    return hashed_bytes.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if plain_password matches the stored bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'), 
        hashed_password.encode('utf-8')
    )


# ---------------------------------------------------------------------------
# JWT Tokens
# ---------------------------------------------------------------------------

def _create_token(
    subject: str,
    token_type: str,
    expires_delta: timedelta,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """
    Low-level JWT factory.
    'sub'  — user UUID (string)
    'type' — 'access' | 'refresh'
    'iat'  — issued-at (UTC)
    'exp'  — expiry (UTC)
    """
    now = datetime.now(tz=timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_access_token(user_id: UUID, role: str = "USER") -> str:
    """Issue a short-lived access token (default 15 min)."""
    return _create_token(
        subject=str(user_id),
        token_type="access",
        expires_delta=settings.access_token_delta,
        extra_claims={"role": role},
    )


def create_refresh_token(user_id: UUID) -> str:
    """Issue a long-lived refresh token (default 7 days)."""
    return _create_token(
        subject=str(user_id),
        token_type="refresh",
        expires_delta=settings.refresh_token_delta,
    )


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT. Raises JWTError on any failure.
    Callers should catch JWTError and map to HTTP 401.
    """
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])


def get_token_subject(token: str) -> Optional[str]:
    """
    Extract the 'sub' claim from a token without raising.
    Returns None on any decode error.
    """
    try:
        payload = decode_token(token)
        return payload.get("sub")
    except JWTError:
        return None
