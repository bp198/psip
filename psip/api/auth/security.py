"""
JWT token creation / validation and password hashing.

Algorithm  : HS256
Token TTL  : 60 minutes (access) — configurable via env vars
Secret key : read from PSIP_SECRET_KEY env var (falls back to a dev default;
             ALWAYS override in production).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt
from passlib.context import CryptContext

# ── Configuration ──────────────────────────────────────────────────────────────
SECRET_KEY: str = os.getenv(
    "PSIP_SECRET_KEY",
    "psip-dev-secret-change-me-in-production-please",  # noqa: S105
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("PSIP_TOKEN_TTL_MINUTES", "60"))

# ── Password hashing ───────────────────────────────────────────────────────────
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


# ── Token helpers ──────────────────────────────────────────────────────────────
def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    """Return a signed JWT access token for *subject* (username)."""
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Decode and validate a JWT.  Raises ``JWTError`` on failure.
    Returns the full payload dict on success.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


__all__ = [
    "SECRET_KEY",
    "ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
]
