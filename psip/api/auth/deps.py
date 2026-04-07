"""
FastAPI dependency: ``get_current_user``.

Usage
-----
Add ``current_user: UserInfo = Depends(get_current_user)`` to any route
that should require a valid JWT.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from .models import UserInfo
from .security import decode_access_token
from .users import USERS_DB

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInfo:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = USERS_DB.get(username)
    if user is None:
        raise credentials_exc

    return UserInfo(username=username, role=user["role"])


__all__ = ["oauth2_scheme", "get_current_user"]
