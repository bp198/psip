"""
Auth router — mounted at /api/auth

POST /api/auth/login   → returns JWT access token
GET  /api/auth/me      → returns current user info (requires valid token)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from .deps import get_current_user
from .models import LoginRequest, TokenResponse, UserInfo
from .security import ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, verify_password
from .users import USERS_DB

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Obtain a JWT access token",
    description=(
        "Exchange a username + password for a signed JWT bearer token. "
        "Pass the token in subsequent requests as `Authorization: Bearer <token>`."
    ),
)
def login(body: LoginRequest) -> TokenResponse:
    user = USERS_DB.get(body.username)
    if user is None or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=body.username, extra={"role": user["role"]})
    return TokenResponse(
        access_token=token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get(
    "/me",
    response_model=UserInfo,
    summary="Return current authenticated user",
)
def me(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
    return current_user


__all__ = ["router"]
