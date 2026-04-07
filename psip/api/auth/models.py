"""Pydantic models for authentication request / response bodies."""
from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(..., examples=["admin"])
    password: str = Field(..., examples=["psip2024"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token TTL in seconds")


class UserInfo(BaseModel):
    username: str
    role: str


__all__ = ["LoginRequest", "TokenResponse", "UserInfo"]
