"""
In-memory user store for Phase 1.

Two built-in accounts:
  admin / psip2024  — role: admin  (full access)
  viewer / view2024 — role: viewer (read-only)

Replace with a real database in production.
"""
from __future__ import annotations

from .security import hash_password

USERS_DB: dict[str, dict] = {
    "admin": {
        "hashed_password": hash_password("psip2024"),
        "role": "admin",
    },
    "viewer": {
        "hashed_password": hash_password("view2024"),
        "role": "viewer",
    },
}

__all__ = ["USERS_DB"]
