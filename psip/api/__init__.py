"""
psip.api — FastAPI application for the Pipeline Security & Integrity Platform.

Registers all engine routers and exposes the application factory.
Import `app` from here or run via:  uvicorn psip.api:app --reload
"""
from psip.api.app import app

__all__ = ["app"]
