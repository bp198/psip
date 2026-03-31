"""
psip.api.app — FastAPI application factory.

All engine routers are registered here under the /api prefix.
Interactive documentation is auto-generated at /docs (Swagger UI)
and /redoc (ReDoc).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import psip
from psip.api.models import HealthResponse
from psip.api.routers import adversarial, fad, game, mc, network

# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PSIP — Pipeline Security & Integrity Platform",
    description=(
        "Physics-Informed Game-Theoretic Defense of Pipeline Infrastructure.\n\n"
        "This API wraps four computation engines developed in the STRATEGOS MSc thesis "
        "(Babak Pirzadi, 2025):\n\n"
        "- **FAD** — BS 7910:2019 Level 2 Failure Assessment Diagram\n"
        "- **MC** — Monte Carlo failure probability "
        "(calibrated against PHMSA 1,996-record dataset)\n"
        "- **Game** — Bayesian Stackelberg Security Equilibrium (LP enumeration / DOBSS)\n"
        "- **Adversarial** — FGSM / BIM / PGD attacks on the WeldDefectMLP NDE classifier\n\n"
        "All endpoints are fully documented below. Click **Try it out** on any endpoint "
        "to run a live calculation directly from this page."
    ),
    version=psip.__version__,
    contact={"name": "Babak Pirzadi", "email": "babak.pirzadi@gmail.com"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {
            "name": "Health",
            "description": "Service health check.",
        },
        {
            "name": "FAD Assessment",
            "description": "BS 7910:2019 Level 2 Failure Assessment Diagram engine. "
            "Determines whether a weld flaw is acceptable under operating conditions.",
        },
        {
            "name": "Monte Carlo Simulation",
            "description": "Monte Carlo failure probability engine. "
            "Propagates uncertainty in flaw size, toughness, and pressure "
            "through the FAD framework to compute P_f.",
        },
        {
            "name": "Stackelberg Game",
            "description": "Bayesian Stackelberg Security Equilibrium solver. "
            "Computes the optimal defender coverage allocation over a pipeline network "
            "against multiple attacker types.",
        },
        {
            "name": "Adversarial Attacks",
            "description": (
                "FGSM / BIM / PGD adversarial attack evaluation on the WeldDefectMLP classifier. "
                "Supports physics-informed ε scaling."
            ),
        },
        {
            "name": "Network",
            "description": "Pipeline network topology and P_f summary.",
        },
    ],
)

# CORS — allow all origins in development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(fad.router, prefix="/api")
app.include_router(mc.router, prefix="/api")
app.include_router(game.router, prefix="/api")
app.include_router(adversarial.router, prefix="/api")
app.include_router(network.router, prefix="/api")


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Returns service status, version, and list of available engines.",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=psip.__version__,
        engines=["fad", "mc", "game", "adversarial", "network"],
    )


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "PSIP API is running.",
        "docs": "/docs",
        "version": psip.__version__,
    }
