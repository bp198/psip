# ─────────────────────────────────────────────────────────────────────────────
# PSIP — Pipeline Security & Integrity Platform
# Multi-stage Dockerfile
#
# Stage 1 (builder): install all Python dependencies into a venv
# Stage 2 (runtime): copy only the venv + app source — no build tools in prod
#
# Usage:
#   docker build -t psip:latest .
#   docker run -p 8000:8000 psip:latest
#   open http://localhost:8000/docs
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Prevent .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install build dependencies (needed to compile some scipy/numpy C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency manifest first (maximises layer cache reuse)
COPY pyproject.toml ./

# Create an isolated virtualenv so Stage 2 can copy just /venv
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install the package and its runtime dependencies
# We exclude optional extras (zone_a torch/yolo, fea) — not needed for the API
RUN pip install --upgrade pip && \
    pip install \
        numpy>=1.24 \
        scipy>=1.11 \
        pandas>=2.0 \
        matplotlib>=3.7 \
        networkx>=3.1 \
        shapely>=2.0 \
        pulp>=2.7 \
        plotly>=5.18 \
        dash>=2.14 \
        openpyxl>=3.1 \
        requests>=2.31 \
        fastapi>=0.100 \
        "uvicorn[standard]>=0.23" \
        httpx \
        pydantic>=2.0 \
        "python-jose[cryptography]>=3.3" \
        "passlib[bcrypt]>=1.7" \
        "bcrypt<4.0.0"


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Point Python to the venv copied from builder
    PATH="/venv/bin:$PATH" \
    # Add /app to PYTHONPATH so `import src.*` and `import psip` both resolve
    PYTHONPATH="/app"

# Non-root user for security — never run production services as root
RUN groupadd --gid 1001 psip && \
    useradd  --uid 1001 --gid psip --shell /bin/bash --create-home psip

WORKDIR /app

# Copy the virtualenv from the builder stage
COPY --from=builder /venv /venv

# Copy application source
# We copy src/ and psip/ (the two importable namespaces) plus main.py
COPY --chown=psip:psip src/      ./src/
COPY --chown=psip:psip psip/     ./psip/
COPY --chown=psip:psip main.py   ./main.py

# Optional: copy data directory (PHMSA calibration files) if present
# Comment this out if you don't need the data files at runtime
COPY --chown=psip:psip data/     ./data/

# Switch to non-root user
USER psip

# Expose the API port
EXPOSE 8000

# Health check — Docker will mark the container unhealthy if /api/health fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" \
    || exit 1

# Default command: run with uvicorn
# --workers 1 keeps memory usage low for pilot deployments;
# increase to match CPU count for production
CMD ["uvicorn", "psip.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
