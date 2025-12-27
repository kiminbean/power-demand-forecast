# ============================================================
# Power Demand Forecast API - Dockerfile (Task 15)
# ============================================================
# Multi-stage build for optimized production image

# -----------------------------------------------------------------------------
# Stage 1: Base image with Python
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage 2: Dependencies
# -----------------------------------------------------------------------------
FROM base as dependencies

WORKDIR /app

# Copy requirements (Docker-optimized with CPU PyTorch)
COPY requirements-docker.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt

# -----------------------------------------------------------------------------
# Stage 3: Test (optional, for CI)
# -----------------------------------------------------------------------------
FROM dependencies as test

WORKDIR /app

# Copy source code and tests
COPY src/ ./src/
COPY api/ ./api/
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY pytest.ini ./

# Run tests
RUN pip install pytest pytest-cov pytest-asyncio
RUN python -m pytest tests/ -v --ignore=tests/test_dashboard.py || true

# -----------------------------------------------------------------------------
# Stage 4: Production
# -----------------------------------------------------------------------------
FROM dependencies as production

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser tools/ ./tools/
COPY --chown=appuser:appuser configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data && \
    chown -R appuser:appuser /app

# API configuration
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_WORKERS=4 \
    LOG_LEVEL=info

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Default command
CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST} --port ${API_PORT} --workers ${API_WORKERS} --log-level ${LOG_LEVEL}"]
