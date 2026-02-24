# ─────────────────────────────────────────────────────────────────────────────
# Customer Intelligence Platform - Multi-stage Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — installs Python deps in /root/.local
# Stage 2: runtime — slim image with only what's needed to run
#
# Build: docker build -t customer-intelligence:latest .
# Run:   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... customer-intelligence
# ─────────────────────────────────────────────────────────────────────────────

# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system dependencies (needed for scikit-learn/numpy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into /root/.local (user-install)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Customer Intelligence Platform" \
      org.opencontainers.image.description="Multi-agent AI API: segmentation, campaigns, recommendations, CRM/loyalty" \
      org.opencontainers.image.source="https://github.com/your-org/multi-agent-project-1" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code (owned by appuser)
COPY --chown=appuser:appuser . .

# Environment configuration
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    LOG_LEVEL=WARNING

# Switch to non-root user
USER appuser

EXPOSE 8000

# ─── Liveness check ──────────────────────────────────────────────────────────
# Polls /health endpoint; allows 30s for MCP server startup
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
        || exit 1

# ─── Entrypoint ──────────────────────────────────────────────────────────────
# Single uvicorn worker: MCP subprocess is in-process.
# Horizontal scaling is handled by Kubernetes HPA (not by uvicorn workers).
CMD ["sh", "-c", \
     "uvicorn app.api:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level warning"]
