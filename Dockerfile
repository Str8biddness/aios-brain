# =========================
# Stage 1: Builder
# =========================
FROM python:3.11-slim AS builder

# Install build dependencies (gcc, libpq, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies into a temporary directory
RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# =========================
# Stage 2: Final Runtime
# =========================
FROM python:3.11-slim

# Metadata labels
LABEL org.opencontainers.image.title="AIOS Brain FastAPI Application" \
      org.opencontainers.image.description="Production-grade reasoning engine with FastAPI, Redis, PostgreSQL, and Prometheus metrics." \
      org.opencontainers.image.authors="AIOS Team" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

# Install runtime dependencies (curl for healthcheck, libpq for Postgres)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application code
COPY . /app

# Create non-root user
RUN useradd -m aiosuser
USER aiosuser

# Expose application port
EXPOSE 8080

# Healthcheck: calls FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Default command: run uvicorn ASGI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
