# ============================================================
# Dockerfile — ABSA Sentiment Analysis API
# Multi-stage: builder + runtime
# ============================================================

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.10.13-slim AS builder

WORKDIR /app

# System deps for scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements_api.txt


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.10.13-slim AS runtime

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY src/          ./src/
COPY serving/      ./serving/
COPY services/     ./services/
COPY absa_models/  ./absa_models/
COPY logs/         ./logs/

# Non-root user for security
RUN useradd -m -u 1000 absa \
 && chown -R absa:absa /app
USER absa

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

CMD ["uvicorn", "serving.absa_api:app", "--host", "0.0.0.0", "--port", "8000"]
