# ==========================================
# HalluciSense Production Dockerfile (Railway Ready)
# ==========================================

# ---------- Builder Stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install \
    --no-cache-dir \
    --prefix=/install \
    -r requirements.txt

# ---------- Runtime Stage ----------
FROM python:3.11-slim

WORKDIR /app

# Production Environment Variables for Railway
ENV APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    ENABLE_TRACING=true \
    ENABLE_DEBUG_API=true \
    API_VERSION=v1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    HF_HOME=/data/cache/huggingface \
    TRANSFORMERS_CACHE=/data/cache/transformers \
    TRACE_DIR=/data/traces \
    MODEL_DIR=/data/models \
    FAISS_DIR=/data/faiss \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create Railway persistent volume directories
RUN mkdir -p /data/traces /data/models /data/cache /data/faiss /data/reports \
    && chmod -R 777 /data

COPY --from=builder /install /usr/local
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["python", "start.py"]