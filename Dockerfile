# ============================================================
# HalluciSense Production Dockerfile (Railway Root Build)
# Memory-safe CPU/ONNX inference runtime
# ============================================================

FROM python:3.10-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel

# CPU-only PyTorch: retained only for non-NLI compatibility paths.
RUN pip install \
    --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Install remaining dependencies, including the ONNX Runtime backend.
RUN pip install \
    --prefix=/install \
    -r requirements.txt

# ============================================================

FROM python:3.10-slim AS runner

WORKDIR /app

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
    ONNXRUNTIME_INTRA_OP_NUM_THREADS=1 \
    ONNXRUNTIME_INTER_OP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=65536 \
    HF_HOME=/data/cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/data/cache/sentence-transformers \
    TRANSFORMERS_CACHE=/data/cache/transformers \
    TRACE_DIR=/data/traces \
    MODEL_DIR=/data/models \
    FAISS_DIR=/data/faiss \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend:/app \
    HALLUCISENSE_ENABLE_RERANKER=false

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data/traces /data/models /data/cache /data/faiss /data/reports \
    && chmod -R 777 /data

COPY --from=builder /install /usr/local

COPY backend/ /app/backend/
COPY start.py /app/start.py

EXPOSE 8000

HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=90s \
    --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["python", "start.py"]
