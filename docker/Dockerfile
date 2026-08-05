# ============================================================
# HalluciSense Production Dockerfile
# Optimized for Railway / Render / Fly.io
# CPU-only PyTorch (keeps Pillar 3 working)
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
    git \
    curl \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# Install CPU-only PyTorch FIRST
# Prevents downloading NVIDIA CUDA wheels on Railway
# ------------------------------------------------------------
RUN pip install \
    --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch \
    torchvision \
    torchaudio

# ------------------------------------------------------------
# Install remaining dependencies
# ------------------------------------------------------------
RUN pip install \
    --prefix=/install \
    -r requirements.txt

# ============================================================

FROM python:3.10-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY backend/ /app/

EXPOSE 8000

HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/hallucisense/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]