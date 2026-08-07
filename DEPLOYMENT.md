# HalluciSense v1.0 Production Deployment Guide

Complete production deployment guide for deploying **HalluciSense v1.0** to Railway PaaS, Docker Compose, or local production environments.

---

## 1. Railway Production Deployment Guide

### Step 1: Connect Repository
1. Log in to [Railway.app](https://railway.app).
2. Click **New Project** $\rightarrow$ **Deploy from GitHub repo** $\rightarrow$ Select `HalluciSense`.
3. Railway automatically detects `railway.toml` and selects `backend/Dockerfile` as the build context.

### Step 2: Configure Environment Variables
In your Railway Service Settings, configure the following variables:

```ini
APP_ENV=production
HOST=0.0.0.0
PORT=${PORT}
LOG_LEVEL=INFO
ENABLE_TRACING=true
ENABLE_DEBUG_API=true
API_VERSION=v1
TOKENIZERS_PARALLELISM=false
HF_HOME=/data/cache/huggingface
TRANSFORMERS_CACHE=/data/cache/transformers
TRACE_DIR=/data/traces
MODEL_DIR=/data/models
FAISS_DIR=/data/faiss
CORS_ORIGINS=*
```

### Step 3: Attach Railway Volume (Persistent Data)
1. In the Railway Dashboard, add a **Volume** to your service.
2. Set the mount path to `/data`.
3. This persists execution traces (`/data/traces`), HuggingFace model cache (`/data/cache`), and FAISS vector indexes (`/data/faiss`) across container restarts.

### Step 4: Verify Deployment & Health Probes
Once Railway completes building the container:
- **Liveness Probe**: `https://<your-railway-app>.up.railway.app/health`
- **Readiness Probe**: `https://<your-railway-app>.up.railway.app/ready`
- **Swagger Documentation**: `https://<your-railway-app>.up.railway.app/docs`
- **Canonical Analysis API**: `https://<your-railway-app>.up.railway.app/api/v1/analyze`

---

## 2. Quick Start with Docker Compose

```bash
git clone https://github.com/akashcodes23/HalluciSense.git
cd HalluciSense
docker-compose up --build -d
```

Access services:
- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Readiness Probe**: `http://localhost:8000/ready`
- **Canonical Analysis API**: `http://localhost:8000/api/v1/analyze`

---

## 3. Production Smoke Testing Verification

Run the automated Railway smoke test suite against your local backend or deployed Railway service:

```bash
# Local backend verification
python3 backend/scripts/run_railway_smoke_tests.py http://127.0.0.1:8000

# Remote Railway deployment verification
python3 backend/scripts/run_railway_smoke_tests.py https://<your-railway-app>.up.railway.app
```

---

## 4. REST API Endpoint Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Service info and OpenAPI schema metadata |
| `/health` | `GET` | System liveness health check |
| `/ready` | `GET` | Deep component readiness check (503 if unready) |
| `/docs` | `GET` | Interactive Swagger OpenAPI UI |
| `/api/v1/analyze` | `POST` | **Canonical HalluciSense Verification Pipeline** |
| `/api/v1/explain` | `POST` | Detailed explainability & passage decomposition |
| `/api/v1/debug/latest` | `GET` | Retrieve latest execution trace JSON |
| `/api/v1/debug/{trace_id}` | `GET` | Retrieve specific execution trace JSON by ID |
| `/api/v1/metrics` | `GET` | Real-time system telemetry and process RAM RSS |
