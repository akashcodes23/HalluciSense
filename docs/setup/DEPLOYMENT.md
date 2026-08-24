# Production Deployment Guide (Railway & Cloud Infrastructure)

This document describes the production deployment topology, persistent volume strategy, and environment configurations for HalluciSense on Railway.

---

## 1. Production Architecture Overview

```
                   Internet / Users
                          │
                          ▼
            Railway Edge HTTPS Router
                          │
       ┌──────────────────┴──────────────────┐
       ▼                                     ▼
Frontend Service                      Backend Service
(Next.js 16 SSR/Static)               (FastAPI ASGI)
Port 3000                             Port 8000
       │                                     │
       │                                     ▼
       └─────────────────────────────► Persistent Volume
                                       Mounted at: /data
                                       ├── traces/
                                       ├── cache/
                                       └── faiss/
```

---

## 2. Railway Service Configuration

### Backend Service (FastAPI)
- **Runtime**: Python 3.10
- **Start Command**:
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 75
  ```
- **Persistent Volume Mount**: `/data` (10 GB)
- **Health Check Path**: `/health`
- **Restart Policy**: `ON_FAILURE` (max 5 retries)

### Frontend Service (Next.js)
- **Runtime**: Node.js 20
- **Build Command**: `npm run build`
- **Start Command**: `npm start`
- **Environment Variables**:
  - `NEXT_PUBLIC_API_BASE_URL`: `https://hallucisense-production.up.railway.app`
  - `NODE_ENV`: `production`

---

## 3. Environment Variables & Security Configuration

| Variable | Recommended Production Value | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `production` | Enables JSON logging and error sanitization |
| `PORT` | `8000` (Injected by Railway) | Binding port |
| `CORS_ORIGINS` | `https://*.railway.app,http://localhost:3000` | Allowed origins |
| `RATE_LIMIT_PER_MINUTE`| `100` | In-memory token bucket rate limit |
| `GEMINI_API_KEY` | `[SECURE_VAULT]` | API key for closed-loop chat generator |
| `OPENAI_API_KEY` | `[SECURE_VAULT]` | Optional external comparator key |

---

## 4. Production Smoke Testing Verification

After deployment, run the following verification sequence:
```bash
# 1. Health Probe
curl -I https://hallucisense-production.up.railway.app/health

# 2. Readiness Probe
curl -I https://hallucisense-production.up.railway.app/ready

# 3. Analyze Pipeline
curl -X POST https://hallucisense-production.up.railway.app/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of Karnataka?", "response": "The capital of Karnataka is Bengaluru."}'
```
