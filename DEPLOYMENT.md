# HalluciSense v1.0 Production Deployment Guide

Guide for deploying **HalluciSense v1.0** to Railway, Docker Compose, or local production environments.

---

## 1. Quick Start with Docker Compose

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

## 2. Railway Deployment Protocol

1. Link your GitHub repository in the Railway Dashboard.
2. Configure Environment Variables:
   - `APP_ENV=production`
   - `PORT=8000`
   - `GEMINI_API_KEY=your_gemini_api_key_here`
   - `DATABASE_URL=postgresql://user:pass@host:5432/dbname`
3. Railway start command:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 4
   ```

---

## 3. Endpoints Reference

| Endpoint | Method | Purpose |
| :--- | :---: | :--- |
| `/` | `GET` | Service info and environment status |
| `/health` | `GET` | System liveness probe |
| `/ready` | `GET` | Deep component readiness probe |
| `/docs` | `GET` | Interactive Swagger OpenAPI UI |
| `/api/v1/analyze` | `POST` | **Canonical HalluciSense Verification API** |
