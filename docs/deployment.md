# Production Cloud Deployment Guide

Deploy **HalluciSense** to production platforms (Railway, AWS ECS, GCP Cloud Run, Kubernetes).

---

## 1. Railway Platform Deployment (PaaS)

HalluciSense includes native `railway.toml` configuration:

```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "backend/Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

Deploy in one command via Railway CLI:
```bash
railway up
```

---

## 2. Docker Compose Local & Server Launch

```bash
docker-compose up -d --build
```
This boots the FastAPI backend container (`:8000`) and Next.js frontend web app (`:3000`).
