# HalluciSense Production Deployment Guide

Complete operational guide for deploying, managing, scaling, and maintaining HalluciSense in production cloud environments.

---

## 1. System Architecture Overview

```
[ Next.js 16 Web UI ]  (Vercel Edge Network)
         │
         ▼  HTTPS / REST API
[ FastAPI Backend Engine ]  (Railway Cloud Container)
   ├── Dynamic Claim Extractor
   ├── Knowledge Retrieval Engine (Wikipedia + BM25 + FAISS + Cross-Encoder)
   ├── Pillar-1 (Evidence Consistency Engine)
   ├── Pillar-2 (Structural NLI Matrix Engine)
   ├── Hybrid Fusion Engine (SET_A_FULL_HYBRID - 19 Features)
   └── Explainability & Attribution Engine
         │
         ├── PostgreSQL 16 (Database / User & Analytics State)
         └── Redis 7 (Async Task Queue & Cache)
```

---

## 2. Environment Variables Reference

### Backend (`backend/.env`)

| Variable Name | Required | Default Value | Description |
| :--- | :---: | :--- | :--- |
| `APP_ENV` | Yes | `production` | Target deployment environment |
| `SECRET_KEY` | Yes | *Required* | 32+ char secret for JWT authentication |
| `PORT` | Yes | `8000` | Dynamic HTTP server binding port |
| `DATABASE_URL` | Yes | *Required* | PostgreSQL connection string (`postgresql+asyncpg://...`) |
| `REDIS_URL` | Yes | *Required* | Redis cache/broker URI (`redis://...`) |
| `CORS_ORIGINS` | Yes | *Vercel URL* | Comma-separated allowed web client domains |

### Frontend (`frontend/.env.local`)

| Variable Name | Required | Description |
| :--- | :---: | :--- |
| `NEXT_PUBLIC_API_URL` | Yes | Target Railway backend API URL (`https://...up.railway.app`) |

---

## 3. Step-by-Step Deployment Instructions

### Step 1: Railway Backend Deployment

1. Install Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```
2. Authenticate & initialize project:
   ```bash
   railway login
   railway init
   ```
3. Deploy backend:
   ```bash
   railway up
   ```
4. Attach PostgreSQL and Redis add-ons via Railway Dashboard.

### Step 2: Vercel Frontend Deployment

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```
2. Deploy from `frontend/` directory:
   ```bash
   cd frontend
   vercel --prod
   ```
3. Set `NEXT_PUBLIC_API_URL` in Vercel project environment settings.

---

## 4. Disaster Recovery & Rollback Procedure

- **Database Backups**: Automatic daily snapshots configured on Supabase / Railway PostgreSQL.
- **Rollback Command (Backend)**:
  ```bash
  railway rollback
  ```
- **Rollback Command (Frontend)**:
  Promote previous instant deployment via Vercel Dashboard or:
  ```bash
  vercel rollback [DEPLOYMENT_ID]
  ```

---

## 5. Scaling & Maintenance Guide

- **Horizontal Pod Autoscaling**: Railway automatically scales container instances when CPU exceeds 80%.
- **Zero-Downtime Migration**: Alembic async schema migrations execute automatically during start command.
