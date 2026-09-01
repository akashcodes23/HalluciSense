# Phase 38.11 — Railway Build & Runtime Forensic Audit

**Repository:** akashcodes23/HalluciSense  
**Project:** `passionate-contentment` (`2c0fdad7-7765-475c-a41a-7315afb700b7`)  
**Environment:** `production` (`b69f4974-053f-4f1f-bbf8-68991e501f39`)  
**Date:** 2026-09-01  

---

## 1. Project Services Topology

The Railway project contains two decoupled microservices:

| Service Name | Service ID | Role / Technology | Current Status | Deployment ID |
|---|---|---|---|---|
| **HalluciSense** | `a449c886-d20f-4eb3-b461-81cb5b9944ea` | Backend API (Python 3.11 / FastAPI / Sklearn / PyTorch) | **● Online / SUCCESS** | `1e3d7963-a3ab-4dbe-99b0-268d8823467f` |
| **enchanting-wonder** | Auxiliary Service | Frontend Web Client (Node.js / Next.js 16) | **● Online / Ready** | Container Active |

---

## 2. Backend Build & Runtime Forensics (`HalluciSense`)

### Docker Configuration
- **Base Image:** `python:3.11-slim`
- **Builder Pattern:** 2-stage multi-stage build:
  - Stage 1 (`builder`): Compiles wheel artifacts into `/install` using `build-essential`, `gcc`, `libpq-dev`.
  - Stage 2 (`runtime`): Copies compiled wheels from Stage 1 into `/usr/local`, resulting in a lean runtime container without build dependencies.
- **Entrypoint:** `CMD ["python", "start.py"]`
- **Port Binding:** Binds dynamically to Railway `$PORT` via `uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, workers=1)`.

### Memory & Health Telemetry
- **Railway Container Memory Limit:** 1024 MB
- **Measured Cold RSS:** 528.8 MB
- **Steady-State RSS:** 538.0 MB
- **Concurrent Load Peak (2 requests):** 539.6 MB
- **Exit 137 / SIGKILL Count:** **0** (All Phase 33+ deployments remain crash-free)
- **Health Probes:**
  - `GET /health` responds immediately with memory telemetry.
  - `GET /ready` reports HTTP 200 once background pipeline warmup completes.

---

## 3. Frontend Build Analysis (`enchanting-wonder`)

- **Runtime:** `Next.js 16.2.11` on Node.js.
- **Build Command:** `next build` $\to$ `next start -p 8080`.
- **Local Static Verification:** `npm run build` generates 23 static pages with **0 TypeScript errors**.
- **Deploy Alert Root Cause:** Occasional Railway deploy notification alerts were caused by frontend port resolution / start timeout race conditions during simultaneous dual-service repository commits, whereas the backend service has consistently built and deployed successfully.

---

## 4. Operational Guarantees

1. **Deterministic Single-Worker Topology:** Confirmed strictly 1 Uvicorn worker process.
2. **Thread Safety:** OpenMP, MKL, BLAS, PyTorch CPU threads, and Tokenizer parallelism are capped to 1.
3. **No Memory Leaks:** 162 adversarial evaluations executed with zero memory drift beyond baseline.
