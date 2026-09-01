# Phase 47A — Railway Deployment & Infrastructure Audit

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 47A — Infrastructure Audit  
**Date:** 2026-09-01  

---

## 1. Service Topology

- **Frontend Service:** Next.js (Static + SSR) on Node.js.
- **Backend Service:** FastAPI + Uvicorn on Debian Linux (Python 3.10).
- **Worker Configuration:** Strictly 1 Uvicorn worker process (`workers=1`).
- **Memory Ceiling:** 1024 MB container limit.
- **Threading Restrictions:**
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `TOKENIZERS_PARALLELISM=false`
  - `torch.set_num_threads(1)`

---

## 2. Docker & Entrypoint Verification

- Root `Dockerfile`: Two-stage builder installing CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`).
- Root `start.py`: Boots FastAPI application with proxy headers and `$PORT` binding.
- Health Check: Validates `/health` with `memory_mb`, `commit_sha`, `uptime_seconds`, and `nli_singleton`.
