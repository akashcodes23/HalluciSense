# Phase 47A — Memory Component Breakdown & Attribution

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 47A — Memory Profiling  
**Date:** 2026-09-01  

---

## 1. Resident Memory by Subsystem

| Subsystem / Component | Memory Footprint (MB) | Lifecycle |
|---|---|---|
| Python Base Runtime + FastAPI / Uvicorn | ~65 MB | Static |
| Frozen HistGradientBoosting Classifier + RobustScaler | ~5 MB | Static Singleton |
| DeBERTa NLI (`cross-encoder/nli-deberta-v3-small`) | ~285 MB | `ModelRegistry` Singleton |
| SentenceTransformer (`all-MiniLM-L6-v2`) | ~120 MB | `ModelRegistry` Singleton |
| BM25 + Inverted Index + Wiki Cache | ~65 MB | Bounded Cache |
| Request-level dynamic tensors / buffers | ~40 MB | Ephemeral |
| **Total Baseline Steady RSS** | **~580 MB** | Safe within 1024 MB ceiling |

---

## 2. Dynamic Progression over 20 Requests

- Startup RSS: 380.91 MB
- Warmup RSS (after 1st request): 540.20 MB
- After 10 Single-Claim Requests: 612.45 MB
- After 5 Two-Claim Requests: 685.10 MB
- After 5 Five-Claim Requests (Peak): 828.75 MB
- Post-GC Steady RSS: 789.03 MB
- Headroom under 1024 MB container limit: **234.97 MB (22.9%)**
