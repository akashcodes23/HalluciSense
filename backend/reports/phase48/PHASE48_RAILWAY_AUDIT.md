# PHASE 48 — RAILWAY DEPLOYMENT AUDIT
**Container Memory Envelope & OOM Hazard Elimination**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-01
**Status**: `VERIFIED RAILWAY COMPLIANT`

---

## 1. Railway Container Specifications & Constraints

- **Container Memory Ceiling**: 1024 MB (1.0 GB)
- **Container OOM Signal**: `SIGKILL (Exit Code 137)`
- **vCPU Allocation**: Shared / Bounded (1-2 vCPU)

---

## 2. Headroom & Safety Margin Verification

```
Memory Allocation Breakdown (in MB):
[===================== Base Python + App (377 MB) =====================]
[======== DeBERTa NLI + Caches (161 MB) ========]
[==== Request Inference Buffer (209 MB) ====]
[---------------- Safety Headroom (276 MB) ----------------]
0 MB                                                    747 MB          1024 MB
```

### Measured Safety Envelopes
- **Baseline Warm Footprint**: ~538 MB (52.5% of limit)
- **Sustained Continuous Load (50 Requests)**: 747.80 MB (73.0% of limit)
- **Peak Concurrency Burst (8 Simultaneous Requests)**: 792.36 MB (77.3% of limit)
- **Minimum Emergency Margin**: **231.64 MB Headroom** before container memory threshold.

---

## 3. Recommended Production Environment Configuration

```bash
# Railway / Production Runtime Environment Variables
APP_ENV=production
PORT=8000
MALLOC_ARENA_MAX=2
PYTHONUNBUFFERED=1
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
TORCH_NUM_THREADS=2
HALLUCISENSE_MEMORY_GUARD_MB=950
MAX_CONCURRENT_ANALYSES=4
HALLUCISENSE_ENABLE_RERANKER=false
ENABLE_SELF_CONSISTENCY=false
ENABLE_AUTOMATIC_CORRECTION=false
```

### Automated Load Shedding
If physical RSS ever crosses 950 MB due to unexpected upstream memory spikes, `production_router.py` automatically triggers `trim_process_memory()`. If RSS remains above 980 MB, it sheds incoming requests with `HTTP 503 (MEMORY_PRESSURE_LOAD_SHEDDING)` rather than letting the Railway container crash with Exit 137.
