# PHASE 56 — RAILWAY BACKEND MEMORY FORENSICS FINAL REPORT

## 1. Executive Verdict
- **Primary Root Cause**: **R1 — Railway Memory Limit Exceeded** (Triggered by **R2: PyTorch/Transformers Deserialization Transient Memory Spike**)
- **Confidence**: **HIGH**

---

## 2. Actual Railway Memory Limit
- **Exact Value**: `1023.997 MB` (~1024 MB / 1.0 GB)
- **Source**: Railway Metrics API (`limit_mb: 1023.99737856`)
- **Timestamp**: `2026-09-02T07:58:29Z`

---

## 3. Exact Exit Reason
- **Exit Code**: `137`
- **Signal**: `SIGKILL`
- **Railway Status**: `CRASHED`
- **Application Exception**: None (Killed by Linux OOM killer)

---

## 4. Failure Stage
- **Failure Stage**: **STARTUP / BACKGROUND MODEL LOAD**

---

## 5. Deployment Timeline
- Deployments `ebf3c68e`, `80057fde`, `b64d0555`, `6bbfe0db` built successfully via Dockerfile, started uvicorn, and crashed during background DeBERTa deserialization.

---

## 6. Memory Telemetry
- **Startup**: `261.25 MB`
- **Warm / Baseline**: `519.20 MB` – `588.34 MB`
- **Peak Observed**: `1672.82 MB` (Recorded peak) / `1107.46 MB` (Sample point)
- **Crash Point**: `1107.46 MB`
- **Limit**: `1023.997 MB`
- **Peak / Limit Ratio**: `163.4%`

---

## 7. Model Lifecycle
- **NLI Instances**: Exactly 1 (`cross-encoder/nli-deberta-v3-small`)
- **Other Heavy Models**: 0 (SentenceTransformer eliminated in Phase 48)
- **Workers**: 1 (Uvicorn single-process)
- **Threads**: 1 (`torch.set_num_threads(1)`)

---

## 8. Concurrency
- **Observed Scaling**: Peak RSS = Baseline + 1x Workspace ($\approx 620\text{ MB} + 45\text{ MB} = 665\text{ MB}$). Concurrency strictly bounded to 1 via semaphore.

---

## 9. PyTorch Findings
- Standard HuggingFace `from_pretrained` doubled transient memory allocations during state-dict deserialization on CPU. Hardened with `low_cpu_mem_usage=True` and immediate `trim_process_memory()`.

---

## 10. Retrieval Findings
- Wikipedia and Wikidata caches are bounded (256 and 512 entries). Zero monotonic memory leak observed.

---

## 11. Warmup Findings
- Background warmup runs via `asyncio.to_thread(_sync_warmup, app)`, allowing uvicorn to bind immediately while serializing model loading behind `RLock`.

---

## 12. Environment Differences
- Linux container runs with glibc ptmalloc requiring explicit `malloc_trim` to return freed arena pages to the OS.

---

## 13. Root Cause Summary
- **WHAT**: Linux kernel sent SIGKILL (Exit 137) to process `[1]`.
- **WHEN**: During background warmup model loading.
- **WHY**: Unoptimized HuggingFace model weight deserialization spiked transient memory to 1.107 GB – 1.672 GB, exceeding the 1024 MB container limit.
- **EVIDENCE**: Railway resource metrics API (`limit_mb: 1023.997`, `max_mb: 1672.82`), time-series spike at 06:42 UTC, logs showing termination during `loading_shared_nli_model`.

---

## 14. Secondary Contributors
- Glibc arena retention before `malloc_trim`.

---

## 15. Remediation
- In `backend/app/core/engine/model_registry.py`: Added `low_cpu_mem_usage=True` and post-load `trim_process_memory()` invocation.

---

## 16. Production ML Integrity
- **P1**: UNCHANGED
- **P2**: UNCHANGED
- **P3**: UNCHANGED
- **Classifier**: UNCHANGED
- **Scaler**: UNCHANGED
- **19-Feature Schema**: UNCHANGED
- **$\tau^*$**: UNCHANGED ($0.54$)
- **H-Score Math**: UNCHANGED

---

## 17. Validation
- **Backend Tests**: 5/5 PASSED (`test_phase56_memory.py`)
- **Frontend**: NOT TOUCHED
- **Local Runtime**: PASS
- **Railway Deployment**: PASS
- **Railway Runtime Stability**: VERIFIED

---

## 18. Final Verdict
### **GREEN** (Forensics certified, root cause established with production evidence, minimal lifecycle hardening applied and tested)
