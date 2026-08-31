# HalluciSense Memory Engineering & OOM Resolution Case Study

## 1. Context & Operational Constraints

- **Deployment Platform**: Railway Linux Containers (x86_64, cgroup v2).
- **Container Hard Memory Limit**: `1024 MB` (1.0 GB).
- **Core Workload**: Real-time NLI cross-encoding (`cross-encoder/nli-deberta-v3-small`), dense semantic embeddings (`all-MiniLM-L6-v2`), and tabular gradient boosting (`HistGradientBoostingClassifier`).

Heavy transformer models typically allocate large dynamic memory buffers during PyTorch weight deserialization. In a constrained 1024 MB environment, improper allocation strategies lead directly to Linux kernel Out-Of-Memory (`SIGKILL` / Exit 137) terminations.

---

## 2. Forensic Timeline of the Memory Regression

| Stage / Deployment | Configuration | Measured Peak RSS | Headroom vs 1024 MB | Incident Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Initial Working Gold (`5b4c5a29`)** | Default `pymalloc` + Singleton NLI | `972 MB` | `+52 MB (5.1%)` | Stable, but constrained |
| **Speculative Hardening (`78c445a`)** | Added `PYTHONMALLOC=malloc` | **`1.22 GB` (1,249 MB)** | **`-225 MB (Breach)`** | **CRASHED (SIGKILL / Exit 137)** |
| **Phase 33 Allocator Fix (`41dbfa06`)** | Removed `PYTHONMALLOC=malloc`, kept `MALLOC_TRIM_THRESHOLD_=65536` | **`774 MB`** | **`+250 MB (24.4%)`** | **ONLINE & STABLE** |
| **Phase 34 Stress Test (`41efbc6e`)** | 2-Request Concurrent Load | **`832 MB`** | **`+192 MB (18.75%)`** | **ONLINE & STABLE** |

---

## 3. Deep Technical Root Cause: Why `PYTHONMALLOC=malloc` Failed

### A. How Python's Default `pymalloc` Operates
Python uses a specialized small-object allocator called `pymalloc` for allocations $\le 512$ bytes:
- Divides memory into 256 KB arenas, subdividing them into 4 KB pools with fixed-size blocks (8, 16, 24, ..., 512 bytes).
- Suballocations within pools have **zero per-object malloc metadata overhead**.

### B. The Glibc Malloc Overhead Explosion
When `PYTHONMALLOC=malloc` is forced via environment variables:
1. Python's `pymalloc` is completely disabled. Every single integer, string, tuple, PyTorch tensor header, and dictionary entry is routed directly to glibc `malloc()`.
2. Glibc `malloc()` attaches an **8-to-16 byte chunk header** and enforces 16-byte address alignment for every allocation.
3. During HuggingFace DeBERTa weight loading (`AutoModelForSequenceClassification.from_pretrained`), the tokenizer and deserializer instantiate hundreds of thousands of small Python strings and configuration dictionaries.
4. Under glibc `malloc`, this created an immediate **~25-30% memory inflation** (~277 MB increase), pushing total heap usage from 972 MB to 1,249 MB and triggering an immediate kernel OOM kill.

---

## 4. Multi-Layered Architectural Defense in Production

To permanently guarantee stability below 1024 MB, HalluciSense implements five synergistic memory safeguards:

### 1. Allocator Tuning in `backend/Dockerfile`
```dockerfile
ENV MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=65536 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
```
- `MALLOC_ARENA_MAX=2`: Prevents glibc from spawning 8 arenas per CPU core, restricting heap sprawl.
- `MALLOC_TRIM_THRESHOLD_=65536`: Forces glibc to release free memory chunks larger than 64 KB back to the operating system immediately via `madvise(MADV_DONTNEED)`.

### 2. PyTorch CPU Thread Confinement in `backend/app/main.py`
```python
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```
Restricting PyTorch CPU parallelism to a single thread prevents thread-local tensor allocation buffers from multiplying across vCPUs.

### 3. Thread-Safe Model Singleton in `backend/app/core/engine/model_registry.py`
Ensures that heavy ML models are instantiated exactly once:
- `ModelRegistry._nli_model`: Shared singleton DeBERTa model (`init_count = 1`).
- `SentenceTransformer` & Reranker: Lazy-loaded only when requested.

### 4. Bounded Concurrency Semaphore
```python
@classmethod
def get_nli_semaphore(cls, max_concurrent: int = 2) -> threading.Semaphore:
    if cls._nli_semaphore is None:
        with cls._lock:
            if cls._nli_semaphore is None:
                cls._nli_semaphore = threading.Semaphore(max_concurrent)
    return cls._nli_semaphore
```
Restricts simultaneous heavy NLI matrix multiplications to 2 concurrent workers, keeping peak concurrent RSS capped at **832 MB**.

### 5. Lazy Core Pipeline Proxy
`backend/app/core/pipeline.py` wraps the master orchestrator in a lightweight proxy that defers sub-engine instantiation until first access, eliminating duplicate memory allocations at startup.
