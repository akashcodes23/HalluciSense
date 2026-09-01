# Phase 39.19 — Memory Safety & Model Concurrency Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 39.19 — Semantic Grounding Memory Safety Verification  
**Container Limit:** 1024 MB  
**Date:** 2026-09-01  

---

## 1. Model Lifecycle & Singleton Guarantee

| Model Name | Physical Role | Initialization Counter | Shared Across Subsystems | In-Memory RSS |
|---|---|---|---|---|
| `cross-encoder/nli-deberta-v3-small` | Claim ↔ Evidence NLI & Pairwise Consistency | `ModelRegistry._init_counts['nli_model'] = 1` | ✅ Shared between Pillar 1 and Pillar 2 | ~280 MB |
| `all-MiniLM-L6-v2` | Dense Semantic Embeddings | `ModelRegistry._init_counts['sentence_transformer'] = 1` | ✅ Pillar 2 similarity | ~90 MB |
| `HistGradientBoostingClassifier` | Hybrid Decision Fusion | Lazy singleton | ✅ Orchestrator | ~218 KB |
| `RobustScaler` (19 features) | Feature Scaling | Lazy singleton | ✅ Orchestrator | ~799 bytes |

---

## 2. Memory Telemetry Across Load Profiles

| Execution Phase | Measured RSS | Phase 38 Baseline | Delta | Safety Margin (vs 1024 MB) |
|---|---|---|---|---|
| **Cold Startup** | **528.8 MB** | 528.8 MB | +0.0 MB | 48.4% (495.2 MB) |
| **Steady State Post-Warmup** | **538.0 MB** | 538.0 MB | +0.0 MB | 47.5% (486.0 MB) |
| **10 Sequential Requests** | **538.2 MB** | 538.16 MB | +0.04 MB | 47.4% (485.8 MB) |
| **2 Concurrent Requests** | **539.6 MB** | 539.56 MB | +0.04 MB | 47.3% (484.4 MB) |
| **202 Golden Cases Run** | **541.2 MB** | 540.8 MB | +0.4 MB | 47.1% (482.8 MB) |

---

## 3. Concurrency & Allocator Controls

- **Semaphore Bound:** `ModelRegistry.get_nli_semaphore(max_concurrent=2)` ensures that simultaneous HTTP requests queue gracefully rather than triggering concurrent PyTorch memory spikes.
- **Worker Topology:** Strictly 1 Uvicorn worker process.
- **Allocator Settings:** `MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_=65536` actively return freed memory to the OS.
- **Exit 137 / SIGKILL Count:** **0**.
