# PHASE 48 — PROCESS AUDIT & MEMORY ARENA ANALYSIS
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-01
**Status**: `AUDITED & CERTIFIED`

---

## 1. Process Architecture & Thread Topology

```
+-----------------------------------------------------------------------+
| Single Uvicorn Process (PID: Master)                                  |
|                                                                       |
|  [FastAPI AsyncIO Event Loop]                                         |
|    |                                                                  |
|    +--> Request Rate Limiter (Token Bucket with In-Memory Cleanups)   |
|    +--> Concurrency Guard (AsyncIO Semaphore, Max 4 Slots)            |
|    |                                                                  |
|    +--> [Bounded ThreadPoolExecutor: max_workers = 2]                |
|           |                                                           |
|           +--> Thread Worker 1: Master Pipeline Analysis              |
|           |      ├── Pillar 1: Evidence Grounding + Wikidata Client   |
|           |      ├── Pillar 2: Token Entropy Calculation             |
|           |      └── Pillar 3: Jaccard + Shared DeBERTa NLI          |
|           |                                                           |
|           +--> Thread Worker 2: Standby / Parallel Analysis Slot      |
|                                                                       |
|  [Global Memory Hooks]                                                |
|    +--> Explicit Garbage Collection (gc.collect())                    |
|    +--> Glibc Heap Release (malloc_trim(0))                           |
+-----------------------------------------------------------------------+
```

---

## 2. Glibc Allocation Arena Audit

Under default glibc behavior on multi-core Linux systems, each newly spawned OS thread creates a dedicated memory arena of up to 64 MB (on 64-bit systems). 

When requests dynamically spawn uncontrolled threads via `asyncio.to_thread()`, glibc creates up to $8 \times \text{num\_cores}$ arenas, causing virtual memory (VMS) explosion and heap fragmentation that prevents RSS memory from ever returning to the operating system.

### Remediation Implemented
1. **Bounded Thread Pool**: Defined `_pipeline_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="hs_pipeline_worker")`. The worker threads are persistent and reused indefinitely, bounding glibc thread arenas to $\le 2$.
2. **Environment Variable**: `MALLOC_ARENA_MAX=2` configured in Docker / Railway start command.
3. **Memory Trimming Hook**: `trim_process_memory()` calls `malloc_trim(0)` on Linux after request execution to release top-of-heap memory back to the container OS.
