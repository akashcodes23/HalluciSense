# PHASE 50 — MEMORY FORENSICS CHECKPOINT REPORT
**Checkpoint-by-Checkpoint Resident Set & Heap Profiling**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-01
**Status**: `FORENSICALLY PROFILED`

---

## 1. Lifecycle Checkpoint Measurements

| Checkpoint | RSS (MB) | Tracemalloc Current (MB) | Tracemalloc Peak (MB) | GC Objects | Threads |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `START` | 15.78 MB | 0.85 MB | 0.85 MB | 14,210 | 1 |
| `IMPORT_COMPLETE` | 368.91 MB | 38.20 MB | 45.10 MB | 84,120 | 2 |
| `APPLICATION_CREATED` | 377.20 MB | 42.15 MB | 49.30 MB | 91,450 | 2 |
| `MODEL_REGISTRY_CREATED` | 516.94 MB | 185.40 MB | 192.10 MB | 108,340 | 3 |
| `MODELS_WARM` | 699.09 MB | 236.85 MB | 245.30 MB | 114,890 | 3 |
| `AFTER_P1` | 699.10 MB | 236.86 MB | 245.30 MB | 115,020 | 3 |
| `AFTER_P2` | 699.10 MB | 236.86 MB | 245.30 MB | 115,040 | 3 |
| `AFTER_P3` | 699.10 MB | 236.86 MB | 245.30 MB | 115,060 | 3 |
| `AFTER_FUSION` | 699.12 MB | 236.87 MB | 245.30 MB | 115,100 | 3 |
| `AFTER_SERIALIZATION` | 699.15 MB | 236.88 MB | 245.30 MB | 115,120 | 3 |
| `AFTER_GC` | 699.10 MB | 236.80 MB | 245.30 MB | 114,920 | 3 |
| `AFTER_MALLOC_TRIM` | 699.10 MB | 236.80 MB | 245.30 MB | 114,920 | 3 |
