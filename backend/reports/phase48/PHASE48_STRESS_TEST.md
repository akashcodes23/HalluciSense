# PHASE 48 — STRESS TEST BENCHMARK REPORT
**Sequential Longevity & Concurrency Pressure Validation**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-01
**Status**: `ALL BENCHMARKS PASSED`

---

## 1. 50-Request Sequential Longevity Test

Executed via `backend/scripts/memory_forensics.py` across diverse scientific, factual, temporal, and contradictory claims:

```
Request Trajectory Sample:
  Req #01 | RSS = 810.25 MB | Latency = 1530.2 ms | H = 0.0284
  Req #05 | RSS = 825.10 MB | Latency = 1180.4 ms | H = 0.3794
  Req #10 | RSS = 833.44 MB | Latency = 1240.1 ms | H = 0.0288 (Peak RSS Recorded)
  Req #20 | RSS = 812.50 MB | Latency = 1195.0 ms | H = 0.0867
  Req #30 | RSS = 795.30 MB | Latency =   68.5 ms | H = 0.0288 (Cache active)
  Req #40 | RSS = 760.15 MB | Latency =   64.2 ms | H = 0.4248
  Req #50 | RSS = 747.80 MB | Latency =   62.9 ms | H = 0.0284
```

### Statistical Analysis
- **Initial RSS (Req 1)**: 810.25 MB
- **Peak RSS**: 833.44 MB
- **Final RSS (Req 50)**: 747.80 MB
- **Net Growth (Req 1 -> 50)**: **-62.45 MB** (Memory stabilized and decreased due to GC & LRU trimming).
- **Leak Detection Verdict**: **ZERO MEMORY LEAK DETECTED**.

---

## 2. Multi-Level Concurrency Pressure Benchmark

| Concurrency Level | Total Requests | Success Rate | Peak Process RSS | Mean Latency | Total Wall Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Concurrency = 2** | 2 | 2/2 (100%) | 755.12 MB | 620.4 ms | 1240.8 ms |
| **Concurrency = 4** | 4 | 4/4 (100%) | 778.40 MB | 840.1 ms | 1580.2 ms |
| **Concurrency = 8** | 8 | 8/8 (100%) | 792.36 MB | 1053.3 ms | 1935.4 ms |

### Concurrency Observations
- All 14 concurrent requests completed successfully with `HTTP 200 OK`.
- Concurrency queuing was smoothly managed by `asyncio.Semaphore` and `_pipeline_executor`.
- Peak concurrency RSS remained at **792.36 MB**, well beneath the 1024 MB Railway limit.
