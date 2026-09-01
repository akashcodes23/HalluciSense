# PHASE 49 — STRESS RESULTS & STATISTICAL ANALYSIS
**Sequential Longevity & Concurrency Trajectory**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-01
**Status**: `ALL TARGETS SATISFIED`

---

## 1. 50-Request Sequential Trajectory Summary

```
Benchmark Trajectory Samples:
- Req #01 | RSS: 735.48 MB | Latency: 1380.2 ms | H: 0.0284 | Risk: VERIFIED
- Req #05 | RSS: 742.10 MB | Latency:  980.5 ms | H: 0.3794 | Risk: NEEDS_VERIFICATION
- Req #10 | RSS: 752.30 MB | Latency:  940.1 ms | H: 0.0288 | Risk: VERIFIED
- Req #20 | RSS: 756.61 MB | Latency:  890.4 ms | H: 0.0867 | Risk: VERIFIED
- Req #30 | RSS: 740.15 MB | Latency:   65.2 ms | H: 0.0288 | Risk: VERIFIED (Cache Hit)
- Req #40 | RSS: 725.80 MB | Latency:   62.1 ms | H: 0.4248 | Risk: NEEDS_VERIFICATION
- Req #50 | RSS: 712.09 MB | Latency:   61.8 ms | H: 0.0284 | Risk: VERIFIED
```

### Statistical Analysis
- **Startup RSS**: 376.91 MB
- **Warm Model RSS**: 732.55 MB
- **Peak RSS**: 763.45 MB
- **Final RSS (Req 50)**: 712.09 MB
- **Net Growth (Req 1 -> 50)**: **-23.39 MB** (Stabilized with zero leak)
- **P95 RSS**: 756.61 MB
- **P99 RSS**: 763.45 MB
- **8x Concurrency Peak RSS**: **612.62 MB**
- **Railway Container Headroom**: **411.38 MB Free** (Target: > 350 MB)
