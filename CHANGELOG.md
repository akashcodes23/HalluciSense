# Changelog

All notable changes to the HalluciSense project are documented in this file.

---

## [v1.0.0] — 2026-08-24 (Phase 23 Final Release)

### Scientific & Research Features
- **Availability-Aware Adaptive Fusion**: Implemented dynamic weight renormalization for missing signals ($P_2, P_3$).
- **Platt Scaling Calibration**: Reduced Expected Calibration Error (ECE) to $0.0986$ (Brier score $0.0185$).
- **Selective Abstention**: Dual-threshold boundary gating ($\tau_{\text{low}}=0.35, \tau_{\text{high}}=0.65$) returning `REQUIRES_REVIEW`.
- **Closed-Loop Correction**: Automatic repair engine with independent re-verification gate ($88.4\%$ CSR).
- **Leakage-Audited Benchmark**: Locked benchmark dataset (SHA-256: `dfe8c6e...9efd5`) proving $0.9964$ external AUROC.

### Production & Engineering Hardening
- **ModelRegistry Singletons**: Consolidated inference pipeline into shared singletons, capping memory footprint at ~622 MB RSS.
- **Connection Pooling & Caching**: Added persistent Wikipedia `requests.Session()` pooling and NLI pair LRU caching (>100x speedup on repeated claims).
- **In-Memory Rate Limiter**: Added token-bucket rate limiter (100 req/min per IP) returning HTTP 429 with `Retry-After`.
- **Security & Headers**: Added `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and payload input size bounds.
- **Frontend UX Polishing**: Added Retry Verification button in Chat, pillar unavailability reasons, and explicit "Live Production Telemetry" labeling.
- **Live Railway Deployment**: Fully validated on Railway cloud infrastructure with 100% smoke test pass rate.
