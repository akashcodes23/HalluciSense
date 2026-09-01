# Phase 47A — Production Baseline & Anomaly Root Cause Analysis

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 47A — Production Runtime Recovery & Root Cause Diagnostic  
**Date:** 2026-09-01  
**Deployed Head:** `ab1659f`  

---

## 1. Executive Incident Summary

In production on Railway, traces previously indicated:
- Pillar 2: "Token logprobs not available" (`unavailable`)
- Pillar 3: "Multi-generation not available for static input" (`unavailable`)
- Root Cause: `Entity Linking Failure`
- High uncalibrated H-score anomalies.
- Frequent process restarts / OOM (Exit 137) events.

---

## 2. Root Cause Breakdown

### A. Missing Root Dockerfile & Start Entrypoint
1. `railway.toml` referenced `dockerfilePath = "Dockerfile"` and `startCommand = "python start.py"`.
2. `Dockerfile` and `start.py` were historically placed under `backend/` and `docker/`, rather than the project root. Railway fell back to the Nixpacks runtime or default container configuration, which lacked CPU-only PyTorch optimization and proper worker caps.
3. Fix: Created `/Dockerfile` with CPU-only PyTorch and `/start.py` binding dynamically to `$PORT` with `workers=1`.

### B. Legacy Model Duplication
1. `backend/app/modules/verification/router.py` was instantiating fresh `HallucinationDetectionPipeline()` and `HybridRetriever()` per invocation on legacy `/verification/verify-text` calls.
2. `backend/evaluation/phase6l/pairwise_nli.py` was initializing a separate SentenceTransformer instance outside `ModelRegistry`.
3. Fix: Routed all model invocations through the `ModelRegistry` singleton and configured single-thread NLI concurrency.

### C. Static P2/P3 Activation
1. Pillar 2 now executes `STATIC_VERIFICATION_CONFIDENCE` using evidence coverage, NLI margin, and retrieval certainty.
2. Pillar 3 now executes `SINGLE_CLAIM_CONSISTENCY` ($CF = 0.0$) and `INTRA_RESPONSE_CONSISTENCY` (pairwise sentence embeddings + DeBERTa NLI contradictions bounded to 15 claims).
3. Old fallback strings ("Token logprobs not available" and "Multi-generation not available for static input") have been removed from the active execution path.
