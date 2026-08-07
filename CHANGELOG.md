# Changelog

All notable changes to HalluciSense are documented in this file.

## [1.0.0-sprint1] - 2026-08-07

### Added
- **Production Metrics Tracker (`MetricsTracker`)**: Thread-safe telemetry engine recording total request counts, latency accumulator, H-score accumulator, success/error rates, and process RAM memory.
- **Canonical API Endpoints**:
  - `POST /api/v1/analyze`: Returns standardized `trace_id`, `overall_h_score`, `risk_level`, `confidence`, `pillar_scores`, `failure_taxonomy`, `processing_time_ms`, and `version`.
  - `POST /api/v1/explain`: Returns full evidence explanations, supporting/contradictory passages, token heatmaps, sentence scores, reasoning chains, and adaptive weights.
  - `GET /api/v1/metrics`: Production metrics computed from real runtime statistics.
- **Deep Readiness Probe (`GET /ready`)**: Verifies `HybridRetriever`, NLI model (`cross-encoder/nli-deberta-v3-small`), CrossEncoder reranker (`ms-marco-MiniLM-L-6-v2`), SentenceTransformer (`all-MiniLM-L6-v2`), and `FusionEngine`.
- **Centralized Structured Exception Handlers**: Enforces structured JSON error formats for validation (422), bad request (400), payload too large (413), and system errors (500) without exposing Python stack traces.
- **Payload Validation**: Strict Pydantic model validation and 100KB body limit check.

### Changed
- Refactored `production_router.py` and `main.py` to route all endpoints through the single master `HallucinationDetectionPipeline` singleton.
- Updated production test suite `test_production_api.py` covering unit, integration, regression, and stress test scenarios.
