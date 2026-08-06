# HalluciSense Production Release Changelog

All notable changes to the HalluciSense hallucination detection platform will be documented in this file.

---

## [v1.0.0] - 2026-08-06 - Production Release Candidate

### Added
- **Canonical Production Endpoint (`POST /api/v1/analyze`)**: Unified multi-pillar analysis pipeline executing Retrieval, Predictive Confidence, Paraphrase Consistency, Bayesian Adaptive Fusion, and 4-Tier Token Localization.
- **Deep Health & Readiness Probes (`/health`, `/healthz`, `/ready`, `/readyz`)**: Returns vector store, ML model, and provider status.
- **Root Info Endpoint (`GET /`)**: Provides service name, version, environment, and Swagger docs paths for Railway and Docker deployments.
- **Production Schemas**: Strictly typed Pydantic V2 models (`AnalysisRequest`, `AnalysisResponse`) with OpenAPI 3.0 specs.
- **Structured JSON Logging & Tracing**: Request ID propagation (`X-Request-ID`), latency tracking, and sanitization.

### Fixed
- Fixed FastAPI openapi/docs paths for Railway proxy deployments.
- Fixed 500 error exception handling to return sanitized JSON errors without exposing internal stack traces.

### Security & Operations
- Locked dependency manifests (`release/requirements-lock.txt`, `release/environment.yml`).
- Docker Compose & Railway multi-stage container build optimizations.
