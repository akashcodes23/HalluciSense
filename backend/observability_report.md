# Sprint 8 — Production Observability & Telemetry Report

## Executive Summary

Sprint 8 instruments production telemetry across HalluciSense, including Prometheus `/metrics`, `/health`, `/ready` readiness probes, correlation IDs (`request_id`, `trace_id`), and latency histograms.

---

## 1. Exposed Telemetry & Metrics Endpoints

| Endpoint | Protocol | Purpose | Expected Status |
| :--- | :--- | :--- | :--- |
| `GET /health` | HTTP | Liveness probe for Kubernetes / Railway orchestrator | `200 OK` (`"status": "healthy"`) |
| `GET /api/v1/health/readiness` | HTTP | Readiness probe checking DB, Redis, and Gemini API | `200 OK` (`"ready": true`) |
| `GET /metrics` | HTTP | Prometheus metrics endpoint for Grafana dashboards | `200 OK` (Prometheus text format) |

---

## 2. Tracked Metrics & Histogram Summary

- **`hallucisense_llm_request_duration_seconds`**: Histogram tracking Gemini API latency.
- **`hallucisense_verification_pipeline_seconds`**: Histogram tracking 3-pillar verification processing time.
- **`hallucisense_redis_latency_seconds`**: Histogram tracking Upstash Redis cache GET/SET latency.
- **`hallucisense_db_query_seconds`**: Histogram tracking PostgreSQL query execution time.
- **`hallucisense_quota_circuit_breaker_tripped_total`**: Counter incremented on HTTP 429 quota events.
- **`hallucisense_llm_calls_per_request`**: Summary tracking LLM budget compliance (`llm_calls <= 1`).

---

## 3. Correlation ID & Distributed Tracing

Every log entry emits structured JSON fields:
```json
{
  "timestamp": 1785904658.65,
  "level": "info",
  "event": "LLM_EXECUTION_REPORT",
  "request_id": "req-1785904658650",
  "trace_id": "tr-1785904658650",
  "total_llm_calls": 1
}
```

---

*Report generated automatically by `scripts/run_observability_audit.py`.*
