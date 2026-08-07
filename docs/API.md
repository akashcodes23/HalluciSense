# HalluciSense v1.0 Production REST API Reference

The HalluciSense production backend exposes a clean REST API for real-time hallucination detection, explainability, trace debugging, and runtime telemetry.

Base URL: `http://localhost:8000/api/v1`

---

## 1. POST /api/v1/analyze

Canonical production endpoint executing the full HalluciSense three-pillar verification pipeline.

### Request Body
```json
{
  "query": "Who invented the telephone?",
  "response": "Alexander Graham Bell invented the telephone in 1876.",
  "model_name": "gpt-4"
}
```

### Response Body (200 OK)
```json
{
  "trace_id": "TRACE_88CFA3E9",
  "overall_h_score": 0.08,
  "risk_level": "VERIFIED",
  "confidence": 0.94,
  "pillar_scores": {
    "retrieval": 0.05,
    "confidence": 0.12,
    "consistency": 0.07
  },
  "failure_taxonomy": "NONE",
  "processing_time_ms": 143.0,
  "version": "1.0.0"
}
```

---

## 2. POST /api/v1/explain

Returns detailed supporting/contradictory evidence, token risk heatmaps, sentence scores, reasoning chains, and adaptive weights.

### Request Body
```json
{
  "query": "What is photosynthesis?",
  "response": "Photosynthesis is the process by which green plants convert sunlight into chemical energy using chlorophyll.",
  "model_name": "gpt-4"
}
```

---

## 3. GET /api/v1/metrics

Returns process RAM memory and cumulative request telemetry computed from real runtime statistics.

### Response Body (200 OK)
```json
{
  "requests": 152,
  "average_latency_ms": 143.0,
  "average_h_score": 0.18,
  "success_rate": 99.7,
  "error_rate": 0.3,
  "memory_mb": 421.0
}
```

---

## 4. System Probes

- **`GET /health`**: Returns `{"status": "healthy"}`
- **`GET /ready`**: Deep readiness probe verifying Hybrid Retriever, NLI (`deberta-v3-small`), SentenceTransformer (`all-MiniLM-L6-v2`), CrossEncoder (`ms-marco-MiniLM-L-6-v2`), and Fusion Engine.
