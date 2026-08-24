# HalluciSense REST API Reference

The HalluciSense backend exposes a high-performance REST API built on FastAPI.

**Base URL (Local)**: `http://localhost:8000`  
**Base URL (Production)**: `https://hallucisense-production.up.railway.app`  
**Interactive Docs**: `/docs` (Swagger UI) | `/redoc` (ReDoc)

---

## 1. System Health & Readiness Endpoints

### `GET /health` | `GET /healthz`
Returns system liveness and memory RSS telemetry.

- **Request**: `GET /health`
- **Response (200 OK)**:
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "memory_mb": 622.72,
    "models": {
      "nli_model": true,
      "sentence_transformer": false,
      "cross_encoder_reranker": false,
      "pipeline": true
    },
    "model_counts": {
      "nli_model": 1,
      "sentence_transformer": 0,
      "cross_encoder_reranker": 0,
      "pipeline": 1
    }
  }
  ```

### `GET /ready` | `GET /readyz`
Deep readiness probe evaluating pipeline singleton availability.

- **Request**: `GET /ready`
- **Response (200 OK)**:
  ```json
  {
    "status": "ready",
    "components": {
      "pipeline": true,
      "nli_model": true,
      "p1_hybrid": true,
      "retriever": true,
      "fusion_engine": true
    },
    "version": "1.0.0"
  }
  ```

---

## 2. Core Verification Endpoints

### `POST /api/v1/analyze`
Executes full multi-pillar verification and adaptive fusion on an LLM response.

- **Rate Limit**: 100 requests / minute per client IP.
- **Request Body**:
  ```json
  {
    "query": "What is the capital of Karnataka?",
    "response": "The capital of Karnataka is Bengaluru.",
    "model_name": "gpt-4o",
    "provided_evidence": null,
    "sample_responses": null,
    "logprobs": null
  }
  ```
  *Field constraints*: `query` (optional, max 2000 chars), `response` (required, 1–10000 chars).

- **Response (200 OK)**:
  ```json
  {
    "trace_id": "TRACE_BD377F830813",
    "overall_h_score": 0.1333,
    "risk_level": "VERIFIED",
    "confidence": 0.6333,
    "pillar_scores": {
      "retrieval": 0.1333,
      "confidence": null,
      "consistency": null
    },
    "failure_taxonomy": "NONE",
    "processing_time_ms": 2176.27,
    "version": "1.0.0",
    "hallucination": false,
    "sentence_scores": [
      {
        "sentence_index": 0,
        "text": "The capital of Karnataka is Bengaluru.",
        "score": 0.1333,
        "risk_level": "VERIFIED"
      }
    ],
    "token_heatmap": [
      {
        "token": "The capital of Karnataka is Bengaluru.",
        "score": 0.1333,
        "tier": "VERIFIED",
        "color_hex": "#10B981"
      }
    ],
    "evidence": [
      {
        "id": "ev_1",
        "title": "Wikipedia: Karnataka",
        "snippet": "Karnataka is a state on the southwestern coast of India...",
        "score": 0.85,
        "source": "Wikipedia: Karnataka"
      }
    ],
    "root_cause_classification": "VERIFIED",
    "pillar_status": {
      "p1_status": "EXECUTED",
      "p2_status": "UNAVAILABLE",
      "p3_status": "UNAVAILABLE",
      "fusion_status": "PARTIAL_ONE_PILLAR",
      "p1_available": true,
      "p2_available": false,
      "p3_available": false,
      "is_full_analysis": false
    },
    "fusion_decomposition": {
      "equation": "H = alpha*P1 + beta*P2 + gamma*P3",
      "fusion_mode": "PARTIAL_RENORMALIZED",
      "configured_weights": { "alpha": 0.45, "beta": 0.3, "gamma": 0.25 },
      "effective_weights": { "alpha": 1.0, "beta": 0.0, "gamma": 0.0 },
      "uncalibrated_h_score": 0.1333,
      "calibrated_h_score": 0.0207
    }
  }
  ```

---

## 3. Closed-Loop AI Chat Endpoint

### `POST /api/v1/chat`
Performs end-to-end question answering, evidence retrieval, verification, and automatic closed-loop repair.

- **Request Body**:
  ```json
  {
    "message": "What causes Type 1 diabetes mellitus?",
    "enable_verification": true,
    "auto_correct": true,
    "model_name": "gemini-2.0-flash"
  }
  ```

- **Response (200 OK)**:
  ```json
  {
    "conversation_id": "conv_b1300fb96749",
    "message_id": "msg_bdb9870085a3",
    "original_response": "Type 1 diabetes mellitus is characterized by autoimmune destruction of pancreatic beta cells.",
    "final_response": "Type 1 diabetes mellitus is characterized by autoimmune destruction of pancreatic beta cells.",
    "verification": {
      "status": "VERIFIED",
      "h_score": 0.0103,
      "risk_level": "VERIFIED",
      "claims_total": 1,
      "claims_flagged": 0,
      "error_message": null
    },
    "correction": {
      "performed": false,
      "reason": "NO_CORRECTION_NEEDED",
      "claims_corrected": [],
      "original_to_corrected": []
    },
    "evidence": [
      {
        "source_name": "Wikipedia: Type 1 diabetes",
        "snippet": "Type 1 diabetes (T1D) is an autoimmune disease where the body destroys beta cells in the pancreas...",
        "claim": "Type 1 diabetes mellitus is characterized by autoimmune destruction of pancreatic beta cells."
      }
    ],
    "sources": ["Wikipedia: Type 1 diabetes", "Wikipedia: Type 2 diabetes"],
    "trace_id": "TRACE_12BABF0E17C5",
    "latency_ms": 61891.32
  }
  ```

---

## 4. Telemetry & Trace Endpoints

### `GET /api/v1/metrics`
Returns streaming runtime analytics (total requests, verified counts, average latency, mean H-score).

### `GET /api/v1/debug/latest`
Returns the most recent execution trace with granular per-stage latency breakdown.

### `GET /api/v1/debug/{trace_id}`
Fetches a specific execution trace by UUID/Trace-ID.

---

## 5. Failure Semantics & Error Responses

| HTTP Status | Error Code | Behavior |
| :---: | :--- | :--- |
| **422** | `VALIDATION_ERROR` | Request payload schema error or missing required fields. |
| **429** | `RATE_LIMIT_EXCEEDED` | Request quota exceeded (>100 req/min). Contains `Retry-After` header. |
| **500** | `INTERNAL_SERVER_ERROR` | Unhandled error. Stack traces are sanitized in production mode. |
| **200 (Structured)** | `status="FAILED"` | Total signal deficit ($\sum m_i = 0$). `h_score=null`, `risk_level=null`. |
