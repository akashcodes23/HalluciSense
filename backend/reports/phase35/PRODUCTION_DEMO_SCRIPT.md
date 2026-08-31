# HalluciSense Viva & Final Demonstration Script

This script provides exact, copy-pasteable, deterministic commands for conducting the live examination demonstration against the production deployment (`https://hallucisense-production.up.railway.app`).

---

## Pre-Requisite
Ensure `curl` and `python3` (or `jq`) are available in your terminal.

```bash
export API_URL="https://hallucisense-production.up.railway.app"
```

---

## Step 1: Verify System Health & Telemetry

### Command:
```bash
curl -s -i "$API_URL/health"
```

### Expected Output:
- **HTTP Status**: `200 OK`
- **JSON Body**:
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "memory_mb": 620.0,
    "active_model": "hybrid",
    "hybrid_available": true,
    "fallback_active": false,
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
- **Viva Talking Point**:
  > *"Notice that memory usage is ~620 MB (well below the 1024 MB container limit), the NLI model is instantiated as a singleton (count=1), and `active_model` confirms that the full 19-feature Hybrid engine is active with zero fallback."*

---

## Step 2: Verify Readiness Status

### Command:
```bash
curl -s -i "$API_URL/ready"
```

### Expected Output:
- **HTTP Status**: `200 OK`
- **JSON Body**:
  ```json
  {
    "status": "ready",
    "ready": true,
    "active_model": "hybrid",
    "hybrid_available": true,
    "fallback_active": false,
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

## Step 3: Factual Claim Analysis (Cold Request)

### Command:
```bash
curl -s -X POST "$API_URL/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of Japan?",
    "response": "Tokyo is the capital and most populous metropolis of Japan."
  }' | python3 -m json.tool
```

### Expected Output Highlights:
- **`risk_level`**: `"VERIFIED"`
- **`overall_h_score`**: `< 0.20` (typically `0.1333`)
- **`hallucination`**: `false`
- **`evidence`**: Contains retrieved Wikipedia passages about Tokyo and Japan.
- **Viva Talking Point**:
  > *"Pillar 1 retrieved reference knowledge from Wikipedia, evaluated NLI entailment using DeBERTa v3, and the Hybrid engine classified this claim as VERIFIED with a low H-Score (0.1333)."*

---

## Step 4: Hallucinated / Contradictory Claim Analysis

### Command:
```bash
curl -s -X POST "$API_URL/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who was the first president of the United States?",
    "response": "The first president of the United States was Winston Churchill in 1945."
  }' | python3 -m json.tool
```

### Expected Output Highlights:
- **`risk_level`**: `"LIKELY_HALLUCINATED"`
- **`overall_h_score`**: `> 0.90` (typically `0.9987`)
- **`hallucination`**: `true`
- **`token_heatmap`**: Sentences flagged in red (`#EF4444`).
- **Viva Talking Point**:
  > *"Pillar 1 detected strong NLI contradiction against reference history, triggering an H-score of 0.9987 and classifying the generation as LIKELY_HALLUCINATED."*

---

## Step 5: Fast In-Memory Cached Query

### Command:
```bash
curl -s -X POST "$API_URL/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of Japan?",
    "response": "Tokyo is the capital and most populous metropolis of Japan."
  }' | python3 -m json.tool
```

### Expected Output Highlights:
- **`processing_time_ms`**: `~10 ms` (sub-15ms cached return).
- **`cache_hit`**: `true`
- **Viva Talking Point**:
  > *"The multi-layer LRU cache avoids redundant embedding and retrieval, dropping latency from ~1.5s down to 10ms on repeated evaluations."*

---

## Step 6: Dedicated Hybrid Classifier Evaluation (`/predict`)

### Command:
```bash
curl -s -X POST "$API_URL/api/v1/hallucisense/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "response_text": "Bengaluru is the capital of Karnataka."
  }' | python3 -m json.tool
```

### Expected Output Highlights:
- **`is_hallucinated`**: `false`
- **`operating_threshold`**: `0.54`
- **`hallucination_probability`**: `0.2973`
- **`explanation.verdict`**: `"FACTUAL"`
- **`explanation.pillar_contributions`**: Full breakdown of Pillar 1 vs Pillar 2 probabilities.
- **Viva Talking Point**:
  > *"This endpoint directly invokes our frozen 19-feature `HistGradientBoostingClassifier` with threshold $\tau^* = 0.54$, providing a complete explainability summary and individual pillar probability contributions."*
