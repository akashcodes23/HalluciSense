# PHASE 48 — MODEL INVENTORY & SINGLETON AUDIT
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-01
**Status**: `VERIFIED SINGLETON COMPLIANCE`

---

## 1. Machine Learning Model Inventory

```
+----------------------------------------------------------------------------------------------------+
| Model Identifier                | Parameter Count | Disk Size | RAM Allocated | Singleton Verified |
+----------------------------------------------------------------------------------------------------+
| cross-encoder/nli-deberta-v3-small | 44.1M          | 176 MB    | ~285 MB       | YES (Init Count: 1)|
| HistGradientBoosting (Frozen)    | N/A             | ~5 MB     | ~8 MB         | YES (Joblib Loaded)|
| RobustScaler Preprocessor       | N/A             | <1 MB     | <1 MB         | YES (Joblib Loaded)|
+----------------------------------------------------------------------------------------------------+
```

### Models Explicitly Excluded from Production Runtime
- **SentenceTransformer (`all-MiniLM-L6-v2`)**: **REMOVED FROM PRODUCTION RUNTIME**. Pillar 3 now executes intra-response and cross-generation semantic consistency using fast lexical token Jaccard alignment combined with the shared DeBERTa NLI cross-encoder.
- **CrossEncoderReranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**: **DISABLED BY DEFAULT** (`HALLUCISENSE_ENABLE_RERANKER=False`). Hybrid retrieval uses BM25 and Wikipedia dense similarity directly.

---

## 2. Model Registry Telemetry Verification

Runtime execution of `ModelRegistry.get_init_counts()` under 50 continuous requests:

```json
{
  "nli_model_init_count": 1,
  "sentence_transformer_init_count": 0,
  "cross_encoder_reranker_init_count": 0,
  "pipeline_init_count": 1
}
```

### Verification Criteria
- [x] Heavy NLI Transformer loaded lazily on first request.
- [x] Process-level initialization count is strictly $\le 1$.
- [x] No thread-local copies or per-worker duplicates spawned.
- [x] Tokenizer vocabulary buffers shared across all requests.
- [x] PyTorch model set to `.eval()` mode with `torch.inference_mode()` context.
