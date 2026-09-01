# PHASE 50 — MODEL SINGLETON & CODEBASE INVENTORY AUDIT
**Repository-Wide Search for Model Instantiations**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-01
**Status**: `SINGLETON COMPLIANT`

---

## 1. Codebase Model Instantiation Audit

Repository-wide search across production and offline scripts:

| Search Target | Occurrences in Production Runtime | Occurrences in Tests / Offline Benchmarks | Status |
| :--- | :--- | :--- | :--- |
| `AutoModelForSequenceClassification` | **1** (`app.core.engine.model_registry:get_nli_model`) | 0 | ✅ STRICT SINGLETON |
| `AutoTokenizer.from_pretrained` | **1** (`app.core.engine.model_registry:get_nli_model`) | 0 | ✅ STRICT SINGLETON |
| `SentenceTransformer(` | **0** (Strictly eliminated from production) | Legacy test fixtures | ✅ ZERO IN PROD |
| `CrossEncoder(` | **0** (Strictly eliminated from production) | Legacy test fixtures | ✅ ZERO IN PROD |
| `HallucinationDetectionPipeline(` | **1** (Shared application lifecycle instance) | Test harnesses | ✅ STRICT SINGLETON |
| `HybridRetriever(` | **1** (Shared inside Pipeline) | Test harnesses | ✅ STRICT SINGLETON |

---

## 2. Model Registry Runtime Counters

- `ModelRegistry.get_init_counts()["nli_model"]`: **1**
- `ModelRegistry.get_init_counts()["sentence_transformer"]`: **0**
- `ModelRegistry.get_init_counts()["cross_encoder_reranker"]`: **0**
