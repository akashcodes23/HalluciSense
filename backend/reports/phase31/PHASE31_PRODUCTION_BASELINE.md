# PHASE 31: PRODUCTION BASELINE REPRODUCIBILITY RECORD

**Document Generated**: August 31, 2026  
**Repository**: `akashcodes23/HalluciSense`  
**Branch**: `main`  
**Production Platform**: Railway (Project: `passionate-contentment`, Service: `HalluciSense`)  
**Production URL**: `https://hallucisense-production.up.railway.app`  

---

## 1. Verified Production Identity

| Parameter | Value |
|---|---|
| **Git Commit Hash** | `b1aafb36b3cbddbd9c6f8cf4b8033ccd5dde393c` |
| **Git Commit Message** | `fix(phase30): make core pipeline a lazy proxy to eliminate startup memory duplicate` |
| **Base Fix Commit** | `bf05043b8d7129fb9aaf726f974e4faae07cdc24` |
| **Railway Deployment ID** | `5b4c5a29-d502-433d-9ec9-99549b585cd7` |
| **Railway Deployment Status** | `SUCCESS (● Online)` |
| **Railway Environment** | `production` (`b69f4974-053f-4f1f-bbf8-68991e501f39`) |

---

## 2. Environment & Dependency Manifest

| Component | Pinned Version | Purpose |
|---|---|---|
| **Python** | `3.11-slim` (Docker) / `3.10.12` (Local test) | Base Runtime |
| **NumPy** | `1.26.4` | Numerical array manipulation |
| **scikit-learn** | `1.7.2` | HistGradientBoostingClassifier inference |
| **joblib** | `1.5.2` | Model serialization / unpickling |
| **PyTorch** | `2.5.1+cpu` | Cross-encoder transformer tensor ops |
| **Transformers** | `4.47.1` | DeBERTa v3 small sequence classification |
| **Accelerate** | `1.2.1` | Low CPU memory loading (`low_cpu_mem_usage=True`) |
| **FastAPI** | `0.115.6` | Async REST API framework |
| **Uvicorn** | `0.32.1` | Production ASGI web server |

---

## 3. Frozen Scientific Artifacts & Cryptographic Checksums

| Artifact File | Size (Bytes) | SHA-256 Checksum |
|---|---:|---|
| `hybrid_meta_classifier.joblib` | 218,104 | `089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad` |
| `hybrid_meta_classifier.joblib.backup` | 218,104 | `cb459fd99b3da606f78c5777cbf87dee482e59ef60e27168f7656306b4a22fbf` |
| `preprocessing.joblib` | 799 | `bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90` |
| `feature_schema.json` | 449 | `942df39475c1cabc54b5f472d2ef111cfa511b3ba24050115b9bb57177db0388` |
| `model_metadata.json` | 895 | `69d8c63219de4fa27a62b0a351d78a1fdea1107775b871fc2f0391f353b11f74` |
| `backend/requirements.txt` | 1,489 | `72ed66de4f3c99d0642fdf95dd948bb5dfb272b862fe55dcc2ca67143d4d0e9a` |
| `benchmark_dataset.jsonl` | 2,829,864 | `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5` |

---

## 4. Scientific Invariants & Model Specification

- **Model Type**: `HistGradientBoostingClassifier` (Meta-Classifier Fusion)
- **Training Samples**: 58,002
- **Ensemble Size**: 100 Trees (`max_iter=100`)
- **Loss Function**: Binary Log-Loss
- **Operating Threshold ($\tau^*$)**: `0.5400`
- **Feature Dimension**: Exactly 19 input features
- **Feature Schema**:
  1. `p1_mean_entailment`
  2. `p1_max_entailment`
  3. `p1_mean_contradiction`
  4. `p1_min_support_margin`
  5. `p1_num_claims`
  6. `p2_max_pairwise_contradiction`
  7. `p2_mean_pairwise_contradiction`
  8. `p2_max_pairwise_similarity`
  9. `p2_fraction_contradictory_pairs`
  10. `p2_num_claims`
  11. `prob_p1`
  12. `prob_p2`
  13. `logit_p1`
  14. `logit_p2`
  15. `prob_disagreement_abs`
  16. `prob_mean`
  17. `prob_max`
  18. `prob_min`
  19. `prob_ratio`
- **Numerical Equivalence**: Max absolute prediction difference between repaired artifact and original backup across 100 validation vectors is $\mathbf{0.0000000000}$.
- **Retraining**: **NO** (Zero synthetic or regenerated data; fitted decision structures preserved identically).

---

## 5. Live Production Health & Readiness

#### `GET /health` (Response Time: `0.86ms`):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "memory_mb": 914.96,
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

#### `GET /ready` (Response Time: `0.67ms`):
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

## 6. Live Production Inference Verification

| Scenario | Input Query & Response | Classification | Overall H-Score | Server Latency | Result |
|---|---|---|---:|---:|---|
| **True Claim (Cold)** | Q: *"Capital of Karnataka?"* <br> R: *"The capital of Karnataka is Bengaluru."* | `VERIFIED` | 0.1333 | 1679.75 ms | **PASS** |
| **False Claim (Cold)** | Q: *"Capital of Karnataka?"* <br> R: *"The capital of Karnataka is Mumbai."* | `LIKELY_HALLUCINATED` | 0.9831 | 2038.54 ms | **PASS** |
| **Cached Repeat** | Q: *"Capital of Karnataka?"* <br> R: *"The capital of Karnataka is Bengaluru."* | `VERIFIED` | 0.1333 | 4.20 ms | **PASS** |
| **Hybrid Direct** | `POST /api/v1/hallucisense/predict` | `FACTUAL` ($P=0.2973 < 0.54$) | 0.2973 | 438.02 ms | **PASS** |

---

## 7. Resource & Resilience Profile

- **Memory Allocation**: 1024 MB
- **Peak RSS**: 972 MB
- **Steady Warm RSS**: 914.96 MB
- **Concurrency Guard**: `MAX_CONCURRENT_ANALYSES = 2` (via async semaphore)
- **Memory Circuit Breaker**: `HALLUCISENSE_MEMORY_GUARD_MB = 1500`
- **PyTorch Threads**: `1` (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`)
- **Fallback Mode**: Graceful fallback to Pillar 1 Evidence Grounding if hybrid directory is removed, reporting `active_model="pillar1_fallback"`, `hybrid_available=false`, `fallback_active=true`.
