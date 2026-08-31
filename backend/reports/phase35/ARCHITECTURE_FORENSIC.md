# HalluciSense Production Architecture Forensic Review

## 1. System Architecture Overview

HalluciSense is a production-grade hallucination detection platform that unifies Multi-Pillar verification signals (Evidence Grounding, Internal Semantic Consistency, and Adaptive Renormalized Fusion) into a frozen 19-dimensional Gradient Boosted metadata decision engine (`HistGradientBoostingClassifier`).

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                    Client Application                   │
                              └───────────────────────────┬─────────────────────────────┘
                                                          │ HTTP POST /api/v1/analyze
                                                          ▼
                              ┌─────────────────────────────────────────────────────────┐
                              │          FastAPI Application (app.main:app)             │
                              └───────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                              ┌─────────────────────────────────────────────────────────┐
                              │      Verification Router (app.modules.verification)     │
                              │        - Payload Validation (ClaimAnalysisRequest)      │
                              │        - Telemetry & Trace ID Generation (TRACE_*)      │
                              └───────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                              ┌─────────────────────────────────────────────────────────┐
                              │         Master Pipeline Proxy (app.core.pipeline)       │
                              │           - Singleton Lazy Forwarding                   │
                              │           - LRU Evidence & Claim In-Memory Cache        │
                              └───────────────────────────┬─────────────────────────────┘
                                                          │
                                  ┌───────────────────────┴───────────────────────┐
                                  │                                               │
                                  ▼                                               ▼
┌──────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────┐
│         PILLAR 1: Evidence Grounding Engine      │   │    PILLAR 2: Semantic Consistency Engine         │
│             (app.core.engine.pillar1)            │   │          (app.core.engine.pillar2)               │
├──────────────────────────────────────────────────┤   ├──────────────────────────────────────────────────┤
│ 1. Sentence/Claim Extraction (spaCy regex)       │   │ 1. Multi-sample / Pairwise Claim Extraction      │
│ 2. Hybrid Retrieval (Dense FAISS + BM25 + Wiki)  │   │ 2. Cross-Sentence Similarity Analysis            │
│ 3. DeBERTa NLI CrossEncoder Entailment/Contra    │   │ 3. Semantic Contradiction & Variance Metric      │
│    (AutoModelForSequenceClassification)          │   │ 4. Internal Agreement Scoring                    │
│ 4. Bounded Concurrency Semaphore (max=2)         │   └────────────────────────┬─────────────────────────┘
└────────────────────────┬─────────────────────────┘                            │
                         │                                                      │
                         └───────────────────────┬──────────────────────────────┘
                                                 │
                                                 ▼
                              ┌─────────────────────────────────────────────────────────┐
                              │      19-Dimensional Feature Vector Construction         │
                              │       - P1 Grounding Features (p1_mean_entailment...)   │
                              │       - P2 Consistency Features (p2_max_contradiction...)│
                              │       - Cross-Pillar Probability & Logit Interactions   │
                              └───────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                              ┌─────────────────────────────────────────────────────────┐
                              │       Model Registry (app.models.registry:ModelRegistry)│
                              │       1. Preprocessing: RobustScaler (799 bytes)        │
                              │       2. Meta-Classifier: HistGradientBoosting (218 KB) │
                              │       3. Decision Boundary: P(H) >= 0.54                │
                              └───────────────────────────┬─────────────────────────────┘
                                                          │
                                  ┌───────────────────────┴───────────────────────┐
                                  │                                               │
                         P(H) < 0.54                                     P(H) >= 0.54
                                  │                                               │
                                  ▼                                               ▼
               ┌─────────────────────────────────────┐         ┌─────────────────────────────────────┐
               │         VERIFIED (Factual)          │         │     LIKELY_HALLUCINATED (Risk)      │
               │   - Token Heatmap (Green/Amber)     │         │   - Token Heatmap (Red/Flagged)     │
               │   - Evidence Attributions           │         │   - Root Cause Taxonomy             │
               │   - Renormalized Pillar Breakdown   │         │   - Mitigation Recommendations      │
               └──────────────────┬──────────────────┘         └──────────────────┬──────────────────┘
                                  │                                               │
                                  └───────────────────────┬───────────────────────┘
                                                          │
                                                          ▼
                              ┌─────────────────────────────────────────────────────────┐
                              │               JSON Output + Trace Persistence           │
                              │             (/data/traces/TRACE_<ID>.json)              │
                              └─────────────────────────────────────────────────────────┘
```

---

## 2. Request Lifecycle Tracing

### Step 1: Ingress & Middleware
- **File**: `backend/app/main.py`
- **Component**: FastAPI application instance configured with structured JSON logging (`structlog`), CORS middleware, GZip compression, and process-level lifespan handlers.
- **Lifespan Initialization** (`lifespan()`):
  - Configures CPU thread confinement: `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`.
  - Creates persistent Railway volume directories (`/data/traces`, `/data/models`, `/data/cache`, `/data/faiss`, `/data/reports`).
  - Spawns background warmup task `_background_warmup()` to pre-load the DeBERTa NLI cross-encoder into singleton memory asynchronously without blocking HTTP readiness.

### Step 2: Route Dispatch & Schema Validation
- **File**: `backend/app/modules/verification/router.py`
- **Route**: `POST /api/v1/analyze` and `POST /api/v1/hallucisense/predict`
- **Schema Validation**: Evaluates incoming payload against Pydantic models (`ClaimAnalysisRequest` / `HalluciSensePredictRequest`). Sanitizes inputs, extracts `query` and `response` fields, and assigns a deterministic `trace_id` (`TRACE_XXXXXXXXXXXX`).

### Step 3: Lazy Pipeline Orchestration
- **File**: `backend/app/core/pipeline.py` and `backend/app/core/engine/model_registry.py`
- **Component**: `HallucinationDetectionPipeline` (lazy proxy pattern).
- `get_pipeline()` delegates to `ModelRegistry.get_pipeline()`, ensuring that heavy sub-components (evidence providers, FAISS indices, NLI tokenizers, and PyTorch models) are instantiated exactly once per container lifecycle.

### Step 4: Pillar 1 Execution (Evidence Grounding)
- **File**: `backend/app/core/engine/pillar1.py`
- **Execution Flow**:
  1. Breaks response text into atomic factual propositions using sentence tokenization and heuristic claim segmentation.
  2. Queries hybrid retrieval sources: In-Memory FAISS Vector Store, BM25 keyword index, and real-time external knowledge APIs (Wikipedia, Wikidata, CrossRef, PubMed, Semantic Scholar).
  3. Formats candidate premise-hypothesis pairs `(evidence_passage, claim)`.
  4. Acquires `ModelRegistry.get_nli_semaphore(max_concurrent=2)` to bound memory usage.
  5. Computes entailment, neutral, and contradiction logit distributions via `cross-encoder/nli-deberta-v3-small`.
  6. Aggregates grounding statistics: `p1_mean_entailment`, `p1_max_entailment`, `p1_mean_contradiction`, `p1_min_support_margin`.

### Step 5: Pillar 2 Execution (Semantic Consistency)
- **File**: `backend/app/core/engine/pillar2.py`
- **Execution Flow**:
  1. Computes pairwise cross-sentence contradiction probabilities and semantic similarities across claims within the generation.
  2. Evaluates token-level variance and information entropy (where logprobs are available).
  3. Derives internal consistency features: `p2_max_pairwise_contradiction`, `p2_mean_pairwise_contradiction`, `p2_max_pairwise_similarity`, `p2_fraction_contradictory_pairs`.

### Step 6: 19-Dimensional Feature Synthesis & Hybrid Meta-Classification
- **File**: `backend/app/models/registry.py`
- **Execution Flow**:
  1. Synthesizes the 19 interaction features (P1 signals, P2 signals, individual pillar probabilities, logits, disagreement delta, ratios, and extremes).
  2. Normalizes the vector using `RobustScaler` (`preprocessing.joblib`).
  3. Evaluates calibrated hallucination risk using the frozen `HistGradientBoostingClassifier` (`hybrid_meta_classifier.joblib`).
  4. Compares $P(\text{hallucination})$ against the empirical optimal decision boundary $\tau^* = 0.54$.

### Step 7: Post-Processing, Attribution & Trace Persistence
- **File**: `backend/app/core/engine/fusion_engine.py` and `backend/app/modules/verification/service.py`
- **Execution Flow**:
  1. Assigns discrete risk level: `VERIFIED` ($H < 0.54$) or `LIKELY_HALLUCINATED` ($H \ge 0.54$).
  2. Constructs token-level color-coded risk heatmaps (`#10B981` green, `#F59E0B` amber, `#EF4444` red).
  3. Diagnoses failure taxonomy root cause (e.g., *Knowledge Base Absence*, *Entity Linking Failure*, *Premise Contradiction*).
  4. Persists the complete audit log to `/data/traces/TRACE_<ID>.json`.
  5. Emits structured JSON response with latency breakdown.

---

## 3. Component Directory & File Responsibility Matrix

| Subsystem / Layer | Source File Path | Primary Class / Function | Architectural Responsibility |
| :--- | :--- | :--- | :--- |
| **API Entry Point** | `backend/app/main.py` | `FastAPI`, `lifespan()`, `_sync_warmup()` | Application factory, thread confinement, background NLI warmup |
| **Router Layer** | `backend/app/modules/verification/router.py` | `analyze_claim()`, `predict_hybrid()` | HTTP endpoints, schema validation, trace lifecycle |
| **Service Layer** | `backend/app/modules/verification/service.py` | `VerificationService` | High-level business logic, response formatting, trace saving |
| **Core Pipeline Proxy** | `backend/app/core/pipeline.py` | `get_pipeline()`, `HallucinationDetectionPipeline` | Thread-safe proxy preventing duplicate pipeline initialization |
| **Singleton Engine Registry** | `backend/app/core/engine/model_registry.py` | `ModelRegistry` | Lazy singleton loaders for DeBERTa NLI, SentenceTransformer, Reranker |
| **Pillar 1 Grounding** | `backend/app/core/engine/pillar1.py` | `Pillar1RetrievalGroundingEngine` | Claim extraction, hybrid retrieval, NLI entailment scoring |
| **Pillar 2 Consistency** | `backend/app/core/engine/pillar2.py` | `Pillar2SelfConsistencyEngine` | Pairwise contradiction analysis, semantic similarity, entropy |
| **Pillar 3 Verification** | `backend/app/core/engine/pillar3.py` | `Pillar3CrossLLMConsistencyEngine` | Multi-LLM consensus verification (when multi-generation enabled) |
| **Hybrid Model Registry** | `backend/app/models/registry.py` | `ModelRegistry`, `safe_joblib_load` | Lazy loading of frozen 19-feature classifier, BitGenerator safe-loader |
| **Fusion & Taxonomy** | `backend/app/core/engine/fusion_engine.py` | `FusionEngine` | Renormalized weighted fusion, failure taxonomy attribution |
| **Health & Telemetry** | `backend/app/main.py` | `/health`, `/ready` endpoints | Live container health, memory telemetry, active model state |

---

## 4. Model Loading & Singleton Lifecycle

To ensure strict adherence to container memory limits (1024 MB), HalluciSense enforces a single-allocation paradigm:

1. **Lazy Initialization**: Heavy models are never imported at global module scope. Imports occur inside thread-safe synchronized blocks (`threading.RLock()`).
2. **Singleton Counts**:
   - `nli_model`: Instantiated exactly once (`init_count = 1`).
   - `pipeline`: Instantiated exactly once (`init_count = 1`).
   - `sentence_transformer`: On-demand singleton (`init_count = 0` during standard NLI mode).
   - `cross_encoder_reranker`: On-demand singleton (`init_count = 0` during standard NLI mode).
3. **Active Model Telemetry**:
   - Every health check inspects `app.models.registry.ModelRegistry.get_model_status()`.
   - The response guarantees transparency regarding whether `hybrid` is active or if `pillar1_fallback` has been engaged.
