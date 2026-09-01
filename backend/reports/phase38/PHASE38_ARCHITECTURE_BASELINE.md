# Phase 38.1 — Repository Forensic Architecture Baseline

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 38 — Adversarial Robustness & Production Reliability  
**Active Production Model:** `HistGradientBoostingClassifier` (19 features, $\tau^* = 0.54$, $N=58,002$)  
**Date:** 2026-09-01  

---

## 1. End-to-End Inference Pipeline

The complete execution path from raw LLM generated text to human-readable verdict and local counterfactual explanation:

```
                  ┌─────────────────────────────────┐
                  │          response_text          │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │ 1. Claim Decomposition          │
                  │    extract_claims()             │
                  │    - abbreviation protection    │
                  │    - regex sentence boundary    │
                  │    - list of {claim_id, text}   │
                  └────────┬───────────────┬────────┘
                           │               │
            ┌──────────────┘               └──────────────┐
            │ (Pillar 1: Evidence Grounding)              │ (Pillar 2: Internal Consistency)
            ▼                                             ▼
┌───────────────────────────────────────┐ ┌───────────────────────────────────────┐
│ 2. Hybrid Retrieval (Wikipedia/BM25)  │ │ 3. Pairwise Claim Analysis            │
│    HybridRetriever.get_evidence()     │ │    (if claim_count >= 2)              │
│    - Batch Wikipedia API search       │ │    - generate_unordered_claim_pairs   │
│    - CrossEncoder NLI scaling         │ │    - DeBERTa pairwise NLI             │
│    - 5 Pillar-1 features (p1_0..p1_4) │ │    - 5 Pillar-2 features (p2_0..p2_4) │
│    - Base P1 logit model (prob_p1)    │ │    - Base P2 logit model (prob_p2)    │
└───────────────────┬───────────────────┘ └───────────────────┬───────────────────┘
                    │                                         │
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │ 4. Meta-Feature Fusion Assembly           │
                  │    - 5 Pillar 1 features                  │
                  │    - 5 Pillar 2 features                  │
                  │    - 4 Base probability & logit signals   │
                  │    - 5 Nonlinear meta interaction signals │
                  │    = 19-dimensional raw feature vector X  │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │ 5. Feature Transformation (RobustScaler)  │
                  │    X_scaled = RobustScaler.transform(X)   │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │ 6. Hybrid Classification                  │
                  │    P(H) = clf.predict_proba(X_scaled)[1]  │
                  │    is_hallucinated = P(H) >= 0.54         │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │ 7. Local Counterfactual Attribution       │
                  │    compute_local_attribution()            │
                  │    - 21 model calls (1 orig, 1 base, 19)  │
                  │    - a_i = P(H|X) - P(H|X_i)              │
                  │    - interaction_gap calculation          │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │ 8. API Serialization & Response           │
                  │    /predict payload & UI rendering        │
                  └───────────────────────────────────────────┘
```

---

## 2. Canonical 19-Feature Schema & Subsystem Dependencies

The 19 features required by the frozen `HistGradientBoostingClassifier` are structured into 4 distinct groups:

| Index | Feature Name | Subsystem Origin | Generation Logic & Dependency | Expected Range |
|---|---|---|---|---|
| `[0]` | `p1_mean_entailment` | Pillar 1 (Retrieval) | Mean NLI entailment score across retrieved evidence passages | $[0.0, 1.0]$ |
| `[1]` | `p1_max_entailment` | Pillar 1 (Retrieval) | Peak NLI entailment score from any single evidence snippet | $[0.0, 1.0]$ |
| `[2]` | `p1_mean_contradiction` | Pillar 1 (Retrieval) | Mean NLI contradiction score across retrieved evidence passages | $[0.0, 1.0]$ |
| `[3]` | `p1_min_support_margin` | Pillar 1 (Retrieval) | Minimum difference between entailment and contradiction ($\text{ent} - \text{con}$) | $[-1.0, 1.0]$ |
| `[4]` | `p1_num_claims` | Pillar 1 (Decomposition)| Number of claims extracted for external evidence retrieval | $[1.0, \infty)$ |
| `[5]` | `p2_max_pairwise_contradiction` | Pillar 2 (Consistency)| Maximum pairwise contradiction score between any two claims (0.0 if $\text{claims} < 2$) | $[0.0, 1.0]$ |
| `[6]` | `p2_mean_pairwise_contradiction` | Pillar 2 (Consistency)| Average pairwise contradiction across all claim pairs (0.0 if $\text{claims} < 2$) | $[0.0, 1.0]$ |
| `[7]` | `p2_max_pairwise_similarity` | Pillar 2 (Consistency)| Peak semantic similarity between any two claims (0.0 if $\text{claims} < 2$) | $[0.0, 1.0]$ |
| `[8]` | `p2_fraction_contradictory_pairs` | Pillar 2 (Consistency)| Proportion of claim pairs exceeding contradiction threshold (0.0 if $\text{claims} < 2$) | $[0.0, 1.0]$ |
| `[9]` | `p2_num_claims` | Pillar 2 (Consistency)| Number of claims extracted for internal consistency analysis | $[1.0, \infty)$ |
| `[10]` | `prob_p1` | Base Model 1 | Calibrated hallucination probability from Pillar 1 LogisticRegression | $(0.0, 1.0)$ |
| `[11]` | `prob_p2` | Base Model 2 | Calibrated hallucination probability from Pillar 2 LogisticRegression | $(0.0, 1.0)$ |
| `[12]` | `logit_p1` | Base Model 1 Transform | $\text{logit}(P_1) = \ln(P_1 / (1 - P_1))$, clipped with $\epsilon = 10^{-7}$ | $(-\infty, +\infty)$ |
| `[13]` | `logit_p2` | Base Model 2 Transform | $\text{logit}(P_2) = \ln(P_2 / (1 - P_2))$, clipped with $\epsilon = 10^{-7}$ | $(-\infty, +\infty)$ |
| `[14]` | `prob_disagreement_abs` | Meta Fusion Signal | Absolute discrepancy between base pillars: $\|P_1 - P_2\|$ | $[0.0, 1.0]$ |
| `[15]` | `prob_mean` | Meta Fusion Signal | Unweighted mean of base probabilities: $(P_1 + P_2) / 2$ | $[0.0, 1.0]$ |
| `[16]` | `prob_max` | Meta Fusion Signal | Upper envelope of base probabilities: $\max(P_1, P_2)$ | $[0.0, 1.0]$ |
| `[17]` | `prob_min` | Meta Fusion Signal | Lower envelope of base probabilities: $\min(P_1, P_2)$ | $[0.0, 1.0]$ |
| `[18]` | `prob_ratio` | Meta Fusion Signal | Regularized probability ratio: $(P_1 + \epsilon) / (P_2 + \epsilon)$ | $(0.0, \infty)$ |

---

## 3. Model Dependencies & Storage

| Model Component | Physical Location | Runtime Class | Size | Initialization Mode |
|---|---|---|---|---|
| **Hybrid Meta Classifier** | `backend/evaluation_results/phase6m/final_hybrid_model/hybrid_meta_classifier.joblib` | `HistGradientBoostingClassifier` | 218 KB | Lazy / Singleton |
| **Hybrid Scaler** | `backend/evaluation_results/phase6m/final_hybrid_model/preprocessing.joblib` | `RobustScaler` (19 features) | 799 bytes | Lazy / Singleton |
| **Pillar 1 Base Classifier** | `backend/evaluation_results/phase6k/final_model/pillar1_logistic_model.joblib` | `LogisticRegression` | 1.8 KB | Lazy / Singleton |
| **Pillar 1 Scaler** | `backend/evaluation_results/phase6k/final_model/robust_scaler.joblib` | `RobustScaler` (5 features) | 700 bytes | Lazy / Singleton |
| **Pillar 2 Base Classifier** | `backend/evaluation_results/phase6l/models/pillar2_logistic_classifier.joblib` | `LogisticRegression` | 1.8 KB | Lazy / Singleton |
| **Pillar 2 Scaler** | `backend/evaluation_results/phase6l/models/pillar2_scaler.joblib` | `RobustScaler` (5 features) | 700 bytes | Lazy / Singleton |
| **DeBERTa NLI CrossEncoder** | HuggingFace cache / `/data/cache` | `AutoModelForSequenceClassification` (`cross-encoder/nli-deberta-v3-small`) | ~280 MB | Startup Warmup / Singleton |
| **SentenceTransformer** | HuggingFace cache / `/data/cache` | `SentenceTransformer` (`all-MiniLM-L6-v2`) | ~90 MB | Lazy / Singleton |

---

## 4. Railway Startup Sequence & Runtime Guarantees

1. **Docker Container Launch:**
   - Base image: `python:3.11-slim`
   - Entrypoint: `python start.py`
   - Uvicorn started with strictly `workers=1` on `$PORT`.
2. **Environment Caps:**
   - `torch.set_num_threads(1)` & `torch.set_num_interop_threads(1)`
   - `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`
   - `TOKENIZERS_PARALLELISM=false`
   - `MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_=65536`
   - `PYTHONMALLOC=malloc` is strictly **excluded**.
3. **Warmup & Health Probes:**
   - Background thread initializes `ModelRegistry.get_pipeline()` asynchronously so HTTP server binds immediately.
   - `/health` responds with memory RSS telemetry and model availability.
   - `/ready` returns HTTP 200 once background initialization succeeds.
