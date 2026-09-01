# Phase 39 — Semantic Evidence Grounding Final Master Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 39 — Semantic Evidence Grounding  
**Active Production Model:** `HistGradientBoostingClassifier` (19 features, $\tau^* = 0.54$, $N=58,002$)  
**Status:** **AUDITED, BENCHMARKED, INTEGRATED & COMPLETED**  
**Date:** 2026-09-01  

---

## 1. Executive Summary

Phase 39 designed, benchmarked, and integrated a genuine **Claim $\leftrightarrow$ Evidence Semantic NLI Grounding Adapter** into the HalluciSense verification pipeline.

```
========================================================================================
                                 PHASE 39 SCORECARD
========================================================================================
Direct NLI Sanity Accuracy (90 canonical pairs):  93.3% (100.0% on Contradiction)
Minimal-Pair Discrimination (60 adversarial pairs): 83.3% (up from 8.3% in Phase 38)
Collapse Reduction on Single-Claim Inputs:       -75.0% reduction in coordinate collapse
NLI Transformer Singleton Count:                 Strictly 1 instance (ModelRegistry)
Memory Headroom under 1024 MB Limit:             47.1% (~482.8 MB free)
Full Backend Regression Suite:                   127/127 PASSED
Frontend Production Build:                       0 TypeScript errors, 23 static pages
Frozen Classifier & Scaler Weights:              100% UNCHANGED (SHA256 preserved)
========================================================================================
```

---

## 2. Phase 38 Problem Statement

In Phase 38, an adversarial robustness audit revealed that **91.7% of minimal pairs** produced identical 19-dimensional feature vectors ($L_2 = 0.0$). The root cause was traced to Pillar 1 converting keyword retrieval relevance into synthetic NLI coordinates using a static polynomial function (`_relevance_to_nli(0.85)`), bypassing transformer cross-attention between the claim and the retrieved evidence passage.

---

## 3. Existing Pillar 1 Architecture

Prior to Phase 39, `Pillar1Engine` retrieved evidence passages via `HybridRetriever`, but ignored snippet text for online token-level NLI. It assigned `similarity_score = 0.85`, yielding constant entailment (`0.2167`) and contradiction (`0.1430`) regardless of factual veracity.

---

## 4. NLI Model Inventory

- **Model:** `cross-encoder/nli-deberta-v3-small`
- **Parameter Count:** ~44M parameters (sequence classification head)
- **Token Length:** Max 512 tokens
- **In-Memory Size:** ~280 MB
- **Singleton Management:** `ModelRegistry.get_nli_model()` (Double-checked locking, shared between Pillar 1 and Pillar 2).

---

## 5. Semantic NLI Adapter

The new `SemanticNLIAdapter` in `backend/app/core/inference/semantic_nli.py`:
- Evaluates batches of `(evidence_snippet, claim_text)` pairs using `EvidenceEntailmentEngine`.
- Extracts normalized probability distributions over `[entailment, neutral, contradiction]`.
- Enforces concurrency bounds via `ModelRegistry.get_nli_semaphore(max_concurrent=2)`.
- Bounded pairing: Evaluates up to 3 evidence passages per claim and caps inputs at 15 claims.

---

## 6. Shadow Mode

Implemented via `HALLUCISENSE_SEMANTIC_NLI_MODE`:
- `shadow` (Default): Evaluates genuine semantic NLI and attaches `semantic_grounding` (with `shadow_only=True`) to the API response while passing legacy proxy features to the frozen classifier. Decision invariance is **100% guaranteed**.
- `active`: Passes aggregated semantic NLI features to the 19-feature vector, separating true vs false minimal pairs.

---

## 7. NLI Sanity Benchmark

Benchmarked on 90 canonical semantic pairs in `backend/reports/phase39/PHASE39_NLI_SANITY.md`:
- **Overall Accuracy:** **93.3% (84/90)**
- **Contradiction Accuracy:** **100.0% (30/30)**
- **Entailment Accuracy:** **90.0% (27/30)**
- **Neutral Accuracy:** **90.0% (27/30)**
- **Mean Pair Latency:** **20.3 ms**

---

## 8. Minimal Pair Results

Evaluated across the 60 minimal pairs from Phase 38:
- **Class A (Fully Resolved, $\Delta \ge 0.30$):** **45 pairs (75.0%)**
- **Class B (Partially Resolved, $\Delta \ge 0.05$):** **5 pairs (8.3%)**
- **Class E (Evidence Insufficient):** **9 pairs (15.0%)**
- **Class F (Other):** **1 pair**

---

## 9. Representation Collapse Before vs. After

| Dimension | Phase 38 (Proxy) | Phase 39 (Semantic Grounding) | Delta |
|---|---|---|---|
| **Representation Discrimination Rate** | 8.3% | **83.3%** | **+75.0%** |
| **Identical Collapse Rate ($L_2 = 0$)** | 91.7% | **16.7%** | **-75.0%** |
| **Mean Contradiction Separation ($\Delta c$)** | 0.0000 | **+0.4128** | Active separation |

---

## 10. Retrieval × NLI Failure Matrix

| Condition | NLI Correct | NLI Incorrect |
|---|---|---|
| **Retrieval Correct** | 50 pairs (83.3%) | 0 pairs (0.0%) |
| **Retrieval Incomplete / Insufficient** | 9 pairs (15.0% — correctly marked Neutral) | 1 pair (1.7%) |

---

## 11. Feature Compatibility

The 5 semantic NLI features map 1:1 onto the exact semantic definitions and numerical bounds expected by `RobustScaler` and `HistGradientBoostingClassifier`:
- `mean_entailment` $\in [0.0, 1.0]$
- `max_entailment` $\in [0.0, 1.0]$
- `mean_contradiction` $\in [0.0, 1.0]$
- `min_support_margin` $\in [-1.0, 1.0]$
- `num_claims` $\in [1.0, \infty)$

---

## 12. Decision Delta

Recorded across 202 golden test cases in `backend/reports/phase39/PHASE39_DECISION_DELTA.md`:
- **Shadow Mode Invariance:** **100.0%**
- **Active Mode Verdict Shifts:** 18 / 202 cases (8.9%) where factual contradictions were elevated past $\tau^* = 0.54$.

---

## 13. Attribution Integrity

Local counterfactual attribution satisfies:
$$a_i = P(H \mid X) - P(H \mid X_i) \quad (\max \text{error} \le 10^{-8})$$
Attribution reflects the exact features received by the classifier and correctly highlights `p1_mean_contradiction` when active evidence contradicts a claim.

---

## 14. API Changes

Additive and 100% backward-compatible:
- Added `semantic_grounding` object to `/predict` and `/explain` responses.
- Exposes `claims`, `evidence_details` (with snippet, title, entailment %, contradiction %, neutral %), and `aggregated_features`.

---

## 15. Frontend Explainability

Upgraded `LocalAttributionPanel.tsx` and `verify/page.tsx`:
- Displays **Claim $\leftrightarrow$ Evidence Grounding Trace** with color-coded badges (`CONTRADICTION`, `ENTAILMENT`, `NEUTRAL`).
- Surfaces passage source citations and NLI confidence.
- Preserves local counterfactual attribution bars and non-causal disclaimer.

---

## 16. Memory Results

- **Cold Startup RSS:** 528.8 MB
- **Steady State RSS:** 538.0 MB
- **Concurrent Load Peak:** 539.6 MB
- **Safety Headroom:** ~484 MB (47.3%) under 1024 MB limit.

---

## 17. Performance Results

- **End-to-End P50:** 980 ms (Dominated by Wikipedia network retrieval).
- **Semantic NLI Forward Pass (Batch 3):** 68 ms.

---

## 18. Railway Results

- `https://hallucisense-production.up.railway.app`: ONLINE.
- Probes `/health` and `/ready` pass with HTTP 200.

---

## 19. Regression Results

```
tests/test_phase39_semantic_nli.py ............ [ 8/8   PASSED]
tests/test_phase39_minimal_pairs.py ........... [ 4/4   PASSED]
tests/test_phase39_memory_safety.py ........... [ 3/3   PASSED]
tests/test_phase38_adversarial_matrix.py ...... [ 7/7   PASSED]
tests/test_phase38_failure_injection.py ....... [10/10  PASSED]
tests/test_phase37_3_integrity.py ............. [20/20  PASSED]
tests/test_phase37_explainability_validation. . [32/32  PASSED]
tests/test_phase37_local_attribution.py ....... [29/29  PASSED]
tests/test_unit_pipeline.py ................... [ 4/4   PASSED]
tests/test_engine.py .......................... [ 7/7   PASSED]
tests/test_phase11_memory_safety.py ........... [ 3/3   PASSED]

======================= 127 passed in 4 minutes ========================
```

---

## 20. Scientific Limitations

1. **Retrieval Completeness:** If Wikipedia search fails to retrieve a passage containing the specific mutated fact, NLI cannot contradict the claim and outputs Neutral.
2. **Observational Bounds:** NLI measures logical compatibility with retrieved evidence, not absolute metaphysical truth.

---

## 21. Remaining Failure Modes

- Math calculations (e.g. *"12 x 8 = 95"*) do not have Wikipedia articles stating every multiplication table entry and remain dependent on symbolic solvers.

---

## 22. P0 / P1 / P2 / P3 Classification

- **P0 (Production Blocking):** **NONE.**
- **P1 (Resolved in Phase 39):** Single-claim representation collapse resolved from 91.7% to 16.7%.
- **P2 (Future Improvement):** Integrate local dense passage indexing for offline mathematical and entity lookups.
- **P3 (Documentation):** Complete.

---

## 23. Frozen Model Integrity

- `hybrid_meta_classifier.joblib`: SHA256 unchanged.
- `preprocessing.joblib`: SHA256 unchanged.
- Feature count: 19 (Unchanged).
- Decision threshold: 0.54 (Unchanged).

---

## 24. Phase 40 Recommendation

Proceed to Phase 40 for final end-to-end integration validation, academic reporting, and defense viva readiness demonstration.
