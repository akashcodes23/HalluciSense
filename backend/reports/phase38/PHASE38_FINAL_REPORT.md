# Phase 38 — Adversarial Robustness & Production Reliability Final Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 38 — Adversarial Robustness & Production Reliability  
**Active Production Model:** `HistGradientBoostingClassifier` (19 features, $\tau^* = 0.54$, $N=58,002$)  
**Status:** **AUDITED, HARDENED & COMPLETED**  
**Date:** 2026-09-01  

---

## 1. Executive Summary

Phase 38 executed a comprehensive forensic investigation into the adversarial robustness, representation discrimination, feature collapse, NLI evidence grounding, local explainability faithfulness, and production runtime stability of HalluciSense.

Using a deterministic **162-case adversarial evaluation matrix** spanning 10 failure categories (Factual Minimal Pairs, Entity Swaps, Numerical Mutations, Negations, Temporal Mutations, Multi-Claim Structural Pairs, Unsupported Claims, Entity-Relationship Swaps, Paraphrases, and Adversarial Framing), the audit discovered:
1. **P1 Architectural Bottleneck (Representation Collapse):** On atomic, single-sentence minimal pairs (e.g., *"Paris is capital of France"* vs *"Berlin is capital of France"*), the pipeline exhibits a **91.7% representation collapse** ($L_2 = 0.0$). This occurs because Pillar 1 maps keyword retrieval relevance via a static polynomial function (`_relevance_to_nli(0.85)`) rather than running transformer cross-attention between evidence and claim text.
2. **Multi-Claim & Structural Robustness:** When inputs contain multiple assertions or repeated adversarial prompts (e.g. Case J05), Pillar 2 pairwise NLI activates topological conflict graphs, correctly elevating hallucination risk to **$P(H) = 0.8175$ (FLAGGED)**.
3. **Local Attribution Integrity:** The local attribution engine operates with **100% mathematical fidelity** ($a_i = P(H \mid X) - P(H \mid X_i)$, error $\le 10^{-8}$), faithfully explaining the exact feature coordinates received from upstream extractors.
4. **Production Runtime Hardening:** 100% pass rate across 10 failure injection scenarios, zero memory leakage, zero Exit 137 crashes, and verified Railway deployment health with ~484 MB safety headroom under the 1024 MB container limit.

---

## 2. System Baseline

- **Hybrid Classifier:** `HistGradientBoostingClassifier` (218 KB, 19 inputs, fitted on $N=58,002$)
- **Preprocessing:** `RobustScaler` (799 bytes, 19 inputs)
- **Decision Threshold:** $\tau^* = 0.54$ (Frozen)
- **Single-Worker Topology:** `workers=1` on `$PORT`
- **Thread Caps:** `torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`
- **Memory Allocator:** `MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_=65536` (`PYTHONMALLOC=malloc` strictly excluded)

---

## 3. Test Matrix

A golden adversarial dataset of **162 cases** was constructed in `backend/tests/test_phase38_adversarial_matrix.py`:
- **Category A (Factual Minimal Pairs):** 10 pairs (20 cases)
- **Category B (Entity Swaps):** 10 pairs (20 cases)
- **Category C (Numerical Mutations):** 10 pairs (20 cases)
- **Category D (Negations):** 10 pairs (20 cases)
- **Category E (Temporal Mutations):** 10 pairs (20 cases)
- **Category F (Multi-Claim Pairs):** 10 pairs (20 cases)
- **Category G (Unsupported Claims):** 10 cases
- **Category H (Entity-Relationship Swaps):** 10 cases
- **Category I (Paraphrases):** 12 cases (3 sets of 4)
- **Category J (Adversarial Framing):** 10 cases

---

## 4. Minimal-Pair Results

Across the 60 minimal pairs evaluated:
- **Representation Discrimination Rate ($L_2 > 0.01$):** **8.3% (5 / 60 pairs)**
- **Identical Representation Rate ($L_2 = 0.0$):** **91.7% (55 / 60 pairs)**
- **Verdict Separation Rate across $\tau^* = 0.54$:** **0.0%**

---

## 5. Feature Representation Analysis

All 162 cases produced valid 19-dimensional feature vectors stored in `backend/reports/phase38/feature_vectors.json`.  
Cosine similarities and Euclidean distances ($L_2$) between paired items confirmed that single-claim mutations share the exact coordinate:
$$X_{\text{single-claim}} = [0.2167, 0.2167, 0.1430, 0.0738, 1.0, 0, 0, 0, 0, 1.0, 0.4879, 0.4341, -0.0483, -0.2650, 0.0538, 0.4610, 0.4879, 0.4341, 1.1239]$$

---

## 6. Feature Collapse Findings

The representation collapse is caused by **three compounding factors**:
1. **Keyword Retrieval Invariance:** Wikipedia search returns articles matching keywords for both true and false claims.
2. **Hardcoded Relevance Default:** Retrieved passages are assigned default similarity `0.85` in `HybridRetriever`.
3. **Polynomial NLI Proxy:** `Pillar1Engine` applies `_relevance_to_nli(0.85)` rather than evaluating `cross-encoder/nli-deberta-v3-small` on `(passage, claim)`.

---

## 7. Retrieval Analysis

- **Cache & Timings:** Wikipedia queries average ~800–1200ms per batch.
- **Diagnostic Behavior:** When Wikipedia search returns 0 articles (Case G01, Case J08), the system enters the negative margin branch (`p1_min_support_margin = -0.2422`), successfully separating feature coordinates ($L_2 = 0.4178$) and elevating hallucination risk ($P(H) = 0.6653$).

---

## 8. NLI Analysis

- **Pillar 1:** Bypasses DeBERTa inference during online prediction to preserve low latency.
- **Pillar 2:** Evaluates DeBERTa NLI and MiniLM embeddings on claim pairs. When multiple claims are present, contradictory claim pairs correctly register pairwise contradiction $\ge 0.80$.

---

## 9. Claim-Level Analysis

- **Decomposition:** `extract_claims()` cleanly protects 21 standard abbreviations (`Dr.`, `U.S.`, `i.e.`) and avoids false sentence fragmentation.
- **Quadratic Capping:** Pillar 2 caps evaluation at 15 claims to prevent $O(N^2)$ NLI pair explosion on long inputs.

---

## 10. Attribution Robustness

- **Formula Verification:** $a_i = P(H \mid X) - P(H \mid X_i)$ holds exactly across all 162 adversarial evaluations ($\text{error} \le 10^{-8}$).
- **Faithfulness:** When given a collapsed feature vector $X$, the attribution engine correctly identifies `p1_mean_contradiction` (+0.0969) as the primary risk driver and `prob_mean` (-0.2509) as the protective factor. It never fabricates explanations that the classifier did not receive.
- **Interaction Gap:** Non-additive tree interaction residuals $\mathcal{I}(X)$ range from $-0.2452$ to $+0.1848$ and are surfaced in all responses.

---

## 11. Failure Injection Test Results

All 10 failure injection tests in `backend/tests/test_phase38_failure_injection.py` passed:
- Empty & whitespace inputs $\to$ Graceful fallback response
- 10,000-character prompt $\to$ Bounded execution
- Unicode, Cyrillic, Chinese, Emojis $\to$ UTF-8 preserved
- Repeated duplicate claims $\to$ Capped and processed
- NaN / Inf / Dimension $\ne 19$ $\to$ Clean `ValueError` rejection

---

## 12. Railway Forensics

- **Backend (`HalluciSense`):** Online, Deployment `1e3d7963-a3ab-4dbe-99b0-268d8823467f` (SUCCESS), 0 OOM crashes.
- **Frontend (`enchanting-wonder`):** Next.js 16 container, 23 static pages.

---

## 13. Runtime Memory Regression

- **Startup RSS:** 528.8 MB
- **Steady State RSS:** 538.0 MB
- **10 Sequential Requests:** 538.16 MB (+0.13 MB drift)
- **2 Concurrent Requests:** 539.56 MB (+1.4 MB peak)
- **Safety Margin:** 484.4 MB (47.3%) under the 1024 MB container limit.

---

## 14. Regression Test Results

```
tests/test_phase38_adversarial_matrix.py ............. [ 7/7   PASSED]
tests/test_phase38_failure_injection.py .............. [10/10  PASSED]
tests/test_phase37_3_integrity.py .................... [20/20  PASSED]
tests/test_phase37_explainability_validation.py ...... [32/32  PASSED]
tests/test_phase37_local_attribution.py .............. [29/29  PASSED]
tests/test_unit_pipeline.py .......................... [ 4/4   PASSED]
tests/test_engine.py ................................. [ 7/7   PASSED]
tests/test_phase11_memory_safety.py .................. [ 7/7   PASSED]

======================= 116 passed in 4 minutes ========================
```

---

## 15. Frontend Validation

- `npm run build`: **0 TypeScript errors, 23 static routes generated**.
- `LocalAttributionPanel.tsx`: Verified to use non-causal counterfactual terminology ("Risk ↑", "Safe ↓"), surface interaction gaps, and include the scientific disclaimer (*"This is local counterfactual attribution, not SHAP"*).

---

## 16. Robustness Scorecard Summary

- **Minimal-Pair Representation Discrimination:** 8.3% (Identified P1 bottleneck)
- **Multi-Claim Conflict Detection:** 100.0%
- **Attribution Mathematical Consistency:** 100.0%
- **Memory Stability & Zero OOM:** 100.0%
- **Failure Injection Resilience:** 100.0%

---

## 17. P0 / P1 / P2 / P3 Issue Classification

- **P0 (Production Blocking):** **NONE.** Production is stable, crash-free, and within memory limits.
- **P1 (Scientifically Significant):** Pillar 1 feature representation collapse on single-sentence minimal pairs due to bypassing DeBERTa cross-encoder evaluation against retrieved evidence.
- **P2 (Robustness Improvement):** Enable true token-level cross-encoder NLI scoring on `(passage, claim)` in Pillar 1 to separate factual vs false single-sentence queries.
- **P3 (Cosmetic / Documentation):** Document that arithmetic checking and entity swap detection require token-level NLI grounding rather than keyword retrieval relevance.

---

## 18. Frozen Model Integrity

- `hybrid_meta_classifier.joblib`: SHA256 unchanged.
- `preprocessing.joblib`: SHA256 unchanged.
- Decision threshold: $\tau^* = 0.54$ (Frozen).
- Schema: 19 canonical features (Frozen).

---

## 19. Scientific Limitations

1. **Keyword Retrieval Boundary:** Keyword-based retrieval cannot distinguish a query from its negation or slight entity mutation without semantic cross-attention.
2. **Observational Attribution:** Explanations faithfully report what reached the classifier; they cannot compensate for upstream representation collapse.

---

## 20. Phase 39 Recommendations

1. **Phase 39 Priority:** Upgrade Pillar 1 evidence grounding to run lightweight cross-encoder NLI scoring between extracted claims and retrieved evidence passages.
2. **Preserve Memory Budget:** Ensure any NLI cross-encoder enhancements remain strictly within the validated 1024 MB Railway memory envelope.
