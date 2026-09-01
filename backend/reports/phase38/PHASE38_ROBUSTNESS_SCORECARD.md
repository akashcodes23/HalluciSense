# Phase 38.16 — HalluciSense Adversarial Robustness Scorecard

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 38 — Adversarial Robustness & Production Reliability  
**Active Production Model:** `HistGradientBoostingClassifier` (19 features, $\tau^* = 0.54$, $N=58,002$)  
**Evaluation Scope:** 162 adversarial test cases spanning 10 distinct failure modes  
**Date:** 2026-09-01  

---

## 1. Quantitative Scorecard Summary

| Evaluation Dimension | Metric Evaluated | Measured Value | Benchmark Target | Status |
|---|---|---|---|---|
| **A. Minimal-Pair Representation** | Proportion of minimal pairs with distinct feature vectors ($L_2 > 0.01$) | **8.3% (5/60 pairs)** | $\ge 80.0\%$ | ⚠️ P1 Bottleneck (Collapse) |
| **B. Probability Separation** | Mean absolute $\Delta P(H)$ on minimal pairs | **0.0000** (single-claim) / **0.0047** (all) | $\ge 0.1500$ | ⚠️ P1 Bottleneck |
| **C. Decision Separation** | Proportion of minimal pairs where verdict changes across $\tau^* = 0.54$ | **0.0% (0/60 pairs)** | $\ge 75.0\%$ | ⚠️ P1 Bottleneck |
| **D. Multi-Claim Discrimination** | Detection of repeated / conflicting multi-claim assertions | **100.0%** (e.g. Case J05: $P(H) = 0.8175$) | $\ge 90.0\%$ | ✅ Robust |
| **E. Retrieval Failure Detection** | Elevation of risk on ungrounded / unsupported queries | **100.0%** (e.g. Case J08: $P(H) = 0.6653$) | $\ge 90.0\%$ | ✅ Robust |
| **F. NLI Subsystem Grounding** | Direct token-level cross-encoder evaluation in Pillar 1 | **Bypassed** (Mapped via `_relevance_to_nli`) | Active NLI | ⚠️ P1 Limitation |
| **G. Attribution Faithfulness** | Numerical consistency $a_i = P(H \mid X) - P(H \mid X_i)$ | **100.0% ($\text{error} \le 10^{-8}$)** | $100.0\%$ | ✅ Exact |
| **H. Non-Additivity Transparency** | Residual interaction gap $\mathcal{I}(X)$ surfaced in API & UI | **100.0%** | $100.0\%$ | ✅ Transparent |
| **I. State Isolation & Repeatability** | Zero prediction caching / state leakage across requests | **100.0% ($\text{dev} < 10^{-12}$)** | $100.0\%$ | ✅ Deterministic |
| **J. Runtime Memory Stability** | Steady-state RSS vs 1024 MB Railway limit | **538.0 MB (47.3% headroom)** | $< 800\text{ MB}$ | ✅ Optimal |
| **K. Boundary / Failure Injection** | Graceful handling of empty, unicode, long, or malformed inputs | **100.0% (10/10 passed)** | $100.0\%$ | ✅ Hardened |
| **L. Deployment Stability** | Live Railway backend availability (`/health`, `/ready`, `/predict`) | **● Online / 0 OOM** | 0 Crashes | ✅ Verified |

---

## 2. Category-by-Category Discrimination Breakdown

| Category | Description | Evaluated Cases | Representation Discrimination | Decision Separation | Primary Diagnostic Note |
|---|---|---|---|---|---|
| **Category A** | Factual Minimal Pairs | 20 cases (10 pairs) | 0.0% | 0.0% | Collapsed: Default Wikipedia relevance (0.85) mapped to identical coordinates |
| **Category B** | Entity Swaps | 20 cases (10 pairs) | 0.0% | 0.0% | Collapsed: Keyword retrieval succeeded for both swapped entities |
| **Category C** | Numerical Mutations | 20 cases (10 pairs) | 10.0% (1/10 pairs) | 0.0% | Slight contradiction variation on Case C01 ($L_2 = 0.0091$), but insufficient to separate verdict |
| **Category D** | Negations | 20 cases (10 pairs) | 0.0% | 0.0% | Collapsed: Negation tokens ("not") bypassed in keyword relevance mapping |
| **Category E** | Temporal Mutations | 20 cases (10 pairs) | 0.0% | 0.0% | Collapsed: Historical years retrieved relevant background articles |
| **Category F** | Multi-Claim Structural Pairs | 20 cases (10 pairs) | 40.0% (4/10 pairs) | 0.0% | Pairwise consistency graph active ($L_2 = 2.38$), but composite claims did not cross $0.54$ |
| **Category G** | Unsupported Claims | 10 cases | 100.0% | Documented | Successfully triggers negative support margins on retrieval failure |
| **Category H** | Entity-Relationship Swaps | 10 cases | 100.0% | Documented | Produces distinct vectors ($L_2 = 0.1018$) based on mixed article retrieval |
| **Category I** | Paraphrases | 12 cases (3 sets) | Invariant | Invariant | Stable representation across syntactic variations |
| **Category J** | Adversarial Wording | 10 cases | 80.0% | 30.0% (3/10 flagged) | Correctly flags repeated claims ($P(H)=0.8175$) and ungrounded jargon ($P(H)=0.6653$) |

---

## 3. Scientific Synthesis

HalluciSense possesses a robust, highly stable multi-pillar fusion architecture and a mathematically faithful local explainability engine. However, the system currently exhibits a **representation collapse bottleneck on single-sentence minimal pairs** because Pillar 1 maps retrieval keyword relevance scores through a polynomial function rather than executing true token-level cross-encoder NLI between the claim and the retrieved evidence.
