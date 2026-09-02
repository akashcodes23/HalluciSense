# PHASE 52 — FALSE NEGATIVE ROOT CAUSE DECOMPOSITION
**Decomposition Across Failure Categories R1 to R11 ($N_{\text{FN}} = 104$)**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & CATALOGUED`

---

## 1. Distribution of False Negative Root Causes ($N=104 / 150$)

| Failure Code | Root Cause Description | Count | Percentage | Primary Impacted Categories |
| :--- | :--- | :--- | :--- | :--- |
| **R7** | **Feature Polarity Inversion in Tree** | **60** | **57.69%** | `B_clearly_false`, `C_direct_contradiction`, `J_entity_swap`, `K_temporal`, `L_negation` |
| **R10** | **Symbolic Verification Path Suppression** | **20** | **19.23%** | `I_numerical_error` |
| **R1** | **Retrieval Missingness / No Evidence** | **10** | **9.62%** | `D_unsupported_claim`, `E_ambiguous_claim` |
| **R2** | **NLI Neutral Dilution / Soft Entailment**| **10** | **9.62%** | `B_clearly_false`, `E_ambiguous_claim` |
| **R9** | **Threshold Conservatism ($\tau^*=0.54$)** | **2** | **1.92%** | `C_direct_contradiction`, `D_unsupported` |
| **R11**| **Unsupported Claim Defaulting** | **2** | **1.92%** | `D_unsupported_claim` |
| **R3-R6, R8**| Claim Seg, P2, P3, Fusion Conflicts | **0** | **0.00%** | — |
| **TOTAL** | **All False Negative Errors** | **104** | **100.00%** | **Over 76.9% caused by R7 + R10** |

---

## 2. Definitive Summary

Over **76.9% of all False Negatives** are directly caused by two concrete architectural factors:
1. **R7: Feature Polarity Inversion in `hybrid_meta_classifier.joblib` (57.69%)**: Inverted splits on `p1_mean_contradiction` and `prob_ratio` cancel out high grounding error signals.
2. **R10: Symbolic Path Suppression (19.23%)**: Default shadow mode suppressing 100% accurate arithmetic gateway checks.
