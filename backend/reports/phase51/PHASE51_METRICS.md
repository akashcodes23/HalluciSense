# PHASE 51 — COMPREHENSIVE PERFORMANCE & CATEGORY BREAKDOWN
**Empirical Metrics Across All 14 Stratified Categories**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & VERIFIED`

---

## 1. Category-by-Category Metric Breakdown

| Category Code | Description | N | Accuracy | Recall | Specificity | Mean $P(H)$ | Errors | Easiest / Hardest |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `A_clearly_factual` | Clearly factual truths | 20 | **100.0%** | 1.0000 | **1.0000** | 0.1688 | 0/20 | 🏆 Easiest Factual |
| `M_paraphrase` | Complex semantic paraphrases | 20 | **100.0%** | 1.0000 | **1.0000** | 0.1814 | 0/20 | 🏆 Robust Alignment |
| `H_numerical_correctness` | Arithmetic correctness | 20 | **100.0%** | 1.0000 | **1.0000** | 0.3021 | 0/20 | ✅ High Specificity |
| `F_multi_claim_contradiction`| Contradictory claim pairs | 20 | **70.0%** | **0.7000** | 0.0000 | 0.6433 | 6/20 | ✅ Strong P3 Detection |
| `J_entity_swap` | Entity-attribute swaps | 20 | 45.0% | 0.4500 | 0.0000 | 0.4893 | 11/20 | ⚠️ Moderate Recall |
| `L_negation` | Syntactic negations | 20 | 45.0% | 0.4500 | 0.0000 | 0.4699 | 11/20 | ⚠️ NLI Polarity Split |
| `K_temporal_mutation` | Historical year mutations | 20 | 45.0% | 0.4500 | 0.0000 | 0.4409 | 11/20 | ⚠️ Temporal Grounding |
| `G_multi_claim_consistency`| Multi-claim consistent sets | 20 | 45.0% | 1.0000 | 0.4500 | 0.5295 | 11/20 | ⚠️ Multi-claim Sensitivity |
| `C_direct_contradiction` | Empirical physical contradictions| 20 | 35.0% | 0.3500 | 0.0000 | 0.4476 | 13/20 | ⚠️ Threshold Sub-0.54 |
| `D_unsupported_claim` | Completely fabricated claims | 20 | 35.0% | 0.3500 | 0.0000 | 0.4784 | 13/20 | ⚠️ Retrieval Missingness |
| `B_clearly_false` | Counterfactual city/planet swaps | 20 | 15.0% | 0.1500 | 0.0000 | 0.3906 | 17/20 | ❌ False Negative Cluster |
| `E_ambiguous_claim` | Unfalsifiable/vague claims | 20 | 10.0% | 0.1000 | 0.0000 | 0.3005 | 18/20 | ❌ High Uncertainty |
| `N_unsupported_causal` | Fabricated causal explanations | 20 | 10.0% | 0.1000 | 0.0000 | 0.3370 | 18/20 | ❌ Causal Grounding Hard |
| `I_numerical_error` | Arithmetic calculation errors | 20 | **0.0%** | **0.0000** | 0.0000 | 0.3023 | 20/20 | 🚨 Hardest Error (Math) |

---

## 2. Hardest and Easiest Categories

- **Easiest Categories**: `A_clearly_factual` (100% accuracy), `M_paraphrase` (100% accuracy), and `H_numerical_correctness` (100% accuracy). The model is exceptionally good at confirming true knowledge and not raising false alarms.
- **Hardest Categories**: `I_numerical_error` (0% recall), `E_ambiguous_claim` (10% recall), and `N_unsupported_causal` (10% recall). Pure language models without symbolic math gateways treat $12 \times 8 = 95$ with generic confidence ($P \approx 0.30$).
