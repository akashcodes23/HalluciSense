# PHASE 51 — STRATIFIED DIAGNOSTIC DATASET SPECIFICATION
**Dataset Metadata, Taxonomy & Distribution Audit**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `DATASET FROZEN & STRATIFIED`

---

## 1. Dataset Taxonomy & Class Distribution

| Category Code | Category Description | Total Items | Factual (y=0) | Hallucinated (y=1) |
| :--- | :--- | :--- | :--- | :--- |
| `A_clearly_factual` | Clearly factual established facts | 20 | 20 | 0 |
| `B_clearly_false` | Clearly false counterfactual claims | 20 | 0 | 20 |
| `C_direct_contradiction` | Direct physical/empirical contradictions | 20 | 0 | 20 |
| `D_unsupported_claim` | Completely unsupported fabricated claims | 20 | 0 | 20 |
| `E_ambiguous_claim` | Vague, unfalsifiable, or ambiguous claims | 20 | 0 | 20 |
| `F_multi_claim_contradiction` | Intra-response contradictory claim pairs | 20 | 0 | 20 |
| `G_multi_claim_consistency` | Multi-claim coherent & consistent responses | 20 | 20 | 0 |
| `H_numerical_correctness` | Arithmetic & mathematical truths | 20 | 20 | 0 |
| `I_numerical_error` | Arithmetic & mathematical errors | 20 | 0 | 20 |
| `J_entity_swap` | Entity-attribute swaps | 20 | 0 | 20 |
| `K_temporal_mutation` | Historical & chronological mutations | 20 | 0 | 20 |
| `L_negation` | Direct syntactic/semantic negations | 20 | 0 | 20 |
| `M_paraphrase` | Complex semantic paraphrases | 20 | 20 | 0 |
| `N_unsupported_causal` | Fabricated causal/mechanistic explanations | 20 | 0 | 20 |
| **TOTAL** | **All 14 Stratified Categories** | **280** | **80 (28.6%)** | **200 (71.4%)** |

---

## 2. Dataset Storage & Artifact Path

- **Artifact Path**: `backend/reports/phase51/diagnostic_dataset.json`
- **Total Diagnostic Samples**: `N = 280`
- **Domain Coverage**: Physical sciences, world history, geography, mathematics, biology, astronomy, literature, and general encyclopedic facts.
