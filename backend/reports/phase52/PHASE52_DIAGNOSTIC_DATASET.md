# PHASE 52 — CONTROLLED 50/50 BALANCED FORENSIC DATASET
**Stratified Diagnostic Dataset Specification ($N=300$)**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `FROZEN 50/50 FORENSIC DATASET`

---

## 1. Class Balance & Stratification Summary

- **Total Samples ($N$)**: **300**
- **Factual / Non-Hallucinated ($y=0$)**: **150 (50.0%)**
- **Hallucinated / Contradictory ($y=1$)**: **150 (50.0%)**
- **Exact Class Ratio**: **1.000 : 1.000** (Zero class imbalance skew)

### Stratification Matrix:

| Class | Category Name | Count | Percentage |
| :--- | :--- | :--- | :--- |
| **Factual ($y=0$)** | `A_clearly_factual` | 40 | 13.33% |
| **Factual ($y=0$)** | `M_paraphrase` | 40 | 13.33% |
| **Factual ($y=0$)** | `H_numerical_correctness` | 35 | 11.67% |
| **Factual ($y=0$)** | `G_multi_claim_consistency` | 35 | 11.67% |
| *Subtotal Factual* | *All 4 Factual Categories* | **150** | **50.00%** |
| **Hallucinated ($y=1$)** | `B_clearly_false` | 20 | 6.67% |
| **Hallucinated ($y=1$)** | `C_direct_contradiction` | 20 | 6.67% |
| **Hallucinated ($y=1$)** | `D_unsupported_claim` | 20 | 6.67% |
| **Hallucinated ($y=1$)** | `E_ambiguous_claim` | 15 | 5.00% |
| **Hallucinated ($y=1$)** | `F_multi_claim_contradiction`| 15 | 5.00% |
| **Hallucinated ($y=1$)** | `I_numerical_error` | 20 | 6.67% |
| **Hallucinated ($y=1$)** | `J_entity_swap` | 15 | 5.00% |
| **Hallucinated ($y=1$)** | `K_temporal_mutation` | 15 | 5.00% |
| **Hallucinated ($y=1$)** | `L_negation` | 10 | 3.33% |
| *Subtotal Hallucinated* | *All 9 Hallucinated Categories*| **150** | **50.00%** |
| **TOTAL** | **All 13 Stratified Categories** | **300** | **100.00%** |

---

## 2. Artifact Path
- `backend/reports/phase52/forensic_50_50_dataset.json`
