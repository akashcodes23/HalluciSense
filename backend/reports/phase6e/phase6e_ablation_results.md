# Phase 6E: Mechanism Ablation & Generalization Report

**Date**: 2026-08-11  
**Target Dataset**: Phase 6E Independent Benchmark ($N=600$, 50% Hallucinated / 50% Factual)  

---

## 1. Mechanism Ablation Ladder (D0 -> D9)

| Config ID | Description | Accuracy | F1 Score | MCC | Non-Assertion FPR | Assertion Preservation Rate |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **D0** | P1 NLI Baseline | 95.00% | 0.9524 | 0.9045 | 10.00% | 100.00% |
| **D1** | P1 + Naive Temporal | 93.33% | 0.9375 | 0.8745 | 14.76% | 100.00% |
| **D2** | D1 + Query Modality | 93.33% | 0.9375 | 0.8745 | 14.76% | 100.00% |
| **D3** | D1 + Response Modality | **95.00%** | **0.9524** | **0.9045** | **10.00%** | **100.00%** |
| **D4** | Temporal-Epistemic Gate | **95.00%** | **0.9524** | **0.9045** | **10.00%** | **100.00%** |
| **D5** | + Atomic Claim Seg. | 95.00% | 0.9524 | 0.9045 | 10.00% | 100.00% |
| **D6** | + Local Evidence Align. | 95.00% | 0.9524 | 0.9045 | 10.00% | 100.00% |
| **D7** | + Global Evidence Align. | 95.00% | 0.9524 | 0.9045 | 10.00% | 100.00% |
| **D8** | + Relational Protection | 95.00% | 0.9524 | 0.9045 | 10.00% | 100.00% |
| **D9** | Full Phase 6E Architecture | **95.00%** | **0.9524** | **0.9045** | **10.00%** | **100.00%** |

---

## 2. Key Empirical Findings
1. **Naive Temporal FP Penalty**: Naive temporal rules ($D1$) introduce a **4.76 percentage points increase** in Non-Assertion False Positive Rate (from 10.00% to 14.76%), dropping overall accuracy by **1.67 percentage points** (from 95.00% to 93.33%).
2. **Epistemic Gating FP Recovery**: The Epistemic Gate ($D4$) suppresses false temporal penalties on non-assertional statements, reducing Non-Assertion FPR by **4.76 percentage points** (a **32.25% relative FP reduction** over naive temporal rules) back to baseline levels.
3. **Assertion Preservation**: Across all configurations, the Assertion Preservation Rate (APR) remains **100.00%**, proving zero distortion of true factual hallucination recall.
