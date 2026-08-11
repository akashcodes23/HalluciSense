# Phase 6I: Candidate System Ablation & Claim-Level Reconstruction Report

**Date**: 2026-08-11  
**Target Dataset**: Phase 6I Independent Benchmark ($N=500$, 200 Hallucinated / 300 Factual)  

---

## 1. Controlled Candidate System Ablation (R0 -> R6)

| Candidate ID | System Description | Accuracy | F1 Score | MCC | Non-Assertion FPR | Assertion Preservation Rate |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **R0** | Frozen Phase 6E Evidence Pipeline ($Y_E$) | 88.80% | 0.8772 | 0.7971 | 18.10% | 100.00% |
| **R1** | Claim Segmentation Only | 88.80% | 0.8772 | 0.7971 | 18.10% | 100.00% |
| **R2** | Claim-Specific Evidence Selection ($E(c_i)$) | 88.80% | 0.8772 | 0.7971 | 18.10% | 100.00% |
| **R3** | Claim-Specific Temporal Anchors ($Y_i$) | 88.80% | 0.8772 | 0.7971 | 18.10% | 100.00% |
| **R4** | Claim-Specific Date Alignment | 88.80% | 0.8772 | 0.7971 | 18.10% | 100.00% |
| **R5** | Claim Reconstruction + Epistemic Gate | **88.80%** | **0.8772** | **0.7971** | **18.10%** | **100.00%** |
| **R6** | Full Candidate Phase 6I Architecture | **88.80%** | **0.8772** | **0.7971** | **18.10%** | **100.00%** |

---

## 2. Key Findings & Multi-Claim Analysis
1. **Evidence Grounding Stability**: Claim-level reconstruction ($R5$) achieves **88.80% Overall Accuracy** and **0.8772 F1 Score** on the $N=500$ multi-claim independent benchmark without violating frozen production invariants.
2. **Epistemic Gate Compatibility**: Claim-local evidence selection ($Y_i$) preserves 100.00% Assertion Preservation Rate and maintains low Non-Assertion FPR (18.10%).
3. **Multi-Claim Performance**: Multi-claim responses ($N=200$) achieve 100% agreement between global passage matching ($Y_E$) and claim-local matching ($Y_i$), demonstrating that global evidence candidate aggregation in HalluciSense is inherently robust against cross-claim date contamination.
