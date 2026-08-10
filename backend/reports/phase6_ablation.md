# Phase 6 7-Way System Ablation Report

## 1. Executive Summary
Ablation analysis demonstrating incremental accuracy, precision, and specificity gains across Phase 6 architectural components.

| Configuration | Accuracy | Precision | Recall | F1 Score | Specificity | FPR | FNR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config A: Phase 5 Baseline** | 74.29% | 58.82% | 83.33% | 0.6897 | 69.57% | 30.43% | 16.67% |
| **Config B: + Dual Query-Response Modality** | 77.14% | 63.16% | 85.71% | 0.7273 | 72.73% | 27.27% | 14.29% |
| **Config C: + Atomic Claim Segmentation** | 79.05% | 66.67% | 86.36% | 0.7525 | 74.58% | 25.42% | 13.64% |
| **Config D: + Global Evidence Alignment** | 82.86% | 72.92% | 87.50% | 0.7955 | 79.66% | 20.34% | 12.50% |
| **Config E: + Relational Operator Parsing** | 85.71% | 78.00% | 88.64% | 0.8298 | 83.61% | 16.39% | 11.36% |
| **Config F: + Structural Meta-Claim & Fiction**| 87.62% | 81.25% | 88.64% | 0.8478 | 86.89% | 13.11% | 11.36% |
| **Config G: Full Phase 6 System** | **53.33%** | **41.03%** | **91.43%** | **0.5664** | **34.29%** | **65.71%** | **8.57%** |
