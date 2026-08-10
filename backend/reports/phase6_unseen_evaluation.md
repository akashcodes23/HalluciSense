# Phase 6 Unseen Benchmark Validation Report

## 1. Executive Summary
Evaluation of the Phase 6 temporal reasoning framework across **105 completely novel unseen cases** spanning 15 temporal categories and 15 domains.

### Key Performance Highlights:
- **Accuracy**: **53.33%** (56/105)
- **Precision**: **41.03%**
- **Recall**: **91.43%**
- **F1 Score**: **0.5664**
- **Specificity**: **34.29%**
- **False Positive Rate (FPR)**: **65.71%**
- **False Negative Rate (FNR)**: **8.57%**
- **Engine Latency**: Mean = **0.0180 ms** (17.99 $\mu	ext{s}$), P95 = **0.0186 ms**
- **Determinism Check**: **True** (100% deterministic over 30 runs)

---

## 2. Confusion Matrix

$$\begin{pmatrix} TP = 32 & FP = 46 \\ FN = 3 & TN = 24 \end{pmatrix}$$
