# Phase 6E: Statistical Analysis & Hypothesis Testing Report

**Date**: 2026-08-11  

---

## 1. McNemar's Test Results
- **D0 (P1 Baseline) vs D4 (Epistemic Gate)**:
  - $b = 0$, $c = 0$, $\chi^2 = 0.0$, $p = 1.0000$ (Not Significant — preserves baseline assertion accuracy).
- **D1 (Naive Temporal) vs D4 (Epistemic Gate)**:
  - $b = 0$, $c = 10$, $\chi^2 = 8.1000$, $p = 0.0044$ (**Statistically Significant** at $\alpha=0.05$).
  - **Interpretation**: D4 significantly outperforms naive temporal rules ($D1$) by eliminating false positive penalties on non-assertion claims ($p=0.0044$).

---

## 2. Bootstrap 95% Confidence Intervals (5,000 Resamples)
- **F1 Score (D4)**: Mean = **0.9524**, 95% CI $[0.9348, 0.9688]$
- **Accuracy (D4)**: Mean = **95.00%**, 95% CI $[93.17\%, 96.67\%]$
