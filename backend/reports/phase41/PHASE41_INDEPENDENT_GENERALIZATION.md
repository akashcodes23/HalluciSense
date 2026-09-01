# Phase 41.9 & 41.11 — Independent Generalization Benchmark Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 41.9/41.11 — 300-Case Independent Holdout Evaluation  
**Holdout Sample Count:** $N = 300$ (150 Factual, 150 Hallucinated)  
**Date:** 2026-09-01  

---

## 1. Multi-Model Benchmark Comparison Table

| Metric | Model A (Production Frozen + Proxy) | Model B (Production Frozen + Semantic NLI) | Model C (Candidate C + Semantic NLI) | Scientific Delta (C vs. A) |
|---|---|---|---|---|
| **ROC-AUC** | 0.5362 | 0.9890 | **0.9091** | **++0.3729** |
| **PR-AUC** | 0.5306 | 0.9896 | **0.9069** | **++0.3763** |
| **F1 Score ($	au=0.54$)** | 0.6667 | 0.7673 | **0.8968** | **++0.2301** |
| **Accuracy** | 0.5200 | 0.6967 | **0.9033** | **++0.3833** |
| **Precision** | 0.5106 | 0.6224 | **0.9618** | **++0.4512** |
| **Recall** | 0.9600 | 1.0000 | **0.8400** | **+-0.1200** |
| **Brier Score (Calibration)** | 0.2760 | 0.1902 | **0.0893** | **-0.1867 (Better)** |
| **Expected Calibration Error** | 0.1651 | 0.3504 | **0.0813** | **-0.0838 (Better)** |

---

## 2. Scientific Analysis

1. **Model A (Legacy):** Suffers from representation collapse due to keyword relevance proxies.
2. **Model B (Forward Compatible):** Semantic grounding immediately unlocks ROC-AUC = 0.9890 without any retraining.
3. **Model C (Candidate C):** Achieves ROC-AUC = 0.9091 and F1 = 0.8968 by calibrating directly to the continuous support margin distributions.
