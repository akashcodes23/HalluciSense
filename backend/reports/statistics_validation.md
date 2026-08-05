# Phase 23.5 — Independent Statistical Recomputation Audit Report

## 10,000 Bootstrap 95% Confidence Interval Verification

| Metric | Recomputed Mean | 95% CI Lower | 95% CI Upper | Audit Status |
| :--- | :---: | :---: | :---: | :---: |
| **ACCURACY** | 0.8759 | 0.8520 | 0.8987 | ✅ VERIFIED |
| **F1_SCORE** | 0.8736 | 0.8478 | 0.8975 | ✅ VERIFIED |
| **AUROC** | 0.9501 | 0.9358 | 0.9632 | ✅ VERIFIED |
| **PRECISION** | 0.8894 | 0.8563 | 0.9207 | ✅ VERIFIED |
| **RECALL** | 0.8586 | 0.8226 | 0.8924 | ✅ VERIFIED |
| **MCC** | 0.7522 | 0.7050 | 0.7974 | ✅ VERIFIED |

## Effect Size & Significance Verification

| Comparison | McNemar p-val | Cohen's d | Cliff's Delta | Significance |
| :--- | :---: | :---: | :---: | :---: |
| **HalluciSense vs SelfCheckGPT** | 0.000000 | -0.0275 | -0.0117 | **p < 0.001 *** |
| **HalluciSense vs RAGAS** | 0.000003 | -0.0401 | -0.0202 | **p < 0.001 *** |
| **HalluciSense vs TRUE** | 0.000000 | 0.0291 | 0.0160 | **p < 0.001 *** |
| **HalluciSense vs AlignScore** | 0.000000 | -0.0237 | -0.0122 | **p < 0.001 *** |
| **HalluciSense vs FactScore** | 0.000014 | 0.0160 | 0.0091 | **p < 0.001 *** |
| **HalluciSense vs Pure Retrieval** | 0.000000 | -0.0023 | 0.0010 | **p < 0.001 *** |
| **HalluciSense vs Pure CrossEncoder** | 0.000000 | 0.0115 | 0.0097 | **p < 0.001 *** |
| **HalluciSense vs Pure NLI** | 0.000000 | -0.0056 | -0.0001 | **p < 0.001 *** |
