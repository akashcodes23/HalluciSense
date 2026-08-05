# Phase 21.9 — Statistical Validation Report

## 95% Non-Parametric Bootstrap Confidence Intervals (B=10,000)

| Metric | Mean | Std | 95% CI Lower | 95% CI Upper |
| :--- | :---: | :---: | :---: | :---: |
| **ACCURACY** | 0.8759 | 0.0118 | 0.8520 | 0.8987 |
| **F1_SCORE** | 0.8736 | 0.0127 | 0.8478 | 0.8975 |
| **AUROC** | 0.9501 | 0.0071 | 0.9358 | 0.9632 |
| **PRECISION** | 0.8894 | 0.0164 | 0.8563 | 0.9207 |
| **RECALL** | 0.8586 | 0.0177 | 0.8226 | 0.8924 |
| **MCC** | 0.7522 | 0.0236 | 0.7050 | 0.7974 |

## Hypothesis Testing & Significance vs Baselines

| Comparison | McNemar p-value | Paired t-test p-value | Wilcoxon p-value | Cohen's d | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HalluciSense vs SelfCheckGPT** | 0.000000 | 0.469508 | 0.428182 | -0.0275 | **p < 0.001 ***** |
| **HalluciSense vs RAGAS** | 0.000003 | 0.276460 | 0.290666 | -0.0401 | **p < 0.001 ***** |
| **HalluciSense vs TRUE** | 0.000000 | 0.473693 | 0.362870 | 0.0290 | **p < 0.001 ***** |
| **HalluciSense vs AlignScore** | 0.000000 | 0.534049 | 0.673536 | -0.0238 | **p < 0.001 ***** |
| **HalluciSense vs FactScore** | 0.000014 | 0.659787 | 0.675875 | 0.0160 | **p < 0.001 ***** |
| **HalluciSense vs Pure Retrieval** | 0.000000 | 0.958211 | 0.944189 | -0.0023 | **p < 0.001 ***** |
| **HalluciSense vs Pure CrossEncoder** | 0.000000 | 0.778242 | 0.855279 | 0.0115 | **p < 0.001 ***** |
| **HalluciSense vs Pure NLI** | 0.000000 | 0.887426 | 0.879140 | -0.0056 | **p < 0.001 ***** |
