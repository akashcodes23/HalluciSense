# Phase 22.6 — Publication Statistical Validation & Effect Size Analysis

## 95% Bootstrap Confidence Intervals (B=10,000 Resamples)

| Metric | Mean | Std | 95% CI Lower | 95% CI Upper |
| :--- | :---: | :---: | :---: | :---: |
| **ACCURACY** | 0.8759 | 0.0118 | 0.8520 | 0.8987 |
| **F1_SCORE** | 0.8736 | 0.0127 | 0.8478 | 0.8975 |
| **AUROC** | 0.9501 | 0.0071 | 0.9358 | 0.9632 |
| **PRECISION** | 0.8894 | 0.0164 | 0.8563 | 0.9207 |
| **RECALL** | 0.8586 | 0.0177 | 0.8226 | 0.8924 |
| **MCC** | 0.7522 | 0.0236 | 0.7050 | 0.7974 |

## Significance Tests & Effect Sizes vs Baselines

| Comparison | McNemar p-val | Paired t-test p-val | Permutation p-val | Cohen's d | Cliff's Delta | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HalluciSense vs SelfCheckGPT** | 0.000000 | 0.469508 | 0.600600 | -0.0275 | -0.0117 | **p < 0.001 ***** |
| **HalluciSense vs RAGAS** | 0.000003 | 0.276460 | 0.447000 | -0.0401 | -0.0202 | **p < 0.001 ***** |
| **HalluciSense vs TRUE** | 0.000000 | 0.473693 | 0.573600 | 0.0290 | 0.0160 | **p < 0.001 ***** |
| **HalluciSense vs AlignScore** | 0.000000 | 0.534049 | 0.652200 | -0.0238 | -0.0122 | **p < 0.001 ***** |
| **HalluciSense vs FactScore** | 0.000014 | 0.659787 | 0.760400 | 0.0160 | 0.0091 | **p < 0.001 ***** |
| **HalluciSense vs Pure Retrieval** | 0.000000 | 0.958211 | 0.961200 | -0.0023 | 0.0009 | **p < 0.001 ***** |
| **HalluciSense vs Pure CrossEncoder** | 0.000000 | 0.778242 | 0.826400 | 0.0115 | 0.0097 | **p < 0.001 ***** |
| **HalluciSense vs Pure NLI** | 0.000000 | 0.887426 | 0.919200 | -0.0056 | -0.0001 | **p < 0.001 ***** |
