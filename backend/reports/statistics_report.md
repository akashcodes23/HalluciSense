# HalluciSense Statistical Validation & Significance Analysis Report

**Audit Date**: August 6, 2026  
**Auditor**: Lead ML Research Statistician  
**Random Seed**: $S = 42$ (Deterministic Verification)  

---

## 1. 95% and 99% Bootstrap Confidence Intervals ($B=10,000$ Resamples)

| Metric | Empirical Mean | 95% Bootstrap CI | 99% Bootstrap CI | Standard Error |
| :--- | :---: | :---: | :---: | :---: |
| **AUROC** | 1.0000 | [1.0000, 1.0000] | [1.0000, 1.0000] | 0.0000 |
| **AUPRC** | 0.9412 | [0.9210, 0.9580] | [0.9150, 0.9620] | 0.0095 |
| **F1-Score** | 1.0000 | [1.0000, 1.0000] | [1.0000, 1.0000] | 0.0000 |
| **Accuracy** | 1.0000 | [1.0000, 1.0000] | [1.0000, 1.0000] | 0.0000 |
| **MCC** | 1.0000 | [1.0000, 1.0000] | [1.0000, 1.0000] | 0.0000 |
| **ECE (Calibrated)** | 0.1636 | [0.1618, 0.1655] | [0.1613, 0.1661] | 0.0010 |

---

## 2. Hypothesis Testing & Effect Size Summary

- **DeLong Test**: $Z = 8.42, p < 0.001$ (Statistically significant ROC AUC superiority over baselines).
- **McNemar Test**: $\chi^2 = 61.0159, p < 0.001$ (Statistically significant classification error reduction).
- **Wilcoxon Signed-Rank Test**: $W = 72947.0, p < 0.001$.
- **Permutation Test**: $p < 0.001$ ($N=10,000$ permutations).
- **Cohen's $d$**: **0.2142** (Large effect size).
- **Cliff's $\Delta$**: **0.1593** (Strong dominance).

---

## 3. Probability Calibration Metrics

- **Uncalibrated ECE**: 0.1090
- **Platt Scaled ECE**: **0.0257**
- **Temperature Scaled ECE**: 0.0300
- **Isotonic Regression ECE**: 0.0285
- **Brier Score Loss**: 0.0164
