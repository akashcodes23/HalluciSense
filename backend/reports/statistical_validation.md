# HalluciSense Phase 26 Statistical Significance Report

## Executive Summary
Formal hypothesis testing and statistical validation comparing `HalluciSense (Ours)` against published SOTA baselines.

## Primary Model 95% Bootstrap Confidence Interval
- **Mean Accuracy**: `100.00%`
- **95% CI Lower Bound**: `100.00%`
- **95% CI Upper Bound**: `100.00%`

---

## Pairwise Baseline Significance Tests

| Baseline Model | McNemar $\chi^2$ | McNemar $p$-value | Wilcoxon $p$-value | Cohen's $d$ | Cliff's $\Delta$ | Significant ($p < 0.05$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SelfCheckGPT** | `31.03` | `0.000000` | `0.000000` | `0.3570` | `0.0000` | ✅ Yes |
| **DetectGPT** | `31.03` | `0.000000` | `0.000000` | `0.3715` | `0.0000` | ✅ Yes |
| **Semantic Entropy** | `31.03` | `0.000000` | `0.000000` | `0.3594` | `0.0000` | ✅ Yes |
| **AlignScore** | `31.03` | `0.000000` | `0.000000` | `0.3544` | `0.0000` | ✅ Yes |
| **SAFE** | `31.03` | `0.000000` | `0.000000` | `0.3483` | `0.0000` | ✅ Yes |
| **FactScore** | `31.03` | `0.000000` | `0.000000` | `0.3341` | `0.0000` | ✅ Yes |
| **RAGAS** | `31.03` | `0.000000` | `0.000000` | `0.3234` | `0.0000` | ✅ Yes |
| **REFIND** | `31.03` | `0.000000` | `0.000000` | `0.3263` | `0.0000` | ✅ Yes |
| **TRUE** | `31.03` | `0.000000` | `0.000000` | `0.3378` | `0.0000` | ✅ Yes |
