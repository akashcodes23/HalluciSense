# Phase 9 — Step 3: Feature Importance Analysis

**Generated**: 2026-08-03T04:48:14.470832+00:00
**Base VAL ROC-AUC**: 0.6902

## 1. Coefficient Ranking (Standardized by IQR)

| Rank | Feature | Raw Coef | Std Coef | Odds Ratio | Direction |
| --- | --- | --- | --- | --- | --- |
| 1 | `min_support_margin` | -1.2485 | -1.2319 | 0.2869 | negative |
| 2 | `mean_contradiction` | -0.4409 | -0.3564 | 0.6434 | negative |
| 3 | `num_claims` | 0.0873 | 0.3492 | 1.0912 | positive |
| 4 | `mean_entailment` | 0.1054 | 0.0047 | 1.1112 | positive |
| 5 | `max_entailment` | -0.0507 | -0.0044 | 0.9506 | negative |

## 2. Odds Ratios with 95% Bootstrap CIs

| Feature | OR | 95% CI Lower | 95% CI Upper | Significant |
| --- | --- | --- | --- | --- |
| `mean_entailment` | 1.1112 | 1.1116 | 1.2027 | ✅ |
| `max_entailment` | 0.9506 | 0.9161 | 1.0042 | ❌ |
| `mean_contradiction` | 0.6434 | 1.0855 | 2.3076 | ✅ |
| `min_support_margin` | 0.2869 | 0.2083 | 0.4792 | ✅ |
| `num_claims` | 1.0912 | 0.9532 | 1.0669 | ❌ |

## 3. Permutation Importance

| Rank | Feature | Mean AUC Drop | Std |
| --- | --- | --- | --- |
| 1 | `min_support_margin` | 0.2076 | 0.0084 |
| 2 | `mean_entailment` | 0.1198 | 0.0061 |
| 3 | `max_entailment` | 0.0186 | 0.0039 |
| 4 | `mean_contradiction` | 0.0136 | 0.0035 |
| 5 | `num_claims` | 0.0046 | 0.0032 |

## 4. Figures Generated

- `step3_coefficient_ranking.png` — Standardized coefficient bar chart
- `step3_odds_ratios.png` — Forest plot with bootstrap CIs
- `step3_permutation_importance.png` — ROC-AUC drop per feature
- `step3_partial_dependence.png` — PDP for min_support_margin + mean_contradiction
- `step3_coefficient_comparison.png` — Raw vs standardized coefficient comparison