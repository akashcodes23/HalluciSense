# HalluciSense Pillar-1: Statistical Tests

*Generated: 2026-08-03T04:49:02.142488+00:00*  
*Phase: 6K (Frozen)*

---

## 1. AUC Significance Test

The held-out ROC-AUC of 0.6902 was compared to random chance (0.5000):
- **Test**: DeLong AUC test (asymptotic normal approximation)
- **Effect size**: 0.6902 - 0.5000 = 0.1902
- **Interpretation**: Statistically and practically significant above chance

## 2. Bootstrap Confidence Intervals

All CIs use 2000 bootstrap iterations with the percentile method:
- **VAL ROC-AUC 95% CI**: See `heldout_bootstrap_ci.json`
- **Feature odds ratio CIs**: See `step3_feature_importance.json`

## 3. McNemar's Test (Pillar-1 vs Single-Feature Baseline)

McNemar's test was performed comparing Pillar-1 predictions (τ=0.56) versus
the single-feature `min_support_margin` baseline on the VAL set:
- **Null hypothesis**: The two models make the same errors
- **Statistic and p-value**: Computed in `full_dev_statistical_tests.json`

## 4. Stability Gate Analysis

32 bootstrap iterations on the full DEV set (Phase 6K):
- **Pass criterion**: All coefficient signs consistent across iterations
- **Result**: 32/32 PASS
- **Interpretation**: The model's feature direction assignments are stable

## 5. Solver Consistency Test

Four solvers (lbfgs, liblinear, newton-cg, saga) fitted on same data:
- **Coefficient agreement**: Max |Δcoef| < 0.001 across warning-free solvers
- **Interpretation**: Optimization is solver-invariant (liblinear selected for zero warnings)

## 6. Partition Leakage Audit

Confirmed zero samples appear in both DEV and VAL:
- SHA-256 fingerprints of all samples verified
- Leakage rate: **0.000%**

## 7. Distribution Shift Analysis

DEV vs VAL Kolmogorov-Smirnov test for each feature:
- All KS p-values reported in `dev_val_distribution_shift.json`
- Positive class ratio shift: 54.3% (DEV) → 47.1% (VAL) — moderate shift
