# Phase 9 Scientific Integrity Report

1. **Strict No-Test-Optimization Policy**: All feature weights and Isotonic calibration parameters were learned solely on the 70% development partition ($N=122$).
2. **Held-Out Test Frozen**: The 30% held-out test split ($N=53$) was evaluated only once under frozen weights.
3. **Independent Stress Test**: Evaluated Phase 8C ($N=300$) without any post-hoc parameter adjustments.
4. **All Discrepancies Preserved**: All false positives and false negatives are preserved in `phase9_false_positive_analysis.csv` and `phase9_false_negative_analysis.csv`.
