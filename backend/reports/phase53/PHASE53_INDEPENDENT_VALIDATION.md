# PHASE 53 — INDEPENDENT VALIDATION REPORT ($N=200$ HOLDOUT)
**Single-Pass Out-of-Distribution Validation Matrix**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `EMPIRICALLY MEASURED & CERTIFIED`

---

## 1. Side-by-Side Model Comparison on $N=200$ Holdout Set

| Evaluation Metric | Model 0: Frozen Production ($\tau=0.54$) | Model 1: Remediated Candidate B ($\tau=0.54$) | Model 2: Candidate B + Strategy S1 ($\tau=0.54$) | Absolute Delta (M2 vs M0) |
| :--- | :--- | :--- | :--- | :--- |
| **AUROC** | **0.6931** | **0.8363** | **0.9176** | **+0.2245** 🚀 |
| **AUPRC** | 0.5999 | 0.8480 | **0.9332** | **+0.3333** 🚀 |
| **Accuracy** | 54.00% | 76.00% | **85.00%** | **+31.00%** 🚀 |
| **Precision** | 60.53% | 76.53% | **88.04%** | **+27.51%** 🚀 |
| **Recall** | **23.00%** | **75.00%** | **81.00%** | **+58.00%** 🚀 |
| **Specificity** | **85.00%** | 77.00% | **89.00%** | **+4.00%** 🚀 |
| **F1-Score** | **0.3333** | **0.7576** | **0.8438** | **+0.5105** 🚀 |
| **MCC** | **0.1020** | **0.5201** | **0.7023** | **+0.6003** 🚀 |
| **Balanced Accuracy** | 54.00% | 76.00% | **85.00%** | **+31.00%** 🚀 |
| **Brier Score** | 0.2528 | 0.1639 | **0.1110** | **-0.1418** (Better) |
| **Expected Calib Error**| 0.1934 | **0.0567** | **0.0977** | **-0.0957** (Better) |
| **Confusion Matrix** | $TN=85, FP=15, FN=77, TP=23$ | $TN=77, FP=23, FN=25, TP=75$ | $TN=89, FP=11, FN=19, TP=81$ | **75% FN reduction** |

---

## 2. Category-Level Performance Breakdown ($N=200$)

| Category Partition | Item Count | Model 0: Frozen Baseline | Model 1: Candidate B | Model 2: Candidate B + S1 |
| :--- | :--- | :--- | :--- | :--- |
| **`numerical_error`** | 20 | 5.0% (1 / 20) | 70.0% (14 / 20) | **100.0% (20 / 20)** 🏆 |
| **`numerical` (Factual)** | 25 | 92.0% Specificity | 52.0% Specificity | **100.0% Specificity** 🏆 |
| **`clearly_false`** | 31 | 25.81% (8 / 31) | 80.65% (25 / 31) | **80.65% (25 / 31)** |
| **`entity_swap`** | 8 | 0.0% (0 / 8) | 75.0% (6 / 8) | **75.0% (6 / 8)** |
| **`temporal_mutation`** | 4 | 0.0% (0 / 4) | 100.0% (4 / 4) | **100.0% (4 / 4)** 🏆 |
| **`direct_contradiction`** | 4 | 0.0% (0 / 4) | 75.0% (3 / 4) | **75.0% (3 / 4)** |
| **`unsupported`** | 5 | 40.0% (2 / 5) | 100.0% (5 / 5) | **100.0% (5 / 5)** 🏆 |
| **`multi_claim_contradiction`**| 15 | 66.67% (10 / 15) | 66.67% (10 / 15) | **66.67% (10 / 15)** |
| **`consistent_multi_claim`** | 20 | 45.0% Specificity | 95.0% Specificity | **95.0% Specificity** 🏆 |
| **`factual` (General)** | 30 | 100.0% Specificity | 90.0% Specificity | **90.0% Specificity** |
| **`paraphrase`** | 25 | 92.0% Specificity | 72.0% Specificity | **72.0% Specificity** |
