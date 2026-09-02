# PHASE 52 — DEVELOPMENT CANDIDATES (5-FOLD STRATIFIED CV)
**Benchmarking Non-Degenerate Alternative Classifiers on Balanced Data**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `DEVELOPMENT ONLY (FROZEN PRODUCTION PRESERVED)`

---

## 1. 5-Fold Stratified Cross-Validation Results ($N=300$)

| Candidate Architecture | Mean MCC | Mean Recall | Mean Specificity | Mean AUROC | Mean AUPRC | Mean Brier | Mean ECE | Non-Degenerate Check |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate B: HistGradientBoosting** | **0.4893** | **0.7000** | **0.7867** | **0.8388** | **0.8459** | **0.1634** | 0.1391 | ✅ Passes All Checks |
| **Candidate A: Calibrated Logistic Regression**| **0.4879** | **0.6533** | **0.8267** | **0.8377** | **0.8525** | **0.1672** | **0.1139** | ✅ Passes All Checks |
| **Candidate C: Random Forest** | **0.4805** | **0.6600** | **0.8133** | **0.8396** | **0.8467** | **0.1617** | 0.1234 | ✅ Passes All Checks |
| **Frozen Production Classifier ($\tau=0.54$)**| 0.1647 | 0.3067 | 0.8333 | 0.6905 | 0.5883 | 0.2516 | 0.2043 | ⚠️ Frozen Baseline |

---

## 2. Scientific Evaluation & Recommendations

1. **Massive Performance Recovery**: Properly fitting an aligned classifier on the 19 features restores AUROC to **0.8388** (matching P1 grounding potential), boosts Recall from **30.67%** to **70.00%**, and triples MCC from **0.1647** to **0.4893**.
2. **Strict Production Freeze**: In compliance with Phase 52 mandates, these models remain **STRICTLY DEVELOPMENT CANDIDATES**. No replacement of `hybrid_meta_classifier.joblib` or threshold modification has occurred in production.
