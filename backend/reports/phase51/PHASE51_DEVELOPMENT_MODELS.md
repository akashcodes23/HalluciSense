# PHASE 51 — DEVELOPMENT CANDIDATE MODELS & CROSS-VALIDATION
**5-Fold Stratified Cross-Validation Benchmarking of Non-Degenerate Classifiers**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `DIAGNOSTIC DEVELOPMENT BENCHMARK`

---

## 1. 5-Fold Stratified Cross-Validation Comparison

To evaluate whether alternative meta-classifiers could resolve the recall/threshold mismatch without degenerate gaming, three candidate architectures were benchmarked on the 19 canonical features across 5-fold stratified CV:

| Candidate Model | Mean MCC | Mean Recall | Mean Specificity | Mean AUROC | Non-Degenerate Check |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoosting (Refit on Diagnostic)** | **0.5929** | **0.9150** | **0.6500** | **0.8586** | ✅ Balanced & Non-Degenerate |
| **Calibrated Logistic Regression** | **0.5860** | **0.9450** | 0.5750 | **0.8562** | ✅ Linear & Interpretable |
| **Random Forest Meta-Classifier** | **0.5810** | **0.9250** | 0.6125 | 0.8489 | ✅ High Robustness |
| **Frozen Production Baseline ($\tau=0.54$)** | 0.1775 | 0.3100 | **0.8625** | 0.7183 | ⚠️ High Specificity / Low Recall |

---

## 2. Scientific Evaluation of Development Models

1. **Improvement Potential**: Refitting a meta-classifier with calibrated thresholds achieves an MCC of **~0.59** and AUROC of **0.8586**, boosting recall from $31\%$ to $>91\%$ while maintaining $>65\%$ specificity.
2. **Strict Production Freeze Rule**: In strict compliance with Phase 51 rules, these candidates remain **DEVELOPMENT BENCHMARKS ONLY**. The frozen production classifier (`hybrid_meta_classifier.joblib`), preprocessing scaler, and threshold $\tau^* = 0.54$ remain **100% UNTOUCHED**.
