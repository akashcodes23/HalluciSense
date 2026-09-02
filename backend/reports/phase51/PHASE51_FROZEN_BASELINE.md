# PHASE 51 — FROZEN DETECTOR BASELINE EVALUATION REPORT
**Scientific Correctness, Diagnostic Set Profile & Baseline Performance**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `EMPIRICALLY EVALUATED (FROZEN UNTOUCHED)`

---

## 1. Executive Summary & Diagnostic Baseline

The frozen HalluciSense detection pipeline (`hybrid_meta_classifier.joblib`, `preprocessing.joblib`, decision threshold $\tau^* = 0.54$, and canonical 19-feature schema) was evaluated across $N=280$ stratified diagnostic examples across 14 scientific categories.

### Baseline Summary Table:
- **Total Samples**: $N = 280$ (80 Factual $y=0$, 200 Hallucinated $y=1$)
- **Accuracy**: **46.79%** (131 / 280 correct)
- **Precision**: **0.8493** (84.93%)
- **Recall**: **0.3100** (31.00% — 62 / 200 hallucinations flagged)
- **Specificity**: **0.8625** (86.25% — 69 / 80 factual verified)
- **F1-Score**: **0.4542**
- **Matthews Correlation Coefficient (MCC)**: **0.1775**
- **Balanced Accuracy**: **0.5863**
- **AUROC**: **0.7183**
- **AUPRC**: **0.7879**
- **Brier Score**: **0.2918**
- **Expected Calibration Error (ECE)**: **0.3438**
- **Confusion Matrix**: $TN=69, FP=11, FN=138, TP=62$

---

## 2. Key Diagnostic Finding

The frozen model exhibits high specificity ($86.25\%$) and precision ($84.93\%$), meaning that when it flags a hallucination, it is almost certainly correct. However, it suffers from a high False Negative rate on short synthetic errors ($FN=138$), resulting in a low Recall ($31.00\%$) and modest MCC ($0.1775$). This occurs primarily because the frozen threshold $\tau^* = 0.54$ was originally fitted on a long-form multi-claim distribution where single-sentence errors hover in the $0.30 - 0.49$ probability range.
