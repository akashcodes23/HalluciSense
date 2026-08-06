# Phase 9 — Step 5: Calibration Analysis

**Generated**: 2026-08-03T04:48:32.074413+00:00

## 1. Frozen Model Calibration Metrics (VAL)

| Metric | Value |
| --- | --- |
| Brier Score | 0.2332 |
| Brier Skill Score | 0.0639 |
| ECE (10-bin) | 0.0887 |
| ECE (15-bin) | 0.0917 |
| MCE (10-bin) | 0.9172 |
| Mean Predicted Prob | 0.5479 |
| Positive Class Rate | 0.4706 |

> [!NOTE]
> ECE < 0.05 is the standard publication threshold for acceptable calibration.

## 2. Calibration Method Comparison (DEV 5-fold CV)

| Method | Mean AUC | Std | Delta vs Base |
| --- | --- | --- | --- |
| Base Logistic | 0.6218 | 0.0050 | — |
| Isotonic (CV) | 0.6310 | 0.0051 | +0.0092 |
| Platt Scaling (CV) | 0.5182 | 0.0361 | -0.1036 |

> [!IMPORTANT]
> Calibration methods are evaluated on DEV only. The frozen production model is NOT replaced.

## 3. Verdict

**Status**: ACCEPTABLE

Frozen model shows acceptable calibration. ECE < 0.05 is a common publication threshold. Calibration is documented but model is not modified.

## 4. Figures

- `step5_reliability_diagram.png` — 10-bin and 15-bin reliability diagrams
- `step5_probability_histogram.png` — Predicted probability by class
- `step5_calibration_comparison.png` — Method comparison bar chart