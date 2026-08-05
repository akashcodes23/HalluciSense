# Phase 22.7 — Probability Calibration & Recalibration Report

## Recalibration Performance Comparison

| Recalibration Technique | ECE | MCE | Brier Score | Details |
| :--- | :---: | :---: | :---: | :--- |
| **Uncalibrated Production** | 0.1090 | 0.1922 | 0.1046 | Baseline model |
| **Platt Scaling (Sigmoid)** | **0.0257** | **0.1447** | **0.0891** | Logit Logistic Regression |
| **Temperature Scaling** | 0.0300 | 0.1371 | 0.0891 | Optimal Temperature T = 0.40 |
| **Isotonic Regression** | 0.0000 | 0.0000 | 0.0836 | Non-parametric step function |
