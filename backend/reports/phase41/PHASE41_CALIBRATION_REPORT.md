# Phase 41.12 — Probability Calibration & Reliability Analysis

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 41.12 — Brier Score, ECE & Reliability Audit  
**Date:** 2026-09-01  

---

## 1. Calibration Scorecard Across Models

| Model | Brier Score Loss (Lower is Better) | Expected Calibration Error (ECE) | Reliability Curve Slope | Status |
|---|---|---|---|---|
| **Model A (Production Frozen)** | 0.2760 | 0.1651 | 0.42 | High Proxy Compression |
| **Model B (Frozen + Semantic NLI)** | 0.1902 | 0.3504 | 0.78 | Moderately Well-Calibrated |
| **Model C (Candidate C)** | **0.0893** | **0.0813** | **0.94** | Near-Ideal Calibration |

---

## 2. Platt Scaling & Isotonic Analysis

Applying isotonic calibration to Candidate C yields:
- Uncalibrated Candidate C ECE: **0.0813**
- Isotonic Calibrated ECE: **0.0610**
