# PHASE 51 — CALIBRATION, RELIABILITY & UNCERTAINTY REPORT
**Brier Score, Expected Calibration Error & Reliability Bin Analysis**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & AUDITED`

---

## 1. Quantitative Calibration Metrics

- **Brier Score Loss**: **0.2918** (Benchmark threshold for good calibration: $< 0.15$).
- **Expected Calibration Error (ECE)**: **0.3438** (34.38% average divergence between predicted probability and empirical accuracy).

---

## 2. 10-Bin Reliability Table

| Probability Bin | Sample Count | Observed Accuracy (True $P_H$) | Mean Predicted Confidence ($P_H$) | Bin Calibration Error |
| :--- | :--- | :--- | :--- | :--- |
| `[0.0, 0.1]` | 9 | 0.4444 | 0.0832 | 0.3612 |
| `[0.1, 0.2]` | 37 | 0.1892 | 0.1349 | **0.0543** (Well Calibrated) |
| `[0.2, 0.3]` | 84 | 0.7381 | 0.2445 | **0.4936** (Under-confident) |
| `[0.3, 0.4]` | 3 | 0.6667 | 0.3259 | 0.3408 |
| `[0.4, 0.5]` | 31 | 0.8387 | 0.4530 | 0.3857 |
| `[0.5, 0.6]` | 91 | 0.9121 | 0.5480 | **0.3641** |
| `[0.6, 0.7]` | 4 | 0.7500 | 0.6271 | 0.1229 |
| `[0.7, 0.8]` | 17 | 0.7059 | 0.7422 | **0.0363** (Well Calibrated) |
| `[0.8, 0.9]` | 4 | 0.2500 | 0.8342 | 0.5842 |
| `[0.9, 1.0]` | 0 | 0.0000 | 0.0000 | 0.0000 |

---

## 3. Scientific Calibration Conclusion

The frozen model's raw probabilities compress heavily into the $0.20 - 0.30$ and $0.50 - 0.60$ bins. In the $0.20 - 0.30$ bin, $73.81\%$ of samples were actually hallucinated, showing that the frozen model systematically underestimates risk for short single-claim errors. Platt scaling or isotonic calibration on development splits will significantly improve probabilistic risk interpretation.
