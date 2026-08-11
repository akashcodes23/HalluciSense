# Phase 6E: Score Calibration Report

**Date**: 2026-08-11  

---

## Metric Summary
- **Brier Score**: **0.0482** (Low squared probability error)
- **Expected Calibration Error (ECE)**: **0.1174** (11.74% mean confidence gap)

---

## Reliability Bins

| Bin Interval | Count | Avg Confidence | Avg Accuracy | Gap |
|:---|:---:|:---:|:---:|:---:|
| `[0.00, 0.10]` | 270 | 0.0000 | 1.0000 | 1.0000 (Protected/Verified Factual) |
| `[0.90, 1.00]` | 330 | 0.9942 | 0.9091 | 0.0851 (Clear Hallucination) |
