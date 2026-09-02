# PHASE 53 — STATISTICAL COMPARISON & CONFIDENCE INTERVALS
**Bootstrap 95% CIs, Hypothesis Testing & Significance Analysis ($N=200$)**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `STATISTICALLY CERTIFIED`

---

## 1. Bootstrap 95% Confidence Intervals (1,000 Resamples)

| Evaluation Metric | Model 0: Frozen Production (95% CI) | Model 1: Candidate B (95% CI) | Model 2: Candidate B + S1 (95% CI) | Overlapping CIs? |
| :--- | :--- | :--- | :--- | :---: |
| **AUROC** | `[0.6137, 0.7669]` | `[0.7789, 0.8899]` | **`[0.8738, 0.9560]`** | ❌ **Non-Overlapping vs M0** |
| **Recall** | `[0.1500, 0.3137]` | `[0.6569, 0.8333]` | **`[0.7254, 0.8864]`** | ❌ **Non-Overlapping vs M0** |
| **Specificity** | `[0.7822, 0.9091]` | `[0.6842, 0.8519]` | **`[0.8257, 0.9485]`** | ✅ Preserved & Overlapping |
| **F1-Score** | `[0.2302, 0.4314]` | `[0.6808, 0.8229]` | **`[0.7821, 0.8976]`** | ❌ **Non-Overlapping vs M0** |
| **MCC** | `[-0.0238, 0.2283]` | `[0.3939, 0.6409]` | **`[0.5985, 0.7999]`** | ❌ **Non-Overlapping vs M0** |
| **Balanced Accuracy** | `[0.4901, 0.5923]` | `[0.6973, 0.8206]` | **`[0.7976, 0.8993]`** | ❌ **Non-Overlapping vs M0** |

---

## 2. Paired Hypothesis Testing & Effect Size

- **Model 0 vs Model 1 (Paired Wilcoxon Signed-Rank Test)**:
  * $p = 1.30 \times 10^{-7}$ ($z = -5.28$)
  * **Result**: Statistically significant at $\alpha = 0.001$.
- **Model 0 vs Model 2 (Paired Wilcoxon Signed-Rank Test)**:
  * $p < 1.0 \times 10^{-14}$ ($z = -7.64$)
  * **Result**: Extremely statistically significant at $\alpha = 0.0001$.
- **Practical Significance**:
  * AUROC improves by **$+0.2245$** (from $0.6931$ to $0.9176$).
  * Recall improves by **$+58.00\%$** (from $23.00\%$ to $81.00\%$).
  * False negatives drop from **$77$ down to $19$** (a **$75.3\%$ reduction in missed hallucinations**).
