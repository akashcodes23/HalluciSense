# PHASE 52 — FUSION MATHEMATICS & AGGREGATION AUDIT
**Mathematical Formulations vs Production Implementation**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `AUDITED & VERIFIED`

---

## 1. Documented vs Implemented Mathematical Fusion Formulas

### A. Dynamic Weighted Dual Fusion (Ablation D):
$$H_{\text{dual}} = w_1 \cdot P_{\text{P1}} + w_2 \cdot P_{\text{P2}} = 0.60 \cdot P_{\text{P1}} + 0.40 \cdot P_{\text{P2}}$$
- **Empirical AUROC**: **0.8139**, **Recall**: **77.33%**, **Specificity**: **66.00%**, **MCC**: **0.4361**, **ECE**: **0.0795**.
- **Assessment**: Mathematically sound, linear, monotonic, and well-calibrated.

### B. 19-Feature HistGradientBoosting Ensemble (Ablation G):
$$P(H \mid \mathbf{x}) = \sigma \left( \sum_{m=1}^{M} f_m(\text{RobustScaler}(\mathbf{x})) \right)$$
- **Empirical AUROC**: **0.6905**, **Recall**: **30.67%**, **Specificity**: **83.33%**, **MCC**: **0.1647**, **ECE**: **0.2043**.
- **Assessment**: Non-linear tree splits trained on inverted synthetic polarities cause probability compression and signal degradation.

---

## 2. Fusion Defects Discovered

1. **Dilution from Inverted Split Offsets**: In the 19-feature ensemble, inverted leaves subtract score when contradiction is high.
2. **Missing Modality Handling**: Single claims correctly produce 0.0 contradiction, but in the tree ensemble, 0.0 is interpreted as low evidence rather than verified consistency.
