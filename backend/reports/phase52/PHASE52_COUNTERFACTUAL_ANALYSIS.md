# PHASE 52 — COUNTERFACTUAL PERTURBATION ANALYSIS
**Measuring Pillar Influence ($dP(H)/dP_i$) on Final Decision**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & AUDITED`

---

## 1. Measured Pillar Gradients ($dP(H)/dP_i$)

We hold all other features fixed at training medians and perturb each pillar probability from $0.05$ to $0.95$:

| Pillar Tested | $P_{\text{low}} = 0.05$ | $P_{\text{high}} = 0.95$ | Measured Sensitivity Gradient $\frac{\partial P_H}{\partial P_i}$ | Effective Influence on Decision |
| :--- | :--- | :--- | :--- | :--- |
| **Pillar 1 ($P_1$)** | $P_H = 0.5716$ | $P_H = 0.6560$ | **$+0.0937$** | 🏆 Primary Active Gradient |
| **Pillar 2 ($P_2$)** | $P_H = 0.6419$ | $P_H = 0.6239$ | **$-0.0200$** | ⚠️ Slight Inverted Damping |
| **Pillar 3 ($P_3$)** | $P_H = 0.6272$ | $P_H = 0.6272$ | **$+0.0000$** | ⚪ Zero Single-Claim Sensitivity |

---

## 2. Key Insights

1. **P1 Dominates Sensitivity**: Pillar 1 produces over **82% of the dynamic gradient** ($\frac{\partial P_H}{\partial P_1} = +0.0937$).
2. **P3 Has Zero Gradient on Median Baseline**: Because pairwise contradiction features for median inputs are below the tree split thresholds, P3 has zero effect on the baseline point.
3. **P2 Slightly Damps Probability**: P2 has a small negative gradient ($-0.0200$), slightly moderating extreme predictions.
