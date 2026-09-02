# PHASE 52 — FEATURE POLARITY & SENSITIVITY AUDIT
**Investigation of Feature Directions, Sign Inversions & Tree Gradients**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & AUDITED`

---

## 1. 19-Feature Direction Audit

| Index | Feature Identifier | Expected Semantic Direction | Tree Response $\Delta P(H)$ | Classifier Behavior | Polarity Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| `[0]` | `p1_mean_entailment` | Higher $\to$ More Factual | $-0.0280$ | Decreases $P(H)$ | ✅ Aligned |
| `[1]` | `p1_max_entailment` | Higher $\to$ More Factual | $-0.0548$ | Decreases $P(H)$ | ✅ Aligned |
| `[2]` | `p1_mean_contradiction` | Higher $\to$ More Hallucinated | **$-0.1048$** | **Decreases $P(H)$** | 🚨 **INVERTED** |
| `[3]` | `p1_min_support_margin` | Higher $\to$ More Factual | **$+0.0411$** | **Increases $P(H)$** | 🚨 **INVERTED** |
| `[4]` | `p1_num_claims` | Higher $\to$ More Hallucinated | $+0.0177$ | Increases $P(H)$ | ✅ Aligned |
| `[5]` | `p2_max_pairwise_contradiction`| Higher $\to$ More Hallucinated | $+0.0033$ | Neutral | ✅ Neutral |
| `[6]` | `p2_mean_pairwise_contradiction`| Higher $\to$ More Hallucinated | $+0.0014$ | Neutral | ✅ Neutral |
| `[7]` | `p2_max_pairwise_similarity` | Higher $\to$ More Factual | **$+0.1102$** | **Increases $P(H)$** | 🚨 **INVERTED** |
| `[8]` | `p2_fraction_contradictory_pairs`| Higher $\to$ More Hallucinated | $+0.0000$ | Neutral | ✅ Neutral |
| `[9]` | `p2_num_claims` | Higher $\to$ More Hallucinated | $+0.0000$ | Neutral | ✅ Neutral |
| `[10]`| `prob_p1` | Higher $\to$ More Hallucinated | $+0.0387$ | Increases $P(H)$ | ✅ Aligned |
| `[11]`| `prob_p2` | Higher $\to$ More Hallucinated | **$-0.0180$** | **Decreases $P(H)$** | 🚨 **INVERTED** |
| `[12]`| `logit_p1` | Higher $\to$ More Hallucinated | $+0.0000$ | Neutral | ✅ Neutral |
| `[13]`| `logit_p2` | Higher $\to$ More Hallucinated | $+0.0000$ | Neutral | ✅ Neutral |
| `[14]`| `prob_disagreement_abs` | Higher $\to$ More Hallucinated | **$-0.0293$** | **Decreases $P(H)$** | 🚨 **INVERTED** |
| `[15]`| `prob_mean` | Higher $\to$ More Hallucinated | **$+0.1971$** | **Increases $P(H)$** | 🏆 **PRIMARY DRIVER** |
| `[16]`| `prob_max` | Higher $\to$ More Hallucinated | **$+0.0891$** | **Increases $P(H)$** | 🏆 **PRIMARY DRIVER** |
| `[17]`| `prob_min` | Higher $\to$ More Hallucinated | $+0.0000$ | Neutral | ✅ Neutral |
| `[18]`| `prob_ratio` | Higher $\to$ More Hallucinated | **$-0.0317$** | **Decreases $P(H)$** | 🚨 **INVERTED** |

---

## 2. Forensic Impact of Polarity Inversion

- **Root Cause Discovered**: In the frozen tree artifact (`hybrid_meta_classifier.joblib`), `p1_mean_contradiction` has an inverted tree gradient ($\Delta = -0.1048$). When an input exhibits high contradiction, this feature actively pushes $P(H)$ *downward*.
- **The Internal Tug-of-War**: While `prob_mean` ($\Delta = +0.1971$) pulls $P(H)$ upward for hallucinations, `p1_mean_contradiction` ($\Delta = -0.1048$) and `prob_ratio` ($\Delta = -0.0317$) pull it back down, trapping true hallucinations in the sub-0.54 range ($0.40 - 0.53$) and causing massive false negatives.
