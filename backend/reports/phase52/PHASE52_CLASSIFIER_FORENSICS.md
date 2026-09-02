# PHASE 52 — FROZEN CLASSIFIER CONTRIBUTION FORENSICS
**Tree Split Behavior, Permutation Importance & Probability Compression**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & AUDITED`

---

## 1. Classifier Behavior on Balanced Dataset ($N=300$)

- **Model Class**: `HistGradientBoostingClassifier`
- **Ensemble Depth**: 32 decision trees (max depth 5)
- **Baseline Probability at Scaler Medians**: **0.6239**
- **Effective Prediction Distribution on $N=300$**:
  * Range: $[0.2973, 0.7342]$
  * Mean on Factual ($y=0$): **0.3120**
  * Mean on Hallucinated ($y=1$): **0.4612**

---

## 2. Top Drivers of True Positives vs False Negatives

- **Top 5 Features Driving True Positives (Raising $P_H > 0.54$)**:
  1. `prob_mean` ($\Delta = +0.1971$)
  2. `p2_max_pairwise_similarity` ($\Delta = +0.1102$)
  3. `prob_max` ($\Delta = +0.0891$)
  4. `p1_min_support_margin` ($\Delta = +0.0411$)
  5. `prob_p1` ($\Delta = +0.0387$)

- **Top 5 Features Driving False Negatives (Suppressing $P_H < 0.54$)**:
  1. `p1_mean_contradiction` ($\Delta = -0.1048$ — Inverted tree splits suppress contradiction)
  2. `p1_max_entailment` ($\Delta = -0.0548$)
  3. `prob_ratio` ($\Delta = -0.0317$)
  4. `prob_disagreement_abs` ($\Delta = -0.0293$)
  5. `p1_mean_entailment` ($\Delta = -0.0280$)
