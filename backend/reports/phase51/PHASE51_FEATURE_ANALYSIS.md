# PHASE 51 — 19-FEATURE STATISTICAL DIAGNOSTICS & RANKING
**Statistical Profile, Standardized Mean Differences & Informative Rankings**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `MEASURED & RANKED`

---

## 1. Complete Statistical Profile of All 19 Canonical Features

| Index | Feature Identifier | Factual Mean $\pm$ SD | Hallucinated Mean $\pm$ SD | SMD (Cohen's $d$) | Univariate AUROC | Correlation ($r$) | Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `[12]` | `logit_p1` | $0.01 \pm 3.28$ | $4.63 \pm 3.89$ | **+1.2832** | **0.8341** | +0.4885 | 🥇 #1 |
| `[15]` | `prob_mean` | $0.27 \pm 0.18$ | $0.47 \pm 0.13$ | **+1.2604** | **0.8370** | +0.5271 | 🥈 #2 |
| `[16]` | `prob_max` | $0.47 \pm 0.36$ | $0.86 \pm 0.25$ | **+1.2522** | **0.8346** | +0.5230 | 🥉 #3 |
| `[0]` | `p1_mean_entailment` | $0.54 \pm 0.37$ | $0.14 \pm 0.25$ | **-1.2451** | **0.8341** | -0.5219 | 🎖️ #4 |
| `[1]` | `p1_max_entailment` | $0.54 \pm 0.37$ | $0.14 \pm 0.25$ | **-1.2451** | **0.8341** | -0.5219 | 🎖️ #5 |
| `[2]` | `p1_mean_contradiction` | $0.46 \pm 0.37$ | $0.86 \pm 0.25$ | **+1.2451** | **0.8341** | +0.5219 | #6 |
| `[3]` | `p1_min_support_margin` | $0.08 \pm 0.74$ | $-0.71 \pm 0.51$ | **-1.2451** | **0.8341** | -0.5219 | #7 |
| `[10]` | `prob_p1` | $0.46 \pm 0.37$ | $0.86 \pm 0.25$ | **+1.2451** | **0.8341** | +0.5219 | #8 |
| `[14]` | `prob_disagreement_abs` | $0.40 \pm 0.35$ | $0.78 \pm 0.25$ | **+1.2372** | **0.8043** | +0.5168 | #9 |
| `[18]` | `prob_ratio` | $6.35 \pm 5.82$ | $11.07 \pm 4.58$ | **+0.9015** | **0.7933** | +0.3947 | #10 |
| `[17]` | `prob_min` | $0.07 \pm 0.03$ | $0.08 \pm 0.02$ | **+0.5232** | **0.6069** | +0.2561 | #11 |
| `[4]` | `p1_num_claims` | $1.25 \pm 0.43$ | $1.11 \pm 0.31$ | **-0.3865** | **0.4275** | -0.1853 | #12 |
| `[9]` | `p2_num_claims` | $1.25 \pm 0.43$ | $1.11 \pm 0.31$ | **-0.3865** | **0.4275** | -0.1853 | #13 |
| `[11]` | `prob_p2` | $0.08 \pm 0.02$ | $0.08 \pm 0.01$ | **+0.2424** | **0.5478** | +0.1134 | #14 |
| `[13]` | `logit_p2` | $-2.50 \pm 0.26$ | $-2.44 \pm 0.21$ | **+0.2440** | **0.5478** | +0.1141 | #15 |
| `[8]` | `p2_fraction_contradictory_pairs` | $0.18 \pm 0.38$ | $0.10 \pm 0.30$ | **-0.2191** | **0.4625** | -0.1037 | #16 |
| `[5]` | `p2_max_pairwise_contradiction` | $0.17 \pm 0.36$ | $0.10 \pm 0.30$ | **-0.2026** | **0.4353** | -0.0951 | #17 |
| `[6]` | `p2_mean_pairwise_contradiction`| $0.17 \pm 0.36$ | $0.10 \pm 0.30$ | **-0.2026** | **0.4353** | -0.0951 | #18 |
| `[7]` | `p2_max_pairwise_similarity` | $0.83 \pm 0.36$ | $0.90 \pm 0.30$ | **+0.2026** | **0.5647** | +0.0951 | #19 |

---

## 2. Top 5 vs Bottom 5 Informative Features

- **TOP 5 INFORMATIVE FEATURES**:
  1. `logit_p1` ($SMD = +1.2832, AUROC = 0.8341$)
  2. `prob_mean` ($SMD = +1.2604, AUROC = 0.8370$)
  3. `prob_max` ($SMD = +1.2522, AUROC = 0.8346$)
  4. `p1_mean_entailment` ($SMD = -1.2451, AUROC = 0.8341$)
  5. `p1_max_entailment` ($SMD = -1.2451, AUROC = 0.8341$)

- **BOTTOM 5 NON-INFORMATIVE FEATURES**:
  1. `p2_max_pairwise_contradiction` ($SMD = -0.2026, AUROC = 0.4353$)
  2. `p2_mean_pairwise_contradiction` ($SMD = -0.2026, AUROC = 0.4353$)
  3. `p2_max_pairwise_similarity` ($SMD = +0.2026, AUROC = 0.5647$)
  4. `p2_fraction_contradictory_pairs` ($SMD = -0.2191, AUROC = 0.4625$)
  5. `prob_p2` ($SMD = +0.2424, AUROC = 0.5478$)
