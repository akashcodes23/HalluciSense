# Phase 40.3 — Feature Semantic Contract

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 40.3 — 19-Feature Mathematical & Semantic Contract Specification  
**Active Schema:** `SET_A_FULL_HYBRID` (19 Features)  
**Date:** 2026-09-01  

---

## 1. The 19 Canonical Feature Contracts

| Index | Feature Name | Physical Range | Mathematical Definition | Original Phase 6K Meaning | Current Phase 39 Meaning | Semantic Drift | Classifier Compatibility |
|---|---|---|---|---|---|---|---|
| `[0]` | `p1_mean_entailment` | $[0.0, 1.0]$ | $\frac{1}{N}\sum \max_j(e_{ij})$ | Polynomial proxy from retrieval score | Mean DeBERTa cross-encoder entailment score across evidence passages | None (Target quantity) | ✅ Compatible |
| `[1]` | `p1_max_entailment` | $[0.0, 1.0]$ | $\max_{i,j}(e_{ij})$ | Max polynomial proxy score | Highest single evidence snippet entailment score | None | ✅ Compatible |
| `[2]` | `p1_mean_contradiction` | $[0.0, 1.0]$ | $\frac{1}{N}\sum \text{mean}_j(c_{ij})$ | Superlinear decay proxy from retrieval score | Mean DeBERTa cross-encoder contradiction score | None (Target quantity) | ✅ Compatible |
| `[3]` | `p1_min_support_margin` | $[-1.0, 1.0]$ | $\min_i(e_i - c_i)$ | Difference between proxy entailment & contradiction | Difference between real NLI entailment & contradiction | None | ✅ Compatible |
| `[4]` | `p1_num_claims` | $[1.0, \infty)$ | $N_{\text{claims}}$ | Claim count for retrieval | Claim count for retrieval | Zero | ✅ Identical |
| `[5]` | `p2_max_pairwise_contradiction` | $[0.0, 1.0]$ | $\max_{i \ne j}(c_{ij})$ | Peak pairwise claim contradiction | Peak pairwise claim contradiction | Zero | ✅ Identical |
| `[6]` | `p2_mean_pairwise_contradiction` | $[0.0, 1.0]$ | $\text{mean}_{i \ne j}(c_{ij})$ | Average pairwise claim contradiction | Average pairwise claim contradiction | Zero | ✅ Identical |
| `[7]` | `p2_max_pairwise_similarity` | $[0.0, 1.0]$ | $\max_{i \ne j}(\text{sim}_{ij})$ | Peak MiniLM cosine similarity | Peak MiniLM cosine similarity | Zero | ✅ Identical |
| `[8]` | `p2_fraction_contradictory_pairs` | $[0.0, 1.0]$ | $\frac{|\text{pairs with } c \ge 0.5|}{N_{\text{pairs}}}$ | Fraction of contradictory pairs | Fraction of contradictory pairs | Zero | ✅ Identical |
| `[9]` | `p2_num_claims` | $[1.0, \infty)$ | $N_{\text{claims}}$ | Claim count for internal consistency | Claim count for internal consistency | Zero | ✅ Identical |
| `[10]` | `prob_p1` | $[0.0, 1.0]$ | $\sigma(\mathbf{w}_1^T \mathbf{x}_1 + b_1)$ | Pillar 1 LogisticRegression probability | Pillar 1 LogisticRegression probability | Calibrated | ✅ Compatible |
| `[11]` | `prob_p2` | $[0.0, 1.0]$ | $\sigma(\mathbf{w}_2^T \mathbf{x}_2 + b_2)$ | Pillar 2 LogisticRegression probability | Pillar 2 LogisticRegression probability | Zero | ✅ Identical |
| `[12]` | `logit_p1` | $(-\infty, +\infty)$ | $\ln(P_1 / (1 - P_1))$ | Logit transform of $P_1$ | Logit transform of $P_1$ | Calibrated | ✅ Compatible |
| `[13]` | `logit_p2` | $(-\infty, +\infty)$ | $\ln(P_2 / (1 - P_2))$ | Logit transform of $P_2$ | Logit transform of $P_2$ | Zero | ✅ Identical |
| `[14]` | `prob_disagreement_abs` | $[0.0, 1.0]$ | $|P_1 - P_2|$ | Absolute pillar disagreement | Absolute pillar disagreement | Calibrated | ✅ Compatible |
| `[15]` | `prob_mean` | $[0.0, 1.0]$ | $(P_1 + P_2) / 2$ | Mean pillar probability | Mean pillar probability | Calibrated | ✅ Compatible |
| `[16]` | `prob_max` | $[0.0, 1.0]$ | $\max(P_1, P_2)$ | Maximum pillar probability | Maximum pillar probability | Calibrated | ✅ Compatible |
| `[17]` | `prob_min` | $[0.0, 1.0]$ | $\min(P_1, P_2)$ | Minimum pillar probability | Minimum pillar probability | Calibrated | ✅ Compatible |
| `[18]` | `prob_ratio` | $(0, \infty)$ | $(P_1 + \epsilon) / (P_2 + \epsilon)$ | Regularized probability ratio | Regularized probability ratio | Calibrated | ✅ Compatible |

---

## 2. Conclusion

Phase 39 semantic grounding does **not** change the semantic definition of the 19 features; it restores them to their intended theoretical meanings by providing real NLI distributions instead of polynomial proxy values.
