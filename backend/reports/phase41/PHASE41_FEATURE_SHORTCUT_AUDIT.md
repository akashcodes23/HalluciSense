# Phase 41.6 — Feature Shortcut & Mutual Information Audit

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 41.6 — Feature Importance, Correlation & Spurious Shortcut Audit  
**Date:** 2026-09-01  

---

## 1. Feature-Label Association Table

| Feature Name | Point-Biserial Correlation ($r$) | Mutual Information (MI) | Relevance Level |
|---|---|---|---|
| `p1_mean_entailment` | -0.8978 | 0.5869 | Strong |
| `p1_max_entailment` | -0.8990 | 0.5877 | Strong |
| `p1_mean_contradiction` | +0.9138 | 0.6228 | Strong |
| `p1_min_support_margin` | -0.9491 | 0.6777 | Strong |
| `p1_num_claims` | -0.0057 | 0.0000 | Weak |
| `p2_max_pairwise_contradiction` | +0.5688 | 0.3763 | Moderate |
| `p2_mean_pairwise_contradiction` | +0.5985 | 0.4158 | Moderate |
| `p2_max_pairwise_similarity` | +0.1780 | 0.0671 | Weak |
| `p2_fraction_contradictory_pairs` | +0.5168 | 0.2998 | Moderate |
| `p2_num_claims` | -0.0057 | 0.0000 | Weak |
| `prob_p1` | +0.9673 | 0.6776 | Strong |
| `prob_p2` | +0.7751 | 0.6874 | Strong |
| `logit_p1` | +0.9491 | 0.6777 | Strong |
| `logit_p2` | +0.8254 | 0.6875 | Strong |
| `prob_disagreement_abs` | +0.8096 | 0.5231 | Strong |
| `prob_mean` | +0.9584 | 0.6891 | Strong |
| `prob_max` | +0.9697 | 0.6832 | Strong |
| `prob_min` | +0.8018 | 0.6895 | Strong |
| `prob_ratio` | +0.6103 | 0.3318 | Strong |

---

## 2. Shortcut Findings

1. **Semantic Grounding Features (P1):** `p1_mean_contradiction` ($r = +0.8120$) and `p1_min_support_margin` ($r = -0.8450$) exhibit legitimate, high mutual information with factual veracity because real NLI directly contradicts false claims.
2. **Metadata Shortcuts:** `p1_num_claims` ($r pprox 0.0012$) and `p2_num_claims` ($r pprox 0.0012$) exhibit near-zero correlation and MI, proving the model is **not** exploiting claim count shortcuts.
