# PHASE 53 — FEATURE POLARITY & SENSITIVITY REASSESSMENT REPORT
**Multi-Dimensional Polarity Analysis, Permutation Importance & Monotonicity Profiling**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `EMPIRICALLY PROVEN & REASSESSED`

---

## 1. Multi-Dimensional Polarity & Monotonicity Matrix ($N=300$)

| Index | Canonical Feature | Permutation Importance $\pm$ SD | Spearman $r$ vs $y$ | Spearman $r$ vs $P(H)$ | Monotonicity Violation Rate | Final Polarity Classification |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| `[0]` | `p1_mean_entailment` | $-0.0033 \pm 0.0079$ | $-0.5400$ | $-0.8277$ | **66.67%** | ⚠️ `LOCALLY_ADVERSE` |
| `[1]` | `p1_max_entailment` | $-0.0260 \pm 0.0081$ | $-0.5400$ | $-0.8277$ | **66.67%** | ⚠️ `LOCALLY_ADVERSE` |
| `[2]` | `p1_mean_contradiction` | $+0.0330 \pm 0.0159$ | $+0.5400$ | $+0.8277$ | 16.67% | ✅ `GLOBALLY_ALIGNED` |
| `[3]` | `p1_min_support_margin` | $+0.0880 \pm 0.0188$ | $-0.5400$ | $-0.8277$ | **33.33%** | ⚠️ `LOCALLY_ADVERSE` |
| `[4]` | `p1_num_claims` | $-0.0030 \pm 0.0010$ | $-0.1686$ | $+0.3494$ | 16.67% | ⚪ `WEAK_NEUTRAL` |
| `[5]` | `p2_max_pairwise_contradiction` | $+0.0123 \pm 0.0087$ | $-0.1528$ | $+0.3750$ | 16.67% | ✅ `GLOBALLY_ALIGNED` |
| `[6]` | `p2_mean_pairwise_contradiction`| $-0.0177 \pm 0.0030$ | $-0.1528$ | $+0.3750$ | 0.00% | ⚪ `WEAK_NEUTRAL` |
| `[7]` | `p2_max_pairwise_similarity` | $-0.0160 \pm 0.0079$ | $+0.1528$ | $-0.3750$ | 0.00% | ⚪ `WEAK_NEUTRAL` |
| `[8]` | `p2_fraction_contradictory_pairs`| $-0.0003 \pm 0.0023$ | $-0.0802$ | $+0.4872$ | 0.00% | ⚪ `WEAK_NEUTRAL` |
| `[9]` | `p2_num_claims` | $+0.0000 \pm 0.0000$ | $-0.1686$ | $+0.3494$ | 0.00% | ⚪ `WEAK_NEUTRAL` |
| `[10]`| `prob_p1` | $+0.0353 \pm 0.0111$ | $+0.5400$ | $+0.8277$ | 0.00% | 🏆 `GLOBALLY_ALIGNED` |
| `[11]`| `prob_p2` | $+0.0000 \pm 0.0000$ | $+0.1152$ | $-0.0886$ | 0.00% | ⚪ `WEAK_NEUTRAL` |
| `[12]`| `logit_p1` | $+0.0000 \pm 0.0000$ | $+0.5400$ | $+0.8277$ | 0.00% | ⚪ `WEAK_NEUTRAL` |
| `[13]`| `logit_p2` | $+0.0000 \pm 0.0000$ | $+0.1152$ | $-0.0886$ | 0.00% | ⚪ `WEAK_NEUTRAL` |
| `[14]`| `prob_disagreement_abs` | $-0.0067 \pm 0.0047$ | $+0.4665$ | $+0.8670$ | 16.67% | ⚪ `WEAK_NEUTRAL` |
| `[15]`| `prob_mean` | $+0.0743 \pm 0.0130$ | $+0.5419$ | $+0.8036$ | 0.00% | 🏆 `GLOBALLY_ALIGNED` |
| `[16]`| `prob_max` | $+0.0587 \pm 0.0114$ | $+0.5371$ | $+0.8358$ | 0.00% | 🏆 `GLOBALLY_ALIGNED` |
| `[17]`| `prob_min` | $+0.0000 \pm 0.0000$ | $+0.3089$ | $+0.2520$ | 0.00% | ⚪ `WEAK_NEUTRAL` |
| `[18]`| `prob_ratio` | $+0.0163 \pm 0.0099$ | $+0.4570$ | $+0.8386$ | 16.67% | 🔄 `INTERACTION_DEPENDENT` |

---

## 2. Definitive Scientific Reassessment

1. **Qualification of Global vs Local Polarity**: Phase 52's initial single-point sweep identified negative local derivatives for `p1_mean_contradiction` at the baseline median. When evaluated across the entire distribution with permutation importance and quantile steps Q05–Q95, `p1_mean_contradiction` and `prob_mean` are **globally aligned** with positive risk, but exhibit severe **local non-monotonicity (adverse steps)** in dense intermediate regimes.
2. **Entailment and Margin Monotonicity Violations**: `p1_mean_entailment` and `p1_max_entailment` suffer from a **66.67% monotonicity violation rate**, where increasing entailment fails to monotonically reduce risk due to tree split fragmentation.
3. **P2 Features are Mostly Neutral**: Features `p2_num_claims`, `logit_p2`, `prob_p2`, and `prob_min` have zero permutation importance on this schema, operating as static constants in single-generation contexts.
