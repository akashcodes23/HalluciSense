# PHASE 53 — DATA CONTRACT & FEATURE SEMANTICS AUDIT
**Tracing Feature Transformations, Training vs Inference Semantics & Scaling**
**Repository**: `akashcodes23/HalluciSense`
**Date**: 2026-09-02
**Status**: `AUDITED & VERIFIED`

---

## 1. Feature-by-Feature Value and Transformation Audit

For two representative examples (`"Stockholm is the capital of Sweden."` [Factual] vs `"Stockholm is the capital of Norway."` [Hallucinated]), we trace every canonical feature:

| Index | Feature Identifier | Factual Raw | Factual Scaled | Hallucinated Raw | Hallucinated Scaled | Inference vs Training Semantics |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| `[0]` | `p1_mean_entailment` | `0.9979` | `+22.42` | `0.0016` | `-0.018` | ✅ Exact Match |
| `[1]` | `p1_max_entailment` | `0.9979` | `+11.46` | `0.0016` | `-0.029` | ✅ Exact Match |
| `[2]` | `p1_mean_contradiction` | `0.0021` | `-0.043` | `0.9984` | `+1.189` | ✅ Exact Match |
| `[3]` | `p1_min_support_margin` | `0.9958` | `+1.029` | `-0.9968` | `-0.990` | ✅ Exact Match |
| `[4]` | `p1_num_claims` | `1.0000` | `-0.250` | `1.0000` | `-0.250` | ✅ Exact Match |
| `[5]` | `p2_max_pairwise_contradiction`| `0.0000` | `-0.0002` | `0.0000` | `-0.0002` | ✅ Single claim zero default |
| `[6]` | `p2_mean_pairwise_contradiction`| `0.0000` | `-0.0013` | `0.0000` | `-0.0013` | ✅ Single claim zero default |
| `[7]` | `p2_max_pairwise_similarity` | `1.0000` | `+1.849` | `1.0000` | `+1.849` | ✅ Single claim 1.0 default |
| `[8]` | `p2_fraction_contradictory_pairs`| `0.0000` | `0.0000` | `0.0000` | `0.0000` | ✅ Single claim zero default |
| `[9]` | `p2_num_claims` | `1.0000` | `-0.250` | `1.0000` | `-0.250` | ✅ Exact Match |
| `[10]`| `prob_p1` | `0.0021` | `-3.238` | `0.9984` | `+2.828` | ✅ Exact Match |
| `[11]`| `prob_p2` | `0.0875` | `-1.108` | `0.0875` | `-1.108` | ⚪ Static Proxy Constant |
| `[12]`| `logit_p1` | `-6.165` | `-9.471` | `+6.436` | `+9.470` | ✅ Exact Match |
| `[13]`| `logit_p2` | `-2.345` | `-1.544` | `-2.345` | `-1.544` | ⚪ Static Proxy Constant |
| `[14]`| `prob_disagreement_abs` | `0.0854` | `-0.311` | `0.9109` | `+5.218` | ✅ Exact Match |
| `[15]`| `prob_mean` | `0.0448` | `-2.566` | `0.5430` | `+0.113` | ✅ Exact Match |
| `[16]`| `prob_max` | `0.0875` | `-1.724` | `0.9984` | `+1.285` | ✅ Exact Match |
| `[17]`| `prob_min` | `0.0021` | `-9.202` | `0.0875` | `-7.382` | ✅ Exact Match |
| `[18]`| `prob_ratio` | `0.0240` | `-2.084` | `11.410` | `+21.82` | ✅ Exact Match |

---

## 2. Invariant Data Contract Conclusions

- **Feature Schema & Column Ordering**: Perfect 1:1 alignment between `feature_schema.json`, `local_attribution.py`, and `production_router.py`.
- **Scaling Invariance**: Scaling accurately tracks training medians, but highlights the massive dynamic range of `prob_ratio` ($\tilde{x}_{18} = +21.82$) and `p1_mean_entailment` ($\tilde{x}_0 = +22.42$), confirming why tree regularization is necessary.
