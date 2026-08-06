# HalluciSense Pillar-1: Feature Engineering

*Generated: 2026-08-03T04:49:02.142488+00:00*  
*Phase: 6K (Frozen)*

---

## 1. Feature Derivation

All features are derived from NLI inference scores computed by
`cross-encoder/nli-deberta-v3-small` over claim-evidence pairs.

For response R with claims C = {c₁, ..., cₙ} and evidence E = {e₁, ..., eₘ}:
- Let s_ent(cᵢ, eⱼ) = P(entailment | cᵢ, eⱼ)
- Let s_con(cᵢ, eⱼ) = P(contradiction | cᵢ, eⱼ)

### Feature Definitions

| Feature | Formula |
| --- | --- |
| `mean_entailment` | mean{s_ent(cᵢ, eⱼ)} over all (i,j) pairs |
| `max_entailment` | max{s_ent(cᵢ, eⱼ)} over all (i,j) pairs |
| `mean_contradiction` | mean{s_con(cᵢ, eⱼ)} over all (i,j) pairs |
| `min_support_margin` | min{s_ent(cᵢ, eⱼ) - s_con(cᵢ, eⱼ)} over all (i,j) |
| `num_claims` | \|C\| = number of atomic claims in R |

## 2. Feature Motivation

- **mean_entailment**: Measures average evidence support across all claims. Low values suggest many claims lack support.
- **max_entailment**: Captures whether at least one claim is strongly supported. Useful for sparse evidence sets.
- **mean_contradiction**: Directly measures average evidence-claim conflict. High values strongly indicate hallucination.
- **min_support_margin**: Identifies the worst-supported claim. Critical for detecting partially hallucinated responses.
- **num_claims**: Longer responses with more claims are harder to fully support; may correlate with hallucination.

## 3. Collinearity Audit Results

| Feature Pair | Pearson |r| | Action |
| --- | --- | --- |
| mean_entailment ↔ max_entailment | ~0.72 | Retained (below threshold 0.85) |
| mean_entailment ↔ mean_contradiction | ~0.80 | Retained (below threshold) |
| Additional pairs | < 0.70 | Retained |

No features were removed by the collinearity filter. All 5 features in SET_D passed
the variance, correlation, and discriminative thresholds.

## 4. Scaling

RobustScaler was applied:
- Center: median (robust to outliers)
- Scale: IQR (interquartile range, Q1 to Q3)
- Motivation: `min_support_margin` has a bimodal distribution with extreme values at ±1

## 5. Feature Importance (from Phase 9 Step 3)

| Feature | Raw Coef | Direction | Permutation AUC Drop |
| --- | --- | --- | --- |
| `min_support_margin` | -1.2485 | Hallucination ↑ when low | Highest |
| `mean_contradiction` | -0.4409 | Hallucination ↑ when high | Second |
| `mean_entailment` | 0.1054 | Minor positive signal | Low |
| `num_claims` | 0.0873 | Weak hallucination signal | Low |
| `max_entailment` | -0.0507 | Minor negative signal | Low |
