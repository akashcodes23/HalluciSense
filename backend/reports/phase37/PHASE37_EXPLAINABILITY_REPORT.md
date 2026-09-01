# Phase 37 — Explainable AI & Decision Transparency

## Objective

Upgrade HalluciSense from a detector that primarily reports an H-score into an **auditable explainable AI system** that can answer:

1. Which claim was evaluated?
2. What evidence was retrieved?
3. Which verification signal drove the decision?
4. Which frozen hybrid-model features pushed the local prediction toward or away from hallucination?
5. How far is the prediction from the operating threshold?
6. What uncertainty or interaction limitations remain?

Production model artifacts remain frozen. Phase 37 changes the explanation and observability layer only.

## 1. Faithful Local Model Attribution

The frozen Phase 6M hybrid model consumes 19 features. Phase 37 implements:

`LOCAL_LEAVE_ONE_FEATURE_AT_BASELINE`

For each feature `x_i`:

`delta_i = P(H | x) - P(H | x with x_i replaced by baseline_i)`

The baseline is the training median stored in `RobustScaler.center_`. The exact production scaler and exact frozen `HistGradientBoostingClassifier` are reused; no surrogate model and no retraining are involved.

### Interpretation

- `delta > 0`: observed feature pushes the local prediction toward hallucination.
- `delta < 0`: observed feature pushes the local prediction away from hallucination.
- `delta = 0`: negligible local effect.
- `relative_strength`: normalized magnitude of the local delta across the 19 features.
- `interaction_gap`: residual between the observed probability and the baseline-plus-independent-deltas reconstruction. This is reported because the tree model is nonlinear.

### Important scientific boundary

These values are **not SHAP values** and must not be presented as additive Shapley attributions or global feature importance. The UI explicitly labels them as local perturbation drivers.

## 2. 19-Feature Provenance

The explanation layer exposes the same frozen feature schema:

1. `p1_mean_entailment`
2. `p1_max_entailment`
3. `p1_mean_contradiction`
4. `p1_min_support_margin`
5. `p1_num_claims`
6. `p2_max_pairwise_contradiction`
7. `p2_mean_pairwise_contradiction`
8. `p2_max_pairwise_similarity`
9. `p2_fraction_contradictory_pairs`
10. `p2_num_claims`
11. `prob_p1`
12. `prob_p2`
13. `logit_p1`
14. `logit_p2`
15. `prob_disagreement_abs`
16. `prob_mean`
17. `prob_max`
18. `prob_min`
19. `prob_ratio`

The feature vector is returned for diagnostic reproducibility when the active model is the 19-feature hybrid classifier.

## 3. Decision Provenance

The explanation payload includes:

- observed hybrid probability
- operating threshold `tau* = 0.54`
- decision margin `P(H) - tau*`
- training-median baseline probability
- top positive hallucination drivers
- top negative/protective drivers
- counterfactual probability for each driver
- nonlinear interaction gap
- exact attribution methodology

The explanation layer never changes `is_hallucinated` or the production decision threshold.

## 4. Existing Evidence & Structural Explainability

The existing explanation engine continues to expose:

- claim-level evidence attribution
- Pillar 1 and Pillar 2 probabilities
- entity / numeric / temporal structural diagnostics
- contradiction graph information
- claim-evidence topology

Phase 37 makes the model-level contribution layer explicit and faithful instead of describing the previous zeroed-feature calculation as SHAP.

## 5. Production API Contract

### Hybrid prediction

`POST /api/v1/hallucisense/predict`

Now returns:

- `feature_vector`
- `feature_schema`
- `explainability`
- `explanation.model_explainability`
- `explanation.decision_rule`

### Hybrid explainability

`POST /api/v1/hallucisense/explain`

Remains on-demand so normal verification latency is not increased. It returns the complete hybrid prediction plus the local attribution payload.

### Verify UI

The Verify page now exposes an **Explain this decision** control. The explanation request is intentionally separate from the normal `/api/v1/analyze` request. This preserves the existing verification latency path while giving an examiner/user a deeper diagnostic view on demand.

## 6. Failure-Mode Transparency

HalluciSense should retain and display real failure cases rather than hiding them. The previously observed `12 × 8 = 95` false negative remains a valuable evaluation case. Explainability can show which features failed to move the hybrid model across `tau* = 0.54` and therefore support diagnosis of the numerical-reasoning limitation.

No claim of perfect detection is made.

## 7. Regression Tests

`backend/tests/phase37_explainability_smoke.py` verifies:

- exactly 19 features are accepted;
- positive and protective feature directions are correctly identified;
- nonlinear interaction residual is surfaced;
- backward-compatible feature importance output matches the new local deltas;
- invalid feature dimensions are rejected.

The test uses deterministic stand-in scaler/classifier objects and does not load production artifacts.

## 8. Production Safety

- No model artifact is retrained or modified.
- No threshold is changed.
- No Docker/Railway memory configuration is changed.
- Normal `/analyze` execution remains unchanged in its decision path.
- Explainability is on-demand for the hybrid diagnostic layer.
- If attribution fails, the underlying verification result remains valid and the UI reports explainability as unavailable.

## 9. Examiner-Facing Explanation

A concise viva answer is:

> HalluciSense does not merely expose a black-box hallucination score. For a specific response, it can trace the decision from claims and retrieved evidence through the 19-feature frozen hybrid vector to a local counterfactual attribution. Each feature is independently replaced by its training-median baseline and the exact classifier is re-evaluated. The resulting probability delta tells us whether that observed feature locally increased or decreased hallucination risk. Because the classifier is nonlinear, we also report the interaction residual rather than pretending the individual effects are additive.
