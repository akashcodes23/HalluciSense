# Phase 37 — Model-Faithful Explainability Report

**Repository:** akashcodes23/HalluciSense
**Phase:** 37 — Robust, Model-Faithful Explainability & Production Hardening
**Date:** 2026-09-01
**Status:** COMPLETE ✅

---

## Production Baseline (Unchanged)

| Parameter | Value |
|-----------|-------|
| Active model | `hybrid` (HistGradientBoostingClassifier) |
| Decision threshold τ* | 0.54 |
| Training samples | 58,002 |
| Feature count | 19 |
| Frozen classifier weights | **UNCHANGED** |
| Frozen scaler | **UNCHANGED** |
| Production deployment | ONLINE (phase 33 verified) |

---

## Explainability Method: Local Counterfactual Attribution

### Definition

For each of the 19 input features, the attribution is defined as:

```
a_i = P(H | X) − P(H | X_i)

where X_i = X with feature i replaced by its training-median value.
```

**Interaction Gap:**
```
interaction_gap = [P(H | X) − P(H | X_baseline)] − Σ a_i
```

Because `HistGradientBoostingClassifier` is nonlinear, one-feature-at-a-time counterfactuals are NOT an additive decomposition. The residual interaction gap captures the nonlinear interaction between features.

### Why NOT SHAP

SHAP (SHapley Additive exPlanations) requires marginalising over all 2^19 feature coalitions via Shapley values. Phase 37 uses Local Counterfactual Attribution — a computationally efficient single-feature replacement method that:
- Makes 21 model evaluations per prediction (1 original + 1 baseline + 19 features)
- Is deterministic and model-faithful
- Does NOT claim to satisfy the Shapley axioms

### Baseline

The counterfactual baseline is the vector of **training medians** sourced from `RobustScaler.center_` (frozen `preprocessing.joblib`, fitted on N=58,002 development instances). This is traceable to a specific artifact and requires zero re-computation.

---

## Files Changed

### New Files
| File | Purpose |
|------|---------|
| `backend/app/core/inference/local_attribution.py` | Canonical attribution engine. 21 frozen model calls per prediction. |
| `backend/tests/test_phase37_local_attribution.py` | 29-test Phase 37 test suite. |
| `frontend/src/components/verification/LocalAttributionPanel.tsx` | Frontend explanation panel component. |

### Modified Files
| File | Change |
|------|--------|
| `backend/app/core/pipeline.py` | Calls `compute_local_attribution()` after every `predict()`. Attaches result as `local_attribution` key. |
| `backend/app/modules/hallucisense/router.py` | Upgraded `/explain` endpoint to return full attribution payload. Added `feature_vector` param for direct-vector attribution. |
| `backend/app/core/inference/explainability.py` | Removed incorrect "SHAP-style" claim. Fixed baseline from scaled-zero to training medians. Renamed function. |
| `frontend/src/types/verification-types.ts` | Added `LocalAttributionFeature`, `LocalAttribution` types. Extended `AnalysisResponse`. |
| `frontend/src/app/(dashboard)/verify/page.tsx` | Added `LocalAttributionPanel` after Token Heatmap section. |

---

## Test Results

### Phase 37 Suite
```
29 passed in 5.10s
```

| Test | Result |
|------|--------|
| test_19_feature_vector_validation | ✅ PASSED |
| test_wrong_feature_count_rejected (×6 params) | ✅ PASSED |
| test_nan_rejected | ✅ PASSED |
| test_infinity_rejected | ✅ PASSED |
| test_baseline_deterministic | ✅ PASSED |
| test_baseline_from_training_median | ✅ PASSED |
| test_training_medians_from_robust_scaler | ✅ PASSED |
| test_original_probability_match | ✅ PASSED |
| test_single_feature_perturbation_isolates_one_index | ✅ PASSED |
| test_positive_attribution_increases_risk | ✅ PASSED |
| test_negative_attribution_decreases_risk | ✅ PASSED |
| test_zero_attribution_neutral_direction | ✅ PASSED |
| test_interaction_gap_calculation | ✅ PASSED |
| test_explanation_does_not_change_decision | ✅ PASSED |
| test_threshold_unchanged | ✅ PASSED |
| test_predict_backward_compatibility | ✅ PASSED |
| test_explain_endpoint_required_fields | ✅ PASSED |
| test_top_drivers_sorted_correctly | ✅ PASSED |
| test_feature_ordering_canonical | ✅ PASSED |
| test_repeated_calls_deterministic | ✅ PASSED |
| test_known_factual_example_structure | ✅ PASSED |
| test_known_hallucination_example_structure | ✅ PASSED |
| test_near_threshold_boundary | ✅ PASSED |
| test_12x8_failure_analysis | ✅ PASSED (documented, not fixed) |

### Production Regression
```
10 passed in 5.19s  (test_unit_pipeline, test_smoke_production)
```

### Frontend Build
```
✓ Compiled successfully in 2.2s
✓ TypeScript: no errors
✓ 23 static pages generated
```

---

## API Changes (Additive, Backward-Compatible)

### `/api/v1/hallucisense/predict` (unchanged contract + new key)

New `local_attribution` field added to response:

```json
{
  "is_hallucinated": true,
  "hallucination_probability": 0.72,
  "operating_threshold": 0.54,
  "explanation": {...},
  "confidence_score": 0.44,
  "local_attribution": {
    "method": "local_counterfactual_attribution",
    "feature_count": 19,
    "baseline_type": "training_median_from_robust_scaler",
    "original_probability": 0.72,
    "baseline_probability": 0.61,
    "threshold": 0.54,
    "decision_margin": 0.18,
    "interaction_gap": 0.003,
    "features": [ ... ],
    "top_hallucination_drivers": [ ... ],
    "top_protective_drivers": [ ... ],
    "inference_count": 21
  }
}
```

### `/api/v1/hallucisense/explain` (upgraded)

Now accepts optional `feature_vector: List[float]` (19 elements) for direct attribution without running the full pipeline.

---

## Failure Analysis: "12 × 8 = 95"

The frozen classifier does not perform independent arithmetic verification. The Hybrid model aggregates NLI-based entailment signals from `cross-encoder/nli-deberta-v3-small` — which was not trained for arithmetic checking. As a result, simple arithmetic errors like "12 × 8 = 95" may not be consistently flagged.

This is a **known limitation** of the retrieval+NLI architecture, not a defect introduced in Phase 37. The failure analysis test (Test 24) records the classifier's actual behavior without asserting a specific P(H) value, correctly documenting the failure mode rather than attempting to force a specific result.

---

## Phase 37 Acceptance Criteria — All Satisfied

- [x] Frozen classifier weights unchanged
- [x] Threshold 0.54 unchanged  
- [x] Canonical 19-feature schema from `model_metadata.json`
- [x] Training-median baseline from `RobustScaler.center_`
- [x] No incorrect "SHAP" labeling (corrected in `explainability.py`)
- [x] Full backward compatibility on all existing endpoints
- [x] Frontend `LocalAttributionPanel` renders only when attribution data present
- [x] All 29 Phase 37 tests pass (including the 24 core tests + 5 parametrized)
- [x] Full production regression suite passes (10/10)
- [x] `npm run build` passes (23 pages, 0 TypeScript errors)
- [x] No production secrets exposed
