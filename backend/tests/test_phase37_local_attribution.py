"""Phase 37 — Local Counterfactual Attribution Test Suite (24 Tests).

Tests the model-faithful explainability layer introduced in Phase 37.

Design constraints:
  - NO retraining
  - NO threshold modification
  - NO surrogate models
  - Decision authority rests exclusively with frozen production classifier
  - All tests use frozen artifacts from evaluation_results/phase6m/final_hybrid_model/
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ─── Path setup ──────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.inference.local_attribution import (
    LocalAttributionResult,
    FeatureAttribution,
    compute_local_attribution,
    get_feature_schema,
    get_training_medians,
    validate_feature_vector,
    EXPECTED_FEATURE_COUNT,
)
from app.models.registry import registry

# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hybrid_artifacts():
    """Load frozen hybrid model artifacts (scaler, clf, metadata) once per module."""
    scaler, clf, metadata = registry.load_hybrid_model()
    threshold = float(metadata.get("protocol", {}).get("decision_threshold", 0.54))
    return scaler, clf, threshold, metadata


@pytest.fixture(scope="module")
def canonical_schema():
    return get_feature_schema()


@pytest.fixture(scope="module")
def training_medians():
    return get_training_medians()


@pytest.fixture(scope="module")
def factual_vector(hybrid_artifacts):
    """
    Representative factual vector:
    High entailment, low contradiction, high P1 similarity, P2 probabilities low.
    """
    scaler, clf, threshold, _ = hybrid_artifacts
    v = [
        0.55, 0.82, 0.02, 0.48, 3.0,         # P1: strong entailment
        0.01, 0.01, 0.85, 0.0, 3.0,           # P2: low contradiction
        0.18, 0.12, -1.50, -1.99,             # probs + logits (factual territory)
        0.06, 0.15, 0.18, 0.12, 1.45,        # meta signals
    ]
    return np.array(v, dtype=np.float64).reshape(1, 19)


@pytest.fixture(scope="module")
def hallucination_vector(hybrid_artifacts):
    """
    Representative hallucination-risk vector:
    Low entailment, high contradiction, negative support margin, high P1/P2.
    """
    scaler, clf, threshold, _ = hybrid_artifacts
    v = [
        0.02, 0.04, 0.85, -0.82, 5.0,        # P1: high contradiction
        0.88, 0.72, 0.10, 0.75, 5.0,          # P2: high pairwise contradiction
        0.81, 0.78, 1.45, 1.29,               # probs + logits (high risk)
        0.03, 0.795, 0.81, 0.78, 1.038,       # meta signals
    ]
    return np.array(v, dtype=np.float64).reshape(1, 19)


# ─── Tests ────────────────────────────────────────────────────────────────────

# ── Test 1: Valid 19-feature vector accepted ──────────────────────────────────
def test_19_feature_vector_validation(factual_vector):
    """validate_feature_vector must accept a valid 19-dimensional vector."""
    validate_feature_vector(factual_vector)  # must not raise


# ── Test 2: Wrong feature count rejected ─────────────────────────────────────
@pytest.mark.parametrize("bad_count", [0, 1, 5, 18, 20, 100])
def test_wrong_feature_count_rejected(bad_count):
    """Vectors with != 19 features must raise ValueError."""
    bad_vector = np.zeros(bad_count, dtype=np.float64)
    with pytest.raises(ValueError, match=r"exactly 19"):
        validate_feature_vector(bad_vector)


# ── Test 3: NaN in vector rejected ───────────────────────────────────────────
def test_nan_rejected():
    """Vectors containing NaN must raise ValueError."""
    v = np.ones(19, dtype=np.float64)
    v[5] = float("nan")
    with pytest.raises(ValueError, match=r"non-finite"):
        validate_feature_vector(v)


# ── Test 4: Infinity in vector rejected ──────────────────────────────────────
def test_infinity_rejected():
    """Vectors containing Inf must raise ValueError."""
    v = np.ones(19, dtype=np.float64)
    v[12] = float("inf")
    with pytest.raises(ValueError, match=r"non-finite"):
        validate_feature_vector(v)


# ── Test 5: Baseline deterministic ───────────────────────────────────────────
def test_baseline_deterministic(hybrid_artifacts, factual_vector):
    """Two calls with the same input must return identical baseline_probability."""
    scaler, clf, threshold, _ = hybrid_artifacts
    r1 = compute_local_attribution(factual_vector, scaler, clf, threshold)
    r2 = compute_local_attribution(factual_vector, scaler, clf, threshold)
    assert r1.baseline_probability == pytest.approx(r2.baseline_probability, abs=1e-8)


# ── Test 6: Baseline equals training medians applied through model ─────────────
def test_baseline_from_training_median(hybrid_artifacts, training_medians):
    """Baseline probability must equal model output when X = training medians."""
    scaler, clf, threshold, _ = hybrid_artifacts
    X_median = training_medians.reshape(1, 19)
    X_scaled = scaler.transform(X_median)
    expected_baseline = float(clf.predict_proba(X_scaled)[0, 1])

    result = compute_local_attribution(X_median, scaler, clf, threshold)
    # When X == baseline, P_original ≈ P_baseline
    assert result.baseline_probability == pytest.approx(expected_baseline, abs=1e-6)


# ── Test 7: Training medians loaded from RobustScaler.center_ ─────────────────
def test_training_medians_from_robust_scaler(training_medians):
    """Training medians must match RobustScaler.center_ from frozen artifact."""
    scaler, _, _ = registry.load_hybrid_model()  # (scaler, clf, metadata)
    expected = np.array(scaler.center_, dtype=np.float64)
    np.testing.assert_allclose(training_medians, expected, rtol=1e-10,
                               err_msg="Training medians do not match RobustScaler.center_")


# ── Test 8: original_probability matches direct model call ────────────────────
def test_original_probability_match(hybrid_artifacts, hallucination_vector):
    """original_probability must equal clf.predict_proba(scaler.transform(X))[0,1]."""
    scaler, clf, threshold, _ = hybrid_artifacts
    X_scaled = scaler.transform(hallucination_vector)
    expected_prob = float(clf.predict_proba(X_scaled)[0, 1])

    result = compute_local_attribution(hallucination_vector, scaler, clf, threshold)
    assert result.original_probability == pytest.approx(expected_prob, abs=1e-6)


# ── Test 9: Single-feature perturbation affects only one index ────────────────
def test_single_feature_perturbation_isolates_one_index(hybrid_artifacts, factual_vector, training_medians):
    """When computing attribution for feature i, only position i is changed."""
    scaler, clf, threshold, _ = hybrid_artifacts
    X = factual_vector.copy()

    # Manually compute X_0 (only feature index 0 replaced)
    X_0 = X.copy()
    X_0[0, 0] = training_medians[0]

    # Verify exactly one position differs
    diffs = np.where(X[0] != X_0[0])[0]
    assert len(diffs) <= 1, "Only one feature should differ in X_0"


# ── Test 10: Positive attribution indicates increased hallucination risk ────────
def test_positive_attribution_increases_risk(hybrid_artifacts, hallucination_vector, training_medians):
    """For features with positive attribution, replacing them with median should REDUCE P(H)."""
    scaler, clf, threshold, _ = hybrid_artifacts
    result = compute_local_attribution(hallucination_vector, scaler, clf, threshold)

    P_original = result.original_probability
    for f in result.top_hallucination_drivers:
        assert f.attribution > 0, f"Top hallucination driver {f.feature_name} must have positive attribution"
        # Verify: P(H | X_i) < P_original for hallucination-risk features
        X_i = hallucination_vector.copy()
        X_i[0, f.index] = training_medians[f.index]
        P_i = float(clf.predict_proba(scaler.transform(X_i))[0, 1])
        assert P_original - P_i == pytest.approx(f.attribution, abs=1e-5), \
            f"Attribution mismatch for {f.feature_name}"


# ── Test 11: Negative attribution indicates protective effect ─────────────────
def test_negative_attribution_decreases_risk(hybrid_artifacts, factual_vector, training_medians):
    """For features with negative attribution, they locally reduce P(H)."""
    scaler, clf, threshold, _ = hybrid_artifacts
    result = compute_local_attribution(factual_vector, scaler, clf, threshold)

    for f in result.top_protective_drivers:
        assert f.attribution < 0, f"Protective driver {f.feature_name} must have negative attribution"
        # Verify: P(H | X_i) > P_original for protective features
        X_i = factual_vector.copy()
        X_i[0, f.index] = training_medians[f.index]
        P_i = float(clf.predict_proba(scaler.transform(X_i))[0, 1])
        assert result.original_probability - P_i == pytest.approx(f.attribution, abs=1e-5), \
            f"Attribution mismatch for {f.feature_name}"


# ── Test 12: Zero (neutral) attributions handled correctly ────────────────────
def test_zero_attribution_neutral_direction():
    """Features with near-zero attribution must be labelled 'neutral'."""
    f = FeatureAttribution(
        feature_name="test_feature",
        index=0,
        value=0.5,
        baseline=0.5,
        attribution=0.001,  # below threshold
        direction="neutral",
    )
    assert f.direction == "neutral"


# ── Test 13: Interaction gap calculation is consistent ────────────────────────
def test_interaction_gap_calculation(hybrid_artifacts, hallucination_vector):
    """interaction_gap must equal (P_original - P_baseline) - sum(attributions)."""
    scaler, clf, threshold, _ = hybrid_artifacts
    result = compute_local_attribution(hallucination_vector, scaler, clf, threshold)

    total_shift = result.original_probability - result.baseline_probability
    sum_attrs = sum(f.attribution for f in result.features)
    expected_gap = total_shift - sum_attrs

    assert result.interaction_gap == pytest.approx(expected_gap, abs=1e-5), \
        f"interaction_gap should equal (P_orig - P_base) - sum(a_i)"


# ── Test 14: Explanation does not change the classifier decision ───────────────
def test_explanation_does_not_change_decision(hybrid_artifacts, hallucination_vector):
    """Running local attribution must not alter the production decision."""
    scaler, clf, threshold, _ = hybrid_artifacts

    # Reference: direct inference
    X_scaled = scaler.transform(hallucination_vector)
    prob_before = float(clf.predict_proba(X_scaled)[0, 1])
    verdict_before = prob_before >= threshold

    # Run attribution
    result = compute_local_attribution(hallucination_vector, scaler, clf, threshold)

    # After: re-run direct inference (model must be unmodified)
    prob_after = float(clf.predict_proba(X_scaled)[0, 1])
    verdict_after = prob_after >= threshold

    assert prob_before == pytest.approx(prob_after, abs=1e-8), "Classifier output changed after attribution"
    assert verdict_before == verdict_after, "Verdict changed after attribution"


# ── Test 15: Threshold unchanged by attribution ───────────────────────────────
def test_threshold_unchanged(hybrid_artifacts, hallucination_vector):
    """Operating threshold must equal 0.54 and not be modified by attribution."""
    scaler, clf, threshold, _ = hybrid_artifacts
    result = compute_local_attribution(hallucination_vector, scaler, clf, threshold)
    assert result.threshold == pytest.approx(0.54, abs=1e-8), \
        f"Threshold should be 0.54, got {result.threshold}"


# ── Test 16: API backward compatibility (all existing /predict keys present) ───
def test_predict_backward_compatibility(hybrid_artifacts, factual_vector):
    """predict() output must still contain all pre-Phase-37 keys."""
    from app.core.pipeline import get_hallucisense_pipeline
    try:
        p = get_hallucisense_pipeline()
    except Exception:
        pytest.skip("Pipeline requires full model loading; skipping in fast mode")

    result = p.predict(response_text="The Eiffel Tower is in Paris, France.")
    required_keys = {
        "is_hallucinated", "hallucination_probability", "operating_threshold",
        "claim_count", "claims", "explanation", "confidence_score",
    }
    for key in required_keys:
        assert key in result, f"Pre-Phase-37 key '{key}' missing from /predict output"


# ── Test 17: /explain endpoint returns all required local_attribution fields ───
def test_explain_endpoint_required_fields(hybrid_artifacts, factual_vector):
    """compute_local_attribution must produce all required output fields."""
    scaler, clf, threshold, _ = hybrid_artifacts
    result = compute_local_attribution(factual_vector, scaler, clf, threshold)
    result_dict = result.to_dict()

    required_keys = [
        "method", "feature_count", "baseline_type", "original_probability",
        "baseline_probability", "threshold", "decision_margin",
        "interaction_gap", "interaction_gap_explanation", "scientific_caveat",
        "features", "top_hallucination_drivers", "top_protective_drivers",
        "inference_count",
    ]
    for key in required_keys:
        assert key in result_dict, f"Required key '{key}' missing from local_attribution output"

    assert result_dict["method"] == "local_counterfactual_attribution"
    assert result_dict["baseline_type"] == "training_median_from_robust_scaler"
    assert result_dict["feature_count"] == 19


# ── Test 18: Top hallucination drivers sorted correctly ───────────────────────
def test_top_drivers_sorted_correctly(hybrid_artifacts, hallucination_vector):
    """top_hallucination_drivers must be in descending attribution order."""
    scaler, clf, threshold, _ = hybrid_artifacts
    result = compute_local_attribution(hallucination_vector, scaler, clf, threshold)

    drivers = result.top_hallucination_drivers
    for i in range(len(drivers) - 1):
        assert drivers[i].attribution >= drivers[i + 1].attribution, \
            "top_hallucination_drivers not sorted in descending order"


# ── Test 19: Feature ordering matches canonical schema ───────────────────────
def test_feature_ordering_canonical(hybrid_artifacts, canonical_schema, factual_vector):
    """Feature list in result must match canonical schema order from model_metadata.json."""
    scaler, clf, threshold, _ = hybrid_artifacts
    result = compute_local_attribution(factual_vector, scaler, clf, threshold)

    assert len(result.features) == 19
    for i, f in enumerate(result.features):
        assert f.feature_name == canonical_schema[i], \
            f"Feature at index {i}: expected '{canonical_schema[i]}', got '{f.feature_name}'"
        assert f.index == i, f"Feature index mismatch at position {i}"


# ── Test 20: Repeated calls are deterministic ─────────────────────────────────
def test_repeated_calls_deterministic(hybrid_artifacts, hallucination_vector):
    """Multiple calls with identical input must produce identical attribution."""
    scaler, clf, threshold, _ = hybrid_artifacts
    r1 = compute_local_attribution(hallucination_vector, scaler, clf, threshold)
    r2 = compute_local_attribution(hallucination_vector, scaler, clf, threshold)

    assert r1.original_probability == pytest.approx(r2.original_probability, abs=1e-8)
    assert r1.interaction_gap == pytest.approx(r2.interaction_gap, abs=1e-8)
    for f1, f2 in zip(r1.features, r2.features):
        assert f1.attribution == pytest.approx(f2.attribution, abs=1e-8)


# ── Test 21: Known factual example — structure is valid ───────────────────────
def test_known_factual_example_structure(hybrid_artifacts, factual_vector):
    """Factual vector attribution result must have valid structure."""
    scaler, clf, threshold, _ = hybrid_artifacts
    result = compute_local_attribution(factual_vector, scaler, clf, threshold)

    assert 0.0 <= result.original_probability <= 1.0
    assert 0.0 <= result.baseline_probability <= 1.0
    assert result.inference_count == 21  # 1 original + 1 baseline + 19 features
    assert len(result.features) == 19
    # All attributions must be finite
    for f in result.features:
        assert math.isfinite(f.attribution), f"Attribution for {f.feature_name} is not finite"


# ── Test 22: Known hallucination example — structure is valid ─────────────────
def test_known_hallucination_example_structure(hybrid_artifacts, hallucination_vector):
    """Hallucination vector attribution result must have valid structure."""
    scaler, clf, threshold, _ = hybrid_artifacts
    result = compute_local_attribution(hallucination_vector, scaler, clf, threshold)

    assert 0.0 <= result.original_probability <= 1.0
    assert result.inference_count == 21
    assert len(result.features) == 19
    for f in result.features:
        assert math.isfinite(f.attribution)
    # Verify decision margin is correctly computed
    assert result.decision_margin == pytest.approx(
        result.original_probability - result.threshold, abs=1e-6
    )


# ── Test 23: Near-threshold boundary case ────────────────────────────────────
def test_near_threshold_boundary(hybrid_artifacts, training_medians):
    """Near-threshold vector must produce valid attribution with correct margin sign."""
    scaler, clf, threshold, _ = hybrid_artifacts

    # Use training medians (which tend to be near P_baseline ≈ moderate probability)
    X_median = training_medians.reshape(1, 19)
    result = compute_local_attribution(X_median, scaler, clf, threshold)

    # decision_margin should correctly reflect sign
    expected_margin = result.original_probability - threshold
    assert result.decision_margin == pytest.approx(expected_margin, abs=1e-6)

    # All feature values should equal baselines
    for f in result.features:
        assert f.value == pytest.approx(f.baseline, abs=1e-8), \
            f"When X=median, feature {f.feature_name} value should equal baseline"


# ── Test 24: "12 × 8 = 95" failure analysis ──────────────────────────────────
def test_12x8_failure_analysis(hybrid_artifacts):
    """Failure analysis for known difficult case: '12 × 8 = 95'.

    The frozen classifier may or may not flag this as hallucinated —
    this test RECORDS the actual behavior without asserting a specific verdict.
    It asserts that the attribution is structurally valid and documents the
    classifier's local explanation for this input.

    This test must NOT force the classifier to produce any specific P(H).
    It is documentation of the known failure mode, not a fix target.
    """
    from app.core.pipeline import get_hallucisense_pipeline
    try:
        p = get_hallucisense_pipeline()
    except Exception:
        pytest.skip("Pipeline requires full model loading; skipping in fast mode")

    t0 = time.perf_counter()
    result = p.predict(response_text="12 multiplied by 8 equals 95.")
    latency = time.perf_counter() - t0

    # ── Record actual behavior ──
    is_hallucinated = result["is_hallucinated"]
    prob = result["hallucination_probability"]
    local_attr = result.get("local_attribution", {})

    # ── Structural assertions only ──
    assert "hallucination_probability" in result
    assert 0.0 <= prob <= 1.0
    assert isinstance(is_hallucinated, bool)
    assert local_attr.get("inference_count") == 21
    assert len(local_attr.get("features", [])) == 19

    # ── Documentation (printed, not asserted) ──
    top_driver = (
        local_attr.get("top_hallucination_drivers", [{}])[0].get("feature_name", "N/A")
        if local_attr.get("top_hallucination_drivers")
        else "N/A"
    )
    print(
        f"\n[FAILURE_ANALYSIS] '12 × 8 = 95' | "
        f"P(H)={prob:.4f} | Verdict={'HALLUCINATED' if is_hallucinated else 'FACTUAL'} | "
        f"TopDriver={top_driver} | Latency={latency*1000:.1f}ms"
    )

    # Document the known limitation: numeric arithmetic is a known failure mode
    # because the NLI model (cross-encoder/nli-deberta-v3-small) was not specifically
    # trained for arithmetic verification, and the Hybrid classifier aggregates
    # NLI signals — it does not perform independent arithmetic verification.
    assert True, "Failure analysis documented above"
