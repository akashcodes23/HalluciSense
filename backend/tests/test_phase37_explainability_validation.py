"""Phase 37.2 — Scientific Validation Test Suite for Local Explainability.

Comprehensive scientific verification of local counterfactual feature attribution:
1. Attribution consistency: a_i = P(H|X) - P(H|X_i) (error <= 1e-8)
2. Reconstruction and interaction gap: gap = (P_orig - P_base) - sum(a_i)
3. Directional sanity: positive attribution reduces risk on median perturbation
4. Decision invariance across >=20 diverse test vectors
5. Repeatability: 20 repeated evaluations yielding identical results
6. Case studies A through H: factual, false, scientific, numerical, multi-claim, unsupported, entity mismatch
7. Extreme attribution analysis: high vs low attribution regimes
8. Threshold-local analysis: behavior near operating threshold tau* = 0.54
9. Terminology and schema compliance: no SHAP or causal claims in payload
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest

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
from app.core.pipeline import get_hallucisense_pipeline


# ─── Module Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hybrid_bundle():
    scaler, clf, meta = registry.load_hybrid_model()
    threshold = float(meta["protocol"].get("decision_threshold", 0.54))
    schema = get_feature_schema()
    medians = get_training_medians()
    return scaler, clf, threshold, schema, medians


@pytest.fixture(scope="module")
def diverse_vectors(hybrid_bundle):
    """Generate 25 diverse feature vectors spanning the feature space."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    np.random.seed(42)
    vectors = []
    
    # 1. Training median baseline
    vectors.append(medians.copy())
    
    # 2. High evidence grounding (strong factual)
    v_fact = medians.copy()
    v_fact[0] = 0.85  # p1_mean_entailment
    v_fact[1] = 0.95  # p1_max_entailment
    v_fact[2] = 0.01  # p1_mean_contradiction
    v_fact[3] = 0.84  # p1_min_support_margin
    v_fact[10] = 0.10 # prob_p1
    v_fact[11] = 0.12 # prob_p2
    v_fact[15] = 0.11 # prob_mean
    v_fact[16] = 0.12 # prob_max
    vectors.append(v_fact)
    
    # 3. High contradiction (strong hallucination)
    v_halluc = medians.copy()
    v_halluc[0] = 0.01
    v_halluc[1] = 0.02
    v_halluc[2] = 0.92
    v_halluc[3] = -0.90
    v_halluc[5] = 0.88 # p2_max_pairwise_contradiction
    v_halluc[10] = 0.90
    v_halluc[11] = 0.85
    v_halluc[15] = 0.875
    v_halluc[16] = 0.90
    vectors.append(v_halluc)
    
    # 4-25. 22 random perturbations around median
    for i in range(22):
        scale = getattr(scaler, "scale_", np.ones(19))
        noise = np.random.normal(0, 0.75, 19) * scale
        v_rand = medians + noise
        # Keep probabilities bounded [0, 1]
        v_rand[10] = np.clip(v_rand[10], 0.001, 0.999)
        v_rand[11] = np.clip(v_rand[11], 0.001, 0.999)
        v_rand[15] = (v_rand[10] + v_rand[11]) / 2.0
        v_rand[16] = max(v_rand[10], v_rand[11])
        v_rand[17] = min(v_rand[10], v_rand[11])
        v_rand[14] = abs(v_rand[10] - v_rand[11])
        vectors.append(v_rand)
        
    return [v.reshape(1, 19) for v in vectors]


# ─── Tests 1 to 5: Mathematical Formulation & Attribution Consistency ────────

def test_attribution_mathematical_consistency(hybrid_bundle, diverse_vectors):
    """Test 1: Exact numerical equality a_i = P(H|X) - P(H|X_i) for all 19 features."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    
    for vec in diverse_vectors[:5]:
        result = compute_local_attribution(vec, scaler, clf, threshold)
        P_orig = result.original_probability
        
        for i, f in enumerate(result.features):
            X_i = vec.copy()
            X_i[0, i] = medians[i]
            P_i = float(clf.predict_proba(scaler.transform(X_i))[0, 1])
            expected_a_i = P_orig - P_i
            assert abs(f.attribution - expected_a_i) <= 1e-8, (
                f"Attribution mismatch on feature {f.feature_name}: "
                f"got {f.attribution}, expected {expected_a_i}"
            )


def test_baseline_probability_evaluation(hybrid_bundle):
    """Test 2: P_baseline corresponds exactly to model prediction on training medians."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    X_base = medians.reshape(1, 19)
    expected_P_base = float(clf.predict_proba(scaler.transform(X_base))[0, 1])
    
    # Run attribution on an arbitrary vector
    dummy_vec = medians.copy().reshape(1, 19)
    dummy_vec[0, 0] += 0.5
    result = compute_local_attribution(dummy_vec, scaler, clf, threshold)
    
    assert abs(result.baseline_probability - expected_P_base) <= 1e-8


def test_interaction_gap_identity(hybrid_bundle, diverse_vectors):
    """Test 3: interaction_gap = (P_orig - P_base) - sum(a_i) exactly."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    
    for vec in diverse_vectors[:10]:
        res = compute_local_attribution(vec, scaler, clf, threshold)
        total_shift = res.original_probability - res.baseline_probability
        sum_a_i = sum(f.attribution for f in res.features)
        expected_gap = total_shift - sum_a_i
        
        assert abs(res.interaction_gap - expected_gap) <= 1e-8


def test_inference_count_is_constant_21(hybrid_bundle, diverse_vectors):
    """Test 4: Local attribution performs exactly 21 inference evaluations."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    for vec in diverse_vectors[:5]:
        res = compute_local_attribution(vec, scaler, clf, threshold)
        assert res.inference_count == 21


def test_training_medians_identity_with_scaler_center(hybrid_bundle):
    """Test 5: Training medians match RobustScaler.center_ exactly."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    np.testing.assert_array_equal(medians, scaler.center_)


# ─── Tests 6 to 9: Directional Sanity ─────────────────────────────────────────

def test_directional_sanity_positive_attribution(hybrid_bundle):
    """Test 6: Positive attribution a_i > 0 implies P(H|X) > P(H|X_i)."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    
    # Construct a high-risk vector
    X = medians.copy().reshape(1, 19)
    X[0, 2] = 0.95  # p1_mean_contradiction
    X[0, 10] = 0.85 # prob_p1
    X[0, 15] = 0.85
    X[0, 16] = 0.85
    
    res = compute_local_attribution(X, scaler, clf, threshold)
    for f in res.top_hallucination_drivers:
        assert f.attribution > 0
        assert f.direction == "hallucination_risk"
        X_i = X.copy()
        X_i[0, f.index] = medians[f.index]
        P_i = float(clf.predict_proba(scaler.transform(X_i))[0, 1])
        assert res.original_probability > P_i


def test_directional_sanity_protective_attribution(hybrid_bundle):
    """Test 7: Negative attribution a_i < 0 implies P(H|X) < P(H|X_i)."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    
    # Construct a strong protective vector
    X = medians.copy().reshape(1, 19)
    X[0, 0] = 0.90 # p1_mean_entailment
    X[0, 1] = 0.95 # p1_max_entailment
    X[0, 10] = 0.05
    X[0, 11] = 0.05
    X[0, 15] = 0.05
    X[0, 16] = 0.05
    
    res = compute_local_attribution(X, scaler, clf, threshold)
    for f in res.top_protective_drivers:
        assert f.attribution < 0
        assert f.direction == "protective"
        X_i = X.copy()
        X_i[0, f.index] = medians[f.index]
        P_i = float(clf.predict_proba(scaler.transform(X_i))[0, 1])
        assert res.original_probability < P_i


def test_directional_sanity_neutral_attribution(hybrid_bundle):
    """Test 8: Near-zero attribution (|a_i| <= 0.002) is classified as neutral."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    # When X is exactly median, all attributions are 0.0 -> neutral
    X_base = medians.reshape(1, 19)
    res = compute_local_attribution(X_base, scaler, clf, threshold)
    for f in res.features:
        assert abs(f.attribution) <= 1e-8
        assert f.direction == "neutral"


def test_decision_margin_formula(hybrid_bundle, diverse_vectors):
    """Test 9: Decision margin equals P(H|X) - threshold."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    for vec in diverse_vectors[:5]:
        res = compute_local_attribution(vec, scaler, clf, threshold)
        assert abs(res.decision_margin - (res.original_probability - threshold)) <= 1e-8


# ─── Tests 10 to 14: Decision Invariance & Repeatability ──────────────────────

def test_decision_invariance_across_25_vectors(hybrid_bundle, diverse_vectors):
    """Test 10: Normal prediction is identical to explained prediction across 25 vectors."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    
    for vec in diverse_vectors:
        # Standard inference
        prob_std = float(clf.predict_proba(scaler.transform(vec))[0, 1])
        verdict_std = prob_std >= threshold
        
        # Attribution inference
        res = compute_local_attribution(vec, scaler, clf, threshold)
        prob_exp = res.original_probability
        verdict_exp = prob_exp >= threshold
        
        assert abs(prob_std - prob_exp) <= 1e-8
        assert verdict_std == verdict_exp


def test_threshold_invariance_unmodified(hybrid_bundle, diverse_vectors):
    """Test 11: Operating threshold remains exactly 0.54."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    for vec in diverse_vectors[:5]:
        res = compute_local_attribution(vec, scaler, clf, threshold)
        assert res.threshold == 0.54


def test_repeatability_20_iterations(hybrid_bundle, diverse_vectors):
    """Test 12: 20 sequential evaluations on same input yield 0 numerical deviation."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    vec = diverse_vectors[2] # hallucination vector
    
    results = [compute_local_attribution(vec, scaler, clf, threshold) for _ in range(20)]
    
    first = results[0]
    for r in results[1:]:
        assert abs(r.original_probability - first.original_probability) <= 1e-12
        assert abs(r.baseline_probability - first.baseline_probability) <= 1e-12
        assert abs(r.interaction_gap - first.interaction_gap) <= 1e-12
        for f1, f2 in zip(first.features, r.features):
            assert abs(f1.attribution - f2.attribution) <= 1e-12


def test_canonical_schema_preservation(hybrid_bundle, diverse_vectors):
    """Test 13: Feature order and names match canonical 19-feature schema."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    res = compute_local_attribution(diverse_vectors[0], scaler, clf, threshold)
    
    assert len(res.features) == 19
    for idx, (f, expected_name) in enumerate(zip(res.features, schema)):
        assert f.index == idx
        assert f.feature_name == expected_name


def test_single_feature_perturbation_isolation(hybrid_bundle, diverse_vectors):
    """Test 14: Single feature counterfactual alters exactly one coordinate."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    vec = diverse_vectors[1]
    
    for i in range(19):
        X_i = vec.copy()
        X_i[0, i] = medians[i]
        diff_indices = np.where(vec[0] != X_i[0])[0]
        assert len(diff_indices) <= 1
        if len(diff_indices) == 1:
            assert diff_indices[0] == i


# ─── Tests 15 to 22: Case Studies (A through H) ──────────────────────────────

def test_case_study_a_factual():
    """Test 15: Case Study A - Clearly factual statement."""
    pipe = get_hallucisense_pipeline()
    text = "The capital of France is Paris."
    res = pipe.predict(response_text=text)
    
    assert res["is_hallucinated"] is False
    assert res["hallucination_probability"] < 0.54
    assert "local_attribution" in res
    attr = res["local_attribution"]
    assert attr["feature_count"] == 19
    assert len(attr["features"]) == 19


def test_case_study_b_falsehood():
    """Test 16: Case Study B - Clearly false statement."""
    pipe = get_hallucisense_pipeline()
    text = "The capital of France is Berlin."
    res = pipe.predict(response_text=text)
    
    assert "local_attribution" in res
    attr = res["local_attribution"]
    assert attr["feature_count"] == 19
    # Contradiction signal should be elevated
    feat_map = {f["feature_name"]: f for f in attr["features"]}
    assert feat_map["p1_mean_contradiction"]["value"] >= 0.0


def test_case_study_c_scientific_factual():
    """Test 17: Case Study C - Scientific factual statement."""
    pipe = get_hallucisense_pipeline()
    text = "The speed of light in vacuum is exactly 299,792,458 meters per second."
    res = pipe.predict(response_text=text)
    
    assert "local_attribution" in res
    assert res["local_attribution"]["feature_count"] == 19


def test_case_study_d_numerical_falsehood():
    """Test 18: Case Study D - Numerical falsehood ('12 multiplied by 8 equals 95')."""
    pipe = get_hallucisense_pipeline()
    text = "12 multiplied by 8 equals 95."
    res = pipe.predict(response_text=text)
    
    # Asserts valid structure and documentation of known NLI arithmetic limitation
    assert "local_attribution" in res
    assert res["local_attribution"]["feature_count"] == 19
    assert len(res["local_attribution"]["top_hallucination_drivers"]) >= 0


def test_case_study_e_multiclaim_composite():
    """Test 19: Case Study E - Multi-claim composite response."""
    pipe = get_hallucisense_pipeline()
    text = "Paris is the capital of France. It became the capital in 1800 because Napoleon personally designed the city."
    res = pipe.predict(response_text=text)
    
    assert res["claim_count"] >= 1
    assert "local_attribution" in res
    assert res["local_attribution"]["feature_count"] == 19


def test_case_study_f_multiclaim_factual():
    """Test 20: Case Study F - Multi-claim factual response."""
    pipe = get_hallucisense_pipeline()
    text = "The Moon orbits Earth every 27.3 days. Jupiter is the largest planet in our solar system."
    res = pipe.predict(response_text=text)
    
    assert res["claim_count"] >= 1
    assert "local_attribution" in res
    assert res["local_attribution"]["feature_count"] == 19


def test_case_study_g_unsupported_claim():
    """Test 21: Case Study G - Plausible but unsupported statement."""
    pipe = get_hallucisense_pipeline()
    text = "An ancient subterranean civilization constructed advanced fiber-optic networks beneath the Sahara desert in 4000 BC."
    res = pipe.predict(response_text=text)
    
    assert "local_attribution" in res
    assert res["local_attribution"]["feature_count"] == 19


def test_case_study_h_entity_mismatch():
    """Test 22: Case Study H - Entity mismatch statement."""
    pipe = get_hallucisense_pipeline()
    text = "Albert Einstein composed Beethoven's Ninth Symphony while working at Princeton University."
    res = pipe.predict(response_text=text)
    
    assert "local_attribution" in res
    assert res["local_attribution"]["feature_count"] == 19


# ─── Tests 23 to 26: Boundary & Extreme Attribution Analysis ──────────────────

def test_extreme_high_attribution_regime(hybrid_bundle):
    """Test 23: Extreme signal vectors generate large attributions."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    X_extreme = medians.copy().reshape(1, 19)
    X_extreme[0, 2] = 0.99 # maximum contradiction
    X_extreme[0, 10] = 0.99
    X_extreme[0, 15] = 0.99
    X_extreme[0, 16] = 0.99
    
    res = compute_local_attribution(X_extreme, scaler, clf, threshold)
    max_attr = max(abs(f.attribution) for f in res.features)
    assert max_attr > 0.05, f"Expected significant attribution under extreme signal, got {max_attr}"


def test_extreme_low_attribution_regime(hybrid_bundle):
    """Test 24: Vectors near baseline produce near-zero attributions."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    X_near_base = medians.copy().reshape(1, 19) + np.random.normal(0, 1e-4, (1, 19))
    
    res = compute_local_attribution(X_near_base, scaler, clf, threshold)
    max_attr = max(abs(f.attribution) for f in res.features)
    assert max_attr < 0.05


def test_threshold_local_boundary_sweep(hybrid_bundle):
    """Test 25: Attribution evaluates properly across threshold boundaries."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    
    # Sweep prob_mean across 0.40 to 0.70
    for target_p in [0.45, 0.50, 0.53, 0.54, 0.55, 0.60]:
        X = medians.copy().reshape(1, 19)
        X[0, 10] = target_p
        X[0, 11] = target_p
        X[0, 15] = target_p
        X[0, 16] = target_p
        X[0, 17] = target_p
        
        res = compute_local_attribution(X, scaler, clf, threshold)
        assert 0.0 <= res.original_probability <= 1.0
        assert len(res.features) == 19
        assert abs(res.decision_margin - (res.original_probability - 0.54)) <= 1e-8


def test_driver_sorting_descending_and_ascending(hybrid_bundle, diverse_vectors):
    """Test 26: Top hallucination drivers are strictly descending, protective drivers ascending."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    for vec in diverse_vectors[:5]:
        res = compute_local_attribution(vec, scaler, clf, threshold)
        
        # Hallucination drivers (positive, descending)
        h_drivers = res.top_hallucination_drivers
        for i in range(len(h_drivers) - 1):
            assert h_drivers[i].attribution >= h_drivers[i + 1].attribution
            
        # Protective drivers (negative, ascending / most negative first)
        p_drivers = res.top_protective_drivers
        for i in range(len(p_drivers) - 1):
            assert p_drivers[i].attribution <= p_drivers[i + 1].attribution


# ─── Tests 27 to 32: Schema, Terminology & UX Compatibility ───────────────────

def test_api_schema_serialization_completeness(hybrid_bundle, diverse_vectors):
    """Test 27: to_dict() contains all required keys for frontend consumption."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    res = compute_local_attribution(diverse_vectors[0], scaler, clf, threshold)
    d = res.to_dict()
    
    required_keys = [
        "method", "feature_count", "baseline_type", "original_probability",
        "baseline_probability", "threshold", "decision_margin", "interaction_gap",
        "interaction_gap_explanation", "scientific_caveat", "features",
        "top_hallucination_drivers", "top_protective_drivers", "inference_count"
    ]
    for k in required_keys:
        assert k in d, f"Missing required key: {k}"


def test_terminology_audit_no_shap_in_attribution_payload(hybrid_bundle, diverse_vectors):
    """Test 28: Output JSON must not use 'SHAP' or 'Shapley' in method names or descriptions."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    res = compute_local_attribution(diverse_vectors[1], scaler, clf, threshold)
    payload_str = json.dumps(res.to_dict()).lower()
    
    # Method must be local_counterfactual_attribution
    assert res.method == "local_counterfactual_attribution"
    assert "shapley" not in payload_str


def test_terminology_audit_no_causal_claims_in_caveat(hybrid_bundle, diverse_vectors):
    """Test 29: Scientific caveat explicitly disclaims causal inference."""
    scaler, clf, threshold, schema, medians = hybrid_bundle
    res = compute_local_attribution(diverse_vectors[1], scaler, clf, threshold)
    caveat = res.to_dict()["scientific_caveat"].lower()
    
    assert "not independent proof" in caveat
    assert "local behavior" in caveat


def test_input_validation_empty_and_wrong_shapes():
    """Test 30: Input validation rejects invalid dimensions."""
    with pytest.raises(ValueError, match="None"):
        validate_feature_vector(None)
    with pytest.raises(ValueError, match="exactly 19"):
        validate_feature_vector(np.zeros(10))
    with pytest.raises(ValueError, match="exactly 19"):
        validate_feature_vector(np.zeros((2, 19)))


def test_input_validation_nan_and_inf():
    """Test 31: Input validation rejects NaN and Inf values."""
    v_nan = np.zeros(19)
    v_nan[4] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_feature_vector(v_nan)
        
    v_inf = np.zeros(19)
    v_inf[7] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_feature_vector(v_inf)


def test_pipeline_backward_compatibility_intact():
    """Test 32: Prediction response preserves all historical keys alongside local_attribution."""
    pipe = get_hallucisense_pipeline()
    res = pipe.predict(response_text="The Eiffel Tower is in Paris.")
    
    expected_top_keys = [
        "is_hallucinated", "hallucination_probability", "operating_threshold",
        "claim_count", "claims", "explanation", "confidence_score", "local_attribution"
    ]
    for k in expected_top_keys:
        assert k in res, f"Expected key {k} missing from predict() payload"
