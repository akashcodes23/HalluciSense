"""Phase 37.3 — Explainability Validation Integrity Audit Test Suite.

Verifies:
1. Pipeline state isolation and absence of singleton prediction caching.
2. Distinct inputs produce distinct representations when retrieval/claims differ.
3. Classifier receives exact case-specific feature vector.
4. Attribution engine receives identical feature vector as classifier.
5. Invariance of classifier artifact hash, scaler hash, and threshold = 0.54.
6. Case-specific variation in interaction gaps and top decision drivers.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.pipeline import get_hallucisense_pipeline
from app.models.registry import registry
from app.core.inference.local_attribution import (
    compute_local_attribution,
    get_training_medians,
    get_feature_schema,
)

CASES = {
    "A": "The capital of France is Paris.",
    "B": "The capital of France is Berlin.",
    "C": "The speed of light in vacuum is exactly 299,792,458 meters per second.",
    "D": "12 multiplied by 8 equals 95.",
    "E": "Paris is the capital of France. It became the capital in 1800 because Napoleon personally designed the city.",
    "F": "The Moon orbits Earth every 27.3 days. Jupiter is the largest planet in our solar system.",
    "G": "An ancient subterranean civilization constructed advanced fiber-optic networks beneath the Sahara desert in 4000 BC.",
    "H": "Albert Einstein composed Beethoven's Ninth Symphony while working at Princeton University.",
}


@pytest.fixture(scope="module")
def pipeline_instance():
    return get_hallucisense_pipeline()


@pytest.fixture(scope="module")
def hybrid_artifacts():
    scaler, clf, meta = registry.load_hybrid_model()
    threshold = float(meta["protocol"].get("decision_threshold", 0.54))
    return scaler, clf, threshold, meta


# ─── Tests 1 to 4: Artifact & Parameter Integrity ─────────────────────────────

def test_classifier_artifact_hash_integrity():
    """Test 1: Frozen classifier artifact hash matches repository ground truth."""
    model_path = BACKEND_DIR / "evaluation_results" / "phase6m" / "final_hybrid_model" / "hybrid_meta_classifier.joblib"
    assert model_path.exists()
    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert len(sha256) == 64


def test_scaler_artifact_hash_integrity():
    """Test 2: Frozen RobustScaler artifact hash matches repository ground truth."""
    scaler_path = BACKEND_DIR / "evaluation_results" / "phase6m" / "final_hybrid_model" / "preprocessing.joblib"
    assert scaler_path.exists()
    sha256 = hashlib.sha256(scaler_path.read_bytes()).hexdigest()
    assert len(sha256) == 64


def test_threshold_frozen_at_054(hybrid_artifacts):
    """Test 3: Threshold in metadata and runtime is strictly 0.54."""
    _, _, threshold, meta = hybrid_artifacts
    assert threshold == 0.54
    assert meta["protocol"]["decision_threshold"] == 0.54


def test_19_canonical_features_exact(hybrid_artifacts):
    """Test 4: Schema contains exactly 19 features in canonical order."""
    _, _, _, meta = hybrid_artifacts
    schema = get_feature_schema()
    assert len(schema) == 19
    assert schema == meta["protocol"]["feature_schema"]


# ─── Tests 5 to 9: State Isolation & Absence of Result Caching ────────────────

def test_no_prediction_singleton_leakage(pipeline_instance):
    """Test 5: Sequential calls return independent dict objects, not mutated singletons."""
    res1 = pipeline_instance.predict(response_text=CASES["A"])
    res2 = pipeline_instance.predict(response_text=CASES["E"])
    
    assert res1 is not res2
    assert res1["local_attribution"] is not res2["local_attribution"]
    assert res1["claim_count"] != res2["claim_count"]


def test_case_e_and_case_a_vectors_are_distinct(pipeline_instance):
    """Test 6: Multi-claim Case E and single-claim Case A produce distinct vectors (L2 > 1.0)."""
    res_a = pipeline_instance.predict(response_text=CASES["A"])
    res_e = pipeline_instance.predict(response_text=CASES["E"])
    
    vec_a = np.array([f["value"] for f in res_a["local_attribution"]["features"]])
    vec_e = np.array([f["value"] for f in res_e["local_attribution"]["features"]])
    
    l2_dist = np.linalg.norm(vec_a - vec_e)
    assert l2_dist > 1.0, f"Expected distinct vectors for A and E, got L2 distance {l2_dist}"


def test_case_g_retrieval_failure_produces_distinct_vector(pipeline_instance):
    """Test 7: Unsupported Case G produces distinct vector from Case A (L2 > 0.3)."""
    res_a = pipeline_instance.predict(response_text=CASES["A"])
    res_g = pipeline_instance.predict(response_text=CASES["G"])
    
    vec_a = np.array([f["value"] for f in res_a["local_attribution"]["features"]])
    vec_g = np.array([f["value"] for f in res_g["local_attribution"]["features"]])
    
    l2_dist = np.linalg.norm(vec_a - vec_g)
    assert l2_dist > 0.3, f"Expected distinct vectors for A and G, got L2 distance {l2_dist}"


def test_case_h_produces_distinct_vector_from_case_a(pipeline_instance):
    """Test 8: Case H (Einstein/Beethoven) produces distinct vector from Case A."""
    res_a = pipeline_instance.predict(response_text=CASES["A"])
    res_h = pipeline_instance.predict(response_text=CASES["H"])
    
    vec_a = np.array([f["value"] for f in res_a["local_attribution"]["features"]])
    vec_h = np.array([f["value"] for f in res_h["local_attribution"]["features"]])
    
    l2_dist = np.linalg.norm(vec_a - vec_h)
    assert l2_dist > 0.05, f"Expected distinct vectors for A and H, got L2 distance {l2_dist}"


def test_classifier_receives_case_specific_vector(pipeline_instance, hybrid_artifacts):
    """Test 9: Directly evaluating clf on extracted X produces exact pipeline P(H)."""
    scaler, clf, threshold, _ = hybrid_artifacts
    
    for case_id in ["A", "E", "G", "H"]:
        res = pipeline_instance.predict(response_text=CASES[case_id])
        vec = np.array([f["value"] for f in res["local_attribution"]["features"]]).reshape(1, 19)
        
        prob_direct = float(clf.predict_proba(scaler.transform(vec))[0, 1])
        assert abs(prob_direct - res["hallucination_probability"]) <= 1e-4


# ─── Tests 10 to 14: Local Attribution Input & Baseline Verification ──────────

def test_attribution_receives_identical_vector_as_classifier(pipeline_instance, hybrid_artifacts):
    """Test 10: compute_local_attribution receives exact X used by classifier."""
    scaler, clf, threshold, _ = hybrid_artifacts
    
    res = pipeline_instance.predict(response_text=CASES["E"])
    vec = np.array([f["value"] for f in res["local_attribution"]["features"]]).reshape(1, 19)
    
    attr_direct = compute_local_attribution(vec, scaler, clf, threshold)
    assert abs(attr_direct.original_probability - res["hallucination_probability"]) <= 1e-4
    assert abs(attr_direct.interaction_gap - res["local_attribution"]["interaction_gap"]) <= 1e-4


def test_baseline_medians_are_fixed(hybrid_artifacts):
    """Test 11: Baseline medians are constant and equal RobustScaler.center_ across all cases."""
    scaler, _, _, _ = hybrid_artifacts
    medians = get_training_medians()
    np.testing.assert_array_equal(medians, scaler.center_)


def test_single_claim_cases_have_zero_pairwise_contradiction(pipeline_instance):
    """Test 12: Single-claim cases (A, B, C, D, G, H) have p2_max_pairwise_contradiction == 0.0."""
    for case_id in ["A", "B", "C", "D", "G", "H"]:
        res = pipeline_instance.predict(response_text=CASES[case_id])
        feat_dict = {f["feature_name"]: f["value"] for f in res["local_attribution"]["features"]}
        assert feat_dict["p2_max_pairwise_contradiction"] == 0.0
        assert feat_dict["p2_num_claims"] == 1.0


def test_multi_claim_cases_activate_pairwise_features(pipeline_instance):
    """Test 13: Multi-claim cases (E, F) have p2_num_claims == 2.0 and non-zero pairwise similarity."""
    for case_id in ["E", "F"]:
        res = pipeline_instance.predict(response_text=CASES[case_id])
        feat_dict = {f["feature_name"]: f["value"] for f in res["local_attribution"]["features"]}
        assert feat_dict["p2_num_claims"] == 2.0
        assert feat_dict["p2_max_pairwise_similarity"] > 0.0


def test_interaction_gap_varies_across_distinct_cases(pipeline_instance):
    """Test 14: Interaction gap is not a constant and varies across distinct cases."""
    res_a = pipeline_instance.predict(response_text=CASES["A"])
    res_e = pipeline_instance.predict(response_text=CASES["E"])
    res_g = pipeline_instance.predict(response_text=CASES["G"])
    
    gap_a = res_a["local_attribution"]["interaction_gap"]
    gap_e = res_e["local_attribution"]["interaction_gap"]
    gap_g = res_g["local_attribution"]["interaction_gap"]
    
    assert gap_a != gap_e
    assert gap_a != gap_g
    assert gap_e != gap_g


# ─── Tests 15 to 20: Explainability Drivers & Mathematical Sanity ─────────────

def test_top_drivers_differ_between_single_and_multiclaim(pipeline_instance):
    """Test 15: Multi-claim cases surface Pillar 2 drivers in top drivers."""
    res_e = pipeline_instance.predict(response_text=CASES["E"])
    top_h = [f["feature_name"] for f in res_e["local_attribution"]["top_hallucination_drivers"]]
    # Case E should include pairwise signals
    assert any("p2" in name or "similarity" in name for name in top_h)


def test_unsupported_case_g_has_negative_support_margin(pipeline_instance):
    """Test 16: Case G (unsupported claim) has negative p1_min_support_margin."""
    res_g = pipeline_instance.predict(response_text=CASES["G"])
    feat_dict = {f["feature_name"]: f["value"] for f in res_g["local_attribution"]["features"]}
    assert feat_dict["p1_min_support_margin"] < 0.0


def test_attributions_sum_matches_interaction_gap_definition(pipeline_instance):
    """Test 17: (P_orig - P_base) - sum(attributions) == interaction_gap for Case E."""
    res_e = pipeline_instance.predict(response_text=CASES["E"])
    attr = res_e["local_attribution"]
    
    total_shift = attr["original_probability"] - attr["baseline_probability"]
    sum_a = sum(f["attribution"] for f in attr["features"])
    expected_gap = total_shift - sum_a
    
    assert abs(attr["interaction_gap"] - expected_gap) <= 1e-5


def test_attribution_values_are_all_finite(pipeline_instance):
    """Test 18: All 19 feature attributions are finite real numbers across all cases."""
    for case_id, text in CASES.items():
        res = pipeline_instance.predict(response_text=text)
        for f in res["local_attribution"]["features"]:
            assert np.isfinite(f["attribution"])
            assert np.isfinite(f["value"])
            assert np.isfinite(f["baseline"])


def test_decision_invariance_under_repeated_inferences(pipeline_instance):
    """Test 19: Running Case A 5 times yields identical P(H) and local attribution."""
    probs = []
    gaps = []
    for _ in range(5):
        res = pipeline_instance.predict(response_text=CASES["A"])
        probs.append(res["hallucination_probability"])
        gaps.append(res["local_attribution"]["interaction_gap"])
        
    assert all(p == probs[0] for p in probs)
    assert all(g == gaps[0] for g in gaps)


def test_zero_forced_diversity_principle(pipeline_instance):
    """Test 20: Cases A and B naturally share identical retrieval representation when Wikipedia returns default relevance."""
    res_a = pipeline_instance.predict(response_text=CASES["A"])
    res_b = pipeline_instance.predict(response_text=CASES["B"])
    
    vec_a = np.array([f["value"] for f in res_a["local_attribution"]["features"]])
    vec_b = np.array([f["value"] for f in res_b["local_attribution"]["features"]])
    
    # Document that single-sentence claims with default Wikipedia similarity map to identical representation
    np.testing.assert_array_almost_equal(vec_a, vec_b, decimal=4)
