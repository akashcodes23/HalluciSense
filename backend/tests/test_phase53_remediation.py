"""Phase 53 — Remediation, Symbolic Integration & Invariants Regression Test Suite."""

import os
import json
import hashlib
import pytest
import numpy as np
import joblib
from pathlib import Path

from app.core.verification.gateway import EvidenceIntelligenceGateway
from app.core.pipeline import HalluciSensePipeline

BASE_DIR = Path(__file__).resolve().parent.parent
FROZEN_DIR = BASE_DIR / "evaluation_results" / "phase6m" / "final_hybrid_model"
CANDIDATE_DIR = BASE_DIR / "evaluation_results" / "phase53" / "candidate"


def test_frozen_artifact_immutability():
    """Verify frozen production classifier and scaler hashes are unchanged."""
    f_clf_path = FROZEN_DIR / "hybrid_meta_classifier.joblib"
    f_scaler_path = FROZEN_DIR / "preprocessing.joblib"

    assert f_clf_path.exists(), "Frozen classifier missing"
    assert f_scaler_path.exists(), "Frozen scaler missing"

    with open(f_clf_path, "rb") as f:
        clf_hash = hashlib.sha256(f.read()).hexdigest()
    with open(f_scaler_path, "rb") as f:
        scaler_hash = hashlib.sha256(f.read()).hexdigest()

    assert clf_hash == "089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad"
    assert scaler_hash == "bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90"


def test_candidate_artifacts_presence_and_serialization():
    """Verify Candidate B artifacts are properly serialized in the candidate directory."""
    c_clf_path = CANDIDATE_DIR / "hybrid_meta_classifier_phase53_candidate.joblib"
    c_scaler_path = CANDIDATE_DIR / "preprocessing_phase53_candidate.joblib"
    c_schema_path = CANDIDATE_DIR / "candidate_schema.json"
    c_meta_path = CANDIDATE_DIR / "candidate_metadata.json"

    assert c_clf_path.exists()
    assert c_scaler_path.exists()
    assert c_schema_path.exists()
    assert c_meta_path.exists()

    clf = joblib.load(c_clf_path)
    scaler = joblib.load(c_scaler_path)

    assert hasattr(clf, "predict_proba")
    assert getattr(scaler, "n_features_in_", 19) == 19


def test_symbolic_gateway_arithmetic_routing():
    """Verify EvidenceIntelligenceGateway accurately detects arithmetic contradictions."""
    res_true = EvidenceIntelligenceGateway.verify_claim("14 multiplied by 5 equals 70.")
    assert res_true["status"] == "verified_symbolically"
    assert res_true["is_consistent"] is True
    assert res_true["contradiction"] == 0.0

    res_false = EvidenceIntelligenceGateway.verify_claim("14 multiplied by 5 equals 75.")
    assert res_false["status"] == "verified_symbolically"
    assert res_false["is_consistent"] is False
    assert res_false["contradiction"] == 1.0


def test_counterfactual_directionality_remediation():
    """Verify Candidate B scores hallucinations higher than factual matched claims."""
    c_clf = joblib.load(CANDIDATE_DIR / "hybrid_meta_classifier_phase53_candidate.joblib")
    c_scaler = joblib.load(CANDIDATE_DIR / "preprocessing_phase53_candidate.joblib")

    # Construct representative factual vs hallucinated vectors
    v_factual = np.zeros((1, 19))
    v_factual[0, 0] = 0.95  # p1_mean_entailment high
    v_factual[0, 2] = 0.05  # p1_mean_contradiction low
    v_factual[0, 10] = 0.05 # prob_p1 low
    v_factual[0, 15] = 0.05 # prob_mean low

    v_hallu = np.zeros((1, 19))
    v_hallu[0, 0] = 0.05   # p1_mean_entailment low
    v_hallu[0, 2] = 0.95   # p1_mean_contradiction high
    v_hallu[0, 10] = 0.95  # prob_p1 high
    v_hallu[0, 15] = 0.95  # prob_mean high

    p_f = float(c_clf.predict_proba(c_scaler.transform(v_factual))[0, 1])
    p_h = float(c_clf.predict_proba(c_scaler.transform(v_hallu))[0, 1])

    assert p_h > p_f, f"Expected p_h > p_f, got p_h={p_h}, p_f={p_f}"
    assert p_h > 0.54, f"Expected hallucination probability above threshold, got {p_h}"
    assert p_f < 0.54, f"Expected factual probability below threshold, got {p_f}"
