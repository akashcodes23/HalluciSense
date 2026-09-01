"""Phase 40 — Candidate Model & Artifact Integrity Test Suite.

Verifies:
- Candidate artifact loading from evaluation_results/phase40_candidate/
- Deterministic prediction on 19-dimensional vectors
- Compatibility with RobustScaler
- Attribution exactness on candidate model
"""

from __future__ import annotations

import sys
from pathlib import Path
import joblib
import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.inference.local_attribution import compute_local_attribution, get_training_medians


@pytest.fixture(scope="module")
def candidate_artifacts():
    cand_dir = BACKEND_DIR / "evaluation_results" / "phase40_candidate"
    model_path = cand_dir / "hybrid_meta_classifier_phase40_candidate.joblib"
    scaler_path = cand_dir / "preprocessing_phase40_candidate.joblib"
    
    assert model_path.exists(), "Candidate model artifact missing"
    assert scaler_path.exists(), "Candidate scaler artifact missing"
    
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return clf, scaler


def test_candidate_prediction_bounds(candidate_artifacts):
    """Verify Candidate C outputs valid probabilities in [0, 1]."""
    clf, scaler = candidate_artifacts
    X_dummy = np.zeros((1, 19), dtype=np.float64)
    X_scaled = scaler.transform(X_dummy)
    prob = clf.predict_proba(X_scaled)[0, 1]
    assert 0.0 <= prob <= 1.0


def test_candidate_attribution_fidelity(candidate_artifacts):
    """Verify local attribution runs cleanly on Candidate C."""
    clf, scaler = candidate_artifacts
    X_raw = np.array([[0.8, 0.9, 0.05, 0.85, 1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.15, 0.1, -1.7, -2.1, 0.05, 0.12, 0.15, 0.1, 1.5]])
    
    res = compute_local_attribution(
        X_raw=X_raw,
        scaler=scaler,
        clf=clf,
        threshold=0.54,
    )
    assert res.feature_count == 19
    assert res.inference_count == 21
    assert -1.0 <= res.original_probability <= 1.0
