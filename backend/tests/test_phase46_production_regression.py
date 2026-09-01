"""Phase 46 — Production Regression & Invariant Tests."""

import hashlib
from pathlib import Path
import pytest
from app.core.pipeline import HalluciSensePipeline

def test_frozen_model_hashes():
    """Verify SHA256 of frozen classifier and scaler are strictly unmodified."""
    results_dir = Path(__file__).resolve().parent.parent / "evaluation_results" / "phase6m" / "final_hybrid_model"
    clf_path = results_dir / "hybrid_meta_classifier.joblib"
    scaler_path = results_dir / "preprocessing.joblib"
    
    assert clf_path.exists()
    assert scaler_path.exists()
    
    clf_sha = hashlib.sha256(clf_path.read_bytes()).hexdigest()
    scaler_sha = hashlib.sha256(scaler_path.read_bytes()).hexdigest()
    
    assert clf_sha == "089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad"
    assert scaler_sha == "bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90"

def test_operating_threshold_invariant():
    """Verify threshold tau* = 0.54 is strictly maintained."""
    pipeline = HalluciSensePipeline()
    assert pipeline.threshold == 0.54
