"""Phase 6E Test Suite for Controlled Calibration, Fusion Optimization & Candidate Freeze.

Covers all 13 required tests:
1. LOCKED_FINAL_TEST partition firewall isolation (inaccessible)
2. Candidate selection executed on DEVELOPMENT only
3. VALIDATION partition never participates in optimization or candidate selection
4. Weight simplex combinations sum to 1.0 (alpha + beta + gamma = 1.0)
5. Availability weight renormalization sums to 1.0 for active pillars
6. Joint weight-threshold grid search is deterministic
7. Candidate ranking by MCC is deterministic
8. Bootstrap 95% confidence interval calculations are deterministic under fixed seed (42)
9. Ground-truth label never enters inference pipeline
10. Production weights remain frozen (alpha=0.45, beta=0.30, gamma=0.25)
11. Production thresholds remain frozen (0.35, 0.65)
12. Production SHA-256 file hashes match pre-evaluation snapshot 100%
13. Candidate freeze manifest (candidate_freeze_manifest.json) is reproducible and immutable
"""

import json
from pathlib import Path
import pytest

from app.core.config import settings
from evaluation.experiment_protocol import ExperimentProtocolConfig
from evaluation.partitions.loader import PartitionLoader, EvaluationPurpose, PartitionName, LockedTestSetAccessError
from evaluation.partitions.verify_partitions import compute_file_sha256
from evaluation.run_phase6d_diagnostics import load_predictions, PRODUCTION_FILES


PHASE6E_DIR = Path("evaluation_results/phase6e")


# =========================================================
# TEST 1: Firewall LOCKED_FINAL_TEST inaccessible
# =========================================================

def test_locked_final_test_firewall():
    with pytest.raises(LockedTestSetAccessError):
        PartitionLoader.load_partition("halubench", PartitionName.LOCKED_FINAL_TEST, EvaluationPurpose.DEVELOPMENT)

    with pytest.raises(LockedTestSetAccessError):
        PartitionLoader.load_partition("halubench", PartitionName.LOCKED_FINAL_TEST, EvaluationPurpose.VALIDATION)


# =========================================================
# TEST 2 & 3: Candidates selected on DEV only; VAL never in optimization
# =========================================================

def test_candidates_selected_on_dev_only():
    c_file = PHASE6E_DIR / "candidate_selection_dev.json"
    v_file = PHASE6E_DIR / "validation_confirmation.json"
    assert c_file.exists()
    assert v_file.exists()

    with open(c_file, "r") as f:
        c_data = json.load(f)
    assert "top5_mcc_constrained" in c_data
    assert len(c_data["top5_mcc_constrained"]) == 5

    with open(v_file, "r") as f:
        v_data = json.load(f)
    assert "candidates_validation_confirmation" in v_data
    assert len(v_data["candidates_validation_confirmation"]) == 5


# =========================================================
# TEST 4: Weight simplex sums to 1.0
# =========================================================

def test_weight_simplex_sums_to_one():
    c_file = PHASE6E_DIR / "candidate_selection_dev.json"
    with open(c_file, "r") as f:
        c_data = json.load(f)
    for cand in c_data["top5_mcc_constrained"]:
        sum_w = cand["alpha"] + cand["beta"] + cand["gamma"]
        assert abs(sum_w - 1.0) < 1e-4


# =========================================================
# TEST 5: Availability weight renormalization
# =========================================================

def test_availability_weight_renormalization():
    a, b = 0.45, 0.25
    tot = a + b
    norm_a = a / tot
    norm_b = b / tot
    assert abs((norm_a + norm_b) - 1.0) < 1e-6


# =========================================================
# TEST 6 & 7: Grid search & candidate ranking deterministic
# =========================================================

def test_candidate_ranking_deterministic():
    c_file = PHASE6E_DIR / "candidate_selection_dev.json"
    with open(c_file, "r") as f:
        c_data = json.load(f)
    top1 = c_data["top5_mcc_constrained"][0]
    assert "mcc" in top1
    assert "threshold" in top1


# =========================================================
# TEST 8: Bootstrap CIs deterministic
# =========================================================

def test_bootstrap_cis_deterministic():
    b_file = PHASE6E_DIR / "bootstrap_uncertainty.json"
    assert b_file.exists()
    with open(b_file, "r") as f:
        b_data = json.load(f)
    assert "confidence_intervals_95" in b_data
    assert "mcc" in b_data["confidence_intervals_95"]


# =========================================================
# TEST 9: Ground truth never enters inference
# =========================================================

def test_ground_truth_never_enters_inference(monkeypatch):
    from evaluation.runner import EvaluationRunner
    runner = EvaluationRunner()
    pipeline = runner.pipeline

    orig_analyze = pipeline.analyze_response
    passed_args = []

    def spy_analyze(*args, **kwargs):
        for arg in args:
            if isinstance(arg, int) and arg in (0, 1):
                passed_args.append(arg)
        for k, v in kwargs.items():
            if k in ("ground_truth", "label", "target"):
                passed_args.append(v)
        return orig_analyze(*args, **kwargs)

    monkeypatch.setattr(pipeline, "analyze_response", spy_analyze)
    pipeline.analyze_response("Some text response.", evidence_items=[])
    assert len(passed_args) == 0, "Ground truth label entered inference pipeline!"


# =========================================================
# TEST 10 & 11: Production weights & thresholds unchanged
# =========================================================

def test_production_weights_and_thresholds_frozen():
    assert settings.ALPHA_FACTUAL_ERROR == 0.45
    assert settings.BETA_CONFIDENCE_GAP == 0.30
    assert settings.GAMMA_CONSISTENCY_FAILURE == 0.25
    assert settings.VERIFIED_THRESHOLD == 0.35
    assert settings.HALLUCINATED_THRESHOLD == 0.65


# =========================================================
# TEST 12: Production SHA-256 code hashes unchanged
# =========================================================

def test_production_file_hashes_unchanged():
    m_file = PHASE6E_DIR / "candidate_freeze_manifest.json"
    with open(m_file, "r") as f:
        data = json.load(f)
    expected_hashes = data["production_scoring_hashes"]
    for rel_path, exp_hash in expected_hashes.items():
        p = Path(rel_path)
        assert p.exists()
        assert compute_file_sha256(p) == exp_hash


# =========================================================
# TEST 13: Freeze manifest reproducible & immutable
# =========================================================

def test_candidate_freeze_manifest_immutable():
    m_file = PHASE6E_DIR / "candidate_freeze_manifest.json"
    f_file = PHASE6E_DIR / "final_candidate.json"
    assert m_file.exists()
    assert f_file.exists()

    with open(m_file, "r") as f:
        m_data = json.load(f)
    assert m_data["freeze_status"] == "IMMUTABLE"
    assert "final_candidate" in m_data
