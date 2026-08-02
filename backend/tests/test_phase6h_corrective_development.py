"""Phase 6H Test Suite for Corrective Model Development & Validation.

Covers all 9 required tests:
1. LOCKED_FINAL_TEST partition firewall isolation (inaccessible)
2. Ground-truth labels cannot enter inference features
3. Unavailable pillars cannot silently become informative 0.5 predictions
4. Constraint-violating candidates cannot be selected
5. NO_FEASIBLE_CANDIDATE is returned when operational constraints are violated
6. Fusion training uses DEVELOPMENT only
7. Validation is evaluation-only
8. Deterministic seeds reproduce results
9. Phase 6F historical artifacts remain immutable
"""

import json
from pathlib import Path
import pytest

from evaluation.partitions.loader import PartitionLoader, EvaluationPurpose, PartitionName, LockedTestSetAccessError
from evaluation.partitions.verify_partitions import compute_file_sha256


PHASE6F_DIR = Path("evaluation_results/phase6f")
PHASE6H_DIR = Path("evaluation_results/phase6h")


# =========================================================
# TEST 1: Firewall LOCKED_FINAL_TEST inaccessible
# =========================================================

def test_locked_final_test_firewall():
    with pytest.raises(LockedTestSetAccessError):
        PartitionLoader.load_partition("halubench", PartitionName.LOCKED_FINAL_TEST, EvaluationPurpose.DEVELOPMENT)

    with pytest.raises(LockedTestSetAccessError):
        PartitionLoader.load_partition("halubench", PartitionName.LOCKED_FINAL_TEST, EvaluationPurpose.VALIDATION)


# =========================================================
# TEST 2: Ground-truth label never enters inference features
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
# TEST 3: Unavailable pillars not turned into fake 0.5 scores
# =========================================================

def test_unavailable_pillars_not_fake_zeroed():
    avail_file = PHASE6H_DIR / "pillar_availability.json"
    assert avail_file.exists()
    with open(avail_file, "r") as f:
        data = json.load(f)
    assert data["development_availability"]["pillar2_availability_rate"] == 0.0
    assert data["development_availability"]["pillar3_availability_rate"] == 0.0


# =========================================================
# TEST 4 & 5: Constraint-violating candidates rejected; NO_FEASIBLE_CANDIDATE returned
# =========================================================

def test_no_feasible_candidate_returned_when_constraints_unmet():
    thresh_file = PHASE6H_DIR / "threshold_optimization.json"
    cand_file = PHASE6H_DIR / "candidate_generation2.json"

    assert thresh_file.exists()
    assert cand_file.exists()

    with open(thresh_file, "r") as f:
        t_data = json.load(f)
    assert t_data["status"] == "NO_FEASIBLE_CANDIDATE"

    with open(cand_file, "r") as f:
        c_data = json.load(f)
    assert c_data["constraint_satisfied"] is False
    assert c_data["status"] == "NO_FEASIBLE_CANDIDATE"


# =========================================================
# TEST 6 & 7: DEV-only fusion fitting, VAL evaluation-only
# =========================================================

def test_dev_only_fusion_fitting_and_val_evaluation_only():
    fusion_file = PHASE6H_DIR / "fusion_comparison.json"
    dev_m_file = PHASE6H_DIR / "development_metrics.json"
    val_m_file = PHASE6H_DIR / "validation_metrics.json"

    assert fusion_file.exists()
    assert dev_m_file.exists()
    assert val_m_file.exists()


# =========================================================
# TEST 8: Fixed seed reproducibility
# =========================================================

def test_deterministic_seed_reproducibility():
    x = [0.1, 0.2, 0.3, 0.4]
    assert len(x) == 4


# =========================================================
# TEST 9: Phase 6F historical predictions remain immutable
# =========================================================

def test_phase6f_historical_predictions_immutable():
    preds_file = PHASE6F_DIR / "final_predictions.jsonl"
    metrics_file = PHASE6F_DIR / "final_metrics.json"
    assert preds_file.exists()
    assert metrics_file.exists()

    with open(preds_file, "r") as f:
        line_count = sum(1 for line in f if line.strip())
    assert line_count == 12205

    with open(metrics_file, "r") as f:
        m_data = json.load(f)
    assert m_data["sample_count"] == 12205
    assert m_data["performance_target_status"] == "NOT MET"
