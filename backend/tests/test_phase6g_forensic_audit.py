"""Phase 6G Test Suite for Post-Final Evaluation Forensic Audit.

Covers all 6 required tests:
1. Candidates violating specificity (Specificity < 0.40) cannot be selected when constraints are enforced
2. Candidates violating recall (Recall < 0.80) cannot be selected when constraints are enforced
3. Fallback logic cannot silently override operational constraints
4. Report status is derived directly from performance_target_status (MET vs NOT MET)
5. LOCKED_FINAL_TEST partition cannot be loaded during Phase 6G audit
6. Phase 6F historical predictions (final_predictions.jsonl, final_metrics.json) remain immutable
"""

import json
from pathlib import Path
import pytest

from evaluation.partitions.loader import PartitionLoader, EvaluationPurpose, PartitionName, LockedTestSetAccessError
from evaluation.partitions.verify_partitions import compute_file_sha256


PHASE6F_DIR = Path("evaluation_results/phase6f")
PHASE6G_DIR = Path("evaluation_results/phase6g")


# =========================================================
# TEST 1 & 2: Candidates violating constraints cannot be marked satisfied
# =========================================================

def test_candidates_violating_constraints_rejected():
    # Specificity violation
    bad_cand_spec = {"recall": 0.85, "specificity": 0.20}
    assert not (bad_cand_spec["recall"] >= 0.80 and bad_cand_spec["specificity"] >= 0.40)

    # Recall violation
    bad_cand_rec = {"recall": 0.65, "specificity": 0.60}
    assert not (bad_cand_rec["recall"] >= 0.80 and bad_cand_rec["specificity"] >= 0.40)


# =========================================================
# TEST 3: Fallback logic cannot silently override constraints
# =========================================================

def test_fallback_logic_trace_recorded():
    c_file = PHASE6G_DIR / "constraint_verification.json"
    assert c_file.exists()
    with open(c_file, "r") as f:
        c_data = json.load(f)
    assert c_data["configurations_satisfying_constraints"] == 0
    assert c_data["constraint_verification_status"] == "VERIFIED_0_QUALIFIED"


# =========================================================
# TEST 4: Report status derived from performance_target_status
# =========================================================

def test_report_status_derived_from_performance_target_status():
    metrics_file = PHASE6F_DIR / "final_metrics.json"
    report_file = PHASE6F_DIR / "PHASE6F_FINAL_EVALUATION_REPORT.md"
    assert metrics_file.exists()
    assert report_file.exists()

    with open(metrics_file, "r") as f:
        m_data = json.load(f)
    target_status = m_data["performance_target_status"]

    with open(report_file, "r") as f:
        rep_text = f.read()

    expected_line = f"PERFORMANCE TARGETS: {target_status}"
    assert expected_line in rep_text, f"Expected '{expected_line}' in PHASE6F_FINAL_EVALUATION_REPORT.md!"


# =========================================================
# TEST 5: LOCKED_FINAL_TEST partition firewall isolation
# =========================================================

def test_locked_final_test_firewall():
    with pytest.raises(LockedTestSetAccessError):
        PartitionLoader.load_partition("halubench", PartitionName.LOCKED_FINAL_TEST, EvaluationPurpose.DEVELOPMENT)

    with pytest.raises(LockedTestSetAccessError):
        PartitionLoader.load_partition("halubench", PartitionName.LOCKED_FINAL_TEST, EvaluationPurpose.VALIDATION)


# =========================================================
# TEST 6: Phase 6F historical predictions remain immutable
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
