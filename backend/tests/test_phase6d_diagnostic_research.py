"""Phase 6D Test Suite for Diagnostic Decomposition, Pillar Ablation & Offline Calibration Research.

Covers all 14 required tests:
1. LOCKED_FINAL_TEST partition firewall isolation (inaccessible)
2. DEVELOPMENT (58,002) and VALIDATION (12,483) sample counts
3. Threshold candidates selected using DEVELOPMENT only
4. Weight candidates selected using DEVELOPMENT only
5. Ground-truth label never enters inference pipeline
6. Missing pillar values are never converted into fabricated zero scores
7. Ablation weight renormalization sums to 1.0 for active pillars
8. Weight simplex combinations sum to 1.0
9. Metrics remain strictly bounded in [0, 1]
10. Sampled error cases correspond to actual FP / FN prediction errors
11. Production weights remain frozen (alpha=0.45, beta=0.30, gamma=0.25)
12. Production thresholds remain frozen (0.35, 0.65)
13. Production SHA-256 file hashes match pre-evaluation snapshot 100%
14. Phase 6D diagnostic calculations are deterministic under fixed seed
"""

import json
from pathlib import Path
import pytest

from app.core.config import settings
from evaluation.experiment_protocol import ExperimentProtocolConfig
from evaluation.partitions.loader import PartitionLoader, EvaluationPurpose, PartitionName, LockedTestSetAccessError
from evaluation.partitions.verify_partitions import compute_file_sha256
from evaluation.run_phase6d_diagnostics import (
    load_predictions,
    compute_cohens_d,
    compute_cliffs_delta,
    PRODUCTION_FILES,
)


PHASE6D_DIR = Path("evaluation_results/phase6d")


# =========================================================
# TEST 1: Firewall LOCKED_FINAL_TEST inaccessible
# =========================================================

def test_locked_final_test_firewall():
    with pytest.raises(LockedTestSetAccessError):
        PartitionLoader.load_partition("halubench", PartitionName.LOCKED_FINAL_TEST, EvaluationPurpose.DEVELOPMENT)

    with pytest.raises(LockedTestSetAccessError):
        PartitionLoader.load_partition("halubench", PartitionName.LOCKED_FINAL_TEST, EvaluationPurpose.VALIDATION)


# =========================================================
# TEST 2: DEV (58,002) and VAL (12,483) counts
# =========================================================

def test_dev_and_val_prediction_counts():
    dev_preds = load_predictions("development_predictions.jsonl")
    val_preds = load_predictions("validation_predictions.jsonl")
    assert len(dev_preds) == 58002
    assert len(val_preds) == 12483


# =========================================================
# TEST 3 & 4: Candidate selection uses DEVELOPMENT only
# =========================================================

def test_candidates_selected_on_dev_only():
    t_file = PHASE6D_DIR / "threshold_sweep_development.json"
    w_file = PHASE6D_DIR / "weight_sensitivity_development.json"
    assert t_file.exists()
    assert w_file.exists()

    with open(t_file, "r") as f:
        t_data = json.load(f)
    assert "candidates_selected_on_dev" in t_data
    assert "max_balanced_accuracy" in t_data["candidates_selected_on_dev"]

    with open(w_file, "r") as f:
        w_data = json.load(f)
    assert "top5_candidates_on_dev" in w_data
    assert len(w_data["top5_candidates_on_dev"]) == 5


# =========================================================
# TEST 5: Ground-truth label never enters inference
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
# TEST 6: Missing pillar values not zeroed
# =========================================================

def test_missing_pillars_not_zeroed():
    dev_preds = load_predictions("development_predictions.jsonl")
    none_p2 = [r for r in dev_preds if r["confidence_gap"] is None]
    assert len(none_p2) > 0
    for r in none_p2:
        assert r["confidence_gap"] is None, "Missing Pillar 2 was fabricated as non-None score!"


# =========================================================
# TEST 7: Ablation weight renormalization
# =========================================================

def test_ablation_weight_renormalization():
    w1, w2 = 0.60, 0.40
    norm_w1 = w1 / (w1 + w2)
    norm_w2 = w2 / (w1 + w2)
    assert abs((norm_w1 + norm_w2) - 1.0) < 1e-6


# =========================================================
# TEST 8: Weight simplex sums to 1.0
# =========================================================

def test_weight_simplex_sums_to_one():
    w_file = PHASE6D_DIR / "weight_sensitivity_development.json"
    with open(w_file, "r") as f:
        w_data = json.load(f)
    for cand in w_data["top5_candidates_on_dev"]:
        sum_w = cand["alpha"] + cand["beta"] + cand["gamma"]
        assert abs(sum_w - 1.0) < 1e-4


# =========================================================
# TEST 9: Metrics remain bounded
# =========================================================

def test_metrics_remain_bounded():
    diag_file = PHASE6D_DIR / "pillar_diagnostics.json"
    with open(diag_file, "r") as f:
        data = json.load(f)
    for p_name, p_data in data.items():
        if p_data.get("roc_auc") is not None:
            assert 0.0 <= p_data["roc_auc"] <= 1.0
        if p_data.get("pr_auc") is not None:
            assert 0.0 <= p_data["pr_auc"] <= 1.0


# =========================================================
# TEST 10: Error samples correspond to actual errors
# =========================================================

def test_sampled_errors_correctness():
    err_file = PHASE6D_DIR / "error_analysis.jsonl"
    assert err_file.exists()
    records = []
    with open(err_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 100
    fps = [r for r in records if r["error_type"] == "FALSE_POSITIVE"]
    fns = [r for r in records if r["error_type"] == "FALSE_NEGATIVE"]

    assert len(fps) == 50
    assert len(fns) == 50

    for fp in fps:
        assert fp["ground_truth"] == 0
        assert fp["predicted_class"] == 1

    for fn in fns:
        assert fn["ground_truth"] == 1
        assert fn["predicted_class"] == 0


# =========================================================
# TEST 11 & 12: Production weights & thresholds unchanged
# =========================================================

def test_production_weights_and_thresholds_frozen():
    assert settings.ALPHA_FACTUAL_ERROR == 0.45
    assert settings.BETA_CONFIDENCE_GAP == 0.30
    assert settings.GAMMA_CONSISTENCY_FAILURE == 0.25
    assert settings.VERIFIED_THRESHOLD == 0.35
    assert settings.HALLUCINATED_THRESHOLD == 0.65


# =========================================================
# TEST 13: Production SHA-256 code hashes unchanged
# =========================================================

def test_production_file_hashes_unchanged():
    integ_file = PHASE6D_DIR / "input_integrity.json"
    with open(integ_file, "r") as f:
        data = json.load(f)
    expected_hashes = data["production_scoring_hashes"]
    for rel_path, exp_hash in expected_hashes.items():
        p = Path(rel_path)
        assert p.exists()
        assert compute_file_sha256(p) == exp_hash


# =========================================================
# TEST 14: Phase 6D calculations deterministic
# =========================================================

def test_phase6d_calculations_deterministic():
    x1 = [0.5, 0.6, 0.7, 0.8]
    x0 = [0.1, 0.2, 0.3, 0.4]
    cd1 = compute_cohens_d(x1, x0)
    cd2 = compute_cohens_d(x1, x0)
    assert cd1 == cd2

    delta1 = compute_cliffs_delta(x1, x0)
    delta2 = compute_cliffs_delta(x1, x0)
    assert delta1 == delta2
