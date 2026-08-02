"""Phase 6C Test Suite for Frozen Baseline Benchmark Evaluation.

Covers Tests 1 through 12 specified in Phase 6C requirements:
- LOCKED_FINAL_TEST partition firewall isolation (inaccessible)
- Ground-truth label isolation from inference pipeline
- Production scoring file SHA-256 hash preservation before/after evaluation
- Environment snapshot recording and serialization
- Deterministic metric computation and 95% bootstrap confidence interval calculation
- Prediction serialization, checkpointing, and failure accounting
- H-score bounds [0, 1] and availability-aware None pillar handling
- Continuous H-score ranking for AUROC/AUPRC (zero threshold/weight fitting)
"""

import json
from pathlib import Path
import pytest

from app.core.config import settings
from evaluation.dataset import BenchmarkSample
from evaluation.experiment_protocol import ExperimentProtocolConfig
from evaluation.metrics import compute_all_metrics
from evaluation.partitions.loader import PartitionLoader, EvaluationPurpose, PartitionName, LockedTestSetAccessError
from evaluation.run_phase6c_baseline import (
    Phase6CBenchmarkRunner,
    record_environment_snapshot,
    verify_production_hashes,
    compute_bootstrap_ci,
    PRODUCTION_FILES,
)


DATASET_ROOT = Path("evaluation_data")
RESULTS_DIR = Path("evaluation_results/phase6c")


# =========================================================
# TEST 1: Locked test partition is inaccessible
# =========================================================

def test_locked_test_inaccessible():
    with pytest.raises(LockedTestSetAccessError, match="FIREWALL DENIAL"):
        PartitionLoader.load_partition(
            dataset_name="halubench",
            partition=PartitionName.LOCKED_FINAL_TEST,
            purpose=EvaluationPurpose.DEVELOPMENT,
        )

    with pytest.raises(LockedTestSetAccessError, match="FIREWALL DENIAL"):
        PartitionLoader.load_partition(
            dataset_name="halubench",
            partition=PartitionName.LOCKED_FINAL_TEST,
            purpose=EvaluationPurpose.VALIDATION,
        )


# =========================================================
# TEST 2: Ground-truth isolation
# =========================================================

def test_ground_truth_isolation(monkeypatch):
    runner = Phase6CBenchmarkRunner()
    pipeline = runner.pipeline
    orig_analyze = pipeline.analyze_response

    labels_passed = []

    def spy_analyze(*args, **kwargs):
        for arg in args:
            if isinstance(arg, int) and arg in (0, 1):
                labels_passed.append(arg)
        for k, v in kwargs.items():
            if k in ("ground_truth", "label", "target"):
                labels_passed.append(v)
        return orig_analyze(*args, **kwargs)

    monkeypatch.setattr(pipeline, "analyze_response", spy_analyze)

    sample = BenchmarkSample(
        id="test_gt_iso",
        prompt="What is the capital of Germany?",
        response="Berlin is the capital of Germany.",
        ground_truth_label=0,
        category="QA",
        metadata={"dataset": "halubench", "passage": "Berlin is the capital of Germany."},
    )

    res = runner.evaluate_sample(sample)
    assert len(labels_passed) == 0, "CRITICAL DATA LEAKAGE: Ground-truth label entered inference pipeline!"
    assert res["ground_truth_label"] == 0
    assert "prediction" in res


# =========================================================
# TEST 3: Frozen production file hashes
# =========================================================

def test_frozen_production_hashes():
    env_snapshot = record_environment_snapshot()
    assert verify_production_hashes(env_snapshot) is True


# =========================================================
# TEST 4: Environment snapshot record
# =========================================================

def test_environment_snapshot_record():
    env_file = RESULTS_DIR / "phase6c_environment.json"
    assert env_file.exists()

    with open(env_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "protocol_fingerprint" in data
    assert data["protocol_fingerprint"] == ExperimentProtocolConfig.get_protocol_fingerprint()
    assert "production_scoring_hashes" in data
    assert "frozen_configuration" in data


# =========================================================
# TEST 5: Deterministic metric computation & bootstrap CIs
# =========================================================

def test_deterministic_metric_computation():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_scores = [0.01, 0.05, 0.10, 0.20, 0.70, 0.80, 0.90, 0.95]
    y_pred = [0, 0, 0, 0, 1, 1, 1, 1]

    m1 = compute_all_metrics(y_true, y_pred, scores=y_scores)
    assert m1["accuracy"] == 1.0
    assert m1["roc_auc"] == 1.0

    ci = compute_bootstrap_ci(y_true, y_scores, y_pred, n_bootstraps=100, seed=42)
    assert "accuracy" in ci
    assert ci["accuracy"]["mean"] == 1.0


# =========================================================
# TEST 6: Prediction serialization
# =========================================================

def test_prediction_serialization():
    runner = Phase6CBenchmarkRunner()
    sample = BenchmarkSample(
        id="ser_001",
        prompt="Who wrote Hamlet?",
        response="Shakespeare wrote Hamlet.",
        ground_truth_label=0,
        category="QA",
        metadata={"dataset": "halubench"},
    )
    res = runner.evaluate_sample(sample)
    json_str = json.dumps(res)
    loaded = json.loads(json_str)
    assert loaded["example_id"] == "ser_001"
    assert loaded["prediction"] in (0, 1)
    assert 0.0 <= loaded["overall_h_score"] <= 1.0


# =========================================================
# TEST 7: Checkpoint and resume
# =========================================================

def test_checkpoint_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.run_phase6c_baseline.RESULTS_DIR", tmp_path)

    runner = Phase6CBenchmarkRunner()
    s1 = BenchmarkSample(id="c1", prompt="p1", response="r1", ground_truth_label=0, category="QA", metadata={"dataset": "halubench"})
    s2 = BenchmarkSample(id="c2", prompt="p2", response="r2", ground_truth_label=1, category="QA", metadata={"dataset": "halubench"})

    # Save mock sample jsonl for dataset
    proc_dir = tmp_path / "processed" / "halubench"
    proc_dir.mkdir(parents=True)
    with open(proc_dir / "benchmark.jsonl", "w") as f:
        f.write(json.dumps(s1.model_dump()) + "\n")
        f.write(json.dumps(s2.model_dump()) + "\n")

    # Mock partition loader
    import evaluation.run_phase6c_baseline as r6c
    monkeypatch.setattr(r6c.PartitionLoader, "load_partition", lambda *args: [s1, s2])

    res1 = runner.run_partition("halubench", PartitionName.DEVELOPMENT, EvaluationPurpose.DEVELOPMENT)
    assert len(res1) == 2

    # Second run should resume and not duplicate
    res2 = runner.run_partition("halubench", PartitionName.DEVELOPMENT, EvaluationPurpose.DEVELOPMENT)
    assert len(res2) == 2


# =========================================================
# TEST 8: Failure accounting
# =========================================================

def test_failure_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.run_phase6c_baseline.RESULTS_DIR", tmp_path)

    runner = Phase6CBenchmarkRunner()
    s_err = BenchmarkSample(id="err_001", prompt="p", response="r", ground_truth_label=0, category="QA", metadata={"dataset": "halubench"})

    def failing_eval(sample):
        raise ValueError("Simulated inference failure")

    monkeypatch.setattr(runner, "evaluate_sample", failing_eval)
    monkeypatch.setattr("evaluation.run_phase6c_baseline.PartitionLoader.load_partition", lambda *args: [s_err])

    runner.run_partition("halubench", PartitionName.DEVELOPMENT, EvaluationPurpose.DEVELOPMENT)

    fail_file = tmp_path / "failures.jsonl"
    assert fail_file.exists()
    content = fail_file.read_text(encoding="utf-8")
    assert "Simulated inference failure" in content


# =========================================================
# TEST 9: H-score bounds [0, 1]
# =========================================================

def test_h_score_bounds():
    runner = Phase6CBenchmarkRunner()
    sample = BenchmarkSample(
        id="b_001",
        prompt="Question",
        response="Response text.",
        ground_truth_label=0,
        category="QA",
        metadata={"dataset": "halubench"},
    )
    res = runner.evaluate_sample(sample)
    assert 0.0 <= res["overall_h_score"] <= 1.0


# =========================================================
# TEST 10: None pillar handling
# =========================================================

def test_none_pillar_handling():
    runner = Phase6CBenchmarkRunner()
    sample = BenchmarkSample(
        id="none_p",
        prompt="Q",
        response="R",
        ground_truth_label=0,
        category="QA",
        metadata={"dataset": "halubench"},
    )
    res = runner.evaluate_sample(sample)

    # In offline benchmark evaluation without logit inputs, P2 confidence gap is None
    assert res["pillar2_available"] is False
    assert res["confidence_gap"] is None
    assert res["overall_h_score"] is not None


# =========================================================
# TEST 11: Continuous H-score used for AUROC
# =========================================================

def test_continuous_h_score_used_for_auroc():
    y_true = [0, 0, 1, 1]
    y_scores = [0.05, 0.15, 0.85, 0.95]
    y_pred = [0, 0, 1, 1]
    metrics = compute_all_metrics(y_true, y_pred, scores=y_scores)
    assert metrics["roc_auc"] == 1.0


# =========================================================
# TEST 12: No threshold or weight fitting
# =========================================================

def test_no_threshold_or_weight_fitting():
    assert settings.ALPHA_FACTUAL_ERROR == 0.45
    assert settings.BETA_CONFIDENCE_GAP == 0.30
    assert settings.GAMMA_CONSISTENCY_FAILURE == 0.25
    assert settings.VERIFIED_THRESHOLD == 0.35
    assert settings.HALLUCINATED_THRESHOLD == 0.65
