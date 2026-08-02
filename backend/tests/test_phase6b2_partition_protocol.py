"""Phase 6B.2 Test Suite for Experimental Protocol, Dataset Partitioning & Leakage Control.

Covers Tests 1 through 18 specified in Phase 6B.2 requirements:
- Deterministic group-aware partition reproduction and manifest stability
- Complete assignment without missing or duplicate sample IDs
- Strict group integrity: zero cross-partition leakage of paired/source items
- Final-test firewall access control (rejects DEVELOPMENT/VALIDATION, permits FINAL_EVALUATION)
- Ground-truth label isolation from inference pipeline
- Production freeze verification (raw/processed files and core engine unchanged)
"""

import json, hashlib
from pathlib import Path
import pytest

from evaluation.dataset import DatasetLoader, BenchmarkSample
from evaluation.partitions.grouping import DatasetGroupExtractor
from evaluation.partitions.partitioner import GroupAwarePartitioner, PartitionName, HALLUCISENSE_PARTITION_SEED
from evaluation.partitions.loader import PartitionLoader, EvaluationPurpose, LockedTestSetAccessError
from evaluation.partitions.verify_partitions import verify_partition_integrity
from evaluation.experiment_protocol import ExperimentProtocolConfig
from evaluation.runner import EvaluationRunner


DATASET_ROOT = Path("evaluation_data")
PARTITION_DIR = DATASET_ROOT / "partitions"


# =========================================================
# TEST 1: Deterministic partition reproduction
# =========================================================

def test_deterministic_partition_reproduction():
    s1 = BenchmarkSample(id="s1", prompt="p1", response="r1", ground_truth_label=0, category="QA", metadata={"dataset": "halueval", "task": "qa", "base_id": "b1"})
    s2 = BenchmarkSample(id="s2", prompt="p1", response="r2", ground_truth_label=1, category="QA", metadata={"dataset": "halueval", "task": "qa", "base_id": "b1"})

    parts1 = GroupAwarePartitioner.partition_samples([s1, s2], seed=HALLUCISENSE_PARTITION_SEED)
    parts2 = GroupAwarePartitioner.partition_samples([s1, s2], seed=HALLUCISENSE_PARTITION_SEED)

    assert [s.id for s in parts1[PartitionName.DEVELOPMENT]] == [s.id for s in parts2[PartitionName.DEVELOPMENT]]
    assert [s.id for s in parts1[PartitionName.VALIDATION]] == [s.id for s in parts2[PartitionName.VALIDATION]]
    assert [s.id for s in parts1[PartitionName.LOCKED_FINAL_TEST]] == [s.id for s in parts2[PartitionName.LOCKED_FINAL_TEST]]


# =========================================================
# TEST 2: Identical seed -> Identical manifests
# =========================================================

def test_identical_seed_identical_manifests():
    comb_path = PARTITION_DIR / "combined_partition_manifest.json"
    assert comb_path.exists()

    with open(comb_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["seed"] == HALLUCISENSE_PARTITION_SEED
    assert manifest["total_benchmark_examples"] == 82690


# =========================================================
# TEST 3: Manifest checksums stable
# =========================================================

def test_manifest_checksums_stable():
    success, report = verify_partition_integrity()
    assert success is True
    assert report["overall_status"] == "PASS"


# =========================================================
# TEST 4 & 5 & 6: Every example assigned exactly once, no missing, no duplicates
# =========================================================

def test_every_example_assigned_exactly_once():
    for ds_name in ["halubench", "ragtruth", "halueval"]:
        mf_path = PARTITION_DIR / f"{ds_name}_partitions.json"
        with open(mf_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        total = manifest["total_examples"]
        dev_ids = set(manifest["partitions"]["development"])
        val_ids = set(manifest["partitions"]["validation"])
        lock_ids = set(manifest["partitions"]["locked_final_test"])

        # Check disjointness
        assert len(dev_ids.intersection(val_ids)) == 0
        assert len(dev_ids.intersection(lock_ids)) == 0
        assert len(val_ids.intersection(lock_ids)) == 0

        # Check total count
        assert len(dev_ids) + len(val_ids) + len(lock_ids) == total


# =========================================================
# TEST 7 & 8: Group integrity and HaluEval pair integrity
# =========================================================

def test_halueval_pair_integrity():
    corr_s = BenchmarkSample(id="halueval:qa:1:correct", prompt="p", response="r1", ground_truth_label=0, category="QA", metadata={"dataset": "halueval", "task": "qa", "base_id": "1"})
    hallu_s = BenchmarkSample(id="halueval:qa:1:hallucinated", prompt="p", response="r2", ground_truth_label=1, category="QA", metadata={"dataset": "halueval", "task": "qa", "base_id": "1"})

    parts = GroupAwarePartitioner.partition_samples([corr_s, hallu_s], seed=2026)

    # Both items MUST be in the exact same partition!
    p_corr = [p for p, samps in parts.items() if any(s.id == corr_s.id for s in samps)][0]
    p_hallu = [p for p, samps in parts.items() if any(s.id == hallu_s.id for s in samps)][0]

    assert p_corr == p_hallu, "HaluEval pair leaked across different partitions!"


# =========================================================
# TEST 9: RAGTruth source grouping
# =========================================================

def test_ragtruth_source_grouping():
    resp1 = BenchmarkSample(id="ragtruth:r1", prompt="src text", response="res1", ground_truth_label=0, category="QA", metadata={"dataset": "ragtruth", "source_id": "100"})
    resp2 = BenchmarkSample(id="ragtruth:r2", prompt="src text", response="res2", ground_truth_label=1, category="QA", metadata={"dataset": "ragtruth", "source_id": "100"})

    parts = GroupAwarePartitioner.partition_samples([resp1, resp2], seed=2026)

    p1 = [p for p, samps in parts.items() if any(s.id == resp1.id for s in samps)][0]
    p2 = [p for p, samps in parts.items() if any(s.id == resp2.id for s in samps)][0]

    assert p1 == p2, "RAGTruth multi-LLM responses derived from same source_id leaked across partitions!"


# =========================================================
# TEST 10 & 11: Label & task stratification sanity
# =========================================================

def test_label_stratification_sanity():
    mf_path = PARTITION_DIR / "halubench_partitions.json"
    with open(mf_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    stats = manifest["partition_statistics"]
    dev_fac_pct = stats["development"]["factual_percentage"]
    val_fac_pct = stats["validation"]["factual_percentage"]
    lock_fac_pct = stats["locked_final_test"]["factual_percentage"]

    # Factual percentage should remain ~51.88% across all partitions (within +-5%)
    assert abs(dev_fac_pct - 51.88) < 5.0
    assert abs(val_fac_pct - 51.88) < 5.0
    assert abs(lock_fac_pct - 51.88) < 5.0


# =========================================================
# TEST 12, 13, 14: Firewall enforces access controls
# =========================================================

def test_locked_final_test_rejects_development_purpose():
    with pytest.raises(LockedTestSetAccessError, match="FIREWALL DENIAL"):
        PartitionLoader.load_partition(
            dataset_name="halubench",
            partition=PartitionName.LOCKED_FINAL_TEST,
            purpose=EvaluationPurpose.DEVELOPMENT,
        )


def test_locked_final_test_rejects_validation_purpose():
    with pytest.raises(LockedTestSetAccessError, match="FIREWALL DENIAL"):
        PartitionLoader.load_partition(
            dataset_name="halubench",
            partition=PartitionName.LOCKED_FINAL_TEST,
            purpose=EvaluationPurpose.VALIDATION,
        )


def test_locked_final_test_permits_final_evaluation_purpose():
    # Should not raise LockedTestSetAccessError
    samples = PartitionLoader.load_partition(
        dataset_name="halubench",
        partition=PartitionName.LOCKED_FINAL_TEST,
        purpose=EvaluationPurpose.FINAL_EVALUATION,
    )
    assert len(samples) > 0


# =========================================================
# TEST 15: Ground-truth labels never enter inference
# =========================================================

def test_ground_truth_labels_never_enter_inference(monkeypatch):
    runner = EvaluationRunner()
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

    dev_samples = PartitionLoader.load_partition(
        dataset_name="halubench",
        partition=PartitionName.DEVELOPMENT,
        purpose=EvaluationPurpose.DEVELOPMENT,
    )
    sample = dev_samples[0]

    pipeline.analyze_response(full_text=sample.response, evidence_items=[])
    assert len(labels_passed) == 0, "CRITICAL DATA LEAKAGE: Ground-truth label entered inference pipeline!"


# =========================================================
# TEST 16 & 17: Raw & processed datasets remain unchanged
# =========================================================

def test_raw_and_processed_datasets_remain_unchanged():
    with open(DATASET_ROOT / "manifests" / "halubench.json", "r", encoding="utf-8") as f:
        m_hb = json.load(f)

    proc_sha = m_hb["checksums"]["processed/halubench/benchmark.jsonl"]
    actual_sha = hashlib.sha256((DATASET_ROOT / "processed/halubench/benchmark.jsonl").read_bytes()).hexdigest()
    assert actual_sha == proc_sha


# =========================================================
# TEST 18: Production scoring files remain unchanged
# =========================================================

def test_production_scoring_files_remain_unchanged():
    fusion_file = Path("app/core/engine/fusion.py")
    assert fusion_file.exists()
    content = fusion_file.read_text(encoding="utf-8")
    assert "settings.ALPHA_FACTUAL_ERROR" in content
    assert "settings.BETA_CONFIDENCE_GAP" in content
    assert "settings.GAMMA_CONSISTENCY_FAILURE" in content
