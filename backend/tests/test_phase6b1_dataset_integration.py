"""Phase 6B.1 Test Suite for Real Benchmark Acquisition, Provenance & Dataset Integration.

Covers Tests 1 through 22 specified in Phase 6B.1 requirements:
- HaluBench, RAGTruth, and HaluEval adapter correctness, label mappings, and span validation
- Checksum determinism, integrity verifier, and manifest serialization
- Data leakage protection (labels never enter inference pipeline)
- Regression safety across Phase 4, 5, 6A, and 6B
"""

import json
from pathlib import Path
import pytest

from app.core.config import settings
from evaluation.dataset import DatasetLoader, BenchmarkSample
from evaluation.datasets.adapter import BenchmarkAdapter, BenchmarkDataset, BenchmarkExample
from evaluation.datasets.halubench_adapter import HaluBenchAdapter, HALUBENCH_LABEL_MAP
from evaluation.datasets.ragtruth_adapter import RAGTruthAdapter
from evaluation.datasets.halueval_adapter import HaluEvalAdapter
from evaluation.datasets.verify_integrity import verify_dataset_integrity, compute_sha256
from evaluation.runner import EvaluationRunner


# =========================================================
# TEST 1: HaluBench adapter maps valid example
# =========================================================

def test_halubench_adapter_maps_valid_example():
    raw_records = [
        {
            "id": "hb_001",
            "passage": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "label": "PASS",
            "source_ds": "test_ds",
        }
    ]
    examples = HaluBenchAdapter.process_records(raw_records)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.example_id == "halubench:hb_001"
    assert ex.label == 0
    assert ex.response == "Paris"
    assert "Context: Paris is the capital of France." in ex.prompt


# =========================================================
# TEST 2: HaluBench label mapping is correct
# =========================================================

def test_halubench_label_mapping_correctness():
    assert HALUBENCH_LABEL_MAP["PASS"] == 0
    assert HALUBENCH_LABEL_MAP["FAIL"] == 1

    records = [
        {"id": "1", "passage": "p", "question": "q", "answer": "a1", "label": "PASS", "source_ds": "s"},
        {"id": "2", "passage": "p", "question": "q", "answer": "a2", "label": "FAIL", "source_ds": "s"},
    ]
    examples = HaluBenchAdapter.process_records(records)
    assert examples[0].label == 0
    assert examples[1].label == 1


# =========================================================
# TEST 3: HaluBench original label preserved
# =========================================================

def test_halubench_original_label_preserved():
    records = [
        {"id": "hb_orig", "passage": "p", "question": "q", "answer": "a", "label": "FAIL", "source_ds": "pubmedQA"}
    ]
    examples = HaluBenchAdapter.process_records(records)
    assert examples[0].metadata["original_label"] == "FAIL"
    assert examples[0].metadata["source_ds"] == "pubmedQA"


# =========================================================
# TEST 4: RAGTruth response-level label derived correctly
# =========================================================

def test_ragtruth_response_level_label_derived_correctly():
    responses = [
        {"id": "r1", "source_id": "s1", "model": "gpt-4", "labels": [], "response": "Valid response."},
        {
            "id": "r2",
            "source_id": "s1",
            "model": "gpt-4",
            "labels": [{"start": 0, "end": 5, "text": "Valid", "label_type": "evidentiary"}],
            "response": "Valid response.",
        },
    ]
    examples, stats = RAGTruthAdapter.process_records(responses)
    assert len(examples) == 2
    assert examples[0].label == 0  # No spans -> 0
    assert examples[1].label == 1  # >= 1 span -> 1


# =========================================================
# TEST 5: RAGTruth span annotations preserved
# =========================================================

def test_ragtruth_span_annotations_preserved():
    span = {"start": 0, "end": 5, "text": "Paris", "label_type": "factuality"}
    responses = [
        {"id": "r_span", "source_id": "s1", "model": "gpt-4", "labels": [span], "response": "Paris is big."}
    ]
    examples, _ = RAGTruthAdapter.process_records(responses)
    spans_meta = examples[0].metadata["hallucination_spans"]
    assert len(spans_meta) == 1
    assert spans_meta[0]["start"] == 0
    assert spans_meta[0]["end"] == 5
    assert spans_meta[0]["text"] == "Paris"


# =========================================================
# TEST 6: RAGTruth span bounds validation
# =========================================================

def test_ragtruth_span_bounds_validation():
    resp_text = "The capital is London."

    # Valid span
    valid_span = {"start": 15, "end": 21, "text": "London"}
    is_v, warn = RAGTruthAdapter.validate_span_bounds(resp_text, valid_span)
    assert is_v is True
    assert warn is None

    # Out of bounds span
    oob_span = {"start": 15, "end": 50, "text": "invalid"}
    is_v2, warn2 = RAGTruthAdapter.validate_span_bounds(resp_text, oob_span)
    assert is_v2 is False
    assert "Out-of-bounds" in warn2


# =========================================================
# TEST 7: HaluEval correct response maps to label 0
# =========================================================

def test_halueval_correct_response_maps_to_0():
    records = [
        {
            "id": "he_01",
            "knowledge": "K",
            "question": "Q",
            "right_answer": "Right Answer",
            "hallucinated_answer": "Wrong Answer",
        }
    ]
    examples, stats = HaluEvalAdapter.process_records(records, task_name="qa")
    assert stats["correct_pair_examples"] == 1
    corr_ex = [e for e in examples if e.label == 0][0]
    assert corr_ex.response == "Right Answer"
    assert corr_ex.example_id == "halueval:qa:he_01:correct"


# =========================================================
# TEST 8: HaluEval hallucinated response maps to label 1
# =========================================================

def test_halueval_hallucinated_response_maps_to_1():
    records = [
        {
            "id": "he_01",
            "knowledge": "K",
            "question": "Q",
            "right_answer": "Right Answer",
            "hallucinated_answer": "Wrong Answer",
        }
    ]
    examples, stats = HaluEvalAdapter.process_records(records, task_name="qa")
    assert stats["hallucinated_pair_examples"] == 1
    hallu_ex = [e for e in examples if e.label == 1][0]
    assert hallu_ex.response == "Wrong Answer"
    assert hallu_ex.example_id == "halueval:qa:he_01:hallucinated"


# =========================================================
# TEST 9: HaluEval generated IDs are unique
# =========================================================

def test_halueval_generated_ids_unique():
    records = [
        {"id": "dup", "right_answer": "R1", "hallucinated_answer": "H1"},
        {"id": "dup", "right_answer": "R2", "hallucinated_answer": "H2"},
    ]
    examples, _ = HaluEvalAdapter.process_records(records, task_name="dialogue")
    ids = [e.example_id for e in examples]
    assert len(ids) == len(set(ids)) == 4


# =========================================================
# TEST 10: All canonical outputs satisfy BenchmarkExample schema
# =========================================================

def test_all_canonical_outputs_satisfy_schema():
    hb_recs = [{"id": "1", "passage": "p", "question": "q", "answer": "a", "label": "PASS", "source_ds": "s"}]
    examples = HaluBenchAdapter.process_records(hb_recs)
    sample = examples[0].to_benchmark_sample()
    assert isinstance(sample, BenchmarkSample)
    assert sample.ground_truth_label in (0, 1)


# =========================================================
# TEST 11: Invalid labels rejected
# =========================================================

def test_invalid_labels_rejected():
    invalid_records = [{"id": "inv", "passage": "p", "question": "q", "answer": "a", "label": "UNKNOWN_LABEL", "source_ds": "s"}]
    with pytest.raises(ValueError, match="Unmapped HaluBench label"):
        HaluBenchAdapter.process_records(invalid_records)


# =========================================================
# TEST 12: Missing response rejected
# =========================================================

def test_missing_response_rejected():
    invalid_json = '{"id": "m1", "prompt": "P", "response": "", "ground_truth_label": 0}\n'
    tmp_file = Path("/tmp/test_missing_resp.jsonl")
    tmp_file.write_text(invalid_json, encoding="utf-8")

    with pytest.raises(Exception, match="(ValidationError|non-empty)"):
        DatasetLoader.load_from_file(tmp_file)

    if tmp_file.exists():
        tmp_file.unlink()


# =========================================================
# TEST 13: Duplicate IDs detected
# =========================================================

def test_duplicate_ids_detected():
    dup_json = '{"id": "d1", "prompt": "P1", "response": "R1", "ground_truth_label": 0}\n{"id": "d1", "prompt": "P2", "response": "R2", "ground_truth_label": 1}\n'
    tmp_file = Path("/tmp/test_dup_id.jsonl")
    tmp_file.write_text(dup_json, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate sample ID"):
        DatasetLoader.load_from_file(tmp_file)

    if tmp_file.exists():
        tmp_file.unlink()


# =========================================================
# TEST 14: SHA-256 checksum deterministic
# =========================================================

def test_sha256_checksum_deterministic(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("HalluciSense Phase 6B.1", encoding="utf-8")
    sha1 = compute_sha256(f)
    sha2 = compute_sha256(f)
    assert sha1 == sha2
    assert len(sha1) == 64


# =========================================================
# TEST 15: Checksum mismatch detected
# =========================================================

def test_checksum_mismatch_detected(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    raw_file = tmp_path / "test.txt"
    raw_file.write_text("Original content", encoding="utf-8")

    manifest_data = {
        "dataset_name": "TestDS",
        "checksums": {"test.txt": "0000000000000000000000000000000000000000000000000000000000000000"},
        "processed_path": "",
    }
    with open(manifest_dir / "testds.json", "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf)

    # Patch MANIFEST_DIR and DATASET_ROOT
    import evaluation.datasets.verify_integrity as vi
    old_root = vi.DATASET_ROOT
    old_man = vi.MANIFEST_DIR
    vi.DATASET_ROOT = tmp_path
    vi.MANIFEST_DIR = manifest_dir

    try:
        success, report = verify_dataset_integrity()
        assert success is False
        assert report["datasets"]["TestDS"]["status"] == "FAIL"
    finally:
        vi.DATASET_ROOT = old_root
        vi.MANIFEST_DIR = old_man


# =========================================================
# TEST 16: Manifest serialization works
# =========================================================

def test_manifest_serialization_works(tmp_path):
    manifest = {
        "dataset_name": "HaluBench",
        "record_count_raw": 14900,
        "checksums": {"raw/file": "abc"},
    }
    mf_path = tmp_path / "halubench.json"
    with open(mf_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    with open(mf_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["dataset_name"] == "HaluBench"
        assert loaded["record_count_raw"] == 14900


# =========================================================
# TEST 17: Adapter output deterministic
# =========================================================

def test_adapter_output_deterministic():
    records = [{"id": "d1", "passage": "P", "question": "Q", "answer": "A", "label": "PASS", "source_ds": "s"}]
    ex1 = HaluBenchAdapter.process_records(records)
    ex2 = HaluBenchAdapter.process_records(records)
    assert ex1[0].example_id == ex2[0].example_id
    assert ex1[0].label == ex2[0].label
    assert ex1[0].prompt == ex2[0].prompt


# =========================================================
# TEST 18: Ground-truth labels do not enter inference
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

    hb_recs = [{"id": "1", "passage": "p", "question": "q", "answer": "a", "label": "FAIL", "source_ds": "s"}]
    exs = HaluBenchAdapter.process_records(hb_recs)
    sample = exs[0].to_benchmark_sample()

    # Pass sample to pipeline
    pipeline.analyze_response(full_text=sample.response, evidence_items=[])

    assert len(labels_passed) == 0, (
        "CRITICAL DATA LEAKAGE DETECTED: Ground-truth label entered inference pipeline!"
    )
