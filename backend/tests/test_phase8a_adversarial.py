"""Test Suite for Phase 8A Scientific Adversarial Benchmark.

Validates:
- 8A dataset structure, balance (175 claims: 5 domains × 7 categories × 5 samples).
- Ground truth labeling rules (GT=0 for TRUE_CONTROL, GT=1 for corruptions).
- Provenance integrity (valid URLs, authoritative sources, documented reasons).
- Failure classification logic and category mapping.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import pytest

from evaluation.phase8a.build_dataset_8a import DOMAINS, CATEGORIES, CLAIMS, build_dataset
from evaluation.phase8a.run_phase8a_evaluation import classify_failure, CATEGORY_EXPECTED_FAILURE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PHASE8A_DIR = PROJECT_ROOT / "backend" / "reports" / "phase8" / "8A"


class TestPhase8ADatasetStructure:
    def test_total_claims_count(self):
        records = build_dataset()
        assert len(records) == 175, f"Expected 175 claims, got {len(records)}"

    def test_domain_distribution(self):
        records = build_dataset()
        domain_counts = {}
        for r in records:
            domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1
        for dom in DOMAINS:
            assert domain_counts[dom] == 35, f"Domain {dom} has {domain_counts.get(dom)} claims, expected 35"

    def test_category_distribution(self):
        records = build_dataset()
        cat_counts = {}
        for r in records:
            cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
        for cat in CATEGORIES:
            assert cat_counts[cat] == 25, f"Category {cat} has {cat_counts.get(cat)} claims, expected 25"

    def test_exact_5x7x5_cell_balance(self):
        records = build_dataset()
        cell_counts = {}
        for r in records:
            key = (r["domain"], r["category"])
            cell_counts[key] = cell_counts.get(key, 0) + 1
        for dom in DOMAINS:
            for cat in CATEGORIES:
                assert cell_counts.get((dom, cat)) == 5, f"Cell ({dom}, {cat}) has {cell_counts.get((dom, cat))}, expected 5"

    def test_true_control_ground_truth(self):
        records = build_dataset()
        controls = [r for r in records if r["category"] == "TRUE_CONTROL"]
        assert len(controls) == 25
        for r in controls:
            assert r["ground_truth"] == 0
            assert r["ground_truth_label"] == "factual"

    def test_adversarial_categories_ground_truth(self):
        records = build_dataset()
        adversarial_cats = [c for c in CATEGORIES if c != "TRUE_CONTROL"]
        for cat in adversarial_cats:
            cat_records = [r for r in records if r["category"] == cat]
            assert len(cat_records) == 25
            for r in cat_records:
                assert r["ground_truth"] == 1, f"Claim {r['id']} in {cat} has GT={r['ground_truth']}, expected 1"
                assert r["ground_truth_label"] == "hallucinated"

    def test_overall_label_distribution(self):
        records = build_dataset()
        factual = [r for r in records if r["ground_truth"] == 0]
        hallucinated = [r for r in records if r["ground_truth"] == 1]
        assert len(factual) == 25
        assert len(hallucinated) == 150

    def test_required_schema_fields(self):
        records = build_dataset()
        required_keys = {
            "id", "domain", "category", "claim", "ground_truth",
            "ground_truth_label", "ground_truth_source", "source_url",
            "source_type", "difficulty", "provenance", "annotation_method",
            "hallucisense_used_for_gt",
        }
        for r in records:
            assert required_keys.issubset(r.keys()), f"Record {r.get('id')} missing keys: {required_keys - set(r.keys())}"

    def test_unique_record_ids(self):
        records = build_dataset()
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids)), "Duplicate record IDs found"

    def test_valid_source_urls(self):
        records = build_dataset()
        for r in records:
            url = r["source_url"]
            assert url.startswith("http://") or url.startswith("https://"), f"Invalid URL in {r['id']}: {url}"

    def test_valid_source_types(self):
        records = build_dataset()
        valid_types = {"encyclopedia", "standards_body", "journal", "academic", "textbook"}
        for r in records:
            assert r["source_type"] in valid_types, f"Unknown source type {r['source_type']} in {r['id']}"

    def test_no_hallucisense_in_gt(self):
        records = build_dataset()
        for r in records:
            assert r["hallucisense_used_for_gt"] is False


class TestPhase8AFailureClassification:
    def test_correct_prediction_no_failure(self):
        rec = {"category": "NUMERICAL_PRECISION", "ground_truth": 1}
        pri, sec = classify_failure(rec, p1_score=0.9, predicted=1, gt=1)
        assert pri == "CORRECT"
        assert sec is None

    def test_fn_numerical_precision(self):
        rec = {"category": "NUMERICAL_PRECISION", "ground_truth": 1}
        pri, sec = classify_failure(rec, p1_score=0.1, predicted=0, gt=1)
        assert pri == "NUMERICAL_REASONING_FAILURE"
        assert sec == "NLI_FAILURE"

    def test_fn_unit_scale(self):
        rec = {"category": "UNIT_SCALE", "ground_truth": 1}
        pri, sec = classify_failure(rec, p1_score=0.1, predicted=0, gt=1)
        assert pri == "UNIT_REASONING_FAILURE"
        assert sec == "NLI_FAILURE"

    def test_fn_negation(self):
        rec = {"category": "NEGATION", "ground_truth": 1}
        pri, sec = classify_failure(rec, p1_score=0.1, predicted=0, gt=1)
        assert pri == "NEGATION_FAILURE"
        assert sec == "NLI_FAILURE"

    def test_fn_causal_inversion(self):
        rec = {"category": "CAUSAL_INVERSION", "ground_truth": 1}
        pri, sec = classify_failure(rec, p1_score=0.1, predicted=0, gt=1)
        assert pri == "CAUSAL_DIRECTION_FAILURE"
        assert sec == "NLI_FAILURE"

    def test_fn_outdated_claim(self):
        rec = {"category": "OUTDATED_SCIENTIFIC_CLAIM", "ground_truth": 1}
        pri, sec = classify_failure(rec, p1_score=0.1, predicted=0, gt=1)
        assert pri == "RETRIEVAL_FAILURE"
        assert sec == "TEMPORAL_REASONING_FAILURE"

    def test_fn_true_core_false_elaboration(self):
        rec = {"category": "TRUE_CORE_FALSE_ELABORATION", "ground_truth": 1}
        pri, sec = classify_failure(rec, p1_score=0.1, predicted=0, gt=1)
        assert pri == "PARTIAL_CLAIM_FAILURE"
        assert sec == "RETRIEVAL_FAILURE"

    def test_fp_true_control(self):
        rec = {"category": "TRUE_CONTROL", "ground_truth": 0}
        pri, sec = classify_failure(rec, p1_score=0.9, predicted=1, gt=0)
        assert pri == "RETRIEVAL_FAILURE"
        assert sec == "NLI_FAILURE"


class TestDatasetFileIntegrity:
    def test_dataset_file_exists(self):
        path = PHASE8A_DIR / "dataset_8a.jsonl"
        assert path.exists(), f"Dataset file {path} does not exist"

    def test_dataset_sha256_matches_manifest(self):
        path = PHASE8A_DIR / "dataset_8a.jsonl"
        manifest_path = PHASE8A_DIR / "dataset_manifest.json"
        assert path.exists()
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest["sha256"] == actual_hash
