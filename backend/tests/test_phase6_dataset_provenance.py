"""Phase 6A Automated Dataset Provenance and Reproducibility Test Suite.

Mandated Test Coverage:
1. Canonical benchmark dataset exists at specified path.
2. Contains exactly N=750 records.
3. Every record has a unique ID (0 duplicates).
4. Covers exactly 15 canonical research domains.
5. Every domain contains exactly 50 samples.
6. Contains exactly 375 factual (0) and 375 hallucinated (1) samples (1:1 balance).
7. Every record conforms to the required BenchmarkExample schema.
8. No required field is null or empty.
9. SHA-256 matches the frozen reference hash (dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5).
10. Provenance manifest fields match the dataset characteristics exactly.
11. Reproduction script reproduces the benchmark with 100% cryptographic parity.
12. Dataset package and zip archive are present and verified.
"""

import json
import hashlib
from pathlib import Path
import pytest

from evaluation.benchmark_dataset.dataset_schema import DOMAINS, BenchmarkExample
from evaluation.reproduce_phase6_dataset import reproduce_benchmark_dataset, FROZEN_BENCHMARK_SHA256

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATASET_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
MANIFEST_PATH = BACKEND_DIR / "reports" / "phase6" / "dataset_provenance_manifest.json"
HASHES_PATH = BACKEND_DIR / "reports" / "phase6" / "dataset_hashes.json"
ZIP_PATH = PROJECT_ROOT / "HalluciSense_Canonical_Benchmark_Dataset.zip"


def test_1_dataset_file_exists():
    """TEST 1: Canonical benchmark dataset exists."""
    assert DATASET_PATH.exists(), f"Benchmark dataset missing at {DATASET_PATH}"


def test_2_exactly_750_records():
    """TEST 2: Contains exactly N=750 records."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 750, f"Expected 750 records, found {len(records)}"


def test_3_unique_ids_zero_duplicates():
    """TEST 3: Every record has a unique ID (0 duplicates)."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {len(ids) - len(set(ids))}"


def test_4_covers_15_canonical_domains():
    """TEST 4: Covers exactly 15 canonical research domains."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    domains = set(r["domain"] for r in records)
    assert len(domains) == 15, f"Expected 15 domains, found {len(domains)}"
    assert domains == set(DOMAINS), f"Domain mismatch: {domains - set(DOMAINS)}"


def test_5_exactly_50_samples_per_domain():
    """TEST 5: Every domain contains exactly 50 samples."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    counts = {}
    for r in records:
        dom = r["domain"]
        counts[dom] = counts.get(dom, 0) + 1
    for dom, count in counts.items():
        assert count == 50, f"Domain {dom} has {count} samples, expected 50"


def test_6_perfect_class_balance():
    """TEST 6: Exactly 375 factual (0) and 375 hallucinated (1) samples."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    factual = sum(1 for r in records if r["ground_truth"] == 0)
    hallucinated = sum(1 for r in records if r["ground_truth"] == 1)
    assert factual == 375, f"Expected 375 factual, got {factual}"
    assert hallucinated == 375, f"Expected 375 hallucinated, got {hallucinated}"


def test_7_conforms_to_schema():
    """TEST 7: Every record conforms to the required BenchmarkExample schema."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    required_keys = {"id", "question", "response", "ground_truth", "domain", "difficulty", "label", "claims"}
    for r in records:
        missing = required_keys - set(r.keys())
        assert not missing, f"Record {r.get('id')} missing keys: {missing}"


def test_8_no_null_required_fields():
    """TEST 8: No required field is null or empty."""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for r in records:
        assert r["id"], "Empty ID"
        assert r["question"], "Empty question"
        assert r["response"], "Empty response"
        assert r["ground_truth"] in (0, 1), f"Invalid ground_truth: {r['ground_truth']}"
        assert r["domain"] in DOMAINS, f"Invalid domain: {r['domain']}"
        assert r["difficulty"] in ("easy", "medium", "hard"), f"Invalid difficulty: {r['difficulty']}"
        assert len(r["claims"]) >= 1, "Empty claims array"


def test_9_cryptographic_sha256_match():
    """TEST 9: SHA-256 matches the frozen reference hash."""
    content = DATASET_PATH.read_bytes()
    computed_sha = hashlib.sha256(content).hexdigest()
    assert computed_sha == FROZEN_BENCHMARK_SHA256, (
        f"SHA-256 mismatch! Computed: {computed_sha}, Expected: {FROZEN_BENCHMARK_SHA256}"
    )


def test_10_provenance_manifest_consistency():
    """TEST 10: Provenance manifest matches dataset characteristics."""
    assert MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["total_samples"] == 750
    assert manifest["domains"] == 15
    assert manifest["samples_per_domain"] == 50
    assert manifest["factual_samples"] == 375
    assert manifest["hallucinated_samples"] == 375
    assert manifest["canonical_benchmark_sha256"] == FROZEN_BENCHMARK_SHA256


def test_11_reproduction_script_passes():
    """TEST 11: Reproduction script achieves 100% cryptographic parity."""
    res = reproduce_benchmark_dataset()
    assert res["records_pass"] is True
    assert res["domains_pass"] is True
    assert res["class_balance_pass"] is True
    assert res["schema_pass"] is True
    assert res["duplicates_pass"] is True
    assert res["sha_pass"] is True


def test_12_dataset_package_and_zip_present():
    """TEST 12: Dataset package and ZIP archive exist."""
    assert ZIP_PATH.exists(), f"ZIP archive missing at {ZIP_PATH}"
    assert ZIP_PATH.stat().st_size > 10000, "ZIP archive suspiciously small"
