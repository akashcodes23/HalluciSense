"""Phase 14 Reproducibility, Benchmark Hash & Manifest Tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
MANIFEST_PATH = BACKEND_DIR / "evaluation" / "phase14" / "phase14_experiment_manifest.json"


class TestPhase14Reproducibility:
    def test_canonical_benchmark_sha256_unaltered(self):
        """Verifies canonical benchmark dataset hash has never changed."""
        hasher = hashlib.sha256()
        with open(BENCHMARK_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"

    def test_phase14_manifest_fields_and_traceability(self):
        """Verifies phase14 manifest has complete reproducibility metadata."""
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["phase"] == 14
        assert manifest["canonical_benchmark_sha256"] == "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"
        assert len(manifest["evaluated_external_benchmarks"]) == 5
        assert manifest["publication_readiness"] == "PUBLICATION_READY"
