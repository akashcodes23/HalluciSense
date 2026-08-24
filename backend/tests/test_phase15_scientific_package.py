"""Phase 15 — Scientific Package, Reproducibility & Gate Invariant Tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from app.core.engine.model_registry import ModelRegistry
from app.core.engine.fusion import FusionEngine
from app.core.engine.calibration import ProbabilityCalibrator

BACKEND_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
REPORTS_DIR = BACKEND_DIR / "reports" / "phase15"
TABLES_DIR = REPORTS_DIR / "tables"


class TestPhase15ScientificPackage:
    def test_canonical_benchmark_hash_strictly_invariant(self):
        """Verifies canonical benchmark dataset hash has never changed."""
        hasher = hashlib.sha256()
        with open(BENCHMARK_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"

    def test_paper_grade_tables_exist(self):
        """Verifies all 11 paper-grade tables exist and are non-empty."""
        expected_tables = [
            "table1_architecture_components.csv",
            "table2_main_benchmark_performance.csv",
            "table3_external_benchmark_performance.csv",
            "table4_baseline_comparison.csv",
            "table5_ablation_study.csv",
            "table6_availability_mask_robustness.csv",
            "table7_calibration.csv",
            "table8_risk_coverage.csv",
            "table9_closed_loop_correction.csv",
            "table10_failure_taxonomy.csv",
            "table11_reproducibility_configuration.csv",
        ]
        for tbl in expected_tables:
            tbl_path = TABLES_DIR / tbl
            assert tbl_path.exists(), f"Missing table: {tbl}"
            assert tbl_path.stat().st_size > 50, f"Empty table: {tbl}"

    def test_reproducibility_manifests_integrity(self):
        """Verifies all 4 reproducibility manifests exist and contain valid JSON metadata."""
        manifests = [
            "REPRODUCIBILITY_MANIFEST.json",
            "ENVIRONMENT_MANIFEST.json",
            "MODEL_MANIFEST.json",
            "DATASET_MANIFEST.json",
        ]
        for mf in manifests:
            mf_path = REPORTS_DIR / mf
            assert mf_path.exists(), f"Missing manifest: {mf}"
            with open(mf_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, dict)
                assert len(data) > 2

    def test_adaptive_fusion_statistical_superiority(self):
        """Verifies that adaptive fusion outperforms fixed fusion under missing signal mask [1,0,1]."""
        fusion = FusionEngine(alpha=0.40, beta=0.30, gamma=0.30)
        # Fixed Score (CG treated as 0)
        h_fixed = 0.40 * 0.85 + 0.30 * 0.0 + 0.30 * 0.65  # = 0.5350
        # Adaptive Score
        h_adapt, eff_w, mask = fusion.compute_adaptive_h_score(fe=0.85, cg=None, cf=0.65)
        assert mask == [1, 0, 1]
        assert eff_w["beta_confidence_gap"] == 0.0
        assert h_adapt > h_fixed
        assert abs(h_adapt - 0.7643) < 0.01

    def test_final_scientific_gate_verdict(self):
        """Verifies final scientific gate report contains SUBMISSION READY classification."""
        gate_path = REPORTS_DIR / "PHASE15_FINAL_SCIENTIFIC_GATE.md"
        assert gate_path.exists()
        with open(gate_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "A — SUBMISSION READY" in content
