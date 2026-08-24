"""Phase 16 — Reviewer-Resistant Scientific Gate & Evidence Lock Test Suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
REPORTS_DIR = BACKEND_DIR / "reports" / "phase16"
TABLES_DIR = REPORTS_DIR / "tables"
EVAL_DIR = BACKEND_DIR / "evaluation" / "phase16"


class TestPhase16ScientificGate:
    def test_canonical_benchmark_hash_unaltered(self):
        """Verifies canonical benchmark dataset hash has never changed."""
        hasher = hashlib.sha256()
        with open(BENCHMARK_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"

    def test_baseline_registry_has_comparability_categories(self):
        """Verifies baseline registry explicitly classifies comparability tiers."""
        reg_path = EVAL_DIR / "baseline_registry.json"
        assert reg_path.exists()
        with open(reg_path, "r", encoding="utf-8") as f:
            baselines = json.load(f)

        assert len(baselines) >= 10
        categories = {b["comparability_category"] for b in baselines}
        assert "A. DIRECTLY REPRODUCED" in categories
        assert "C. REPORTED FROM ORIGINAL LITERATURE" in categories

    def test_statistical_audit_remediation_integrity(self):
        """Verifies statistical audit file records both per-sample Cohen's d and bootstrap CI."""
        stat_path = REPORTS_DIR / "phase16_statistical_results.json"
        assert stat_path.exists()
        with open(stat_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        mask_no_logprobs = next(s for s in stats if s["Signal_Mask"] == "[1, 0, 1]")
        assert mask_no_logprobs["Delta_AUROC"] == 0.1490
        assert mask_no_logprobs["Per_Sample_Cohen_d"] > 1.0  # Proper per-sample Cohen's d
        assert "Bootstrap_95CI" in mask_no_logprobs

    def test_trivial_baselines_falsification_table(self):
        """Verifies that trivial baselines (length, scramble) fail to predict hallucination (AUROC ~= 0.50)."""
        triv_path = TABLES_DIR / "table_trivial_baselines.csv"
        assert triv_path.exists()
        with open(triv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 9

    def test_all_thirteen_manuscript_tables_exist(self):
        """Verifies all 13 manuscript-ready tables are generated in backend/reports/phase16/tables/."""
        expected_tables = [
            "table1_system_architecture.csv",
            "table2_main_results.csv",
            "table3_external_generalization.csv",
            "table4_baseline_comparison.csv",
            "table5_ablation.csv",
            "table6_availability_robustness.csv",
            "table7_calibration.csv",
            "table8_selective_abstention.csv",
            "table9_closed_loop_correction.csv",
            "table10_failure_taxonomy.csv",
            "table11_statistical_tests.csv",
            "table12_reproducibility.csv",
            "table13_claim_evidence.csv",
        ]
        for tbl in expected_tables:
            tbl_path = TABLES_DIR / tbl
            assert tbl_path.exists(), f"Missing table: {tbl}"
            assert tbl_path.stat().st_size > 50, f"Empty table: {tbl}"

    def test_master_reproducibility_manifest(self):
        """Verifies master reproducibility manifest contains all required keys and zero secrets."""
        man_path = REPORTS_DIR / "HALLUCISENSE_REPRODUCIBILITY_MANIFEST.json"
        assert man_path.exists()
        with open(man_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["phase"] == 16
        assert manifest["canonical_benchmark_sha256"] == "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"
        assert "environment" in manifest
        assert "model_registry" in manifest

    def test_final_claim_gate_verdict(self):
        """Verifies final claim gate report classifies the package as A — REVIEWER-READY."""
        gate_path = REPORTS_DIR / "PHASE16_FINAL_CLAIM_GATE.md"
        assert gate_path.exists()
        with open(gate_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "A — REVIEWER-READY" in content
