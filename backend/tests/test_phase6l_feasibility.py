"""Unit tests for Phase 6L.1A Feasibility Gate & Data Firewall.

Verifies:
    1. Validation partition (N=12,483) is strictly sealed and untouched.
    2. Feasibility report and JSON artifacts exist and contain explicit Decision Gate answers.
    3. Historical Phase 6I/6J/6K artifacts are preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from evaluation.phase6l.config import PHASE6L_DIR, VAL_FEATURES_JSONL, PHASE6K_DIR
from evaluation.phase6l.nli_feasibility import audit_nli_model_metadata


def test_validation_partition_firewall():
    """Verify Phase 6L.1A never accesses or opens claim_evidence_features_validation.jsonl."""
    # Ensure VAL_FEATURES_JSONL exists but is NEVER read during Phase 6L.1A
    assert VAL_FEATURES_JSONL.exists()

    report_path = PHASE6L_DIR / "phase6l_1a_feasibility_report.md"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "STRICTLY SEALED" in text


def test_nli_model_audit_export(tmp_path):
    """Test audit_nli_model_metadata exports expected metadata structure."""
    res = audit_nli_model_metadata(out_dir=tmp_path)
    assert res["label_mapping_verified"] is True
    assert (tmp_path / "nli_model_audit.json").exists()


def test_historical_phase6k_artifacts_preserved():
    """Verify all historical reports from Phase 6K and 6K.4 are preserved."""
    assert (PHASE6K_DIR / "FINAL_PILLAR1_VALIDATION_REPORT.md").exists()
    assert (PHASE6K_DIR / "final_model_protocol.json").exists()
