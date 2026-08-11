import pytest, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports" / "phase6j"

def test_phase6j_preimplementation_audit():
    audit_path = REPORTS_DIR / "phase6j_preimplementation_audit.md"
    assert audit_path.exists()

def test_phase6j_architecture_freeze():
    freeze_path = REPORTS_DIR / "phase6j_architecture_freeze.md"
    assert freeze_path.exists()

def test_phase6j_production_invariants():
    inv_path = REPORTS_DIR / "phase6j_production_invariants.json"
    assert inv_path.exists()
    invariants = json.loads(inv_path.read_text())
    assert all(inv["status"] == "PASS" for inv in invariants)

def test_phase6j_dataset_integrity():
    ds_path = REPORTS_DIR / "phase6j_dataset_integrity.json"
    assert ds_path.exists()
    data = json.loads(ds_path.read_text())
    assert data["cross_phase_overlap"]["status"] == "PASS"

def test_phase6j_phase6i_reproduction():
    repro_path = REPORTS_DIR / "phase6i_reproduction_results.json"
    assert repro_path.exists()
    res = json.loads(repro_path.read_text())
    assert res["accuracy"] == 0.888

def test_phase6j_reproducibility_script():
    script_path = ROOT / "scripts" / "reproduce_final_validation.sh"
    assert script_path.exists()

def test_phase6j_final_summary():
    sum_path = REPORTS_DIR / "phase6j_final_summary.json"
    assert sum_path.exists()
    s = json.loads(sum_path.read_text())
    assert s["decision_gate"] == "A — PASS"
    assert s["regression_tests"]["passed"] == 81
