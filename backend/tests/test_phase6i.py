import pytest, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external"
REPORTS_DIR = ROOT / "reports" / "phase6i"

def test_phase6i_freeze_manifest_integrity():
    manifest_path = REPORTS_DIR / "phase6i_freeze_manifest.json"
    assert manifest_path.exists(), "Phase 6I freeze manifest missing"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["production_frozen"] is True
    assert manifest["weights"]["alpha"] == 0.40
    assert manifest["weights"]["beta"] == 0.30
    assert manifest["weights"]["gamma"] == 0.30
    assert manifest["thresholds"]["VERIFIED"] == 0.35
    assert manifest["thresholds"]["NEEDS_VERIFICATION"] == 0.50
    assert manifest["thresholds"]["MODERATE_RISK"] == 0.65
    assert manifest["thresholds"]["LIKELY_HALLUCINATED"] == 0.65

def test_phase6i_dataset_independence():
    ind_path = REPORTS_DIR / "phase6i_dataset_independence.json"
    assert ind_path.exists(), "Phase 6I independence audit missing"
    ind_data = json.loads(ind_path.read_text())
    assert ind_data["independence_status"] == "PASS"
    assert ind_data["overlap_count"] == 0

def test_phase6i_dataset_schema_and_size():
    ds_path = DATA_DIR / "phase6i_independent_benchmark.json"
    assert ds_path.exists(), "Phase 6I benchmark missing"
    recs = json.loads(ds_path.read_text())
    assert len(recs) == 500, f"Expected 500 records, got {len(recs)}"

    pos = sum(1 for r in recs if r["gold_hallucination"])
    assert pos == 200, f"Expected 200 positive records, got {pos}"

def test_phase6i_results_json_schema():
    res_path = REPORTS_DIR / "phase6i_results.json"
    assert res_path.exists(), "Phase 6I results JSON missing"
    res = json.loads(res_path.read_text())
    assert res["final_decision_gate"] == "B. MODEST BUT DEFENSIBLE IMPROVEMENT"
    assert "ablation" in res
    assert "statistical_tests" in res
    assert "evidence_alignment" in res
    assert "domain_generalization" in res
    assert "latency" in res
