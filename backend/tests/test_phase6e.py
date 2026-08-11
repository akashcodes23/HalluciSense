import pytest, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external"
REPORTS_DIR = ROOT / "reports" / "phase6e"

def test_phase6e_freeze_manifest_integrity():
    manifest_path = REPORTS_DIR / "phase6e_freeze_manifest.json"
    assert manifest_path.exists(), "Freeze manifest missing"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["production_frozen"] is True
    assert manifest["weights"]["alpha"] == 0.40
    assert manifest["weights"]["beta"] == 0.30
    assert manifest["weights"]["gamma"] == 0.30
    assert manifest["thresholds"]["VERIFIED"] == 0.35
    assert manifest["thresholds"]["NEEDS_VERIFICATION"] == 0.50
    assert manifest["thresholds"]["MODERATE_RISK"] == 0.65
    assert manifest["thresholds"]["LIKELY_HALLUCINATED"] == 0.65

def test_phase6e_dataset_independence():
    ind_path = REPORTS_DIR / "phase6e_dataset_independence.json"
    assert ind_path.exists(), "Independence audit report missing"
    ind_data = json.loads(ind_path.read_text())
    assert ind_data["independence_status"] == "PASS"
    assert ind_data["overlap_count"] == 0

def test_phase6e_dataset_schema_and_size():
    ds_path = DATA_DIR / "phase6e_independent_benchmark.json"
    assert ds_path.exists(), "Phase 6E benchmark dataset missing"
    recs = json.loads(ds_path.read_text())
    assert len(recs) == 600, f"Expected 600 records, got {len(recs)}"

    pos = sum(1 for r in recs if r["gold_hallucination"])
    neg = len(recs) - pos
    assert pos == 300 and neg == 300, f"Expected 50/50 balance, got pos={pos}, neg={neg}"

    required_keys = {"id", "domain", "query", "response", "context", "gold_hallucination", "epistemic_category", "evidence_noise_category", "split"}
    for rec in recs[:20]:
        assert required_keys.issubset(rec.keys()), f"Missing keys in record {rec['id']}"

def test_phase6e_counterfactual_pairs_completeness():
    pairs_path = REPORTS_DIR / "phase6e_counterfactual_pairs.json"
    assert pairs_path.exists(), "Counterfactual pairs missing"
    pairs = json.loads(pairs_path.read_text())
    assert len(pairs) == 6, f"Expected 6 counterfactual pairs, got {len(pairs)}"

def test_phase6e_results_json_schema():
    res_path = REPORTS_DIR / "phase6e_results.json"
    assert res_path.exists(), "Phase 6E results JSON missing"
    res = json.loads(res_path.read_text())
    assert "dataset_size" in res
    assert res["dataset_size"] == 600
    assert "ablation" in res
    assert "statistical_tests" in res
    assert "counterfactual_pairs" in res
    assert "domain_generalization" in res
    assert "latency" in res
    assert "calibration" in res
