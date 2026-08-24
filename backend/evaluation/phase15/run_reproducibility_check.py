"""Phase 15 Clean-Room Reproduction Verification Engine.

Validates:
1. Dataset SHA-256 Checksums.
2. Configuration & Manifest Integrity.
3. Model Registry Singleton Integrity.
4. Availability-Aware Adaptive Fusion Invariants.
5. Platt Scaling Monotonicity & Bounded Tolerances.
6. Outputs: PASS / PARTIAL / FAIL with diagnostic trace.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.engine.model_registry import ModelRegistry
from app.core.engine.fusion import FusionEngine
from app.core.engine.calibration import ProbabilityCalibrator

BENCHMARK_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
MANIFEST_PATH = BACKEND_DIR / "reports" / "phase15" / "REPRODUCIBILITY_MANIFEST.json"
EXPECTED_BENCHMARK_SHA = "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_clean_room_check() -> Dict[str, Any]:
    print("=" * 80)
    print("HALLUCISENSE PHASE 15 CLEAN-ROOM REPRODUCIBILITY CHECK")
    print("=" * 80)

    checks = []
    all_passed = True

    # 1. Verify Dataset Hash
    observed_hash = compute_sha256(BENCHMARK_PATH)
    hash_pass = observed_hash == EXPECTED_BENCHMARK_SHA
    checks.append({
        "check": "Canonical Benchmark SHA-256 Hash",
        "expected": EXPECTED_BENCHMARK_SHA,
        "observed": observed_hash,
        "status": "PASS" if hash_pass else "FAIL",
    })
    if not hash_pass:
        all_passed = False

    # 2. Verify Manifest Integrity
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_pass = manifest["canonical_benchmark_sha256"] == EXPECTED_BENCHMARK_SHA and manifest["phase"] == 15
    checks.append({
        "check": "Reproducibility Manifest Consistency",
        "expected": f"Phase 15 + SHA {EXPECTED_BENCHMARK_SHA[:8]}...",
        "observed": f"Phase {manifest['phase']} + SHA {manifest['canonical_benchmark_sha256'][:8]}...",
        "status": "PASS" if manifest_pass else "FAIL",
    })
    if not manifest_pass:
        all_passed = False

    # 3. Verify Model Registry Singleton Architecture
    p1 = ModelRegistry.get_pipeline()
    p2 = ModelRegistry.get_pipeline()
    singleton_pass = p1 is p2 and ModelRegistry.get_init_counts()["pipeline"] == 1
    checks.append({
        "check": "ModelRegistry Singleton Pipeline Identity",
        "expected": "Identity Match (p1 is p2) + init_count=1",
        "observed": f"Identity={'MATCH' if p1 is p2 else 'MISMATCH'}, init_count={ModelRegistry.get_init_counts()['pipeline']}",
        "status": "PASS" if singleton_pass else "FAIL",
    })
    if not singleton_pass:
        all_passed = False

    # 4. Minimal Inference Execution & Invariant Verification
    fusion = FusionEngine(alpha=0.40, beta=0.30, gamma=0.30)
    h_full, eff_w_full, mask_full = fusion.compute_adaptive_h_score(fe=0.85, cg=0.75, cf=0.65)
    h_blackbox, eff_w_bb, mask_bb = fusion.compute_adaptive_h_score(fe=0.85, cg=None, cf=0.65)

    adaptive_pass = (
        mask_full == [1, 1, 1]
        and mask_bb == [1, 0, 1]
        and eff_w_bb["beta_confidence_gap"] == 0.0
        and round(sum(eff_w_bb.values()), 4) == 1.0
        and abs(h_blackbox - 0.7643) < 0.01
    )
    checks.append({
        "check": "Adaptive Fusion Invariant & Weight Renormalization",
        "expected": "Mask [1,0,1], beta=0, sum=1.0, H ~= 0.7643",
        "observed": f"Mask {mask_bb}, sum={round(sum(eff_w_bb.values()), 4)}, H={h_blackbox}",
        "status": "PASS" if adaptive_pass else "FAIL",
    })
    if not adaptive_pass:
        all_passed = False

    # 5. Calibration Monotonicity & Bounded Range
    calibrator = ProbabilityCalibrator(method="platt", platt_a=1.82, platt_b=-0.45)
    c1 = calibrator.calibrate(0.20).calibrated_probability
    c2 = calibrator.calibrate(0.50).calibrated_probability
    c3 = calibrator.calibrate(0.80).calibrated_probability
    calib_pass = (0.0 <= c1 < c2 < c3 <= 1.0) and (abs(c2 - 0.50) < 0.15)
    checks.append({
        "check": "Platt Calibration Range & Monotonicity",
        "expected": "0.0 <= c(0.2) < c(0.5) < c(0.8) <= 1.0",
        "observed": f"c(0.2)={c1:.3f}, c(0.5)={c2:.3f}, c(0.8)={c3:.3f}",
        "status": "PASS" if calib_pass else "FAIL",
    })
    if not calib_pass:
        all_passed = False

    final_verdict = "PASS" if all_passed else "FAIL"

    print("\n--- Diagnostic Check Summary ---")
    for ch in checks:
        print(f"[{ch['status']}] {ch['check']}: Expected={ch['expected']} | Observed={ch['observed']}")

    print(f"\nFinal Clean-Room Reproduction Verdict: {final_verdict}")
    return {
        "verdict": final_verdict,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks,
    }


if __name__ == "__main__":
    res = run_clean_room_check()
    if res["verdict"] != "PASS":
        sys.exit(1)
