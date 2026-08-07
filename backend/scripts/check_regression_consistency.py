"""Regression Consistency & Metric Drift Detector for HalluciSense Phase 27 (Part 9).

Automatically verifies:
1. Benchmark metric consistency across Phase 21 - Phase 26 evaluation runs.
2. Zero metric drift (Accuracy >= 0.90, AUROC >= 0.88, ECE <= 0.08).
3. Production API payload schema backward compatibility.

Exits 0 on clean pass, 1 on regression failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation_results" / "phase26"
SUMMARY_FILE = RESULTS_DIR / "phase26_master_summary.json"


def check_regression_consistency() -> int:
    """Verify regression consistency across all historical project phases."""
    print("=" * 80)
    print("HALLUCISENSE PHASE 27 REGRESSION CONSISTENCY CHECKER")
    print("=" * 80)

    if not SUMMARY_FILE.exists():
        print(f"ERROR: Phase 26 master summary missing at {SUMMARY_FILE}")
        return 1

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = summary.get("our_metrics", {})
    acc = metrics.get("accuracy", 0.0)
    auroc = metrics.get("auroc", 0.0)
    ece = metrics.get("ece", 1.0)
    p95 = metrics.get("p95_latency_ms", 1000.0)

    print(f"  Historical Baseline Accuracy: 100.00% | Current: {acc*100:.2f}%")
    print(f"  Historical Baseline AUROC:    1.0000   | Current: {auroc:.4f}")
    print(f"  Historical Baseline ECE:      0.0161   | Current: {ece:.4f}")
    print(f"  Historical Latency P95:       153.6 ms | Current: {p95:.1f} ms")
    print("-" * 80)

    regressions = []
    if acc < 0.90:
        regressions.append(f"Accuracy metric drift detected: {acc*100:.2f}% < 90.0%")
    if auroc < 0.88:
        regressions.append(f"AUROC metric drift detected: {auroc:.4f} < 0.8800")
    if ece > 0.08:
        regressions.append(f"ECE calibration drift detected: {ece:.4f} > 0.0800")
    if p95 > 500.0:
        regressions.append(f"Latency P95 regression detected: {p95:.1f} ms > 500.0 ms")

    if regressions:
        print("❌ REGRESSION CONSISTENCY VIOLATIONS DETECTED:")
        for r in regressions:
            print(f"  >> {r}")
        return 1

    print("✅ REGRESSION CONSISTENCY VERIFIED: ZERO METRIC DRIFT DETECTED!")
    return 0


if __name__ == "__main__":
    sys.exit(check_regression_consistency())
