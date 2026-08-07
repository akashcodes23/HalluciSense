"""CI Quality Gate Enforcer for HalluciSense Phase 26 (Part 15).

Asserts:
1. HalluciSense AUROC >= 0.88
2. HalluciSense Accuracy >= 0.90
3. HalluciSense F1-Score >= 0.85
4. HalluciSense ECE <= 0.08
5. HalluciSense Latency P95 <= 500.0 ms

Exits 0 on pass, 1 on quality gate violation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation_results" / "phase26"
SUMMARY_FILE = RESULTS_DIR / "phase26_master_summary.json"


def check_quality_gates() -> int:
    """Enforce CI Quality Gates for Phase 26."""
    if not SUMMARY_FILE.exists():
        print(f"ERROR: Quality gate check failed. Master summary missing at {SUMMARY_FILE}")
        return 1

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    our = summary.get("our_metrics", {})
    acc = our.get("accuracy", 0.0)
    auroc = our.get("auroc", 0.0)
    f1 = our.get("f1_score", 0.0)
    ece = our.get("ece", 1.0)
    p95_lat = our.get("p95_latency_ms", 1000.0)

    print("=" * 80)
    print("HALLUCISENSE PHASE 26 CI QUALITY GATE ENFORCER")
    print("=" * 80)
    print(f"  Accuracy:       {acc*100:.2f}% (Target: >= 90.0%)")
    print(f"  AUROC:          {auroc:.4f} (Target: >= 0.8800)")
    print(f"  F1-Score:       {f1:.4f} (Target: >= 0.8500)")
    print(f"  ECE:            {ece:.4f} (Target: <= 0.0800)")
    print(f"  P95 Latency:    {p95_lat:.1f} ms (Target: <= 500.0 ms)")
    print("=" * 80)

    violations = []
    if acc < 0.90:
        violations.append(f"Accuracy {acc*100:.2f}% < 90.0%")
    if auroc < 0.88:
        violations.append(f"AUROC {auroc:.4f} < 0.8800")
    if f1 < 0.85:
        violations.append(f"F1-Score {f1:.4f} < 0.8500")
    if ece > 0.08:
        violations.append(f"ECE {ece:.4f} > 0.0800")
    if p95_lat > 500.0:
        violations.append(f"P95 Latency {p95_lat:.1f} ms > 500.0 ms")

    if violations:
        print("❌ PHASE 26 QUALITY GATE VIOLATIONS DETECTED:")
        for v in violations:
            print(f"  >> {v}")
        return 1

    print("✅ ALL PHASE 26 CI QUALITY GATES PASSED CLEANLY!")
    return 0


if __name__ == "__main__":
    sys.exit(check_quality_gates())
