"""CI Quality Gate Enforcer for HalluciSense Phase 25 (Part 14).

Asserts:
1. Regression Suite v2 Accuracy >= 90.0%
2. Long-Form Scientific QA Accuracy >= 85.0%
3. Retrieval Recall@5 >= 0.85
4. Calibration ECE <= 0.08
5. Evidence Coverage >= 0.80

Exits with code 0 on pass, or code 1 on quality gate violation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation_results" / "phase25"
SUMMARY_FILE = RESULTS_DIR / "phase25_master_summary.json"


def check_quality_gates() -> int:
    """Enforce CI Quality Gates."""
    if not SUMMARY_FILE.exists():
        print(f"ERROR: Quality gate check failed. Master summary file missing at {SUMMARY_FILE}")
        return 1

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    reg_acc = summary.get("regression_v2_accuracy", 0.0)
    lf_acc = summary.get("longform_accuracy", 0.0)
    ir_metrics = summary.get("ir_metrics", {})
    recall_5 = ir_metrics.get("recall_at_5", 0.0)
    ece = summary.get("confidence_metrics", {}).get("expected_calibration_error_ece", 0.0)
    coverage = ir_metrics.get("evidence_coverage", 0.0)

    print("=" * 80)
    print("HALLUCISENSE PHASE 25 CI QUALITY GATE ENFORCER")
    print("=" * 80)
    print(f"  Regression Suite v2 Accuracy: {reg_acc*100:.2f}% (Target: >= 90.0%)")
    print(f"  Long-Form QA Accuracy:        {lf_acc*100:.2f}% (Target: >= 85.0%)")
    print(f"  Retrieval Recall@5:           {recall_5:.4f} (Target: >= 0.8500)")
    print(f"  Calibration ECE:              {ece:.4f} (Target: <= 0.0800)")
    print(f"  Evidence Coverage:            {coverage:.4f} (Target: >= 0.8000)")
    print("=" * 80)

    violations = []
    if reg_acc < 0.90:
        violations.append(f"Regression Accuracy {reg_acc*100:.2f}% < 90.0%")
    if lf_acc < 0.85:
        violations.append(f"Long-Form Accuracy {lf_acc*100:.2f}% < 85.0%")
    if recall_5 < 0.85:
        violations.append(f"Retrieval Recall@5 {recall_5:.4f} < 0.85")
    if ece > 0.08:
        violations.append(f"Calibration ECE {ece:.4f} > 0.08")
    if coverage < 0.80:
        violations.append(f"Evidence Coverage {coverage:.4f} < 0.80")

    if violations:
        print("❌ QUALITY GATE VIOLATIONS DETECTED:")
        for v in violations:
            print(f"  >> {v}")
        return 1

    print("✅ ALL PHASE 25 QUALITY GATES PASSED CLEANLY!")
    return 0


if __name__ == "__main__":
    sys.exit(check_quality_gates())
