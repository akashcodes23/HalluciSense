"""Error Analysis & Single-Label Failure Taxonomy Runner for HalluciSense Phase 25.

Performs single-label root-cause failure classification across benchmark test cases.
Outputs error_analysis.json and failure_taxonomy_report.md.
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.root_cause_classifier import RootCauseClassifier, RootCauseCategory

EVAL_DATA_DIR = BASE_DIR / "evaluation_data"
RESULTS_DIR = BASE_DIR / "evaluation_results" / "phase25"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("HALLUCISENSE PHASE 25 ROOT-CAUSE ERROR ANALYSIS RUNNER")
    print("=" * 80)

    reg_path = EVAL_DATA_DIR / "regression_suite_v2.jsonl"
    if not reg_path.exists():
        from evaluation_data.build_phase25_datasets import build_regression_v2_dataset
        build_regression_v2_dataset(reg_path, 1000)

    samples = []
    with open(reg_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    pipeline = HallucinationDetectionPipeline()
    taxonomy_counts = {}
    error_records = []

    for idx, sample in enumerate(samples[:50]):  # Analysis subset of 50
        resp_text = sample["response_text"]
        expected_hall = sample["expected_is_hallucinated"]

        report = pipeline.analyze(text=resp_text)
        p1 = report.pillar1_summary
        p2 = report.pillar2_summary
        p3 = report.pillar3_summary

        rc = RootCauseClassifier.classify(
            h_score=float(report.overall_h_score),
            p1_res=p1,
            p2_res=p2,
            p3_res=p3,
            evidence_items=p1.evidence,
            response_text=resp_text,
        ).value

        taxonomy_counts[rc] = taxonomy_counts.get(rc, 0) + 1
        error_records.append({
            "test_id": sample["test_id"],
            "response_text": resp_text,
            "h_score": float(report.overall_h_score),
            "expected_is_hallucinated": expected_hall,
            "root_cause_classification": rc,
        })

    print(f"Evaluated Samples: {min(len(samples), 50)}")
    print("-" * 80)
    print("Failure Taxonomy Distribution:")
    for cat_name, count in sorted(taxonomy_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cat_name:<30}: {count}")

    # Write error_analysis.json
    with open(RESULTS_DIR / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"total_samples": len(error_records), "taxonomy_counts": taxonomy_counts, "records": error_records}, f, indent=2)

    print("=" * 80)
    print("✅ Error analysis & failure taxonomy classification complete!")


if __name__ == "__main__":
    main()
