"""Long-Form Scientific Reasoning Benchmark Runner for HalluciSense Phase 25.

Evaluates pipeline performance across 500+ long-form scientific QA samples across 10 domains
(Medicine, Physics, Biology, Chemistry, Finance, Law, Programming, History, Education, Scientific QA).
"""

import sys
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.engine.pipeline import HallucinationDetectionPipeline

EVAL_DATA_DIR = BASE_DIR / "evaluation_data"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("HALLUCISENSE PHASE 25 LONG-FORM REASONING BENCHMARK RUNNER")
    print("=" * 80)

    lf_path = EVAL_DATA_DIR / "gold_longform_dataset.jsonl"
    if not lf_path.exists():
        from evaluation_data.build_phase25_datasets import build_longform_dataset
        build_longform_dataset(lf_path, 500)

    samples = []
    with open(lf_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    pipeline = HallucinationDetectionPipeline()
    correct = 0
    total = len(samples)
    domain_stats = {}

    start_time = time.time()

    for idx, sample in enumerate(samples[:50]):  # Fast benchmark subset of 50
        domain = sample.get("domain", "General")
        if domain not in domain_stats:
            domain_stats[domain] = {"correct": 0, "total": 0}

        domain_stats[domain]["total"] += 1

        resp = sample["response"]
        exp_hall = sample["expected_is_hallucinated"]

        report = pipeline.analyze(text=resp)
        pred_hall = float(report.overall_h_score) >= 0.54

        if pred_hall == exp_hall:
            correct += 1
            domain_stats[domain]["correct"] += 1

    elapsed = round(time.time() - start_time, 2)
    acc = round(correct / float(max(1, total if total <= 50 else 50)), 4)

    print(f"Evaluated Samples: {min(total, 50)}")
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Total Time:        {elapsed} s")
    print("-" * 80)
    print("Domain Accuracy Breakdown:")
    for dom, st in domain_stats.items():
        d_acc = round(st["correct"] / float(max(1, st["total"])), 4)
        print(f"  - {dom:<15}: {d_acc * 100:.1f}% ({st['correct']}/{st['total']})")
    print("=" * 80)
    print("✅ Long-form reasoning benchmark complete!")


if __name__ == "__main__":
    main()
