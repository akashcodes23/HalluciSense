"""Frozen Baseline Benchmark Runner for HalluciSense Phase 6B.

Usage:
    python -m evaluation.run_benchmark --dataset <path> --output-dir evaluation_results/phase6b --seed 42

Executes frozen production HalluciSense pipeline over an independent benchmark dataset,
guaranteeing data leakage protection, zero network calls, reproducible fixed seeding,
and complete scientific artifact export.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any, Dict, List, Optional

from app.core.config import settings
from evaluation.datasets.adapter import BenchmarkAdapter, BenchmarkDataset
from evaluation.runner import EvaluationRunner, EvaluationSampleResult


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HalluciSense Phase 6B Frozen Baseline Benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to benchmark dataset file (.jsonl, .json, .csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results/phase6b",
        help="Directory where evaluation artifacts and reports will be saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def set_reproducible_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def execute_benchmark(
    dataset_path: str, output_dir: str, seed: int = 42
) -> Dict[str, Any]:
    """Main benchmark execution function."""
    set_reproducible_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    benchmark_dataset: BenchmarkDataset = BenchmarkAdapter.load_dataset(dataset_path)
    samples = benchmark_dataset.to_benchmark_samples()

    # 2. Run EvaluationRunner on frozen pipeline
    runner = EvaluationRunner()
    results = runner.evaluate_dataset(samples)

    # 3. Process per-example results & error classification
    per_example_rows = []
    false_positives = []
    false_negatives = []
    risk_crosstab = {
        "0": {"VERIFIED": 0, "NEEDS_VERIFICATION": 0, "LIKELY_HALLUCINATED": 0},
        "1": {"VERIFIED": 0, "NEEDS_VERIFICATION": 0, "LIKELY_HALLUCINATED": 0},
    }

    raw_sample_results: List[EvaluationSampleResult] = [
        runner.evaluate_sample(s) for s in samples
    ]

    sample_map = {s.id: s for s in samples}

    for sr in raw_sample_results:
        # Binary prediction: 0 if VERIFIED else 1
        pred_binary = 0 if sr.predicted_risk == "VERIFIED" else 1
        gt = sr.ground_truth
        is_correct = pred_binary == gt

        if gt == 1 and pred_binary == 1:
            err_type = "TP"
        elif gt == 0 and pred_binary == 0:
            err_type = "TN"
        elif gt == 0 and pred_binary == 1:
            err_type = "FP"
        else:
            err_type = "FN"

        risk_crosstab[str(gt)][sr.predicted_risk] += 1

        ex_sample = sample_map[sr.sample_id]

        ex_dict = {
            "example_id": sr.sample_id,
            "prompt": ex_sample.prompt,
            "response": ex_sample.response,
            "ground_truth": gt,
            "predicted_binary_label": pred_binary,
            "predicted_risk_level": sr.predicted_risk,
            "overall_h_score": sr.h_score,
            "factual_error": sr.p1_factual_error,
            "confidence_gap": sr.p2_confidence_gap,
            "consistency_failure": sr.p3_consistency_failure,
            "pillar1_available": sr.p1_available,
            "pillar2_available": sr.p2_available,
            "pillar3_available": sr.p3_available,
            "weights_used": sr.effective_weights,
            "is_correct": is_correct,
            "error_type": err_type,
            "category": sr.category,
            "processing_time_ms": sr.processing_time_ms,
        }
        per_example_rows.append(ex_dict)

        if err_type == "FP":
            false_positives.append(ex_dict)
        elif err_type == "FN":
            false_negatives.append(ex_dict)

    # Sort FPs highest H-Score first, FNs lowest H-Score first
    false_positives.sort(key=lambda x: x["overall_h_score"], reverse=True)
    false_negatives.sort(key=lambda x: x["overall_h_score"], reverse=False)

    # 4. Generate metadata
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_name": benchmark_dataset.dataset_name,
        "dataset_file": benchmark_dataset.file_path,
        "dataset_checksum_sha256": benchmark_dataset.checksum,
        "dataset_size": benchmark_dataset.total_count,
        "synthetic_test_fixture": benchmark_dataset.synthetic_test_fixture,
        "class_distribution": {
            "factual_count": benchmark_dataset.factual_count,
            "hallucinated_count": benchmark_dataset.hallucinated_count,
        },
        "random_seed": seed,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "frozen_hallucisense_configuration": {
            "alpha_factual_error": settings.ALPHA_FACTUAL_ERROR,
            "beta_confidence_gap": settings.BETA_CONFIDENCE_GAP,
            "gamma_consistency_failure": settings.GAMMA_CONSISTENCY_FAILURE,
            "verified_threshold": settings.VERIFIED_THRESHOLD,
            "hallucinated_threshold": settings.HALLUCINATED_THRESHOLD,
        },
    }

    # 5. Risk level analysis
    total_samples = len(samples)
    risk_counts = {"VERIFIED": 0, "NEEDS_VERIFICATION": 0, "LIKELY_HALLUCINATED": 0}
    for sr in raw_sample_results:
        risk_counts[sr.predicted_risk] += 1

    risk_level_analysis = {
        "counts": risk_counts,
        "percentages": {
            k: round((v / total_samples) * 100, 2) if total_samples > 0 else 0.0
            for k, v in risk_counts.items()
        },
        "cross_tabulation": risk_crosstab,
        "binary_mapping_definition": {
            "0_factual": "predicted_risk == 'VERIFIED' (H-Score < 0.35)",
            "1_hallucinated": "predicted_risk in ('NEEDS_VERIFICATION', 'LIKELY_HALLUCINATED') (H-Score >= 0.35)",
        },
    }

    # 6. Save JSON/JSONL/CSV artifacts
    # per_example_results.jsonl
    with open(output_path / "per_example_results.jsonl", "w", encoding="utf-8") as f:
        for row in per_example_rows:
            f.write(json.dumps(row) + "\n")

    # run_metadata.json
    with open(output_path / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # metrics.json
    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results["overall_metrics"], f, indent=2)

    # confusion_matrix.json
    with open(output_path / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(results["confusion_matrix"], f, indent=2)

    # risk_level_analysis.json
    with open(output_path / "risk_level_analysis.json", "w", encoding="utf-8") as f:
        json.dump(risk_level_analysis, f, indent=2)

    # pillar_availability.json
    with open(output_path / "pillar_availability.json", "w", encoding="utf-8") as f:
        json.dump(results["availability_analysis"], f, indent=2)

    # ablation_results.json
    with open(output_path / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results["ablation_results"], f, indent=2)

    # calibration.json
    with open(output_path / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(results["calibration_results"], f, indent=2)

    # false_positives.json & false_negatives.json
    with open(output_path / "false_positives.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(false_positives),
                "false_positives": false_positives,
            },
            f,
            indent=2,
        )

    with open(output_path / "false_negatives.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(false_negatives),
                "false_negatives": false_negatives,
            },
            f,
            indent=2,
        )

    # threshold_sweep.csv
    with open(output_path / "threshold_sweep.csv", "w", encoding="utf-8") as f:
        f.write(
            "threshold,precision,recall,specificity,false_positive_rate,false_negative_rate,f1,youden_j,status\n"
        )
        sweep = results["threshold_analysis"]
        opt_f1 = sweep.get("optimal_f1_threshold")
        opt_yj = sweep.get("optimal_youden_j_threshold")
        prod_th = 0.35

        for p in sweep["sweep_points"]:
            t = p["threshold"]
            st_list = []
            if t == prod_th:
                st_list.append("CURRENT_PRODUCTION")
            if t == opt_f1:
                st_list.append("OPTIMAL_F1_NOT_DEPLOYED")
            if t == opt_yj:
                st_list.append("OPTIMAL_YOUDEN_NOT_DEPLOYED")
            st_str = ";".join(st_list) if st_list else "ANALYSIS_ONLY"

            f.write(
                f"{t},{p['precision']},{p['recall']},{p['specificity']},"
                f"{p['false_positive_rate']},{p['false_negative_rate']},"
                f"{p['f1']},{p['youden_j']},{st_str}\n"
            )

    # 7. Generate benchmark_report.md
    generate_markdown_report(
        output_path / "benchmark_report.md",
        metadata,
        results["overall_metrics"],
        results["confusion_matrix"],
        results["score_distributions"],
        risk_level_analysis,
        results["availability_analysis"],
        results["ablation_results"],
        sweep,
        results["calibration_results"],
        false_positives,
        false_negatives,
    )

    return {
        "metadata": metadata,
        "metrics": results["overall_metrics"],
        "output_directory": str(output_path),
    }


def generate_markdown_report(
    report_path: Path,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    confusion: Dict[str, Any],
    distributions: Dict[str, Any],
    risk_analysis: Dict[str, Any],
    availability: Dict[str, Any],
    ablation: Dict[str, Any],
    threshold_sweep: Dict[str, Any],
    calibration: Dict[str, Any],
    false_positives: List[Dict[str, Any]],
    false_negatives: List[Dict[str, Any]],
) -> None:
    """Generates a human-readable benchmark report (benchmark_report.md)."""
    syn_warning = ""
    if metadata.get("synthetic_test_fixture"):
        syn_warning = """
> [!WARNING]
> **DEVELOPMENT FIXTURE ONLY**: This benchmark run was executed using a synthetic development test fixture (`synthetic_test_fixture = true`).
> Results are for verifying benchmark runner infrastructure correctness and must **NOT** be reported as real-world academic performance.
"""

    md = f"""# HalluciSense Phase 6B — Frozen Baseline Benchmark Report

**Timestamp**: {metadata['timestamp']}  
**Dataset**: `{metadata['dataset_name']}` ({metadata['dataset_file']})  
**Dataset Size**: {metadata['dataset_size']} samples (Factual: {metadata['class_distribution']['factual_count']}, Hallucinated: {metadata['class_distribution']['hallucinated_count']})  
**Dataset SHA256**: `{metadata['dataset_checksum_sha256'][:16]}...`  
**Random Seed**: `{metadata['random_seed']}`  
{syn_warning}

---

## A. Frozen Production Configuration
- **Base Weights**: $\\alpha_{{FE}} = {metadata['frozen_hallucisense_configuration']['alpha_factual_error']}$, $\\beta_{{CG}} = {metadata['frozen_hallucisense_configuration']['beta_confidence_gap']}$, $\\gamma_{{CF}} = {metadata['frozen_hallucisense_configuration']['gamma_consistency_failure']}$
- **Risk Thresholds**: VERIFIED ($H < 0.35$), NEEDS_VERIFICATION ($0.35 \\le H < 0.65$), LIKELY_HALLUCINATED ($H \\ge 0.65$)
- **Decision Rule**: `VERIFIED` $\\rightarrow$ Factual ($0$); `NEEDS_VERIFICATION` / `LIKELY_HALLUCINATED` $\\rightarrow$ Hallucinated ($1$).

---

## B. Main Classification Metrics
- **Accuracy**: `{metrics.get('accuracy')}`
- **Balanced Accuracy**: `{metrics.get('balanced_accuracy')}`
- **Precision**: `{metrics.get('precision')}`
- **Recall / Sensitivity**: `{metrics.get('recall')}`
- **Specificity**: `{metrics.get('specificity')}`
- **F1 Score**: `{metrics.get('f1')}`
- **False Positive Rate (FPR)**: `{metrics.get('false_positive_rate')}`
- **False Negative Rate (FNR)**: `{metrics.get('false_negative_rate')}`
- **Youden's J Statistic**: `{metrics.get('youden_j')}`
- **ROC-AUC**: `{metrics.get('roc_auc')}`
- **PR-AUC**: `{metrics.get('pr_auc')}`

---

## C. Confusion Matrix

| Actual \\ Predicted | Factual (0) | Hallucinated (1) |
|---|---|---|
| **Factual (0)** | TN = **{confusion['tn']}** | FP = **{confusion['fp']}** |
| **Hallucinated (1)** | FN = **{confusion['fn']}** | TP = **{confusion['tp']}** |

---

## D. H-Score Distribution Analysis

| Ground Truth | Count | Mean | Median | Std Dev | Min | Max | Q1 (P25) | Q3 (P75) |
|---|---|---|---|---|---|---|---|---|
| **Factual (0)** | {distributions['factual']['count']} | {distributions['factual']['mean']} | {distributions['factual']['median']} | {distributions['factual']['std']} | {distributions['factual']['min']} | {distributions['factual']['max']} | {distributions['factual']['p25']} | {distributions['factual']['p75']} |
| **Hallucinated (1)** | {distributions['hallucinated']['count']} | {distributions['hallucinated']['mean']} | {distributions['hallucinated']['median']} | {distributions['hallucinated']['std']} | {distributions['hallucinated']['min']} | {distributions['hallucinated']['max']} | {distributions['hallucinated']['p25']} | {distributions['hallucinated']['p75']} |

---

## E. Risk-Level Distribution & Cross-Tabulation

### Risk-Level Output Breakdown
- **VERIFIED**: {risk_analysis['counts']['VERIFIED']} ({risk_analysis['percentages']['VERIFIED']}%)
- **NEEDS_VERIFICATION**: {risk_analysis['counts']['NEEDS_VERIFICATION']} ({risk_analysis['percentages']['NEEDS_VERIFICATION']}%)
- **LIKELY_HALLUCINATED**: {risk_analysis['counts']['LIKELY_HALLUCINATED']} ({risk_analysis['percentages']['LIKELY_HALLUCINATED']}%)

### Cross-Tabulation Matrix (Ground Truth × Risk Level)
| Ground Truth | VERIFIED | NEEDS_VERIFICATION | LIKELY_HALLUCINATED |
|---|---|---|---|
| **Factual (0)** | {risk_analysis['cross_tabulation']['0']['VERIFIED']} | {risk_analysis['cross_tabulation']['0']['NEEDS_VERIFICATION']} | {risk_analysis['cross_tabulation']['0']['LIKELY_HALLUCINATED']} |
| **Hallucinated (1)** | {risk_analysis['cross_tabulation']['1']['VERIFIED']} | {risk_analysis['cross_tabulation']['1']['NEEDS_VERIFICATION']} | {risk_analysis['cross_tabulation']['1']['LIKELY_HALLUCINATED']} |

---

## F. Pillar Availability Breakdown

| Pillar Condition | Count | Mean H-Score | Accuracy | F1 Score |
|---|---|---|---|---|
| **All Pillars Available** | {availability['all_pillars_available']['sample_count']} | {availability['all_pillars_available'].get('mean_h_score')} | {availability['all_pillars_available']['metrics'].get('accuracy') if availability['all_pillars_available']['metrics'] else 'N/A'} | {availability['all_pillars_available']['metrics'].get('f1') if availability['all_pillars_available']['metrics'] else 'N/A'} |
| **Pillar 2 Unavailable** | {availability['p2_unavailable']['sample_count']} | {availability['p2_unavailable'].get('mean_h_score')} | {availability['p2_unavailable']['metrics'].get('accuracy') if availability['p2_unavailable']['metrics'] else 'N/A'} | {availability['p2_unavailable']['metrics'].get('f1') if availability['p2_unavailable']['metrics'] else 'N/A'} |
| **Pillar 3 Unavailable** | {availability['p3_unavailable']['sample_count']} | {availability['p3_unavailable'].get('mean_h_score')} | {availability['p3_unavailable']['metrics'].get('accuracy') if availability['p3_unavailable']['metrics'] else 'N/A'} | {availability['p3_unavailable']['metrics'].get('f1') if availability['p3_unavailable']['metrics'] else 'N/A'} |
| **P2 + P3 Unavailable** | {availability['p2_and_p3_unavailable']['sample_count']} | {availability['p2_and_p3_unavailable'].get('mean_h_score')} | {availability['p2_and_p3_unavailable']['metrics'].get('accuracy') if availability['p2_and_p3_unavailable']['metrics'] else 'N/A'} | {availability['p2_and_p3_unavailable']['metrics'].get('f1') if availability['p2_and_p3_unavailable']['metrics'] else 'N/A'} |

---

## G. Pillar Ablation Analysis

| Configuration | Samples | Precision | Recall | Specificity | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **P1 ONLY** | {ablation['P1_ONLY']['sample_count']} | {ablation['P1_ONLY']['metrics'].get('precision') if ablation['P1_ONLY']['metrics'] else 'N/A'} | {ablation['P1_ONLY']['metrics'].get('recall') if ablation['P1_ONLY']['metrics'] else 'N/A'} | {ablation['P1_ONLY']['metrics'].get('specificity') if ablation['P1_ONLY']['metrics'] else 'N/A'} | {ablation['P1_ONLY']['metrics'].get('f1') if ablation['P1_ONLY']['metrics'] else 'N/A'} | {ablation['P1_ONLY']['metrics'].get('roc_auc') if ablation['P1_ONLY']['metrics'] else 'N/A'} |
| **P2 ONLY** | {ablation['P2_ONLY']['sample_count']} | {ablation['P2_ONLY']['metrics'].get('precision') if ablation['P2_ONLY']['metrics'] else 'N/A'} | {ablation['P2_ONLY']['metrics'].get('recall') if ablation['P2_ONLY']['metrics'] else 'N/A'} | {ablation['P2_ONLY']['metrics'].get('specificity') if ablation['P2_ONLY']['metrics'] else 'N/A'} | {ablation['P2_ONLY']['metrics'].get('f1') if ablation['P2_ONLY']['metrics'] else 'N/A'} | {ablation['P2_ONLY']['metrics'].get('roc_auc') if ablation['P2_ONLY']['metrics'] else 'N/A'} |
| **P3 ONLY** | {ablation['P3_ONLY']['sample_count']} | {ablation['P3_ONLY']['metrics'].get('precision') if ablation['P3_ONLY']['metrics'] else 'N/A'} | {ablation['P3_ONLY']['metrics'].get('recall') if ablation['P3_ONLY']['metrics'] else 'N/A'} | {ablation['P3_ONLY']['metrics'].get('specificity') if ablation['P3_ONLY']['metrics'] else 'N/A'} | {ablation['P3_ONLY']['metrics'].get('f1') if ablation['P3_ONLY']['metrics'] else 'N/A'} | {ablation['P3_ONLY']['metrics'].get('roc_auc') if ablation['P3_ONLY']['metrics'] else 'N/A'} |
| **P1 + P2** | {ablation['P1_P2']['sample_count']} | {ablation['P1_P2']['metrics'].get('precision') if ablation['P1_P2']['metrics'] else 'N/A'} | {ablation['P1_P2']['metrics'].get('recall') if ablation['P1_P2']['metrics'] else 'N/A'} | {ablation['P1_P2']['metrics'].get('specificity') if ablation['P1_P2']['metrics'] else 'N/A'} | {ablation['P1_P2']['metrics'].get('f1') if ablation['P1_P2']['metrics'] else 'N/A'} | {ablation['P1_P2']['metrics'].get('roc_auc') if ablation['P1_P2']['metrics'] else 'N/A'} |
| **P1 + P3** | {ablation['P1_P3']['sample_count']} | {ablation['P1_P3']['metrics'].get('precision') if ablation['P1_P3']['metrics'] else 'N/A'} | {ablation['P1_P3']['metrics'].get('recall') if ablation['P1_P3']['metrics'] else 'N/A'} | {ablation['P1_P3']['metrics'].get('specificity') if ablation['P1_P3']['metrics'] else 'N/A'} | {ablation['P1_P3']['metrics'].get('f1') if ablation['P1_P3']['metrics'] else 'N/A'} | {ablation['P1_P3']['metrics'].get('roc_auc') if ablation['P1_P3']['metrics'] else 'N/A'} |
| **P2 + P3** | {ablation['P2_P3']['sample_count']} | {ablation['P2_P3']['metrics'].get('precision') if ablation['P2_P3']['metrics'] else 'N/A'} | {ablation['P2_P3']['metrics'].get('recall') if ablation['P2_P3']['metrics'] else 'N/A'} | {ablation['P2_P3']['metrics'].get('specificity') if ablation['P2_P3']['metrics'] else 'N/A'} | {ablation['P2_P3']['metrics'].get('f1') if ablation['P2_P3']['metrics'] else 'N/A'} | {ablation['P2_P3']['metrics'].get('roc_auc') if ablation['P2_P3']['metrics'] else 'N/A'} |
| **P1 + P2 + P3** | {ablation['P1_P2_P3']['sample_count']} | {ablation['P1_P2_P3']['metrics'].get('precision') if ablation['P1_P2_P3']['metrics'] else 'N/A'} | {ablation['P1_P2_P3']['metrics'].get('recall') if ablation['P1_P2_P3']['metrics'] else 'N/A'} | {ablation['P1_P2_P3']['metrics'].get('specificity') if ablation['P1_P2_P3']['metrics'] else 'N/A'} | {ablation['P1_P2_P3']['metrics'].get('f1') if ablation['P1_P2_P3']['metrics'] else 'N/A'} | {ablation['P1_P2_P3']['metrics'].get('roc_auc') if ablation['P1_P2_P3']['metrics'] else 'N/A'} |

---

## H. Threshold Sweep Observations (Analysis Only)

- **Current Production Threshold**: `0.35` (F1 Score: `{metrics.get('f1')}`)
- **Optimal F1 Threshold (NOT DEPLOYED)**: `{threshold_sweep.get('optimal_f1_threshold')}` (F1 Score: `{threshold_sweep.get('optimal_f1_score')}`)
- **Optimal Youden's J Threshold (NOT DEPLOYED)**: `{threshold_sweep.get('optimal_youden_j_threshold')}` (Youden J: `{threshold_sweep.get('optimal_youden_j_score')}`)

> [!NOTE]
> Threshold sweep observations are for experimental analysis only. Production thresholds remain frozen at $0.35$ for `VERIFIED` and $0.65$ for `LIKELY_HALLUCINATED`.

---

## I. Calibration Results

- **Brier Score**: `{calibration.get('brier_score')}`
- **Expected Calibration Error (ECE)**: `{calibration.get('ece')}`

---

## J. Failure Analysis Summary

- **False Positives**: {len(false_positives)} samples (factual responses misclassified as hallucinated)
- **False Negatives**: {len(false_negatives)} samples (hallucinated responses misclassified as factual)

### Top False Positives (Ranked by Highest H-Score)
"""

    if not false_positives:
        md += "\n*No False Positives observed.*"
    else:
        for fp in false_positives[:5]:
            md += f"\n- **ID `{fp['example_id']}`** ({fp['category']}): H-Score={fp['overall_h_score']} (FE={fp['factual_error']}, CG={fp['confidence_gap']}, CF={fp['consistency_failure']})\n  - *Prompt*: {fp['prompt']}\n  - *Response*: {fp['response']}\n"

    md += "\n\n### Top False Negatives (Ranked by Lowest H-Score)\n"
    if not false_negatives:
        md += "\n*No False Negatives observed.*"
    else:
        for fn in false_negatives[:5]:
            md += f"\n- **ID `{fn['example_id']}`** ({fn['category']}): H-Score={fn['overall_h_score']} (FE={fn['factual_error']}, CG={fn['confidence_gap']}, CF={fn['consistency_failure']})\n  - *Prompt*: {fn['prompt']}\n  - *Response*: {fn['response']}\n"

    verdict_text = "HALLUCISENSE PHASE 6B FROZEN BASELINE: PASS"
    if metadata.get("synthetic_test_fixture"):
        verdict_text = "PHASE 6B INFRASTRUCTURE READY — REAL BENCHMARK DATASET REQUIRED"

    md += f"""

---

## K. Reproducibility & Environment
- **Python**: `{metadata['environment']['python_version']}`
- **Platform**: `{metadata['environment']['platform']}`
- **Seed**: `{metadata['random_seed']}`

---

## L. Final Phase 6B Verdict

```
{verdict_text}
```
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    args = parse_args()
    execute_benchmark(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed,
    )
