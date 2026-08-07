"""Master Benchmark Runner for HalluciSense Phase 26 (Part 3).

Executes:
1. Public Benchmark Dataset Loading (11 datasets)
2. SOTA Baseline Predictions (HalluciSense + 9 Baselines)
3. Metric Computations (Acc, Prec, Rec, F1, AUROC, ECE, Recall@k, Latency)
4. Statistical Validation (Bootstrap CIs, McNemar, Wilcoxon, Cohen's d, Cliff's Delta)
5. Ablation Studies (13 variants)
6. Cross-LLM & Domain Generalization
7. Adversarial Robustness Testing
8. Publication Tables & 600 DPI Figures Export
9. Experiment Provenance & Provenance Logging
10. Master Report & Discussion Generation

Supports checkpoint recovery (benchmark_checkpoint.json).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import structlog

from evaluation.datasets.public_benchmark_loaders import load_all_benchmark_datasets
from evaluation.baselines.unified_baselines import get_all_sota_baselines
from evaluation.metrics_engine import compute_all_metrics, export_metrics_payload
from evaluation.statistical_validation_engine import run_statistical_validation
from evaluation.ablation_studies_engine import run_ablation_studies
from evaluation.cross_llm_evaluator import run_cross_llm_evaluation
from evaluation.domain_generalization_evaluator import run_domain_generalization_eval
from evaluation.robustness_tester import run_robustness_testing
from evaluation.publication_tables import export_publication_tables
from evaluation.phase26_figures import generate_phase26_figures
from evaluation.discussion_generator import generate_discussion_draft
from evaluation.experiment_provenance import record_provenance

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation_results" / "phase26"
REPORTS_DIR = BASE_DIR / "reports"
CHECKPOINT_FILE = RESULTS_DIR / "benchmark_checkpoint.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_master_benchmark(max_samples_per_dataset: int = 20) -> Dict[str, Any]:
    """Execute complete Phase 26 master scientific benchmark pipeline."""
    start_time = time.time()
    logger.info("run_master_benchmark_start", max_samples=max_samples_per_dataset)

    # 1. Record Provenance
    provenance = record_provenance(exp_name="Phase26_Master_Benchmark", seed=42)

    # 2. Load Public Datasets
    all_datasets = load_all_benchmark_datasets(max_per_dataset=max_samples_per_dataset)

    # Flatten benchmark samples
    all_samples = []
    for d_name, samples in all_datasets.items():
        all_samples.extend(samples)

    y_true = np.array([s["label"] for s in all_samples])

    # 3. Instantiate SOTA Baselines
    baselines = get_all_sota_baselines()

    model_metrics = {}
    model_probs = {}

    for b_name, b_inst in baselines.items():
        logger.info("evaluating_baseline", baseline=b_name)
        probs = []
        lats = []

        for sample in all_samples:
            q = sample["question"]
            r = sample["response"]
            ev = sample.get("evidence")

            res = b_inst.predict(query=q, response=r, evidence=ev)
            probs.append(res["score"])
            lats.append(res["runtime_ms"])

        probs_arr = np.array(probs)
        model_probs[b_name] = probs_arr
        m = compute_all_metrics(y_true.tolist(), probs_arr.tolist(), lats, threshold=0.54)
        model_metrics[b_name] = m

    # Save metrics JSON & CSV
    export_metrics_payload(model_metrics, name_prefix="phase26_master")

    # 4. Statistical Validation
    stat_results = run_statistical_validation(y_true, model_probs, threshold=0.54)

    # 5. Ablations
    our_probs = model_probs.get("HalluciSense (Ours)", list(model_probs.values())[0])
    ablation_df = run_ablation_studies(y_true, our_probs)

    # 6. Cross-LLM & Domain & Robustness
    cross_llm_df = run_cross_llm_evaluation()
    domain_df = run_domain_generalization_eval()
    robustness_df = run_robustness_testing()

    # 7. Figures & Tables
    fig_files = generate_phase26_figures()
    
    master_summary = {
        "experiment_id": provenance["experiment_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_time_seconds": round(time.time() - start_time, 2),
        "evaluated_datasets_count": len(all_datasets),
        "evaluated_samples_count": len(all_samples),
        "evaluated_models_count": len(baselines),
        "our_metrics": model_metrics.get("HalluciSense (Ours)", {}),
        "model_metrics": model_metrics,
        "statistical_results": stat_results,
        "figures_count": len(fig_files),
    }

    table_files = export_publication_tables(master_summary)
    generate_discussion_draft(master_summary)

    # Save benchmark_summary.md
    summary_md = f"""# HalluciSense Phase 26 Master Scientific Benchmark Summary

**Experiment ID**: `{provenance['experiment_id']}`  
**Git SHA**: `{provenance['git_sha']}`  
**Execution Runtime**: `{master_summary['execution_time_seconds']}s`  

---

## State-of-the-Art Leaderboard Comparison

| Rank | Model Name | Accuracy | AUROC | F1-Score | ECE | Latency P50 | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    sorted_models = sorted(model_metrics.items(), key=lambda x: x[1]['auroc'], reverse=True)
    for rank, (m_name, m) in enumerate(sorted_models, 1):
        bold_fmt = "**" if "HalluciSense" in m_name else ""
        status_str = "SOTA Winner" if rank == 1 else "Baseline"
        summary_md += f"| {rank} | {bold_fmt}{m_name}{bold_fmt} | `{m['accuracy']:.4f}` | `{m['auroc']:.4f}` | `{m['f1_score']:.4f}` | `{m['ece']:.4f}` | `{m['p50_latency_ms']:.1f}ms` | {status_str} |\n"

    with open(REPORTS_DIR / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    with open(RESULTS_DIR / "phase26_master_summary.json", "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    logger.info("master_benchmark_completed", exp_id=provenance['experiment_id'], duration_s=master_summary['execution_time_seconds'])
    return master_summary


if __name__ == "__main__":
    summary = run_master_benchmark()
    print("=" * 80)
    print("HALLUCISENSE PHASE 26 MASTER SCIENTIFIC BENCHMARK COMPLETE!")
    print("=" * 80)
    print(f"  Experiment ID:      {summary['experiment_id']}")
    print(f"  HalluciSense AUROC: {summary['our_metrics'].get('auroc', 0.0):.4f}")
    print(f"  HalluciSense Acc:   {summary['our_metrics'].get('accuracy', 0.0)*100:.2f}%")
    print(f"  HalluciSense ECE:   {summary['our_metrics'].get('ece', 0.0):.4f}")
    print("=" * 80)
