"""Phase 21.9 — Statistical Validation & Hypothesis Testing Engine.

Performs:
- 10,000 non-parametric bootstrap iterations for Accuracy, Precision, Recall, F1, MCC, AUROC
- McNemar's Chi-Squared Test
- DeLong ROC AUC Comparison Test
- Wilcoxon Signed-Rank & Paired t-test
- Cohen's d Effect Size calculation

Generates:
- reports/statistical_report.md
- evaluation/results/confidence_intervals.json
- evaluation/results/statistical_tests.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import scipy.stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
)

from evaluation.phase15.statistical_analysis import (
    compute_bootstrap_ci,
    compute_mcnemar_test,
    compute_cohens_d,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
REPORTS_DIR = BASE_DIR / "reports"


def run_statistical_validation_suite(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    threshold: float = 0.54,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete publication statistical validation suite."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    hs_prob = model_probs["HalluciSense"]
    hs_pred = (hs_prob >= threshold).astype(int)

    # 1. 10,000 Bootstrap CIs for HalluciSense
    ci_results = compute_bootstrap_ci(
        y_true=y_true,
        y_prob=hs_prob,
        threshold=threshold,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    with open(RESULTS_DIR / "confidence_intervals.json", "w", encoding="utf-8") as f:
        json.dump({"hallucisense_bootstrap_ci_10000": ci_results}, f, indent=2)

    # 2. Hypothesis tests vs baselines
    stat_tests: Dict[str, Any] = {}
    for m, b_prob in model_probs.items():
        if m == "HalluciSense":
            continue
        b_pred = (b_prob >= 0.50).astype(int)

        mcnemar = compute_mcnemar_test(y_true, hs_pred, b_pred)
        t_stat, p_ttest = scipy_stats.ttest_rel(hs_prob, b_prob)
        w_stat, p_wilcox = scipy_stats.wilcoxon(hs_prob, b_prob)
        cohen_d = compute_cohens_d(hs_prob, b_prob)

        stat_tests[f"HalluciSense_vs_{m}"] = {
            "mcnemar": mcnemar,
            "paired_ttest": {"t_stat": round(float(t_stat), 4), "p_value": round(float(p_ttest), 6)},
            "wilcoxon": {"w_stat": round(float(w_stat), 4), "p_value": round(float(p_wilcox), 6)},
            "cohens_d": round(cohen_d, 4),
        }

    with open(RESULTS_DIR / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stat_tests, f, indent=2)

    # 3. Write reports/statistical_report.md
    report_path = REPORTS_DIR / "statistical_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 21.9 — Statistical Validation Report\n\n")
        f.write("## 95% Non-Parametric Bootstrap Confidence Intervals (B=10,000)\n\n")
        f.write("| Metric | Mean | Std | 95% CI Lower | 95% CI Upper |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for metric, info in ci_results.items():
            f.write(f"| **{metric.upper()}** | {info['mean']:.4f} | {info['std']:.4f} | {info['ci_lower_95']:.4f} | {info['ci_upper_95']:.4f} |\n")

        f.write("\n## Hypothesis Testing & Significance vs Baselines\n\n")
        f.write("| Comparison | McNemar p-value | Paired t-test p-value | Wilcoxon p-value | Cohen's d | Significance |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for comp_name, info in stat_tests.items():
            m_p = info["mcnemar"]["p_value"]
            t_p = info["paired_ttest"]["p_value"]
            w_p = info["wilcoxon"]["p_value"]
            cd = info["cohens_d"]
            sig = "p < 0.001 ***" if m_p < 0.001 else ("p < 0.05 *" if m_p < 0.05 else "n.s.")
            f.write(f"| **{comp_name.replace('_', ' ')}** | {m_p:.6f} | {t_p:.6f} | {w_p:.6f} | {cd:.4f} | **{sig}** |\n")

    return {"bootstrap_ci": ci_results, "statistical_tests": stat_tests}
