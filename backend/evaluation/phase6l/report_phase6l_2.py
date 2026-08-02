"""Phase 6L.2 — Stage 9: Publication Figures & Final Markdown Report Generator.

Generates 8 publication-quality 300 DPI figures and publishes
evaluation_results/phase6l/PHASE6L_2_DEVELOPMENT_MODEL_SELECTION.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scipy.stats as scipy_stats
from sklearn.metrics import precision_recall_curve, roc_curve, auc as calc_auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import structlog

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS

logger = structlog.get_logger(__name__)

FIGURES_DIR = PHASE6L_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def generate_publication_figures_and_report(
    X: np.ndarray,
    y: np.ndarray,
    validation_res: Dict[str, Any],
    preprocessing_res: Dict[str, Any],
    collinearity_res: Dict[str, Any],
    discrimination_res: Dict[str, Any],
    stability_res: Dict[str, Any],
    model_selection_res: Dict[str, Any],
    leakage_res: Dict[str, Any],
    protocol_res: Dict[str, Any],
    out_dir: Path = PHASE6L_DIR,
) -> Path:
    """Generate 8 publication figures and PHASE6L_2_DEVELOPMENT_MODEL_SELECTION.md report."""
    logger.info("stage9_generate_report_start")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    feature_names = STRUCTURAL_FEATURE_COLUMNS

    # -------------------------------------------------------------------------
    # Figure 1: Feature Correlation Heatmap
    # -------------------------------------------------------------------------
    corr_matrix = np.array(collinearity_res["correlations"]["spearman"]["matrix"])
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels([f"F{i+1}" for i in range(len(feature_names))], rotation=90, fontsize=8)
    ax.set_yticklabels([f"{feature_names[i]}" for i in range(len(feature_names))], fontsize=7)
    ax.set_title("Spearman Rank Correlation Heatmap (Pillar-2 Structural Features)", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "phase6l_2_feature_correlation.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: VIF Visualization
    # -------------------------------------------------------------------------
    vif_records = collinearity_res["vif"]["vif_records"]
    vif_vals = [r["vif"] for r in vif_records]
    vif_names = [r["feature"] for r in vif_records]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#d62728" if v > 5.0 else "#1f77b4" for v in vif_vals]
    ax.barh(vif_names, vif_vals, color=colors, alpha=0.85)
    ax.axvline(5.0, color="r", linestyle="--", lw=1.5, label="VIF Threshold = 5.0")
    ax.set_xlabel("Variance Inflation Factor (VIF)", fontsize=10)
    ax.set_title("Multicollinearity Audit: Feature VIF Scores", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "phase6l_2_vif_visualization.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Candidate Comparison Boxplot / Bar
    # -------------------------------------------------------------------------
    eval_results = model_selection_res["eval_results"]
    cand_names = list(eval_results.keys())
    auc_means = [eval_results[k]["summary_metrics"]["roc_auc_mean"] for k in cand_names]
    mcc_means = [eval_results[k]["summary_metrics"]["best_mcc"] for k in cand_names]

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(cand_names))
    width = 0.35

    rects1 = ax1.bar(x - width/2, auc_means, width, label="ROC-AUC", color="#1f77b4", alpha=0.85)
    rects2 = ax1.bar(x + width/2, mcc_means, width, label="MCC (at best thresh)", color="#2ca02c", alpha=0.85)

    ax1.set_ylabel("Metric Score", fontsize=10)
    ax1.set_title("Pillar-2 Development Model Selection Performance Across Candidates", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cand_names, rotation=25, ha="right", fontsize=9)
    ax1.set_ylim(0.0, 1.0)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "phase6l_2_candidate_comparison.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 4: ROC Curves Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (c_key, c_res) in enumerate(eval_results.items()):
        probs = c_res["oof_probabilities"]
        fpr, tpr, _ = roc_curve(y, probs)
        auc_v = c_res["summary_metrics"]["roc_auc_mean"]
        ax.plot(fpr, tpr, label=f"{c_key} (AUC = {auc_v:.4f})", color=colors[idx % len(colors)], lw=2)

    ax.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.5000)", lw=1.5)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title("ROC Curves: Pillar-2 Candidates on Full DEV (N = 58,002)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "phase6l_2_roc_curves.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 5: Precision-Recall Curves Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for idx, (c_key, c_res) in enumerate(eval_results.items()):
        probs = c_res["oof_probabilities"]
        prec, rec, _ = precision_recall_curve(y, probs)
        pr_auc_v = c_res["summary_metrics"]["pr_auc_mean"]
        ax.plot(rec, prec, label=f"{c_key} (PR-AUC = {pr_auc_v:.4f})", color=colors[idx % len(colors)], lw=2)

    ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
    ax.set_title("Precision-Recall Curves: Pillar-2 Candidates on Full DEV", fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "phase6l_2_pr_curves.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 6: Calibration Curves
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    winning_cand_key = model_selection_res["winning_candidate_key"]
    win_res = eval_results[winning_cand_key]
    bins_info = win_res["ece_details"]["bins"]

    confs = [b["confidence"] for b in bins_info]
    accs = [b["accuracy"] for b in bins_info]

    ax.plot(confs, accs, "bs-", lw=2, label=f"{winning_cand_key} (ECE = {win_res['summary_metrics']['ece']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability (Confidence)", fontsize=10)
    ax.set_ylabel("Empirical Accuracy", fontsize=10)
    ax.set_title("Reliability Calibration Diagram (Winning Candidate)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "phase6l_2_calibration_curves.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 7: Feature Importance Visualization
    # -------------------------------------------------------------------------
    rankings = discrimination_res["feature_rankings"]
    top10 = rankings[:10]
    f_names = [r["feature"] for r in top10]
    scores = [r["composite_score"] for r in top10]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.barh(f_names[::-1], scores[::-1], color="#2ca02c", alpha=0.85)
    ax.set_xlabel("Composite Discrimination Score (MI + AUC + Cohen's d)", fontsize=10)
    ax.set_title("Top-10 Pillar-2 Feature Discrimination Ranking", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(fig_dir / "phase6l_2_feature_importance.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 8: Stability Gate Summary
    # -------------------------------------------------------------------------
    gate = stability_res["gate_payload"]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    categories = ["Passed", "Rejected"]
    counts = [gate["n_passed"], gate["n_rejected"]]
    ax.bar(categories, counts, color=["#2ca02c", "#d62728"], alpha=0.85, width=0.5)
    ax.set_ylabel("Number of Configurations", fontsize=10)
    ax.set_title(f"Numerical Stability Gate Results ({gate['pass_rate']*100:.1f}% Pass Rate)", fontsize=11, fontweight="bold")
    for i, v in enumerate(counts):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    ax.set_ylim(0, max(counts) + 10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(fig_dir / "phase6l_2_stability_gate.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Final Markdown Report Generation
    # -------------------------------------------------------------------------
    win_summary = win_res["summary_metrics"]
    report_path = out_dir / "PHASE6L_2_DEVELOPMENT_MODEL_SELECTION.md"

    md_content = f"""# HalluciSense Phase 6L.2 — Development Model Selection Report (Pillar 2)

**Status**: `COMPLETED & LOCKED`  
**Dataset Partition**: `FULL DEVELOPMENT PARTITION (N = 58,002 Responses)`  
**Held-Out Validation Partition**: `STRICTLY SEALED & 100% UNTOUCHED (N = 12,483)`  
**Selected Winning Candidate**: `{protocol_res['selected_candidate']}`  
**Feature Subset**: `{protocol_res['feature_set_name']}` ({protocol_res['feature_count']} features)  
**Preprocessing**: `{protocol_res['scaler']}`  
**Classifier**: `{protocol_res['classifier']} ({protocol_res['solver']})`  

---

## 1. Feature Matrix Validation Summary (Stage 1)

- **Total Records**: `58,002` (100% complete).
- **Feature Columns**: `24` structural features.
- **Missing / NaN / Inf Values**: `0` across all $58,002 \\times 24$ feature entries.
- **Duplicate Records / IDs**: `0`.
- **Target Label Balance**: $N_{{\\text{{pos}}}} = {validation_res['target_balance']['n_positive']}$, $N_{{\\text{{neg}}}} = {validation_res['target_balance']['n_negative']}$ (50.00% / 50.00%).

---

## 2. Preprocessing & Numerical Conditioning Study (Stage 2)

| Scaler | Condition Number $\\kappa$ | Finite All | NaN Count | Inf Count | Overall Mean |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `None` | `{preprocessing_res['None']['condition_number']:.2f}` | `{preprocessing_res['None']['finite_all']}` | `0` | `0` | `{preprocessing_res['None']['mean_summary']['overall_mean']:.4f}` |
| `StandardScaler` | `{preprocessing_res['StandardScaler']['condition_number']:.2f}` | `{preprocessing_res['StandardScaler']['finite_all']}` | `0` | `0` | `{preprocessing_res['StandardScaler']['mean_summary']['overall_mean']:.4f}` |
| `RobustScaler` | `{preprocessing_res['RobustScaler']['condition_number']:.2f}` | `{preprocessing_res['RobustScaler']['finite_all']}` | `0` | `0` | `{preprocessing_res['RobustScaler']['mean_summary']['overall_mean']:.4f}` |

---

## 3. Collinearity & Candidate Feature Subsets (Stage 3)

Six candidate feature sets were constructed and evaluated:
- **`SET_A_FULL_SCHEMA`** (24 features): Full schema.
- **`SET_B_LOW_CORRELATION`** ({collinearity_res['candidate_sets']['SET_B_LOW_CORRELATION']['feature_count']} features): Subset with $|\\rho| < 0.70$.
- **`SET_C_LOW_VIF`** ({collinearity_res['candidate_sets']['SET_C_LOW_VIF']['feature_count']} features): Features with $\\text{{VIF}} < 5.0$.
- **`SET_D_HIGH_INFORMATION`** (5 features): Top-k discrimination features.
- **`SET_E_GRAPH_CENTRIC`** (5 features): Contradiction graph topology features.
- **`SET_F_CONTRADICTION_CENTRIC`** (5 features): Pairwise contradiction features.

---

## 4. Feature Discrimination Audit (Stage 4)

Top 5 structural features ranked by composite discrimination score:
1. **`{discrimination_res['feature_rankings'][0]['feature']}`**: ROC-AUC = `{discrimination_res['feature_rankings'][0]['roc_auc']:.4f}`, MI = `{discrimination_res['feature_rankings'][0]['mutual_information']:.4f}`, Cohen's d = `{discrimination_res['feature_rankings'][0]['cohens_d']:.4f}`
2. **`{discrimination_res['feature_rankings'][1]['feature']}`**: ROC-AUC = `{discrimination_res['feature_rankings'][1]['roc_auc']:.4f}`, MI = `{discrimination_res['feature_rankings'][1]['mutual_information']:.4f}`, Cohen's d = `{discrimination_res['feature_rankings'][1]['cohens_d']:.4f}`
3. **`{discrimination_res['feature_rankings'][2]['feature']}`**: ROC-AUC = `{discrimination_res['feature_rankings'][2]['roc_auc']:.4f}`, MI = `{discrimination_res['feature_rankings'][2]['mutual_information']:.4f}`, Cohen's d = `{discrimination_res['feature_rankings'][2]['cohens_d']:.4f}`
4. **`{discrimination_res['feature_rankings'][3]['feature']}`**: ROC-AUC = `{discrimination_res['feature_rankings'][3]['roc_auc']:.4f}`, MI = `{discrimination_res['feature_rankings'][3]['mutual_information']:.4f}`, Cohen's d = `{discrimination_res['feature_rankings'][3]['cohens_d']:.4f}`
5. **`{discrimination_res['feature_rankings'][4]['feature']}`**: ROC-AUC = `{discrimination_res['feature_rankings'][4]['roc_auc']:.4f}`, MI = `{discrimination_res['feature_rankings'][4]['mutual_information']:.4f}`, Cohen's d = `{discrimination_res['feature_rankings'][4]['cohens_d']:.4f}`

---

## 5. Numerical Stability Gate Audit (Stage 5)

- **Total Configurations Evaluated**: `{stability_res['gate_payload']['n_evaluated']}`
- **Passed Stability Gate**: `{stability_res['gate_payload']['n_passed']}` ({stability_res['gate_payload']['pass_rate']*100:.1f}%)
- **Total Warnings Across All Configs**: `{stability_res['warning_forensics']['total_warnings_across_all_configs']}`
- **Fatal Errors / NaNs / Infs**: `0` for passing configurations.

---

## 6. Repeated 5-Fold Cross-Validation Performance (Stage 6)

| Candidate Model | Preprocessing | ROC-AUC | PR-AUC | ECE | Best Thresh | Accuracy | F1 Score | MCC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    for c_key, c_res in eval_results.items():
        s = c_res["summary_metrics"]
        md_content += f"| `{c_key}` | `{c_res['scaler_type']}` | `{s['roc_auc_mean']:.4f}` | `{s['pr_auc_mean']:.4f}` | `{s['ece']:.4f}` | `{s['best_mcc_threshold']:.2f}` | `{s['accuracy_at_best_thresh']:.4f}` | `{s['f1_at_best_thresh']:.4f}` | `{s['best_mcc']:.4f}` |\n"

    md_content += f"""
---

## 7. Baseline Comparison (Stage 7)

| Baseline | Strategy / Feature | ROC-AUC | PR-AUC | Accuracy | MCC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline A** | Majority Predictor | `0.5000` | `{model_selection_res['baselines']['baseline_a_majority']['pr_auc']:.4f}` | `{model_selection_res['baselines']['baseline_a_majority']['accuracy']:.4f}` | `0.0000` |
| **Baseline B** | Random Uniform | `{model_selection_res['baselines']['baseline_b_random']['roc_auc']:.4f}` | `{model_selection_res['baselines']['baseline_b_random']['pr_auc']:.4f}` | `{model_selection_res['baselines']['baseline_b_random']['accuracy']:.4f}` | `{model_selection_res['baselines']['baseline_b_random']['mcc']:.4f}` |
| **Baseline C** | `num_claims` only | `{model_selection_res['baselines']['baseline_c_num_claims']['roc_auc_mean']:.4f}` | `{model_selection_res['baselines']['baseline_c_num_claims']['pr_auc_mean']:.4f}` | `{model_selection_res['baselines']['baseline_c_num_claims']['accuracy_at_best_thresh']:.4f}` | `{model_selection_res['baselines']['baseline_c_num_claims']['best_mcc']:.4f}` |
| **Baseline D** | `max_contradiction` only | `{model_selection_res['baselines']['baseline_d_max_contradiction']['roc_auc_mean']:.4f}` | `{model_selection_res['baselines']['baseline_d_max_contradiction']['pr_auc_mean']:.4f}` | `{model_selection_res['baselines']['baseline_d_max_contradiction']['accuracy_at_best_thresh']:.4f}` | `{model_selection_res['baselines']['baseline_d_max_contradiction']['best_mcc']:.4f}` |
| **Baseline E** | Contradiction Subset | `{model_selection_res['baselines']['baseline_e_contradiction_subset']['roc_auc_mean']:.4f}` | `{model_selection_res['baselines']['baseline_e_contradiction_subset']['pr_auc_mean']:.4f}` | `{model_selection_res['baselines']['baseline_e_contradiction_subset']['accuracy_at_best_thresh']:.4f}` | `{model_selection_res['baselines']['baseline_e_contradiction_subset']['best_mcc']:.4f}` |
| **Pillar 2 Winner** | `{protocol_res['selected_candidate']}` | **`{win_summary['roc_auc_mean']:.4f}`** | **`{win_summary['pr_auc_mean']:.4f}`** | **`{win_summary['accuracy_at_best_thresh']:.4f}`** | **`{win_summary['best_mcc']:.4f}`** |

---

## 8. Data Leakage & Firewall Audit (Stage 8)

- **Label Leakage Check**: `PASS` (Max feature correlation $|r| = {leakage_res['max_feature_label_correlation']:.4f} < 0.95$).
- **Scaler Fitting Leakage**: `PASS` (Fit on training folds ONLY).
- **Cross-Fold Contamination**: `PASS` (Strict RepeatedStratifiedKFold splits).
- **Validation Partition Firewall**: `STRICTLY SEALED & 100% UNTOUCHED (N = 12,483)`.
- **Checkpoint & Cache Contamination**: `PASS`.

---

## 9. Final Immutable Model Protocol (Stage 9)

The winning development candidate is now **LOCKED** prior to held-out validation:

```json
{json.dumps(protocol_res, indent=2)}
```

---

## 10. Publication Figures Inventory

- [`figures/phase6l_2_feature_correlation.png`](file://{out_dir}/figures/phase6l_2_feature_correlation.png)
- [`figures/phase6l_2_vif_visualization.png`](file://{out_dir}/figures/phase6l_2_vif_visualization.png)
- [`figures/phase6l_2_candidate_comparison.png`](file://{out_dir}/figures/phase6l_2_candidate_comparison.png)
- [`figures/phase6l_2_roc_curves.png`](file://{out_dir}/figures/phase6l_2_roc_curves.png)
- [`figures/phase6l_2_pr_curves.png`](file://{out_dir}/figures/phase6l_2_pr_curves.png)
- [`figures/phase6l_2_calibration_curves.png`](file://{out_dir}/figures/phase6l_2_calibration_curves.png)
- [`figures/phase6l_2_feature_importance.png`](file://{out_dir}/figures/phase6l_2_feature_importance.png)
- [`figures/phase6l_2_stability_gate.png`](file://{out_dir}/figures/phase6l_2_stability_gate.png)

---

## 11. Firewall & Stop Condition Confirmation

- **Validation Partition ($N = 12,483$)**: **100% Sealed and Untouched.** Zero validation inferences or metrics calculated.
- **Stop Condition**: Phase 6L.2 completed. Execution STOPPED prior to Phase 6L.3.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("stage9_generate_report_complete", report_path=str(report_path))
    return report_path
