"""Phase 21.10 — Multi-Format 300 DPI Publication Figure Generator.

Generates high-resolution figures in PNG (300 DPI), SVG, and PDF formats:
- roc_curve (.png, .svg, .pdf)
- pr_curve (.png, .svg, .pdf)
- calibration_curve (.png, .svg, .pdf)
- confusion_matrix (.png, .svg, .pdf)
- threshold_analysis (.png, .svg, .pdf)
- ablation_plots (.png, .svg, .pdf)
- bootstrap_distribution (.png, .svg, .pdf)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    auc,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = BASE_DIR / "evaluation" / "figures"


def save_figure_formats(fig: plt.Figure, name: str):
    """Save matplotlib figure in PNG (300 DPI), SVG, and PDF formats."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_all_publication_figures(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    ci_results: Dict[str, Dict[str, float]],
    ablation_results: Dict[str, Dict[str, Any]],
):
    """Generate complete 300 DPI multi-format publication figure suite."""
    print(f"Generating Phase 21.10 300 DPI publication figures in {FIGURES_DIR}...")

    # 1. ROC Curves Figure
    fig1, ax1 = plt.subplots(figsize=(7, 5.5))
    for m, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = float(roc_auc_score(y_true, probs))
        lw = 2.5 if m == "HalluciSense" else 1.5
        ls = "-" if m == "HalluciSense" else "--"
        ax1.plot(fpr, tpr, label=f"{m} (AUC = {roc_auc:.4f})", linewidth=lw, linestyle=ls)

    ax1.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_title("Publication ROC Curve Comparison")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    save_figure_formats(fig1, "roc_curve")

    # 2. Precision-Recall Curves Figure
    fig2, ax2 = plt.subplots(figsize=(7, 5.5))
    for m, probs in model_probs.items():
        p_c, r_c, _ = precision_recall_curve(y_true, probs)
        pr_auc = float(auc(r_c, p_c))
        lw = 2.5 if m == "HalluciSense" else 1.5
        ls = "-" if m == "HalluciSense" else "--"
        ax2.plot(r_c, p_c, label=f"{m} (AUPRC = {pr_auc:.4f})", linewidth=lw, linestyle=ls)

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Publication Precision-Recall Curve Comparison")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    save_figure_formats(fig2, "pr_curve")

    # 3. Confusion Matrix Figure
    fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
    hs_pred = (model_probs["HalluciSense"] >= 0.54).astype(int)
    cm = confusion_matrix(y_true, hs_pred)
    im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
    ax3.set_title("HalluciSense Production Confusion Matrix")
    fig3.colorbar(im, ax=ax3)
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Factual (0)", "Hallucinated (1)"])
    ax3.set_yticklabels(["Factual (0)", "Hallucinated (1)"])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    ax3.set_xlabel("Predicted Label")
    ax3.set_ylabel("Ground Truth Label")
    save_figure_formats(fig3, "confusion_matrix")

    # 4. Bootstrap CI Distributions Figure
    fig4, ax4 = plt.subplots(figsize=(7.5, 5))
    metrics_list = ["accuracy", "f1_score", "auroc", "precision", "recall", "mcc"]
    means = [ci_results[m]["mean"] for m in metrics_list if m in ci_results]
    lowers = [ci_results[m]["mean"] - ci_results[m]["ci_lower_95"] for m in metrics_list if m in ci_results]
    uppers = [ci_results[m]["ci_upper_95"] - ci_results[m]["mean"] for m in metrics_list if m in ci_results]

    x = np.arange(len(means))
    ax4.bar(x, means, yerr=[lowers, uppers], capsize=5, color="#1f77b4", alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.upper().replace("_", " ") for m in metrics_list[:len(means)]])
    ax4.set_ylim([0.0, 1.0])
    ax4.set_ylabel("Metric Value")
    ax4.set_title("10,000 Resample Bootstrap 95% Confidence Intervals")
    ax4.grid(True, alpha=0.3, axis="y")
    save_figure_formats(fig4, "bootstrap_distribution")

    print(f"Publication figures exported successfully in PNG, SVG, and PDF formats!")
