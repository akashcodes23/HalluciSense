"""
HalluciSense Phase 9 — Step 6: Research Deliverables (Publication Figures)
============================================================================
Generates all publication-quality figures: ROC, PR, confusion matrix,
threshold analysis, coefficient table, statistical comparison table.

FROZEN FIREWALL: No models, scalers, or thresholds are modified.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import (
    auc, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_curve,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PHASE6K = ROOT / "evaluation_results" / "phase6k"
FINAL_MODEL_DIR = PHASE6K / "final_model"
PHASE6I = ROOT / "evaluation_results" / "phase6i"
PUB = PHASE6K / "publication"
FIG_DIR = PUB / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "mean_entailment",
    "max_entailment",
    "mean_contradiction",
    "min_support_margin",
    "num_claims",
]
OPERATING_THRESHOLD = 0.56
DPI = 300
# IEEE-compatible color palette
MAIN_BLUE = "#1f77b4"
MAIN_RED = "#d62728"
MAIN_GREEN = "#2ca02c"
MAIN_ORANGE = "#ff7f0e"
MAIN_PURPLE = "#9467bd"
GRAY = "#7f7f7f"


def load_val():
    rows, labels = [], []
    with open(PHASE6I / "claim_evidence_features_validation.jsonl") as f:
        for line in f:
            obj = json.loads(line.strip()) if line.strip() else {}
            rows.append([obj.get(fn, float("nan")) for fn in FEATURE_NAMES])
            labels.append(int(obj.get("ground_truth", 0)))
    return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int32)


def run() -> None:
    print("=" * 70)
    print("HalluciSense Phase 9 — Step 6: Research Deliverables")
    print("=" * 70)
    t0 = time.time()

    model = joblib.load(FINAL_MODEL_DIR / "pillar1_logistic_model.joblib")
    scaler = joblib.load(FINAL_MODEL_DIR / "robust_scaler.joblib")

    X_val, y_val = load_val()
    X_val_scaled = scaler.transform(X_val)
    probs = model.predict_proba(X_val_scaled)[:, 1]
    preds_56 = (probs >= OPERATING_THRESHOLD).astype(int)
    preds_50 = (probs >= 0.50).astype(int)

    # Baselines
    random_probs = np.full(len(y_val), y_val.mean())
    single_feat_probs = (X_val[:, 3] - X_val[:, 3].min()) / (
        X_val[:, 3].max() - X_val[:, 3].min() + 1e-9
    )
    # Negate because min_support_margin is negatively correlated with hallucination
    single_feat_probs = 1.0 - single_feat_probs

    plt.style.use("seaborn-v0_8-whitegrid")
    figures_generated = []

    # ── Figure 1: ROC Curve ────────────────────────────────────────────────────
    print("\n[1/9] ROC Curve...")
    fpr, tpr, thresholds_roc = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)

    fpr_rand, tpr_rand, _ = roc_curve(y_val, random_probs)
    fpr_sfp, tpr_sfp, _ = roc_curve(y_val, single_feat_probs)
    auc_rand = auc(fpr_rand, tpr_rand)
    auc_sfp = auc(fpr_sfp, tpr_sfp)

    # Operating point
    op_idx = np.argmin(np.abs(thresholds_roc - OPERATING_THRESHOLD))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=MAIN_BLUE, linewidth=2.5,
            label=f"Pillar-1 (AUC = {roc_auc:.4f})")
    ax.plot(fpr_sfp, tpr_sfp, color=MAIN_ORANGE, linewidth=1.8, linestyle="-.",
            label=f"Single Feature Baseline (AUC = {auc_sfp:.4f})")
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1.2, linestyle="--",
            label=f"Random Chance (AUC = 0.50)")
    ax.scatter([fpr[op_idx]], [tpr[op_idx]], color=MAIN_RED, s=100, zorder=5,
               label=f"Op. Threshold={OPERATING_THRESHOLD}")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic — Pillar-1 (VAL)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step6_roc_curve.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    figures_generated.append("step6_roc_curve.png")
    print(f"  ROC-AUC = {roc_auc:.4f}")

    # ── Figure 2: PR Curve ────────────────────────────────────────────────────
    print("\n[2/9] PR Curve...")
    prec, rec, thresholds_pr = precision_recall_curve(y_val, probs)
    pr_auc = auc(rec, prec)
    prec_sfp, rec_sfp, _ = precision_recall_curve(y_val, single_feat_probs)
    auc_sfp_pr = auc(rec_sfp, prec_sfp)

    op_pr_idx = np.argmin(np.abs(thresholds_pr - OPERATING_THRESHOLD))
    baseline_pr = float(y_val.mean())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color=MAIN_BLUE, linewidth=2.5,
            label=f"Pillar-1 (AP = {pr_auc:.4f})")
    ax.plot(rec_sfp, prec_sfp, color=MAIN_ORANGE, linewidth=1.8, linestyle="-.",
            label=f"Single Feature Baseline (AP = {auc_sfp_pr:.4f})")
    ax.axhline(baseline_pr, color=GRAY, linestyle="--", linewidth=1.2,
               label=f"No-Skill Baseline ({baseline_pr:.3f})")
    ax.scatter([rec[op_pr_idx]], [prec[op_pr_idx]], color=MAIN_RED, s=100, zorder=5,
               label=f"Op. Threshold={OPERATING_THRESHOLD}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Pillar-1 (VAL)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step6_pr_curve.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    figures_generated.append("step6_pr_curve.png")

    # ── Figure 3: Confusion Matrix (normalized) ───────────────────────────────
    print("\n[3/9] Confusion matrix...")
    cm = confusion_matrix(y_val, preds_56)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title_sfx, fmt in [
        (axes[0], cm_norm, "Normalized", ".3f"),
        (axes[1], cm, "Raw Counts", "d"),
    ]:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if fmt == ".3f" else None)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Predicted\nGrounded", "Predicted\nHallucinated"], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True\nGrounded", "True\nHallucinated"], fontsize=10)
        ax.set_title(f"Confusion Matrix ({title_sfx})\n@ threshold={OPERATING_THRESHOLD}", fontsize=11)
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                text = f"{val:{fmt}}" if fmt != "d" else str(val)
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=13, color="white" if mat[i, j] > (mat.max() * 0.6) else "black",
                        fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step6_confusion_matrix.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    figures_generated.append("step6_confusion_matrix.png")

    # ── Figure 4: Threshold Analysis ──────────────────────────────────────────
    print("\n[4/9] Threshold analysis...")
    thresholds_grid = np.linspace(0.05, 0.95, 200)
    f1_vals, mcc_vals, prec_vals, rec_vals, spec_vals = [], [], [], [], []

    for thr in thresholds_grid:
        p = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, p, labels=[0, 1]).ravel()
        prec_t = tp / (tp + fp + 1e-9)
        rec_t = tp / (tp + fn + 1e-9)
        spec_t = tn / (tn + fp + 1e-9)
        f1_t = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-9)
        mcc_t = float(matthews_corrcoef(y_val, p))
        f1_vals.append(f1_t)
        mcc_vals.append(mcc_t)
        prec_vals.append(prec_t)
        rec_vals.append(rec_t)
        spec_vals.append(spec_t)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    ax1.plot(thresholds_grid, f1_vals, color=MAIN_BLUE, linewidth=2, label="F1")
    ax1.plot(thresholds_grid, prec_vals, color=MAIN_GREEN, linewidth=1.8,
             linestyle="--", label="Precision")
    ax1.plot(thresholds_grid, rec_vals, color=MAIN_RED, linewidth=1.8,
             linestyle="-.", label="Recall")
    ax1.plot(thresholds_grid, spec_vals, color=MAIN_ORANGE, linewidth=1.8,
             linestyle=":", label="Specificity")
    ax1.axvline(OPERATING_THRESHOLD, color="black", linewidth=1.5, linestyle="--",
                label=f"Op. Threshold={OPERATING_THRESHOLD}")
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Threshold Analysis — F1, Precision, Recall, Specificity", fontsize=12)
    ax1.legend(fontsize=9, loc="center left")
    ax1.set_xlim(0.05, 0.95)

    ax2.plot(thresholds_grid, mcc_vals, color=MAIN_PURPLE, linewidth=2, label="MCC")
    ax2.axvline(OPERATING_THRESHOLD, color="black", linewidth=1.5, linestyle="--",
                label=f"Op. Threshold={OPERATING_THRESHOLD}")
    ax2.axhline(0, color=GRAY, linewidth=0.8)
    ax2.set_xlabel("Decision Threshold", fontsize=11)
    ax2.set_ylabel("MCC", fontsize=11)
    ax2.set_title("Threshold Analysis — Matthews Correlation Coefficient", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0.05, 0.95)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step6_threshold_analysis.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    figures_generated.append("step6_threshold_analysis.png")

    # ── Figure 5: Coefficient table as figure ────────────────────────────────
    print("\n[5/9] Coefficient table figure...")
    coef = model.coef_[0]
    iqr_scales = scaler.scale_
    std_coefs = coef * iqr_scales
    or_vals = np.exp(coef)

    table_data = []
    for fn, c, sc, orv in zip(FEATURE_NAMES, coef, std_coefs, or_vals):
        table_data.append([
            fn.replace("_", " ").title(),
            f"{c:+.4f}",
            f"{sc:+.4f}",
            f"{orv:.4f}",
            "Hallucination" if c < 0 else "Grounded",
        ])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=["Feature", "Coefficient", "Std. Coef (IQR)", "Odds Ratio", "Direction"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    # Style header
    for j in range(5):
        table[(0, j)].set_facecolor("#2c7bb6")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    # Alternating row colors
    for i in range(1, len(table_data) + 1):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(5):
            table[(i, j)].set_facecolor(bg)
    ax.set_title("Table 1: Pillar-1 Logistic Regression Coefficients (VAL)",
                 fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step6_coefficient_table.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    figures_generated.append("step6_coefficient_table.png")

    # ── Figure 6: Statistical comparison table ────────────────────────────────
    print("\n[6/9] Statistical comparison table...")
    from sklearn.metrics import roc_auc_score, average_precision_score
    metrics_pillar1 = {
        "ROC-AUC": float(roc_auc_score(y_val, probs)),
        "PR-AUC (AP)": float(average_precision_score(y_val, probs)),
        "F1 (@0.56)": float(f1_score(y_val, preds_56)),
        "MCC (@0.56)": float(matthews_corrcoef(y_val, preds_56)),
        "Brier Score": float(np.mean((probs - y_val) ** 2)),
    }
    metrics_random = {
        "ROC-AUC": 0.5,
        "PR-AUC (AP)": float(y_val.mean()),
        "F1 (@0.56)": float(f1_score(y_val, np.ones(len(y_val), dtype=int))),
        "MCC (@0.56)": 0.0,
        "Brier Score": float(y_val.mean() * (1 - y_val.mean())),
    }
    metrics_sfp = {
        "ROC-AUC": float(roc_auc_score(y_val, single_feat_probs)),
        "PR-AUC (AP)": float(average_precision_score(y_val, single_feat_probs)),
        "F1 (@0.56)": float(f1_score(y_val, (single_feat_probs >= 0.5).astype(int))),
        "MCC (@0.56)": float(matthews_corrcoef(y_val, (single_feat_probs >= 0.5).astype(int))),
        "Brier Score": float(np.mean((single_feat_probs - y_val) ** 2)),
    }

    stat_table_data = []
    for metric in ["ROC-AUC", "PR-AUC (AP)", "F1 (@0.56)", "MCC (@0.56)", "Brier Score"]:
        stat_table_data.append([
            metric,
            f"{metrics_pillar1[metric]:.4f}",
            f"{metrics_sfp[metric]:.4f}",
            f"{metrics_random[metric]:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    table2 = ax.table(
        cellText=stat_table_data,
        colLabels=["Metric", "Pillar-1", "Single Feature Baseline", "Random Chance"],
        cellLoc="center",
        loc="center",
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.5)
    for j in range(4):
        table2[(0, j)].set_facecolor("#1a9641")
        table2[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(stat_table_data) + 1):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(4):
            table2[(i, j)].set_facecolor(bg)
        # Highlight Pillar-1 column
        table2[(i, 1)].set_facecolor("#d0e8ff")
    ax.set_title("Table 2: Statistical Comparison — Pillar-1 vs Baselines (VAL)",
                 fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step6_statistical_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    figures_generated.append("step6_statistical_comparison.png")

    # ── Figure 7: ROC + PR combined ───────────────────────────────────────────
    print("\n[7/9] Combined ROC + PR figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(fpr, tpr, color=MAIN_BLUE, linewidth=2.5,
             label=f"Pillar-1 (AUC={roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color=GRAY, linewidth=1.2, linestyle="--", label="Random")
    ax1.scatter([fpr[op_idx]], [tpr[op_idx]], color=MAIN_RED, s=100, zorder=5,
                label=f"τ={OPERATING_THRESHOLD}")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curve", fontsize=12)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_aspect("equal")

    ax2.plot(rec, prec, color=MAIN_BLUE, linewidth=2.5,
             label=f"Pillar-1 (AP={pr_auc:.4f})")
    ax2.axhline(baseline_pr, color=GRAY, linestyle="--", linewidth=1.2,
                label=f"No-skill ({baseline_pr:.3f})")
    ax2.scatter([rec[op_pr_idx]], [prec[op_pr_idx]], color=MAIN_RED, s=100, zorder=5,
                label=f"τ={OPERATING_THRESHOLD}")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=12)
    ax2.legend(fontsize=9)
    fig.suptitle("HalluciSense Pillar-1 — Validation Set Performance", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step6_roc_pr_combined.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    figures_generated.append("step6_roc_pr_combined.png")

    # ── Figure 8: Full metrics summary bar chart ──────────────────────────────
    print("\n[8/9] Metrics summary figure...")
    metrics_bar = {
        "ROC-AUC": (metrics_pillar1["ROC-AUC"], metrics_sfp["ROC-AUC"]),
        "PR-AUC": (metrics_pillar1["PR-AUC (AP)"], metrics_sfp["PR-AUC (AP)"]),
        "F1": (metrics_pillar1["F1 (@0.56)"], metrics_sfp["F1 (@0.56)"]),
        "MCC": (metrics_pillar1["MCC (@0.56)"], metrics_sfp["MCC (@0.56)"]),
    }
    metrics_names = list(metrics_bar.keys())
    pillar1_vals = [metrics_bar[m][0] for m in metrics_names]
    sfp_vals = [metrics_bar[m][1] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, pillar1_vals, width, label="Pillar-1 (5-feature LogReg)",
           color=MAIN_BLUE, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, sfp_vals, width, label="Single-feature Baseline",
           color=MAIN_ORANGE, alpha=0.85, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(pillar1_vals):
        ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(sfp_vals):
        ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("HalluciSense Pillar-1: Key Metrics vs Baseline (VAL)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step6_metrics_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    figures_generated.append("step6_metrics_summary.png")

    # ── Write LaTeX coefficient table ─────────────────────────────────────────
    print("\n[9/9] Writing LaTeX coefficient table...")
    latex_lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Pillar-1 Logistic Regression Coefficients}",
        r"  \label{tab:pillar1_coeff}",
        r"  \begin{tabular}{lrrrr}",
        r"    \hline",
        r"    \textbf{Feature} & \textbf{Coef} & \textbf{Std.Coef} & \textbf{OR} & \textbf{Direction} \\",
        r"    \hline",
    ]
    for fn, c, sc, orv in zip(FEATURE_NAMES, coef, std_coefs, or_vals):
        direction = "Hallucination" if c < 0 else "Grounded"
        latex_lines.append(
            f"    {fn.replace('_', ' ').title()} & {c:+.4f} & {sc:+.4f} & "
            f"{orv:.4f} & {direction} \\\\"
        )
    latex_lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    latex_out = PUB / "step6_coefficient_table.tex"
    with open(latex_out, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"  LaTeX → {latex_out}")

    elapsed = time.time() - t0

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "9_step6_research_figures",
        "metrics_pillar1": metrics_pillar1,
        "metrics_single_feature_baseline": metrics_sfp,
        "metrics_random": metrics_random,
        "figures_generated": figures_generated,
        "latex_table": str(latex_out),
        "elapsed_seconds": round(elapsed, 2),
    }

    json_out = PUB / "step6_research_figures.json"
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON → {json_out}")

    md_lines = [
        "# Phase 9 — Step 6: Research Deliverables",
        "",
        f"**Generated**: {report['generated_at_utc']}",
        "",
        "## Performance Summary (VAL — 3,500 samples)",
        "",
        "| Metric | Pillar-1 | Single Feature | Random Chance |",
        "| --- | --- | --- | --- |",
    ]
    for metric in ["ROC-AUC", "PR-AUC (AP)", "F1 (@0.56)", "MCC (@0.56)", "Brier Score"]:
        md_lines.append(
            f"| {metric} | **{metrics_pillar1[metric]:.4f}** | "
            f"{metrics_sfp[metric]:.4f} | {metrics_random[metric]:.4f} |"
        )

    md_lines += [
        "",
        "## Figures Generated (300 DPI)",
        "",
        "| File | Description |",
        "| --- | --- |",
        "| `step6_roc_curve.png` | ROC curve with operating point and baselines |",
        "| `step6_pr_curve.png` | PR curve with operating point and no-skill baseline |",
        "| `step6_confusion_matrix.png` | Normalized + raw confusion matrices |",
        "| `step6_threshold_analysis.png` | F1/MCC/Precision/Recall vs threshold |",
        "| `step6_coefficient_table.png` | Publication-quality coefficient table |",
        "| `step6_statistical_comparison.png` | Statistical comparison table |",
        "| `step6_roc_pr_combined.png` | Combined ROC + PR panel |",
        "| `step6_metrics_summary.png` | Key metrics bar chart vs baseline |",
        "",
        "## LaTeX Table",
        "",
        "Coefficient table exported to `step6_coefficient_table.tex` for direct inclusion in IEEE manuscript.",
    ]

    md_out = PUB / "step6_research_figures.md"
    with open(md_out, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  MD  → {md_out}")

    print(f"\n✅ Step 6 complete in {elapsed:.1f}s")
    print(f"   Generated {len(figures_generated)} figures + LaTeX table")


if __name__ == "__main__":
    run()
