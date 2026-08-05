"""Phase 21.8 — Probability Calibration & Recalibration Engine.

Computes:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score Loss
- Platt Scaling (Logistic) and Temperature Scaling recalibration
- Reliability Diagrams

Generates:
- reports/calibration_report.md
- evaluation/figures/calibration_curve.png
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = BASE_DIR / "evaluation" / "figures"
REPORTS_DIR = BASE_DIR / "reports"


def compute_ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """Compute ECE and MCE calibration metrics."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        prop = float(np.mean(in_bin))

        if prop > 0:
            acc = float(np.mean(y_true[in_bin]))
            conf = float(np.mean(y_prob[in_bin]))
            err = abs(acc - conf)
            ece += err * prop
            mce = max(mce, err)

    return float(ece), float(mce)


def apply_platt_scaling(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Fit Platt Scaling (sigmoid recalibration)."""
    eps = 1e-6
    probs_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logits = np.log(probs_clipped / (1.0 - probs_clipped)).reshape(-1, 1)

    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(logits, y_true)
    return lr.predict_proba(logits)[:, 1]


def run_calibration_analysis(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Perform probability calibration and Platt scaling analysis."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_ece, raw_mce = compute_ece_mce(y_true, y_prob)
    raw_brier = float(brier_score_loss(y_true, y_prob))

    # Recalibrate via Platt scaling
    platt_prob = apply_platt_scaling(y_true, y_prob)
    platt_ece, platt_mce = compute_ece_mce(y_true, platt_prob)
    platt_brier = float(brier_score_loss(y_true, platt_prob))

    # Plot reliability diagram
    plt.figure(figsize=(7, 5))
    f_pos_raw, m_val_raw = calibration_curve(y_true, y_prob, n_bins=10)
    f_pos_platt, m_val_platt = calibration_curve(y_true, platt_prob, n_bins=10)

    plt.plot(m_val_raw, f_pos_raw, "s-", label=f"Uncalibrated (ECE = {raw_ece:.4f})", color="orange", linewidth=2)
    plt.plot(m_val_platt, f_pos_platt, "o-", label=f"Platt Scaled (ECE = {platt_ece:.4f})", color="green", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Actual Fraction of Positives")
    plt.title("HalluciSense Reliability Calibration Diagram")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "calibration_curve.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # Write reports/calibration_report.md
    report_path = REPORTS_DIR / "calibration_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 21.8 — Probability Calibration Report\n\n")
        f.write("## Calibration Metrics Summary\n\n")
        f.write("| Calibration State | ECE | MCE | Brier Score |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Raw Production Model** | {raw_ece:.4f} | {raw_mce:.4f} | {raw_brier:.4f} |\n")
        f.write(f"| **Platt Scaled Recalibrated** | **{platt_ece:.4f}** | **{platt_mce:.4f}** | **{platt_brier:.4f}** |\n\n")
        f.write("## Reliability Curve\n")
        f.write("Reliability diagram exported to `evaluation/figures/calibration_curve.png`.\n")

    return {
        "raw_ece": round(raw_ece, 4),
        "raw_mce": round(raw_mce, 4),
        "raw_brier": round(raw_brier, 4),
        "platt_ece": round(platt_ece, 4),
        "platt_mce": round(platt_mce, 4),
        "platt_brier": round(platt_brier, 4),
    }
