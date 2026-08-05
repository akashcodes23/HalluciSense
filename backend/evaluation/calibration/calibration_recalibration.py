"""Phase 22.7 — Probability Calibration & Recalibration Comparison Engine.

Compares:
- Uncalibrated Production Model
- Platt Scaling (Logistic Sigmoid)
- Temperature Scaling
- Isotonic Regression

Computes before vs after:
ECE, MCE, Brier Score, and exports Reliability Diagrams.
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
from sklearn.isotonic import IsotonicRegression

from evaluation.phase14.evaluator import compute_ece

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = BASE_DIR / "evaluation" / "figures"
REPORTS_DIR = BASE_DIR / "reports"


def apply_temperature_scaling(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, float]:
    """Optimize Temperature T > 0 for logit scaling logit / T."""
    eps = 1e-6
    probs_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logits = np.log(probs_clipped / (1.0 - probs_clipped))

    best_t = 1.0
    best_loss = 1e9

    for t in np.linspace(0.1, 5.0, 50):
        scaled_probs = 1.0 / (1.0 + np.exp(-logits / t))
        brier = float(brier_score_loss(y_true, scaled_probs))
        if brier < best_loss:
            best_loss = brier
            best_t = float(t)

    scaled_probs = 1.0 / (1.0 + np.exp(-logits / best_t))
    return scaled_probs, best_t


def run_recalibration_suite(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Run full calibration and recalibration comparative analysis."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Uncalibrated
    raw_ece, raw_mce = compute_ece(y_true, y_prob)
    raw_brier = float(brier_score_loss(y_true, y_prob))

    # 2. Platt Scaling
    eps = 1e-6
    probs_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logits = np.log(probs_clipped / (1.0 - probs_clipped)).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(logits, y_true)
    platt_prob = lr.predict_proba(logits)[:, 1]

    platt_ece, platt_mce = compute_ece(y_true, platt_prob)
    platt_brier = float(brier_score_loss(y_true, platt_prob))

    # 3. Temperature Scaling
    temp_prob, opt_temp = apply_temperature_scaling(y_true, y_prob)
    temp_ece, temp_mce = compute_ece(y_true, temp_prob)
    temp_brier = float(brier_score_loss(y_true, temp_prob))

    # 4. Isotonic Regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso_prob = iso.fit_transform(y_prob, y_true)
    iso_ece, iso_mce = compute_ece(y_true, iso_prob)
    iso_brier = float(brier_score_loss(y_true, iso_prob))

    # Plot Reliability Curve
    plt.figure(figsize=(7, 5))
    f_raw, m_raw = calibration_curve(y_true, y_prob, n_bins=10)
    f_platt, m_platt = calibration_curve(y_true, platt_prob, n_bins=10)
    f_temp, m_temp = calibration_curve(y_true, temp_prob, n_bins=10)
    f_iso, m_iso = calibration_curve(y_true, iso_prob, n_bins=10)

    plt.plot(m_raw, f_raw, "s-", label=f"Uncalibrated (ECE = {raw_ece:.4f})", color="orange", linewidth=1.5)
    plt.plot(m_platt, f_platt, "o-", label=f"Platt Scaling (ECE = {platt_ece:.4f})", color="green", linewidth=2)
    plt.plot(m_temp, f_temp, "^-", label=f"Temp Scaling (T={opt_temp:.2f}, ECE = {temp_ece:.4f})", color="blue", linewidth=1.5)
    plt.plot(m_iso, f_iso, "d-", label=f"Isotonic Regression (ECE = {iso_ece:.4f})", color="purple", linewidth=1.5)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.title("Probability Calibration & Recalibration Comparison")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=300)
    plt.close()

    # Write reports/calibration_report.md
    with open(REPORTS_DIR / "calibration_report.md", "w", encoding="utf-8") as f:
        f.write("# Phase 22.7 — Probability Calibration & Recalibration Report\n\n")
        f.write("## Recalibration Performance Comparison\n\n")
        f.write("| Recalibration Technique | ECE | MCE | Brier Score | Details |\n")
        f.write("| :--- | :---: | :---: | :---: | :--- |\n")
        f.write(f"| **Uncalibrated Production** | {raw_ece:.4f} | {raw_mce:.4f} | {raw_brier:.4f} | Baseline model |\n")
        f.write(f"| **Platt Scaling (Sigmoid)** | **{platt_ece:.4f}** | **{platt_mce:.4f}** | **{platt_brier:.4f}** | Logit Logistic Regression |\n")
        f.write(f"| **Temperature Scaling** | {temp_ece:.4f} | {temp_mce:.4f} | {temp_brier:.4f} | Optimal Temperature T = {opt_temp:.2f} |\n")
        f.write(f"| **Isotonic Regression** | {iso_ece:.4f} | {iso_mce:.4f} | {iso_brier:.4f} | Non-parametric step function |\n")

    return {
        "raw": {"ece": raw_ece, "mce": raw_mce, "brier": raw_brier},
        "platt": {"ece": platt_ece, "mce": platt_mce, "brier": platt_brier},
        "temperature": {"ece": temp_ece, "mce": temp_mce, "brier": temp_brier, "opt_t": opt_temp},
        "isotonic": {"ece": iso_ece, "mce": iso_mce, "brier": iso_brier},
    }
