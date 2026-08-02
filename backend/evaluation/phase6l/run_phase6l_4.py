"""Master Orchestrator for HalluciSense Phase 6L.4: Root Cause Analysis of Pillar-2 Generalization Failure.

Executes Stages 1 through 9:
1. Feature Distribution Shift Decomposition
2. Pairwise NLI Score Drift Analysis
3. Structural Complexity Analysis
4. Detector Activation Audit
5. Feature Stability Audit
6. Probability Compression Mechanics
7. Error Cluster Analysis
8. Root Cause Hierarchy Synthesis
9. Publication Figure & Report Generation

Strict Scientific Rule:
    * DIAGNOSTIC ONLY — 100% Read-Only.
    * NO model retraining, threshold tuning, feature engineering, preprocessing changes,
      hyperparameter optimization, classifier changes, or hybrid fusion.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import structlog

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS
from evaluation.phase6l.dataset import load_and_validate_full_dev_matrix
from evaluation.phase6l.val_feature_reconstruction import load_val_structural_features
from evaluation.phase6l.root_cause_analysis import (
    LOCKED_FEATURE_NAMES,
    PRIMARY_THRESHOLD,
    decompose_feature_distribution_shift,
    analyze_pairwise_nli_score_drift,
    analyze_structural_complexity,
    analyze_detector_activations,
    analyze_feature_stability,
    analyze_probability_compression,
    analyze_error_clusters,
    synthesize_root_cause,
    generate_publication_figures,
    generate_root_cause_markdown_report,
)

logger = structlog.get_logger(__name__)


def run_phase6l_4(out_dir: Path = PHASE6L_DIR) -> Dict[str, Any]:
    """Execute Phase 6L.4 master orchestrator pipeline."""
    start_time = time.time()
    logger.info("phase6l_4_orchestrator_start")

    print("\n" + "=" * 85)
    print("HalluciSense Phase 6L.4 — Root Cause Analysis of Pillar-2 Generalization Failure")
    print("=" * 85)

    # 1. Load DEV and VAL datasets
    print("=== Loading DEV ($N=58,002$) and VAL ($N=12,483$) Feature Matrices ===")
    dev_data = load_and_validate_full_dev_matrix(out_dir=out_dir)
    X_dev, y_dev = dev_data["X"], dev_data["y"]

    val_data = load_val_structural_features()
    X_val, y_val = val_data["X"], val_data["y"]

    # 2. Load Frozen Scaler and Classifier
    model_dir = out_dir / "final_model"
    scaler = joblib.load(model_dir / "preprocessing.joblib")
    clf = joblib.load(model_dir / "classifier.joblib")
    print(f"  Loaded frozen model: {type(clf).__name__} with {scaler.__class__.__name__}")

    # Stage 1: Feature Distribution Shift Decomposition
    print("\n=== Stage 1: Feature Distribution Shift Decomposition ===")
    shift_data = decompose_feature_distribution_shift(X_dev, X_val, out_dir=out_dir)

    # Stage 2: Pairwise NLI Score Drift Analysis
    print("\n=== Stage 2: Pairwise NLI Score Drift Analysis ===")
    nli_drift = analyze_pairwise_nli_score_drift(X_dev, X_val, out_dir=out_dir)

    # Stage 3: Structural Complexity Analysis
    print("\n=== Stage 3: Structural Complexity Analysis ===")
    complexity = analyze_structural_complexity(X_dev, X_val)

    # Stage 4: Detector Activation Audit
    print("\n=== Stage 4: Detector Activation & Dormancy Audit ===")
    activation = analyze_detector_activations(X_dev, X_val, out_dir=out_dir)

    # Stage 5: Feature Stability Audit
    print("\n=== Stage 5: Feature Contribution & Importance Stability ===")
    stability = analyze_feature_stability(X_dev, y_dev, X_val, y_val, scaler, clf, out_dir=out_dir)

    # Stage 6: Probability Compression Mechanics
    print("\n=== Stage 6: Probability Compression Mechanics ===")
    prob_comp = analyze_probability_compression(X_dev, y_dev, X_val, y_val, scaler, clf, out_dir=out_dir)

    # Stage 7: Error Cluster Analysis
    print("\n=== Stage 7: Error Cluster Analysis ===")
    error_clusters = analyze_error_clusters(X_val, y_val, scaler, clf, out_dir=out_dir)

    # Stage 8: Root Cause Hierarchy Synthesis
    print("\n=== Stage 8: Root Cause Hierarchy Synthesis ===")
    hierarchy = synthesize_root_cause(shift_data, nli_drift, prob_comp, out_dir=out_dir)

    # Stage 9: Publication Figures & Final Markdown Report
    print("\n=== Stage 9: Publication Figures & Final Markdown Report Generation ===")
    fig_paths = generate_publication_figures(X_dev, X_val, y_val, scaler, clf, shift_data, prob_comp, out_dir=out_dir)
    report_path = generate_root_cause_markdown_report(
        shift_data=shift_data,
        nli_drift=nli_drift,
        complexity=complexity,
        activation=activation,
        stability=stability,
        prob_comp=prob_comp,
        error_clusters=error_clusters,
        hierarchy=hierarchy,
        out_dir=out_dir,
    )

    total_time = time.time() - start_time
    print("\n" + "=" * 85)
    print(f"Phase 6L.4 Execution Completed Successfully — {total_time:.2f}s elapsed")
    print(f"Primary Root Cause : {hierarchy['primary_root_cause']['title']}")
    print(f"Report Generated   : {report_path}")
    print(f"Figures Generated  : {len(fig_paths)} 300 DPI figures")
    print("Firewall Status    : READ-ONLY DIAGNOSTIC COMPLETE — ZERO MODEL RETRAINING")
    print("=" * 85 + "\n")

    logger.info("phase6l_4_orchestrator_complete", elapsed_s=round(total_time, 2))
    return {
        "shift_data": shift_data,
        "nli_drift": nli_drift,
        "complexity": complexity,
        "activation": activation,
        "stability": stability,
        "prob_comp": prob_comp,
        "error_clusters": error_clusters,
        "hierarchy": hierarchy,
        "report_path": str(report_path),
    }


def main():
    _ = run_phase6l_4()


if __name__ == "__main__":
    main()
