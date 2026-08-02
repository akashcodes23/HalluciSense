"""Master Orchestrator for HalluciSense Phase 6L.2: Development Model Selection for Pillar-2 Structural Consistency.

Executes Stages 1 through 9:
1. Feature Matrix Validation
2. Preprocessing Study
3. Collinearity Analysis
4. Feature Discrimination Audit
5. Numerical Stability Gate
6. Full DEV Model Selection (Repeated Stratified 5-Fold CV)
7. Baseline Comparison
8. Data Leakage & Firewall Audit
9. Protocol Lock & Report Generation

Strict Data Firewall Rule:
    * DEV partition ONLY (N = 58,002).
    * HELD-OUT VAL partition (N = 12,483) is 100% SEALED and UNTOUCHED.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

import structlog

from evaluation.phase6l.config import PHASE6L_DIR
from evaluation.phase6l.dataset import load_and_validate_full_dev_matrix
from evaluation.phase6l.collinearity import evaluate_preprocessing_scalers, run_collinearity_analysis
from evaluation.phase6l.discrimination import run_feature_discrimination_audit
from evaluation.phase6l.stability_gate import run_numerical_stability_gate
from evaluation.phase6l.model_selection import run_development_model_selection
from evaluation.phase6l.leakage import run_data_leakage_audit
from evaluation.phase6l.protocol import export_final_model_protocol
from evaluation.phase6l.report_phase6l_2 import generate_publication_figures_and_report

logger = structlog.get_logger(__name__)


def run_phase6l_2(out_dir: Path = PHASE6L_DIR) -> Dict[str, Any]:
    """Execute Phase 6L.2 master orchestrator pipeline."""
    start_time = time.time()
    logger.info("phase6l_2_orchestrator_start")

    print("\n" + "=" * 85)
    print("HalluciSense Phase 6L.2 — Pillar-2 Development Model Selection (DEV ONLY)")
    print("=" * 85)

    # Stage 1: Feature Matrix Validation
    print("=== Stage 1: Feature Matrix Validation ($N = 58,002$) ===")
    val_data = load_and_validate_full_dev_matrix(out_dir=out_dir)
    X, y = val_data["X"], val_data["y"]

    # Stage 2: Preprocessing Study
    print("=== Stage 2: Preprocessing & Numerical Conditioning Study ===")
    prep_res = evaluate_preprocessing_scalers(X)

    # Stage 3: Collinearity Analysis
    print("=== Stage 3: Collinearity Analysis & Candidate Feature Sets ===")
    collin_res = run_collinearity_analysis(X, out_dir=out_dir)

    # Stage 4: Feature Discrimination Audit
    print("=== Stage 4: Feature Discrimination Audit ===")
    discrim_res = run_feature_discrimination_audit(X, y, out_dir=out_dir)

    # Stage 5: Numerical Stability Gate
    print("=== Stage 5: Corrected Numerical Stability Gate ===")
    stab_res = run_numerical_stability_gate(X, y, candidate_sets=collin_res["candidate_sets"], out_dir=out_dir)

    # Stage 6 & 7: Development Model Selection & Baselines
    print("=== Stage 6 & 7: Development Model Selection & Baselines (15 Folds per Model) ===")
    model_res = run_development_model_selection(X, y, candidate_sets=collin_res["candidate_sets"], out_dir=out_dir)

    # Stage 8: Leakage Audit
    print("=== Stage 8: Data Leakage & Firewall Audit ===")
    leak_res = run_data_leakage_audit(X, y, out_dir=out_dir)

    # Stage 9: Protocol Lock & Publication Figures / Report
    print("=== Stage 9: Protocol Lock & Publication Report Generation ===")
    protocol_res = export_final_model_protocol(winning_candidate=model_res["winning_candidate"], out_dir=out_dir)
    report_path = generate_publication_figures_and_report(
        X=X,
        y=y,
        validation_res=val_data["validation_payload"],
        preprocessing_res=prep_res,
        collinearity_res=collin_res,
        discrimination_res=discrim_res,
        stability_res=stab_res,
        model_selection_res=model_res,
        leakage_res=leak_res,
        protocol_res=protocol_res,
        out_dir=out_dir,
    )

    total_time = time.time() - start_time
    print("\n" + "=" * 85)
    print(f"Phase 6L.2 Execution Completed Successfully — {total_time:.2f}s elapsed")
    print(f"Selected Pillar-2 Model : {protocol_res['selected_candidate']}")
    print(f"Feature Subset Locked   : {protocol_res['feature_set_name']} ({protocol_res['feature_count']} features)")
    print(f"Decision Threshold      : {protocol_res['decision_threshold']}")
    print(f"Report Generated        : {report_path}")
    print(f"Firewall Status         : HELD-OUT VAL PARTITION (N = 12,483) 100% UNTOUCHED")
    print("=" * 85 + "\n")

    logger.info("phase6l_2_orchestrator_complete", elapsed_s=round(total_time, 2), winner=protocol_res["selected_candidate"])

    return {
        "validation": val_data["validation_payload"],
        "preprocessing": prep_res,
        "collinearity": collin_res,
        "discrimination": discrim_res,
        "stability": stab_res,
        "model_selection": model_res,
        "leakage": leak_res,
        "protocol": protocol_res,
        "report_path": str(report_path),
    }


def main():
    _ = run_phase6l_2()


if __name__ == "__main__":
    main()
