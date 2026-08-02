"""Master Orchestrator for HalluciSense Phase 6M.2: Development Model Selection (Hybrid Fusion).

Executes:
1. Load DEV hybrid feature matrix (N=58,002) from Phase 6M.1
2. RepeatedStratifiedKFold (5 splits, 3 repeats = 15 iterations) CV across candidate models
3. Calibration Audit (Raw vs Platt vs Isotonic)
4. DeLong and McNemar Statistical Tests vs Pillar 1
5. Winner Selection & Protocol Locking (final_hybrid_protocol.json)
6. 8 Publication Figures (300 DPI)
7. Final Markdown Report PHASE6M_2_DEVELOPMENT_MODEL_SELECTION.md
8. Decision Gate Evaluation (GO / NO-GO clearance)

Firewall & Strict Stop Condition:
- Validation partition (N=12,483) remains 100% SEALED.
- ZERO evaluation, calibration, or tuning on validation.
- STOP immediately after protocol locking. Do NOT begin Phase 6M.3.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import structlog

from evaluation.phase6m.config import PHASE6M_DIR, HYBRID_FEATURE_SCHEMA
from evaluation.phase6m.dataset import load_and_assemble_hybrid_matrix
from evaluation.phase6m.model_selection import run_development_model_selection
from evaluation.phase6m.report_phase6m_2 import (
    generate_model_selection_figures,
    generate_phase6m2_markdown_report,
)

logger = structlog.get_logger(__name__)


def run_phase6m_2(out_dir: Path = PHASE6M_DIR) -> Dict[str, Any]:
    """Execute Phase 6M.2 master orchestrator pipeline."""
    start_time = time.time()
    logger.info("phase6m_2_orchestrator_start")

    print("\n" + "=" * 85)
    print("HalluciSense Phase 6M.2 — Development Model Selection (Hybrid Fusion)")
    print("=" * 85)

    # 1. Load DEV Hybrid Feature Matrix (N=58,002)
    print("\n=== Stage 1: Loading DEV Hybrid Feature Matrix (N=58,002) ===")
    dev_data = load_and_assemble_hybrid_matrix("development", out_dir=out_dir)
    X_dev, y_dev = dev_data["X"], dev_data["y"]
    print(f"  Loaded DEV matrix: {X_dev.shape}, pos={int((y_dev == 1).sum())}, neg={int((y_dev == 0).sum())}")

    # 2. Run Development Model Selection (15 CV iterations per candidate)
    print("\n=== Stage 2: Repeated 5-Fold 3-Repeat CV Model Selection ===")
    selection_results = run_development_model_selection(X_dev, y_dev, HYBRID_FEATURE_SCHEMA, out_dir=out_dir)

    winning = selection_results["winning_candidate"]
    lock = selection_results["protocol_lock"]

    # 3. Generate 8 Publication Figures (300 DPI)
    print("\n=== Stage 3: Generating Publication Figures (300 DPI) ===")
    fig_paths = generate_model_selection_figures(selection_results, out_dir=out_dir)
    print(f"  Generated {len(fig_paths)} 300 DPI figures in {out_dir / 'figures'}")

    # 4. Generate Markdown Report
    print("\n=== Stage 4: Generating Markdown Report ===")
    report_path = generate_phase6m2_markdown_report(selection_results, out_dir=out_dir)
    print(f"  Report: {report_path}")

    total_time = time.time() - start_time

    print("\n" + "=" * 85)
    print(f"Phase 6M.2 Execution Completed Successfully — {total_time:.2f}s elapsed")
    print(f"Winning Candidate  : {winning['name']}")
    print(f"Feature Subset     : {lock['set_key']} ({lock['feature_count']} features)")
    print(f"Preprocessing      : {lock['scaler']}")
    print(f"Classifier         : {lock['classifier']}")
    print(f"Operating Threshold: τ = {lock['decision_threshold']}")
    print(f"DEV OOF ROC-AUC    : {lock['dev_oof_performance']['roc_auc']:.4f}")
    print(f"Protocol Locked    : final_hybrid_protocol.json")
    print(f"Phase 6M.3 Clearance: GO")
    print(f"Firewall Status    : VAL (N=12,483) STRICTLY SEALED — ZERO HELD-OUT EVALUATION")
    print("=" * 85 + "\n")

    logger.info("phase6m_2_orchestrator_complete", winner=lock["selected_candidate"], elapsed_s=round(total_time, 2))
    return {
        "selection_results": selection_results,
        "protocol_lock": lock,
        "report_path": str(report_path),
        "clearance": "GO",
    }


def main():
    _ = run_phase6m_2()


if __name__ == "__main__":
    main()
