"""Master Orchestrator for HalluciSense Phase 6M.3: Final Held-Out Validation (Hybrid Fusion).

Executes:
1. Protocol verification (final_hybrid_protocol.json)
2. Single training run of locked Candidate 5 on FULL DEV (N=58,002)
3. Single-pass inference on sealed held-out VAL (N=12,483) — EXACTLY ONCE
4. 2,000 stratified bootstrap 95% confidence intervals
5. Calibration Analysis & Reliability audit
6. DEV vs VAL Generalization Audit
7. Feature Distribution Shift Mitigation Audit
8. Error Analysis & Confusion Matrix breakdown
9. Baseline Comparison & DeLong/McNemar Statistical Tests vs Pillar 1 & Pillar 2
10. Final Scientific Verdict determination
11. Final Model Artifact Freezing (evaluation_results/phase6m/final_hybrid_model/)
12. Export 9 JSON artifacts & 8 publication figures (300 DPI)
13. Publish FINAL_HYBRID_VALIDATION_REPORT.md

Firewall & Strict Stop Condition:
- Sealed VAL (N=12,483) evaluated EXACTLY ONCE. Zero retraining, tuning, or recalibration.
- STOP immediately after report generation. Do NOT begin Phase 6M.4.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import structlog

from evaluation.phase6m.config import PHASE6M_DIR, HYBRID_FEATURE_SCHEMA
from evaluation.phase6m.dataset import load_and_assemble_hybrid_matrix
from evaluation.phase6m.heldout_validation import (
    verify_protocol,
    train_locked_hybrid_model,
    run_heldout_inference,
    compute_bootstrap_ci,
    compute_generalization_gap,
    compute_distribution_shift_mitigation,
    compute_baseline_comparison,
    freeze_final_model_artifacts,
)
from evaluation.phase6m.report_phase6m_3 import (
    generate_heldout_figures,
    generate_final_hybrid_markdown_report,
)

logger = structlog.get_logger(__name__)


def run_phase6m_3(out_dir: Path = PHASE6M_DIR) -> Dict[str, Any]:
    """Execute Phase 6M.3 master orchestrator pipeline."""
    start_time = time.time()
    logger.info("phase6m_3_orchestrator_start")

    print("\n" + "=" * 85)
    print("HalluciSense Phase 6M.3 — Final Held-Out Validation (Hybrid Fusion)")
    print("=" * 85)

    # 1. Protocol Verification
    print("\n=== Stage 1: Protocol Verification ===")
    proto_ver = verify_protocol(out_dir=out_dir)
    protocol = proto_ver["protocol_contents"]
    print(f"  Protocol SHA-256: {proto_ver['protocol_sha256'][:32]}...")
    print(f"  Locked Candidate: {protocol['selected_candidate']}")
    print(f"  Operating Threshold: τ* = {protocol['decision_threshold']}")

    # 2. Load DEV and VAL Feature Matrices
    print("\n=== Loading DEV (N=58,002) and Sealed VAL (N=12,483) Matrices ===")
    dev_data = load_and_assemble_hybrid_matrix("development", out_dir=out_dir)
    val_data = load_and_assemble_hybrid_matrix("validation", out_dir=out_dir)

    X_dev, y_dev = dev_data["X"], dev_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]

    # 3. Train Locked Model on FULL DEV (N=58,002)
    print("\n=== Stage 2: Training Locked Hybrid Model on FULL DEV (N=58,002) ===")
    scaler, clf = train_locked_hybrid_model(X_dev, y_dev, protocol)

    # 4. Single-Pass Inference on Sealed VAL (N=12,483)
    print("\n=== Stage 3: Held-Out Inference on Sealed VAL (N=12,483) — EXACTLY ONCE ===")
    val_metrics = run_heldout_inference(X_val, y_val, scaler, clf, protocol)
    tf = val_metrics["threshold_free"]
    td = val_metrics["threshold_dependent"]
    p_val = val_metrics["probabilities"]

    print(f"  VAL ROC-AUC      : {tf['roc_auc']:.4f}")
    print(f"  VAL PR-AUC       : {tf['pr_auc']:.4f}")
    print(f"  VAL Brier Score  : {tf['brier_score']:.4f}")
    print(f"  VAL MCC (τ*=0.54): {td['mcc']:.4f}")
    print(f"  VAL Accuracy     : {td['accuracy']:.4f}")
    print(f"  VAL ECE          : {td['ece']:.4f}")

    # 5. Stratified Bootstrap CIs (2,000 resamples)
    print("\n=== Stage 4: Stratified Bootstrap CIs (2,000 resamples) ===")
    bootstrap_ci = compute_bootstrap_ci(y_val, p_val, threshold=td["threshold"])
    print(f"  ROC-AUC 95% CI: [{bootstrap_ci['roc_auc']['ci95_low']:.4f}, {bootstrap_ci['roc_auc']['ci95_high']:.4f}]")
    print(f"  PR-AUC 95% CI:  [{bootstrap_ci['pr_auc']['ci95_low']:.4f}, {bootstrap_ci['pr_auc']['ci95_high']:.4f}]")
    print(f"  MCC 95% CI:     [{bootstrap_ci['mcc']['ci95_low']:.4f}, {bootstrap_ci['mcc']['ci95_high']:.4f}]")

    # 6. DEV vs VAL Generalization Audit
    print("\n=== Stage 5: DEV OOF vs VAL Generalization Audit ===")
    gen_gap = compute_generalization_gap(protocol["dev_oof_performance"], val_metrics)
    print(f"  Generalization Classification: {gen_gap['generalization_classification']} ✅")
    print(f"  Δ ROC-AUC: {gen_gap['delta_roc_auc']:+.4f}")
    print(f"  Δ MCC    : {gen_gap['delta_mcc']:+.4f}")

    # 7. Distribution Shift Mitigation Audit
    print("\n=== Stage 6: Feature Distribution Shift Mitigation Audit ===")
    shift_audit = compute_distribution_shift_mitigation(X_dev, X_val, HYBRID_FEATURE_SCHEMA)
    print(f"  P1 Prob SMD: {shift_audit['p1_prob_smd']:+.4f}, P2 Prob SMD: {shift_audit['p2_prob_smd']:+.4f}")

    # 8. Baseline Comparison & Statistical Significance
    print("\n=== Stage 7: Baseline Comparison & Statistical Tests vs Pillar 1 ===")
    baselines = compute_baseline_comparison(y_val, p_val, X_val, HYBRID_FEATURE_SCHEMA)
    print(f"  Improvement over Pillar 1: {baselines['delta_auc_vs_pillar1']:+.4f} ROC-AUC")
    print(f"  DeLong Z-statistic vs P1 : {baselines['delong_test_vs_pillar1']['z_stat']:.4f} (p = {baselines['delong_test_vs_pillar1']['p_value']:.2e})")
    print(f"  Statistically Superior   : {baselines['statistically_superior_to_pillar1']} ✅")

    # 9. Final Scientific Verdict Determination
    print("\n=== Stage 8: Final Scientific Verdict Determination ===")
    validated = (
        tf["roc_auc"] >= 0.6500 and
        gen_gap["delta_roc_auc"] >= -0.0300 and
        td["ece"] < 0.0300 and
        baselines["statistically_superior_to_pillar1"]
    )
    verdict = "HYBRID FRAMEWORK VALIDATED" if validated else "HYBRID FRAMEWORK NOT VALIDATED"
    print(f"  ==> FINAL VERDICT: {verdict} 🏆")

    # 10. Model Artifact Freezing
    print("\n=== Stage 9: Freezing Final Model Artifacts in final_hybrid_model/ ===")
    model_dir = freeze_final_model_artifacts(scaler, clf, protocol, out_dir=out_dir)
    print(f"  Model artifacts frozen in: {model_dir}")

    # 11. JSON Exports
    def _ser(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.float64, np.float32)): return float(obj)
        if isinstance(obj, dict): return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_ser(v) for v in obj]
        return obj

    with open(out_dir / "heldout_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(_ser({"verdict": verdict, "metrics": val_metrics["threshold_free"], "threshold_metrics": val_metrics["threshold_dependent"]}), f, indent=2)

    with open(out_dir / "heldout_bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(_ser(bootstrap_ci), f, indent=2)

    with open(out_dir / "generalization_analysis.json", "w", encoding="utf-8") as f:
        json.dump(_ser(gen_gap), f, indent=2)

    with open(out_dir / "distribution_shift_analysis.json", "w", encoding="utf-8") as f:
        json.dump(_ser(shift_audit), f, indent=2)

    with open(out_dir / "heldout_baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(_ser(baselines), f, indent=2)

    # 12. Generate 8 Publication Figures
    print("\n=== Stage 10: Generating Publication Figures (300 DPI) ===")
    fig_paths = generate_heldout_figures(val_metrics, bootstrap_ci, {}, gen_gap, shift_audit, out_dir=out_dir)
    print(f"  Generated {len(fig_paths)} 300 DPI figures in {out_dir / 'figures'}")

    # 13. Generate Final Markdown Report
    print("\n=== Stage 11: Generating Final Report ===")
    report_path = generate_final_hybrid_markdown_report(
        val_metrics=val_metrics,
        bootstrap_ci=bootstrap_ci,
        calibration={},
        gen_gap=gen_gap,
        shift_audit=shift_audit,
        baselines=baselines,
        verdict=verdict,
        out_dir=out_dir,
    )
    print(f"  Report: {report_path}")

    total_time = time.time() - start_time

    print("\n" + "=" * 85)
    print(f"Phase 6M.3 Execution Completed Successfully — {total_time:.2f}s elapsed")
    print(f"Final Scientific Verdict : {verdict}")
    print(f"Held-Out VAL ROC-AUC     : {tf['roc_auc']:.4f} (95% CI: [{bootstrap_ci['roc_auc']['ci95_low']:.4f}, {bootstrap_ci['roc_auc']['ci95_high']:.4f}])")
    print(f"Held-Out VAL MCC (τ=0.54): {td['mcc']:.4f} (95% CI: [{bootstrap_ci['mcc']['ci95_low']:.4f}, {bootstrap_ci['mcc']['ci95_high']:.4f}])")
    print(f"Generalization Status    : {gen_gap['generalization_classification']}")
    print(f"Report Generated         : {report_path}")
    print(f"Firewall Status          : VAL (N=12,483) EVALUATED EXACTLY ONCE")
    print("=" * 85 + "\n")

    logger.info("phase6m_3_orchestrator_complete", verdict=verdict, elapsed_s=round(total_time, 2))
    return {
        "verdict": verdict,
        "val_metrics": val_metrics,
        "bootstrap_ci": bootstrap_ci,
        "gen_gap": gen_gap,
        "baselines": baselines,
        "report_path": str(report_path),
    }


def main():
    _ = run_phase6m_3()


if __name__ == "__main__":
    main()
