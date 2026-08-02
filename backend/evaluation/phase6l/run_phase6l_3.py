"""Master Orchestrator for Phase 6L.3: Final Held-Out Validation of Pillar-2 Structural Consistency.

Executes:
    Stage 0:  VAL structural feature extraction (if not already done)
    Stage 1:  Protocol verification
    Stage 2:  Train locked model on DEV ONLY
    Stage 3:  Held-out inference ONCE on VAL
    Stage 4:  Bootstrap confidence intervals (2,000 resamples)
    Stage 5:  Calibration analysis
    Stage 6:  Generalization audit (DEV CV → VAL)
    Stage 7:  Baseline confirmation
    Stage 8:  Feature distribution shift
    Stage 9:  Error analysis
    Stage 10: Numerical health audit
    Stage 11: Final Pillar-2 verdict + report + figures

Strict Data Firewall:
    * VAL (N=12,483) is INFERENCE-ONLY. Evaluated EXACTLY ONCE.
    * No fitting, tuning, or selection on VAL.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import structlog

from evaluation.phase6l.config import PHASE6L_DIR, STRUCTURAL_FEATURE_COLUMNS
from evaluation.phase6l.dataset import load_and_validate_full_dev_matrix
from evaluation.phase6l.val_feature_reconstruction import (
    reconstruct_val_structural_features,
    load_val_structural_features,
)
from evaluation.phase6l.heldout_validation import (
    LOCKED_FEATURE_NAMES,
    PRIMARY_THRESHOLD,
    verify_protocol,
    train_locked_model_on_dev,
    run_heldout_inference,
    compute_bootstrap_ci,
    compute_calibration,
    compute_generalization_gap,
    compute_baseline_comparison,
    compute_distribution_shift,
    compute_error_analysis,
    compute_numerical_health,
    generate_figures,
    generate_final_report,
)

logger = structlog.get_logger(__name__)


def run_phase6l_3(out_dir: Path = PHASE6L_DIR) -> Dict[str, Any]:
    """Execute Phase 6L.3 master orchestrator."""
    start_time = time.time()
    logger.info("phase6l_3_orchestrator_start")

    print("\n" + "=" * 85)
    print("HalluciSense Phase 6L.3 — Final Pillar-2 Held-Out Validation")
    print("=" * 85)

    # -------------------------------------------------------------------
    # Stage 0: VAL Structural Feature Extraction
    # -------------------------------------------------------------------
    val_structural_path = out_dir / "structural_features_full_val.jsonl"
    if not val_structural_path.exists():
        print("\n=== Stage 0: VAL Structural Feature Extraction (N=12,483) ===")
        recon_stats = reconstruct_val_structural_features(out_dir=out_dir)
        print(f"  Completed: {recon_stats['total_responses']:,} responses, {recon_stats['elapsed_s']:.1f}s")
    else:
        print("\n=== Stage 0: VAL Structural Features — Already extracted (cached) ===")

    # -------------------------------------------------------------------
    # Stage 1: Protocol Verification
    # -------------------------------------------------------------------
    print("\n=== Stage 1: Protocol Verification ===")
    protocol_ver = verify_protocol(out_dir=out_dir)
    print(f"  Protocol SHA-256: {protocol_ver['protocol_sha256'][:32]}...")
    print(f"  All fields verified: ✅")

    # -------------------------------------------------------------------
    # Load DEV and VAL data
    # -------------------------------------------------------------------
    print("\n=== Loading DEV (N=58,002) and VAL (N=12,483) Feature Matrices ===")
    dev_data = load_and_validate_full_dev_matrix(out_dir=out_dir)
    X_dev, y_dev = dev_data["X"], dev_data["y"]

    val_data = load_val_structural_features()
    X_val, y_val = val_data["X"], val_data["y"]

    print(f"  DEV: {X_dev.shape}, pos={int((y_dev == 1).sum())}, neg={int((y_dev == 0).sum())}")
    print(f"  VAL: {X_val.shape}, pos={int((y_val == 1).sum())}, neg={int((y_val == 0).sum())}")

    # -------------------------------------------------------------------
    # Stage 2: Train Locked Model on DEV ONLY
    # -------------------------------------------------------------------
    print("\n=== Stage 2: Train Locked Model on FULL DEV (N=58,002) ===")
    scaler, clf, training_info = train_locked_model_on_dev(X_dev, y_dev, out_dir=out_dir)
    print(f"  Feature importances: {training_info['feature_importances']}")
    print(f"  Training warnings: {len(training_info['training_warnings'])}")

    # -------------------------------------------------------------------
    # Stage 3: Held-Out Inference (EXACTLY ONCE)
    # -------------------------------------------------------------------
    print("\n=== Stage 3: Held-Out Inference on VAL (N=12,483) — ONCE ===")
    val_metrics = run_heldout_inference(X_val, y_val, scaler, clf)
    tf = val_metrics["threshold_free_metrics"]
    mp = val_metrics["primary_threshold_metrics"]
    print(f"  ROC-AUC:  {tf['roc_auc']:.4f}")
    print(f"  PR-AUC:   {tf['pr_auc']:.4f}")
    print(f"  Brier:    {tf['brier_score']:.4f}")
    print(f"  MCC@{PRIMARY_THRESHOLD}: {mp['mcc']:.4f}")
    print(f"  F1@{PRIMARY_THRESHOLD}:  {mp['f1']:.4f}")

    p_val = val_metrics["probabilities"]
    X_val_selected = val_metrics["X_val_selected"]

    # -------------------------------------------------------------------
    # Stage 4: Bootstrap Confidence Intervals
    # -------------------------------------------------------------------
    print("\n=== Stage 4: Bootstrap CIs (2,000 stratified resamples) ===")
    bootstrap_ci = compute_bootstrap_ci(y_val, p_val, threshold=PRIMARY_THRESHOLD)
    print(f"  ROC-AUC 95% CI: [{bootstrap_ci['roc_auc']['ci95_low']:.4f}, {bootstrap_ci['roc_auc']['ci95_high']:.4f}]")
    print(f"  MCC 95% CI:     [{bootstrap_ci['mcc']['ci95_low']:.4f}, {bootstrap_ci['mcc']['ci95_high']:.4f}]")

    # -------------------------------------------------------------------
    # Stage 5: Calibration Analysis
    # -------------------------------------------------------------------
    print("\n=== Stage 5: Calibration Analysis ===")
    calibration = compute_calibration(y_val, p_val)
    print(f"  ECE: {calibration['ece']:.4f} ({'PASS' if calibration['calibration_pass'] else 'FAIL'})")
    print(f"  MCE: {calibration['mce']:.4f}")

    # -------------------------------------------------------------------
    # Stage 6: Generalization Audit
    # -------------------------------------------------------------------
    print("\n=== Stage 6: Generalization Audit (DEV CV → VAL) ===")
    protocol = protocol_ver["protocol_contents"]
    dev_summary = protocol.get("dev_performance_summary", {})
    gen_gap = compute_generalization_gap(dev_summary, val_metrics, calibration)
    print(f"  Classification: {gen_gap['generalization_classification']}")
    print(f"  Δ ROC-AUC: {gen_gap['gap_auc']:+.4f}")
    print(f"  Δ MCC:     {gen_gap['gap_mcc']:+.4f}")

    # -------------------------------------------------------------------
    # Stage 7: Baseline Confirmation
    # -------------------------------------------------------------------
    print("\n=== Stage 7: Baseline Confirmation on VAL ===")
    baselines = compute_baseline_comparison(X_dev, y_dev, X_val, y_val, p_val, tf["roc_auc"])
    print(f"  Improvement over majority:        {baselines['improvement_over_majority']:+.4f}")
    print(f"  Improvement over best single feat: {baselines['improvement_over_best_single']:+.4f}")

    # -------------------------------------------------------------------
    # Stage 8: Feature Distribution Shift
    # -------------------------------------------------------------------
    print("\n=== Stage 8: Feature Distribution Shift ===")
    shift_data = compute_distribution_shift(X_dev, X_val)
    for fname in LOCKED_FEATURE_NAMES:
        s = shift_data["features"][fname]
        flag = "⚠️" if s["flagged"] else "✅"
        print(f"  {fname}: SMD={s['standardized_mean_difference']:+.4f}, KS={s['ks_statistic']:.4f} {flag}")

    # -------------------------------------------------------------------
    # Stage 9: Error Analysis
    # -------------------------------------------------------------------
    print("\n=== Stage 9: Error Analysis ===")
    error_analysis = compute_error_analysis(y_val, p_val, X_val_selected)
    for g in ["TP", "TN", "FP", "FN"]:
        ea_g = error_analysis["group_statistics"][g]
        print(f"  {g}: {ea_g['count']:,} samples")

    # -------------------------------------------------------------------
    # Stage 10: Numerical Health
    # -------------------------------------------------------------------
    print("\n=== Stage 10: Numerical Health Audit ===")
    numerical_health = compute_numerical_health(scaler, clf, p_val, training_info, val_metrics["inference_warnings"])
    print(f"  Health Pass: {'✅' if numerical_health['numerical_health_pass'] else '❌'}")
    print(f"  Total Warnings: {numerical_health['training_warnings_total'] + numerical_health['inference_warnings_total']}")

    # -------------------------------------------------------------------
    # Stage 11: Final Verdict
    # -------------------------------------------------------------------
    print("\n=== Stage 11: Final Pillar-2 Verdict ===")

    gen_pass = gen_gap["generalization_classification"] in ["STABLE", "MINOR DEGRADATION"]
    base_pass = tf["roc_auc"] > baselines["baseline_c_max_contradiction"]["val_roc_auc"]
    cal_pass = calibration["calibration_pass"]
    num_pass = numerical_health["numerical_health_pass"]

    if gen_pass and base_pass and num_pass:
        if cal_pass:
            verdict = "PILLAR 2 VALIDATED"
        else:
            verdict = "PILLAR 2 VALIDATED WITH LIMITATIONS"
    else:
        verdict = "PILLAR 2 NOT VALIDATED"

    print(f"  Generalization:   {'PASS' if gen_pass else 'FAIL'}")
    print(f"  Baseline Beat:    {'PASS' if base_pass else 'FAIL'}")
    print(f"  Calibration:      {'PASS' if cal_pass else 'FAIL'}")
    print(f"  Numerical Health: {'PASS' if num_pass else 'FAIL'}")
    print(f"  ==> VERDICT: {verdict}")

    # Save model metadata
    model_dir = out_dir / "final_model"
    model_metadata = {
        "model_verdict": verdict,
        "training_sample_count": int(X_dev.shape[0]),
        "validation_sample_count": int(X_val.shape[0]),
        "feature_names": LOCKED_FEATURE_NAMES,
        "feature_importances": training_info["feature_importances"],
        "classifier": "RandomForestClassifier(n_estimators=100, max_depth=6)",
        "scaler": "StandardScaler",
        "operating_threshold": PRIMARY_THRESHOLD,
        "val_roc_auc": tf["roc_auc"],
        "val_pr_auc": tf["pr_auc"],
        "val_mcc": mp["mcc"],
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(model_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # -------------------------------------------------------------------
    # Export all JSON results
    # -------------------------------------------------------------------
    def _ser(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    val_results_export = {
        "verdict": verdict,
        "threshold_free_metrics": tf,
        "primary_threshold_metrics": mp,
        "reference_threshold_metrics": val_metrics["reference_threshold_metrics"],
    }
    with open(out_dir / "heldout_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(val_results_export, f, indent=2, default=_ser)

    with open(out_dir / "heldout_bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(bootstrap_ci, f, indent=2, default=_ser)

    with open(out_dir / "heldout_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2, default=_ser)

    with open(out_dir / "dev_val_generalization.json", "w", encoding="utf-8") as f:
        json.dump(gen_gap, f, indent=2, default=_ser)

    with open(out_dir / "heldout_baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2, default=_ser)

    with open(out_dir / "dev_val_distribution_shift.json", "w", encoding="utf-8") as f:
        json.dump(shift_data, f, indent=2, default=_ser)

    with open(out_dir / "heldout_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, default=_ser)

    with open(out_dir / "numerical_health_audit.json", "w", encoding="utf-8") as f:
        json.dump(numerical_health, f, indent=2, default=_ser)

    # -------------------------------------------------------------------
    # Generate Figures and Report
    # -------------------------------------------------------------------
    print("\n=== Generating Publication Figures (300 DPI) ===")
    fig_paths = generate_figures(
        y_val, p_val, X_dev, X_val, val_metrics, calibration,
        gen_gap, shift_data, error_analysis, out_dir=out_dir,
    )
    print(f"  Generated {len(fig_paths)} figures")

    print("\n=== Generating Final Report ===")
    report_path = generate_final_report(
        protocol_ver, training_info, val_metrics, bootstrap_ci,
        calibration, gen_gap, baselines, shift_data, error_analysis,
        numerical_health, verdict, out_dir=out_dir,
    )
    print(f"  Report: {report_path}")

    total_time = time.time() - start_time

    print("\n" + "=" * 85)
    print(f"Phase 6L.3 Execution Completed — {total_time:.2f}s elapsed")
    print(f"Final Verdict     : {verdict}")
    print(f"VAL ROC-AUC       : {tf['roc_auc']:.4f}")
    print(f"VAL MCC@{PRIMARY_THRESHOLD}     : {mp['mcc']:.4f}")
    print(f"Generalization    : {gen_gap['generalization_classification']}")
    print(f"Report            : {report_path}")
    print(f"Firewall Status   : VAL (N=12,483) EVALUATED EXACTLY ONCE")
    print("=" * 85 + "\n")

    logger.info("phase6l_3_orchestrator_complete", verdict=verdict, elapsed_s=round(total_time, 2))
    return {"verdict": verdict, "val_roc_auc": tf["roc_auc"], "report_path": str(report_path)}


def main():
    run_phase6l_3()


if __name__ == "__main__":
    main()
