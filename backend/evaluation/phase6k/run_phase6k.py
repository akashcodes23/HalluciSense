"""Phase 6K — Orchestrator for Stable Feature Selection & Model Recovery.

Orchestrates cache discovery, schema validation, preprocessing numerical audit,
collinearity analysis, feature subset selection, 1,000-example stability gate,
leakage & shortcut audit, warning forensics, corrected stability gate (Phase 6K.2),
full development model selection & cross-validation (Phase 6K.3), and
final locked-model held-out validation (Phase 6K.4).

Usage:
    python -m evaluation.phase6k.run_phase6k
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import structlog

from evaluation.phase6j.utils import _serializable
from evaluation.phase6k.config import (
    PHASE6I_DIR,
    PHASE6K_DIR,
    FEATURE_COLUMNS,
    ExperimentConfig,
)
from evaluation.phase6k.cache_loader import load_phase6i_cache, LoadedCache
from evaluation.phase6k.preprocessing import audit_preprocessing_strategies, PreprocessingAuditReport
from evaluation.phase6k.collinearity import analyze_collinearity, CollinearityAuditReport, CollinearityDecisionsReport
from evaluation.phase6k.feature_selection import construct_candidate_feature_sets, CandidateFeatureSetsReport
from evaluation.phase6k.benchmark import run_stability_gate_1000, StabilityGate1000Report
from evaluation.phase6k.feasibility import run_leakage_shortcut_audit, LeakageShortcutReport
from evaluation.phase6k.forensics import run_warning_forensics, generate_warning_forensics_report
from evaluation.phase6k.corrected_stability_gate import run_corrected_stability_gate
from evaluation.phase6k.model_selection import run_phase6k3_model_selection
from evaluation.phase6k.validation import run_phase6k4_heldout_validation
from evaluation.phase6k.report import generate_phase6k_report

logger = structlog.get_logger(__name__)


def _export_no_feasible_candidate_artifacts(out_dir: Path) -> None:
    """Export model comparison and validation JSON artifacts reflecting precondition failure."""
    out_dir.mkdir(parents=True, exist_ok=True)

    no_candidate_reason = (
        "Precondition failed: 1,000-example numerical stability gate produced "
        "STABILITY GATE: FAIL with 0 passing configurations due to internal matrix multiplication "
        "overflow/divide-by-zero warnings."
    )

    model_comp = {
        "status": "STOPPED_PRECONDITION_FAILED",
        "precondition_passed": False,
        "verdict": "NO FEASIBLE CANDIDATE",
        "reason": no_candidate_reason,
        "candidates_evaluated": 0,
        "results": [],
    }

    selected_cand = {
        "selected_candidate_id": "NONE",
        "verdict": "NO FEASIBLE CANDIDATE",
        "reason": "Precondition failed: 1,000-example numerical stability gate failed. No candidate was selected.",
        "candidate": None,
    }

    val_eval = {
        "val_evaluated": False,
        "verdict": "NO FEASIBLE CANDIDATE",
        "reason": "Precondition failed: 1,000-example numerical stability gate failed. Validation partition remained completely untouched.",
        "val_metrics": None,
    }

    with open(out_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(_serializable(model_comp), f, indent=2)

    with open(out_dir / "selected_candidate.json", "w", encoding="utf-8") as f:
        json.dump(_serializable(selected_cand), f, indent=2)

    with open(out_dir / "validation_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(_serializable(val_eval), f, indent=2)

    logger.info("phase6k_no_feasible_candidate_artifacts_exported", out_dir=str(out_dir))


def run_phase6k() -> LoadedCache:
    """Orchestrate Phase 6K cache discovery, audits, feature selection, stability gate, leakage audit, forensics, corrected gate, full DEV model selection, and held-out validation.

    Returns:
        LoadedCache object containing DEV and VAL data partitions.
    """
    logger.info("phase6k_orchestrator_start")
    t0 = time.time()

    print(f"\n{'=' * 85}")
    print("HalluciSense Phase 6K — Stable Feature Selection, Forensics & Held-Out Validation")
    print(f"{'=' * 85}")

    # 1. Discover and load Phase 6I cache matrices
    cache = load_phase6i_cache(cache_dir=PHASE6I_DIR, feature_columns=FEATURE_COLUMNS)

    print(f"  DEV Source       : {cache.dev_path}")
    print(f"  VAL Source       : {cache.val_path}")
    print(f"  Output Directory : {PHASE6K_DIR}")
    print(f"{'=' * 85}\n")

    print("=== Phase 6I Cache Verification ===")
    print(f"  DEV Shape        : {cache.dev.n_samples:,} samples × {cache.dev.n_features} features")
    print(f"  DEV Targets      : pos={cache.dev.n_positive:,} ({cache.dev.positive_ratio:.2%}), neg={cache.dev.n_negative:,}")
    print(f"  VAL Shape        : {cache.val.n_samples:,} samples × {cache.val.n_features} features")
    print(f"  VAL Targets      : pos={cache.val.n_positive:,} ({cache.val.positive_ratio:.2%}), neg={cache.val.n_negative:,}\n")

    # 2. Run Preprocessing Numerical Conditioning Audit
    print("=== Module 1: Preprocessing Numerical Conditioning Audit ===")
    audit_report = audit_preprocessing_strategies(
        X_dev=cache.dev.X,
        X_val=cache.val.X,
        feature_names=cache.dev.feature_names,
        out_dir=PHASE6K_DIR,
    )

    hdr1 = f"{'Rank':<5} {'Strategy':<22} {'DEV Cond No (κ)':<18} {'VAL Cond No (κ)':<18} {'Rank':<6} {'Finite'}"
    print(hdr1)
    print("-" * 85)

    for sname in audit_report.ranked_strategy_names:
        res = audit_report.strategies[sname]
        tr_cond = f"{res.train_stats.condition_number:.2e}"
        v_cond = f"{res.val_stats.condition_number:.2e}" if res.val_stats else "N/A"
        r_num = f"{res.train_stats.matrix_rank}"
        fin = "Yes" if res.train_stats.is_finite else "No"
        print(f"{res.rank:<5} {sname:<22} {tr_cond:<18} {v_cond:<18} {r_num:<6} {fin}")

    print(f"\n  Recommended Strategy : {audit_report.recommended_strategy}")
    print(f"  Reasoning            : {audit_report.recommendation_reasoning}\n")

    # 3. Run Collinearity & Redundancy Audit (DEV ONLY)
    print("=== Module 2: Collinearity & Feature Redundancy Audit (DEV Partition Only) ===")
    col_audit, col_decisions = analyze_collinearity(
        X_dev=cache.dev.X,
        y_dev=cache.dev.y,
        feature_names=cache.dev.feature_names,
        threshold=0.90,
        out_dir=PHASE6K_DIR,
    )

    print(f"  Matrix Rank      : {col_audit.matrix_rank} / {col_audit.n_features}")
    print(f"  Condition Number : {col_audit.condition_number:.2e}")
    print(f"  Redundant Pairs  : {col_audit.redundant_pair_count} pairs with |Pearson r| >= 0.90\n")

    print(f"  Proposed Retained Feature Set ({len(col_decisions.proposed_retained_features)} features):")
    print(f"    {', '.join(col_decisions.proposed_retained_features)}")
    print(f"  Proposed Removed Feature Set ({len(col_decisions.proposed_removed_features)} features):")
    print(f"    {', '.join(col_decisions.proposed_removed_features)}\n")

    # 4. Construct Candidate Feature Sets (DEV ONLY)
    print("=== Module 3: Candidate Feature Subset Construction (DEV Partition Only) ===")
    feature_sets_report = construct_candidate_feature_sets(
        X_dev=cache.dev.X,
        y_dev=cache.dev.y,
        feature_names=cache.dev.feature_names,
        delinearized_retained_features=col_decisions.proposed_retained_features,
        out_dir=PHASE6K_DIR,
    )

    hdr3 = f"  {'Set Key':<38} {'Count':<7} {'Rank':<6} {'Unscaled κ':<14} {'Robust Scaled κ':<16} {'Mean |r|':<10} {'Max |r|'}"
    print(hdr3)
    print("  " + "-" * 105)

    for set_key, set_meta in feature_sets_report.candidate_sets.items():
        u_cond = f"{set_meta.condition_number_unscaled:.2e}"
        s_cond = f"{set_meta.condition_number_robust_scaled:.2e}"
        m_r = f"{set_meta.mean_pairwise_abs_correlation:.4f}"
        mx_r = f"{set_meta.max_pairwise_abs_correlation:.4f}"
        print(f"  {set_key:<38} {set_meta.feature_count:<7} {set_meta.matrix_rank:<6} {u_cond:<14} {s_cond:<16} {m_r:<10} {mx_r}")
    print()

    # 5. Run Original 1,000-Example Numerical Stability Gate (DEV ONLY)
    print("=== Module 4: Original 1,000-Example Numerical Stability Gate (DEV Partition Only) ===")
    gate_report = run_stability_gate_1000(
        X_dev=cache.dev.X,
        y_dev=cache.dev.y,
        feature_names=cache.dev.feature_names,
        out_dir=PHASE6K_DIR,
    )

    hdr4 = f"  {'Config ID':<45} {'Cond No (κ)':<14} {'Warns':<7} {'Iters':<7} {'Train Acc':<10} {'Status'}"
    print(hdr4)
    print("  " + "-" * 95)

    for cid in gate_report.ranked_config_ids:
        cfg = gate_report.configs[cid]
        cond_str = f"{cfg.condition_number:.2e}"
        w_cnt = f"{len(cfg.warnings_captured)}"
        status_str = "PASS" if cfg.pass_status else "FAIL"
        print(f"  {cid:<45} {cond_str:<14} {w_cnt:<7} {cfg.n_iter:<7} {cfg.training_accuracy:<10.4f} {status_str}")

    print(f"\n  Stability Gate Tested : {gate_report.total_configs_tested} configurations")
    print(f"  Passing Configurations: {gate_report.passing_configs_count}")
    print(f"  Failing Configurations: {gate_report.failing_configs_count}\n")
    print(f"  >>> {gate_report.overall_verdict} <<<\n")

    # 6. Run Leakage & Shortcut Audit
    print("=== Module 5: Leakage & Shortcut Audit ===")
    leakage_report = run_leakage_shortcut_audit(
        X_dev=cache.dev.X,
        y_dev=cache.dev.y,
        X_val=cache.val.X,
        y_val=cache.val.y,
        feature_names=cache.dev.feature_names,
        out_dir=PHASE6K_DIR,
    )

    print(f"  Target Leakage Detected  : {'Yes' if leakage_report.target_leakage_detected else 'No'}")
    print(f"  Max Feature-Target |r|   : {leakage_report.max_feature_target_correlation:.4f} ({leakage_report.leakage_feature_name})")
    print(f"  DEV-VAL Overlap Count    : {leakage_report.dev_val_overlap_count} ({leakage_report.dev_val_overlap_ratio:.2%})")
    print(f"  Single Feature Dominance : {'Yes' if leakage_report.single_feature_dominance_detected else 'No'}")
    print(f"  Label Permuted ROC-AUC   : {leakage_report.permutation_test.mean_permuted_roc_auc:.4f} ± {leakage_report.permutation_test.std_permuted_roc_auc:.4f}")
    print(f"  Collapsed to Chance      : {'Yes' if leakage_report.permutation_test.collapsed_to_chance else 'No'}")
    print(f"  Catastrophic Ablations   : {len(leakage_report.catastrophic_ablation_features)} features")
    print(f"\n  >>> LEAKAGE AUDIT VERDICT: {leakage_report.overall_verdict} <<<\n")

    # 7. Run Numerical Warning Forensics (Phase 6K.1 Audit)
    print("=== Module 6: Numerical Warning Forensics (Phase 6K.1 Audit) ===")
    forensic_data = run_warning_forensics(out_dir=PHASE6K_DIR)
    forensic_report_path = generate_warning_forensics_report(forensic_data, out_dir=PHASE6K_DIR)

    s1 = forensic_data["step1_warning_counting"]
    s4 = forensic_data["step4_solver_isolation"]
    s8 = forensic_data["step8_standalone_reproduction"]

    print(f"  Raw Recorded Warnings (Step 1) : {s1['raw_recorded_warning_count']}")
    print(f"  Mutually Exclusive Categories  : {s1['mutually_exclusive_counts']}")
    print("  Solver Isolation Warnings (Step 4):")
    for srow in s4:
        print(f"    * Solver '{srow['solver']:<10}': warnings={srow['warning_count']}, accuracy={srow.get('training_accuracy', 0.0):.4f}")
    print(f"  Standalone Reproduction (Step 8): {s8['standalone_warning_count']} warnings (Reproduced: {s8['reproduced_in_standalone']})")
    print(f"  Forensic Report Exported       : {forensic_report_path}\n")

    # 8. Run Corrected 1,000-Example Numerical Stability Gate (Phase 6K.2)
    print("=== Module 7: Corrected 1,000-Example Numerical Stability Gate (Phase 6K.2) ===")
    corrected_gate_data, consistency_data = run_corrected_stability_gate(
        X_dev=cache.dev.X,
        y_dev=cache.dev.y,
        feature_names=cache.dev.feature_names,
        out_dir=PHASE6K_DIR,
    )

    print(f"  Total Configurations Tested (liblinear & saga) : {corrected_gate_data['total_configs_tested']}")
    print(f"  Passing Configurations                         : {corrected_gate_data['passing_configs_count']}")
    print(f"  Failing Configurations                         : {corrected_gate_data['failing_configs_count']}")
    print(f"  Cross-Solver Material Equivalence               : {consistency_data['materially_equivalent_count']} / {consistency_data['total_comparisons']}")
    print(f"\n  >>> CORRECTED STABILITY GATE VERDICT: {corrected_gate_data['overall_verdict']} <<<\n")

    # 9. Run Full Development Model Selection & Cross-Validation (Phase 6K.3)
    print("=== Module 8: Full Development Model Selection & CV (Phase 6K.3, N=58,002) ===")
    p6k3_results = run_phase6k3_model_selection(out_dir=PHASE6K_DIR)
    fin_cand = p6k3_results["final_candidate"]

    print(f"  Full DEV Partition Samples (N) : {cache.dev.n_samples:,} rows")
    print(f"  Selected Candidate Key         : {fin_cand['selected_candidate_key']}")
    print(f"  Selected Candidate Name        : {fin_cand['display_name']}")
    print(f"  Features Included ({len(fin_cand['feature_names'])})          : {', '.join(fin_cand['feature_names'])}")
    print(f"  Mean CV ROC-AUC (15 Folds)     : {fin_cand['mean_cv_metrics']['roc_auc']:.4f}")
    print(f"  Mean CV PR-AUC (15 Folds)      : {fin_cand['mean_cv_metrics']['pr_auc']:.4f}")
    print(f"  Mean CV MCC (15 Folds)         : {fin_cand['mean_cv_metrics']['mcc']:.4f}")
    print(f"  Expected Calibration Error     : {fin_cand['ece']:.4f}")
    print(f"  Optimal Decision Threshold     : {fin_cand['best_mcc_threshold']:.2f}")
    print(f"\n  >>> OVERALL DEV SELECTION VERDICT: {fin_cand['acceptance_criteria']['overall_verdict']} <<<\n")

    # 10. Run Final Locked-Model Held-Out Validation (Phase 6K.4)
    print("=== Module 9: Final Locked-Model Held-Out Validation (Phase 6K.4, VAL N=12,483) ===")
    val_p6k4_data = run_phase6k4_heldout_validation(out_dir=PHASE6K_DIR)
    v_results = val_p6k4_data["val_results"]
    v_tf = v_results["threshold_free_metrics"]
    v_m56 = v_results["primary_threshold_056_metrics"]

    print(f"  Held-Out VAL Samples (N)       : {cache.val.n_samples:,} rows")
    print(f"  Protocol Lock Exported         : final_model_protocol.json (PRE-EVALUATION)")
    print(f"  Held-Out ROC-AUC               : {v_tf['roc_auc']:.4f} (95% CI: [{val_p6k4_data['bootstrap_ci']['roc_auc']['ci95_low']:.4f}, {val_p6k4_data['bootstrap_ci']['roc_auc']['ci95_high']:.4f}])")
    print(f"  Held-Out PR-AUC                : {v_tf['pr_auc']:.4f}")
    print(f"  Held-Out MCC (@ 0.56)          : {v_m56['mcc']:.4f}")
    print(f"  Held-Out Accuracy (@ 0.56)     : {v_m56['accuracy']:.4f}")
    print(f"  Expected Calibration Error     : {val_p6k4_data['generalization_gap']['val_ece']:.4f}")
    print(f"  Generalization Classification  : {val_p6k4_data['generalization_gap']['generalization_classification']} (Delta ROC-AUC = {val_p6k4_data['generalization_gap']['gap_auc']:+.4f})")
    print(f"\n  >>> FINAL PILLAR-1 VALIDATION VERDICT: {val_p6k4_data['verdict']} <<<\n")

    # 11. EXPORT ARTIFACTS AND SUMMARY
    _export_no_feasible_candidate_artifacts(PHASE6K_DIR)
    report_meta = generate_phase6k_report(cache=cache, out_dir=PHASE6K_DIR)

    elapsed = time.time() - t0
    print(f"Phase 6K Execution Completed — {elapsed:.2f}s elapsed")
    print(f"Artifacts Exported:")
    print(f"  - {PHASE6K_DIR / 'preprocessing_audit.json'}")
    print(f"  - {PHASE6K_DIR / 'collinearity_audit.json'}")
    print(f"  - {PHASE6K_DIR / 'collinearity_decisions.json'}")
    print(f"  - {PHASE6K_DIR / 'feature_sets.json'}")
    print(f"  - {PHASE6K_DIR / 'stability_gate_1000.json'}")
    print(f"  - {PHASE6K_DIR / 'leakage_shortcut_audit.json'}")
    print(f"  - {PHASE6K_DIR / 'warning_forensics.json'}")
    print(f"  - {PHASE6K_DIR / 'stability_gate_1000_corrected.json'}")
    print(f"  - {PHASE6K_DIR / 'solver_consistency_1000.json'}")
    print(f"  - {PHASE6K_DIR / 'full_dev_cv_results.json'}")
    print(f"  - {PHASE6K_DIR / 'full_dev_candidate_comparison.json'}")
    print(f"  - {PHASE6K_DIR / 'full_dev_statistical_tests.json'}")
    print(f"  - {PHASE6K_DIR / 'full_dev_threshold_analysis.json'}")
    print(f"  - {PHASE6K_DIR / 'full_dev_calibration.json'}")
    print(f"  - {PHASE6K_DIR / 'full_dev_error_analysis.json'}")
    print(f"  - {PHASE6K_DIR / 'final_dev_candidate.json'}")
    print(f"  - {PHASE6K_DIR / 'final_model_protocol.json'}")
    print(f"  - {PHASE6K_DIR / 'heldout_validation_results.json'}")
    print(f"  - {PHASE6K_DIR / 'heldout_bootstrap_ci.json'}")
    print(f"  - {PHASE6K_DIR / 'dev_val_generalization.json'}")
    print(f"  - {PHASE6K_DIR / 'heldout_calibration.json'}")
    print(f"  - {PHASE6K_DIR / 'heldout_baseline_comparison.json'}")
    print(f"  - {PHASE6K_DIR / 'heldout_error_analysis.json'}")
    print(f"  - {PHASE6K_DIR / 'dev_val_distribution_shift.json'}")
    print(f"  - {PHASE6K_DIR / 'PHASE6K_WARNING_FORENSICS.md'}")
    print(f"  - {PHASE6K_DIR / 'PHASE6K_CORRECTED_STABILITY_GATE.md'}")
    print(f"  - {PHASE6K_DIR / 'PHASE6K_AMENDMENT.md'}")
    print(f"  - {PHASE6K_DIR / 'PHASE6K_FULL_DEV_MODEL_SELECTION.md'}")
    print(f"  - {PHASE6K_DIR / 'FINAL_PILLAR1_VALIDATION_REPORT.md'}")
    print(f"  - {PHASE6K_DIR / 'final_model/robust_scaler.joblib'}")
    print(f"  - {PHASE6K_DIR / 'final_model/pillar1_logistic_model.joblib'}")
    print(f"  - {PHASE6K_DIR / 'final_model/feature_schema.json'}")
    print(f"  - {PHASE6K_DIR / 'final_model/model_metadata.json'}")
    print(f"  - {PHASE6K_DIR / 'predictions/candidate_3_val_predictions.jsonl'}")
    print(f"  - {PHASE6K_DIR / 'predictions/baseline_single_feature_val_predictions.jsonl'}")
    print(f"  - {PHASE6K_DIR / 'figures/phase6k_final_val_roc.png'}")
    print(f"  - {PHASE6K_DIR / 'figures/phase6k_final_val_pr.png'}")
    print(f"  - {PHASE6K_DIR / 'figures/phase6k_final_val_calibration.png'}")
    print(f"  - {PHASE6K_DIR / 'figures/phase6k_dev_val_metric_comparison.png'}")
    print(f"  - {PHASE6K_DIR / 'figures/phase6k_val_confusion_matrix.png'}")
    print(f"  - {PHASE6K_DIR / 'figures/phase6k_dev_val_feature_shift.png'}")
    print(f"  - {PHASE6K_DIR / 'figures/phase6k_val_error_distributions.png'}")
    print(f"{'=' * 85}\n")

    logger.info("phase6k_orchestrator_complete", elapsed_s=round(elapsed, 2))
    return cache


def main() -> None:
    """CLI entry point for Phase 6K."""
    parser = argparse.ArgumentParser(
        description="HalluciSense Phase 6K — Preprocessing, Feature Selection, Forensics, Full DEV Model Selection & Held-Out Validation",
    )
    _ = parser.parse_args()
    _ = run_phase6k()


if __name__ == "__main__":
    main()
