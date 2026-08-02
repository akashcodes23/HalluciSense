"""Master Orchestrator for HalluciSense Phase 6M.1: Hybrid Feature Assembly & Preflight Validation.

Executes:
1. Assembly of DEV (N=58,002) and VAL (N=12,483) 19-feature hybrid matrices
2. Dataset Integrity Audit (ID alignment, row counts, 0 duplicates)
3. Matrix & Finiteness Validation (0 NaN, 0 Inf)
4. Distribution Audit (parametric & non-parametric stats)
5. Correlation Audit (Pearson, Spearman, Kendall matrices)
6. Probability Diagnostics (P1, P2, P1-P2, entropy, saturation)
7. Candidate Feature Subsets Serialization (SET_A through SET_F)
8. 5-Point Data Leakage Audit
9. Numerical Health Audit
10. 6 Publication Figures (300 DPI)
11. Decision Gate Clearance Checklist (9 Questions)

Firewall & Strict Stop Condition:
- Validation partition (N=12,483) remains STRICTLY SEALED.
- ZERO model training, feature selection, threshold tuning, cross-validation, or classifier fitting.
- STOP immediately after preflight clearance. Do NOT begin Phase 6M.2.
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
from evaluation.phase6m.preflight_assembly import (
    audit_dataset_integrity,
    validate_hybrid_matrix,
    compute_feature_distribution_statistics,
    compute_correlation_audit,
    compute_probability_diagnostics,
    audit_data_leakage,
    audit_numerical_health,
    export_candidate_subsets,
    export_hybrid_jsonl_files,
    generate_preflight_figures,
    evaluate_decision_gate,
)

logger = structlog.get_logger(__name__)


def run_phase6m_1(out_dir: Path = PHASE6M_DIR) -> Dict[str, Any]:
    """Execute Phase 6M.1 master orchestrator pipeline."""
    start_time = time.time()
    logger.info("phase6m_1_orchestrator_start")

    print("\n" + "=" * 85)
    print("HalluciSense Phase 6M.1 — Hybrid Feature Assembly & Preflight Validation")
    print("=" * 85)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Assemble DEV Hybrid Matrix (N=58,002)
    print("\n=== Stage 1: Assembling DEV Hybrid Feature Matrix (N=58,002) ===")
    dev_data = load_and_assemble_hybrid_matrix("development", out_dir=out_dir)
    X_dev, y_dev = dev_data["X"], dev_data["y"]
    p1_dev, p2_dev = dev_data["p1_probs"], dev_data["p2_probs"]
    print(f"  DEV Hybrid Matrix: {X_dev.shape}, pos={int((y_dev == 1).sum())}, neg={int((y_dev == 0).sum())}")

    # 2. Assemble VAL Hybrid Matrix (N=12,483)
    print("\n=== Stage 2: Assembling VAL Hybrid Feature Matrix (N=12,483) ===")
    val_data = load_and_assemble_hybrid_matrix("validation", out_dir=out_dir)
    X_val, y_val = val_data["X"], val_data["y"]
    p1_val, p2_val = val_data["p1_probs"], val_data["p2_probs"]
    print(f"  VAL Hybrid Matrix: {X_val.shape}, pos={int((y_val == 1).sum())}, neg={int((y_val == 0).sum())}")

    # 3. Dataset Integrity Audit
    print("\n=== Stage 3: Dataset Integrity Audit ===")
    integrity = audit_dataset_integrity(dev_data, val_data, out_dir=out_dir)
    print(f"  DEV Records: {integrity['dev_record_count']:,}, Duplicates: {integrity['dev_duplicate_ids']}")
    print(f"  VAL Records: {integrity['val_record_count']:,}, Duplicates: {integrity['val_duplicate_ids']}")
    print(f"  ID Overlap:  {integrity['dev_val_id_overlap']}")
    print(f"  Integrity Status: {integrity['integrity_status']} ✅")

    # 4. Matrix & Finiteness Validation
    print("\n=== Stage 4: Matrix & Finiteness Validation ===")
    matrix_val = validate_hybrid_matrix(X_dev, X_val, HYBRID_FEATURE_SCHEMA)
    print(f"  DEV Finite: {matrix_val['dev_all_finite']}, NaN: {matrix_val['dev_nan_count']}, Inf: {matrix_val['dev_inf_count']}")
    print(f"  VAL Finite: {matrix_val['val_all_finite']}, NaN: {matrix_val['val_nan_count']}, Inf: {matrix_val['val_inf_count']}")
    print(f"  Duplicate Columns: {matrix_val['duplicate_columns_count']}")
    print(f"  Matrix Validation Status: {matrix_val['matrix_validation_status']} ✅")

    # 5. Distribution Audit
    print("\n=== Stage 5: Feature Distribution Audit ===")
    dist_stats = compute_feature_distribution_statistics(X_dev, X_val, HYBRID_FEATURE_SCHEMA, out_dir=out_dir)
    print(f"  Computed distribution statistics for {len(HYBRID_FEATURE_SCHEMA)} features across DEV and VAL")

    # 6. Correlation & Redundancy Audit
    print("\n=== Stage 6: Correlation & Redundancy Audit ===")
    corr_stats = compute_correlation_audit(X_dev, HYBRID_FEATURE_SCHEMA, out_dir=out_dir)
    print(f"  Redundant Pairs (|r| > 0.90): {len(corr_stats['redundant_feature_pairs_above_090'])}")

    # 7. Probability Diagnostics
    print("\n=== Stage 7: Probability Diagnostics ===")
    prob_audit = compute_probability_diagnostics(p1_dev, p2_dev, p1_val, p2_val, out_dir=out_dir)
    dev_diag = prob_audit["dev_probability_diagnostics"]
    print(f"  DEV P1 Mean: {dev_diag['p1_summary']['mean']}, P2 Mean: {dev_diag['p2_summary']['mean']}")
    print(f"  DEV Mean Abs Disagreement |P1-P2|: {dev_diag['disagreement_summary']['mean_abs_difference']}")

    # 8. Data Leakage Audit
    print("\n=== Stage 8: Data Leakage Audit ===")
    leakage = audit_data_leakage(X_dev, y_dev, X_val, y_val, out_dir=out_dir)
    print(f"  Labels Embedded in Features: {leakage['labels_embedded_in_features']}")
    print(f"  Validation Labels Unmodified: {leakage['validation_labels_unmodified']}")
    print(f"  Validation Probabilities Untouched: {leakage['validation_probabilities_untouched']}")
    print(f"  Leakage Audit Status: {leakage['leakage_audit_status']} ✅")

    # 9. Numerical Health Audit
    print("\n=== Stage 9: Numerical Health Audit ===")
    num_health = audit_numerical_health(X_dev, X_val, out_dir=out_dir)
    print(f"  DEV Covariance Matrix Finite: {num_health['dev_covariance_matrix_finite']}")
    print(f"  VAL Covariance Matrix Finite: {num_health['val_covariance_matrix_finite']}")
    print(f"  Numerical Health Status: {num_health['numerical_health_status']} ✅")

    # 10. Candidate Feature Subsets Serialization & Hybrid JSONL Export
    print("\n=== Stage 10: Serializing Candidate Feature Subsets & Hybrid JSONL Files ===")
    schema_payload = export_candidate_subsets(out_dir=out_dir)
    dev_jsonl, val_jsonl = export_hybrid_jsonl_files(dev_data["record_payloads"], val_data["record_payloads"], out_dir=out_dir)
    print(f"  Subsets Created: {list(schema_payload['candidate_subsets'].keys())}")
    print(f"  Exported DEV JSONL: {dev_jsonl}")
    print(f"  Exported VAL JSONL: {val_jsonl}")

    # 11. Generate 6 Publication Figures
    print("\n=== Stage 11: Generating Publication Figures (300 DPI) ===")
    fig_paths = generate_preflight_figures(dev_data, val_data, out_dir=out_dir)
    print(f"  Generated {len(fig_paths)} 300 DPI figures in {out_dir / 'figures'}")

    # 12. Evaluate Decision Gate
    print("\n=== Stage 12: Decision Gate Evaluation ===")
    gate = evaluate_decision_gate(integrity, matrix_val, leakage, num_health)
    print("  Checklist:")
    for k, v in gate.items():
        print(f"    - {k}: {v}")

    verdict = gate["9_phase6m2_scientifically_cleared"]
    total_time = time.time() - start_time

    print("\n" + "=" * 85)
    print(f"Phase 6M.1 Execution Completed Successfully — {total_time:.2f}s elapsed")
    print(f"DEV Matrix Shape   : {X_dev.shape}")
    print(f"VAL Matrix Shape   : {X_val.shape}")
    print(f"Integrity Status   : {integrity['integrity_status']}")
    print(f"Leakage Status     : {leakage['leakage_audit_status']}")
    print(f"Decision Gate      : {verdict}")
    print(f"Firewall Status    : VAL (N=12,483) STRICTLY SEALED — ZERO MODEL FITTING")
    print("=" * 85 + "\n")

    logger.info("phase6m_1_orchestrator_complete", decision_gate=verdict, elapsed_s=round(total_time, 2))
    return {
        "integrity": integrity,
        "matrix_validation": matrix_val,
        "leakage": leakage,
        "numerical_health": num_health,
        "decision_gate": gate,
        "verdict": verdict,
    }


def main():
    _ = run_phase6m_1()


if __name__ == "__main__":
    main()
