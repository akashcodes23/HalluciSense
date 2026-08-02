"""Phase 6L.1C — Master Orchestrator for Full DEV Structural Feature Reconstruction.

Usage:
    python -m evaluation.phase6l.run_phase6l_1c
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import structlog

from evaluation.phase6l.config import DEV_FEATURES_JSONL, PHASE6L_DIR
from evaluation.phase6l.preflight_activation import run_rare_feature_activation_preflight
from evaluation.phase6l.full_dev_reconstruction import (
    execute_full_dev_sharded_reconstruction,
    run_post_reconstruction_audits,
)

logger = structlog.get_logger(__name__)


def run_phase6l_1c() -> Dict[str, Any]:
    """Orchestrate Phase 6L.1C full DEV structural feature matrix reconstruction.

    Returns:
        Dict containing reconstruction summary and decision gate answers.
    """
    logger.info("phase6l_1c_orchestrator_start")
    t0 = time.time()

    print(f"\n{'=' * 85}")
    print("HalluciSense Phase 6L.1C — Full DEV Structural Feature Reconstruction Gate")
    print(f"{'=' * 85}")

    # 1. Execute Critical Preflight: Rare-Feature Activation Audit (Label-Free)
    print("=== Task 1: Label-Free Rare-Feature Activation Preflight Audit ===")
    preflight_res = run_rare_feature_activation_preflight(dev_path=DEV_FEATURES_JSONL, out_dir=PHASE6L_DIR)
    print(f"  Responses Scanned  : {preflight_res['total_responses_scanned']:,}")
    print(f"  Entity Conflicts   : {preflight_res['activation_counts']['entity_conflict_activations']:,} ({preflight_res['activation_prevalence_pct']['entity_conflict_pct']:.2f}%)")
    print(f"  Numeric Conflicts  : {preflight_res['activation_counts']['numeric_conflict_activations']:,} ({preflight_res['activation_prevalence_pct']['numeric_conflict_pct']:.2f}%)")
    print(f"  Temporal Conflicts : {preflight_res['activation_counts']['temporal_conflict_activations']:,} ({preflight_res['activation_prevalence_pct']['temporal_conflict_pct']:.2f}%)")
    print(f"  Preflight Decision : {preflight_res['preflight_status']}\n")

    # 2. Execute Resumable Full DEV Sharded Reconstruction (N=58,002)
    print("=== Task 2: Executing Resumable Full DEV Reconstruction (N=58,002) ===")
    recon_payload = execute_full_dev_sharded_reconstruction(dev_path=DEV_FEATURES_JSONL, out_dir=PHASE6L_DIR)
    print(f"  Total Records Saved: {recon_payload['total_records_reconstructed']:,}")
    print(f"  Shards Completed   : {recon_payload['shards_count']}")
    print(f"  Reconstruction Time: {recon_payload['total_elapsed_seconds']:.2f}s\n")

    # 3. Post-Reconstruction Label-Free Audits & Reports
    print("=== Task 3: Running Label-Free Post-Reconstruction Audits & Generating Reports ===")
    audit_res = run_post_reconstruction_audits(recon_payload, out_dir=PHASE6L_DIR)

    elapsed = time.time() - t0
    print(f"\n{'=' * 85}")
    print(f"Phase 6L.1C Execution Completed — {elapsed:.2f}s elapsed")
    print(f"Full DEV Feature Matrix: {recon_payload['final_merged_path']}")
    print(f"Report Generated: {PHASE6L_DIR / 'PHASE6L_1C_FULL_DEV_RECONSTRUCTION.md'}")
    print(f"{'=' * 85}\n")

    logger.info("phase6l_1c_orchestrator_complete", elapsed_s=round(elapsed, 2))
    return {
        "preflight_res": preflight_res,
        "recon_payload": recon_payload,
        "audit_res": audit_res,
    }


def main() -> None:
    """CLI entry point for Phase 6L.1C."""
    parser = argparse.ArgumentParser(
        description="HalluciSense Phase 6L.1C — Full DEV Structural Feature Reconstruction Gate",
    )
    _ = parser.parse_args()
    _ = run_phase6l_1c()


if __name__ == "__main__":
    main()
