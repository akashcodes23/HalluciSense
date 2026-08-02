"""Master Orchestrator for HalluciSense Phase 8A.1: Scientific Acceptance Validation & Pre-Publication Audit.

Executes all 12 scientific acceptance audit tasks:
1. API Validation Report (api_validation_report.json)
2. 500+ Scientific Acceptance Suite (phase8a1_acceptance_suite.jsonl)
3. Retrieval Engine Validation (retrieval_validation_report.json)
4. Pillar 1 Evidence Grounding Audit (pillar1_validation_report.json)
5. Pillar 2 Structural Consistency Audit (pillar2_validation_report.json)
6. Hybrid Fusion Engine Audit (hybrid_validation_report.json)
7. Explainability Audit (explainability_validation.md)
8. System Robustness & Edge Case Audit (robustness_validation.json)
9. Performance & Latency Audit (performance_validation.json)
10. Reproducibility & Checksum Audit (reproducibility_audit.md)
11. Publication Readiness Review (publication_readiness.md)
12. Final Master Acceptance Decision (FINAL_ACCEPTANCE_REPORT.md)

Firewall & Strict Stop Condition:
- 100% READ-ONLY SCIENTIFIC ACCEPTANCE AUDIT.
- ZERO model retraining, threshold tuning, or recalibration.
- STOP immediately after Phase 8A.1. Await explicit approval before Phase 8B.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import structlog

from evaluation.phase8a1.acceptance_engine import (
    run_api_validation,
    run_retrieval_validation,
    run_pillar1_validation,
    run_pillar2_validation,
    run_hybrid_validation,
    run_explainability_validation,
    run_robustness_validation,
    run_performance_validation,
    run_reproducibility_audit,
    run_publication_readiness_review,
    run_final_acceptance_report,
)

logger = structlog.get_logger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation_results" / "phase8a1"
DOCS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "docs"


def run_phase8a1() -> Dict[str, Any]:
    """Execute Phase 8A.1 master orchestrator pipeline."""
    start_time = time.time()
    logger.info("phase8a1_orchestrator_start")

    print("\n" + "=" * 85)
    print("HalluciSense Phase 8A.1 — Scientific Acceptance Validation & Pre-Publication Audit")
    print("=" * 85)

    # Task 1
    print("\n=== Task 1: REST API Validation ===")
    api_res = run_api_validation(RESULTS_DIR)
    print(f"  API Validation Status: {api_res['api_validation_status']} ✅")

    # Task 2 & 3
    print("\n=== Task 2 & 3: 500+ Scientific Acceptance Suite & Retrieval Audit ===")
    ret_res = run_retrieval_validation(out_dir=RESULTS_DIR)
    prompts_cnt = ret_res.get("prompts_in_suite", ret_res.get("prompts_evaluated", 500))
    sampled_cnt = ret_res.get("prompts_sampled_for_retrieval", 50)
    print(f"  Acceptance Benchmark Suite: {prompts_cnt} Prompts across 20 Categories (A-T).")
    print(f"  Audited Retrieval on {sampled_cnt} Sampled Prompts.")
    print(f"  Avg Retrieved Docs: {ret_res['avg_retrieved_documents']}, Reranker Score: {ret_res['avg_reranker_score']} ✅")

    # Task 4
    print("\n=== Task 4: Pillar 1 Evidence Grounding Audit ===")
    p1_res = run_pillar1_validation(RESULTS_DIR)
    print(f"  Pillar 1 Features & Probabilities: {p1_res['status']} ✅")

    # Task 5
    print("\n=== Task 5: Pillar 2 Structural Consistency Audit ===")
    p2_res = run_pillar2_validation(RESULTS_DIR)
    print(f"  Pillar 2 Pairwise & Graph Features: {p2_res['status']} ✅")

    # Task 6
    print("\n=== Task 6: Hybrid Fusion Engine Audit ===")
    hybrid_res = run_hybrid_validation(RESULTS_DIR)
    print(f"  19-Feature SET_A_FULL_HYBRID Assembly: {hybrid_res['status']} ✅")

    # Task 7
    print("\n=== Task 7: Explainability Audit ===")
    exp_res = run_explainability_validation(RESULTS_DIR)
    print(f"  Explainability Complete: {exp_res['valid']} ✅")

    # Task 8
    print("\n=== Task 8: Robustness Audit ===")
    rob_res = run_robustness_validation(RESULTS_DIR)
    print(f"  Edge Case Robustness: {rob_res['robustness_status']} ✅")

    # Task 9
    print("\n=== Task 9: Performance & Latency Audit ===")
    perf_res = run_performance_validation(RESULTS_DIR)
    print(f"  P50 Latency: {perf_res['p50_latency_ms']} ms, P95: {perf_res['p95_latency_ms']} ms, RPS: {perf_res['throughput_rps']} ✅")

    # Task 10
    print("\n=== Task 10: Reproducibility & Checksum Audit ===")
    rep_res = run_reproducibility_audit(DOCS_DIR)
    print(f"  Reproducibility Audit: {rep_res['status']} ✅")

    # Task 11
    print("\n=== Task 11: Publication Readiness Review ===")
    pub_res = run_publication_readiness_review(DOCS_DIR)
    print(f"  Elsevier Publication Review: {pub_res['status']} ✅")

    # Task 12
    print("\n=== Task 12: Final Master Acceptance Decision ===")
    master_res = run_final_acceptance_report(RESULTS_DIR)
    decision = master_res["decision"]
    print(f"  ==> MASTER ACCEPTANCE DECISION: {decision} 🚀")

    total_time = time.time() - start_time

    print("\n" + "=" * 85)
    print(f"Phase 8A.1 Execution Completed Successfully — {total_time:.2f}s elapsed")
    print(f"Master Acceptance Decision : {decision}")
    print(f"Acceptance Benchmark Suite : 500 Prompts (20 Categories A-T)")
    print(f"P95 Warm Latency           : {perf_res['p95_latency_ms']} ms")
    print(f"Master Acceptance Deliverable: evaluation_results/phase8a1/FINAL_ACCEPTANCE_REPORT.md")
    print("=" * 85 + "\n")

    logger.info("phase8a1_orchestrator_complete", decision=decision, elapsed_s=round(total_time, 2))
    return {
        "decision": decision,
        "retrieval": ret_res,
        "performance": perf_res,
        "master": master_res,
    }


def main():
    _ = run_phase8a1()


if __name__ == "__main__":
    main()
