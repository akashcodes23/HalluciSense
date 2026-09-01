"""Phase 46 — Diagnostic Script for Capital of France & Multi-Claim Cases."""

import os
import sys
from pathlib import Path

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.pipeline import HalluciSensePipeline
from app.core.engine.pipeline import HallucinationDetectionPipeline

def run_diagnostic():
    print("=" * 70)
    print("PHASE 46 PRODUCTION TRACE DIAGNOSTIC")
    print("=" * 70)

    test_cases = [
        ("Case 1 (Question)", "What is the capital of France?"),
        ("Case 2 (True Fact)", "The capital of France is Paris."),
        ("Case 3 (False Fact)", "The capital of France is Berlin."),
        ("Case 4 (True Fact Inv)", "Paris is the capital of France."),
        ("Case 5 (False Fact Inv)", "Berlin is the capital of France."),
        ("Case 6 (True Fact Alt)", "France has Paris as its capital."),
        ("Case 7 (Contradictory Multi-Claim)", "Paris is the capital of France. Berlin is the capital of France."),
        ("Case 8 (Consistent Multi-Claim)", "Paris is the capital of France. Berlin is the capital of Germany."),
    ]

    u_pipe = HalluciSensePipeline()
    h_pipe = HallucinationDetectionPipeline()

    for label, text in test_cases:
        print(f"\n--- {label}: '{text}' ---")
        
        # 1. Unified Inference Pipeline
        u_res = u_pipe.predict(response_text=text)
        print(f"[Unified Pipeline]")
        print(f"  P(H) = {u_res['hallucination_probability']:.4f} | Is Hallucinated: {u_res['is_hallucinated']}")
        print(f"  Claims Extracted ({u_res['claim_count']}): {u_res['claims']}")
        v_summary = u_res.get('verification_summary', {})
        print(f"  Verification Status: {v_summary.get('primary_status')} (Verified: {v_summary.get('verified_claims')}, Contradicted: {v_summary.get('contradicted_claims')}, Insufficient: {v_summary.get('unsupported_claims')})")

        # 2. Hallucination Detection Pipeline (Analyze Engine)
        h_res = h_pipe.analyze(text=text)
        p1 = h_res.pillar1_summary
        p2 = h_res.pillar2_summary
        p3 = h_res.pillar3_summary
        print(f"[Master Analyze Engine]")
        print(f"  Overall H-Score: {h_res.overall_h_score:.4f} | Risk Level: {h_res.overall_risk_level}")
        print(f"  Pillar 1 FE: {getattr(p1, 'factual_error_score', None):.4f}")
        print(f"  Pillar 2 CG: {getattr(p2, 'confidence_gap_score', None)} | Mode: {getattr(p2, 'mode', None)} | Available: {getattr(p2, 'available', None)}")
        print(f"  Pillar 3 CF: {getattr(p3, 'consistency_failure_score', None)} | Mode: {getattr(p3, 'mode', None)} | Available: {getattr(p3, 'available', None)}")

if __name__ == "__main__":
    run_diagnostic()
