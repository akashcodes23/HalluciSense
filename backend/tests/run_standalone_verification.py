"""
Standalone Verification Script for Module 1 (No external dependencies required).
Verifies math formulas, classes, data structures, and pipeline execution.
"""
import sys
import os
from pathlib import Path

# Ensure backend directory is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.engine.types import (
    RiskLevel, EvidenceItem, TokenAnalysis,
    Pillar1Result, Pillar2Result, Pillar3Result, SentenceAnalysis, HallucinationReport
)
from app.core.engine.pillar1_retrieval import Pillar1RetrievalEngine
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.engine.fusion import FusionEngine
from app.core.engine.pipeline import HallucinationDetectionPipeline

def run_verification():
    print("==================================================")
    print(" HALLUCISENSE MODULE 1 VERIFICATION ")
    print("==================================================")

    # 1. Test Config & Weights
    print("\n[1/5] Validating Config & Settings...")
    assert settings.ALPHA_FACTUAL_ERROR == 0.45
    assert settings.BETA_CONFIDENCE_GAP == 0.30
    assert settings.GAMMA_CONSISTENCY_FAILURE == 0.25
    print("   -> Config defaults loaded successfully (alpha=0.45, beta=0.30, gamma=0.25).")

    # 2. Test Pillar 1 Retrieval Verification
    print("\n[2/5] Testing Pillar 1 (Retrieval Verification)...")
    p1 = Pillar1RetrievalEngine()
    evidence = [
        EvidenceItem(
            claim="Paris is the capital of France",
            snippet="Paris is the capital and largest city of France.",
            source_name="Wikipedia: Paris",
            similarity_score=0.92,
            is_supporting=True
        )
    ]
    res1 = p1.analyze("Paris is the capital of France.", evidence)
    assert res1.factual_error_score < 0.15
    print(f"   -> Pillar 1 FE Score: {res1.factual_error_score} ({res1.reasoning})")

    # 3. Test Pillar 2 Confidence Analysis
    print("\n[3/5] Testing Pillar 2 (Confidence Analysis & Entropy)...")
    p2 = Pillar2ConfidenceEngine()
    tokens = ["Paris", "is", "the", "capital"]
    probs = [0.98, 0.95, 0.99, 0.92]
    res2 = p2.analyze(tokens, probs)
    assert res2.avg_probability > 0.90
    assert res2.confidence_gap_score < 0.20
    print(f"   -> Pillar 2 CG Score: {res2.confidence_gap_score} (Avg Prob: {res2.avg_probability})")

    # 4. Test Pillar 3 Consistency Checking
    print("\n[4/5] Testing Pillar 3 (Consistency Checking)...")
    p3 = Pillar3ConsistencyEngine()
    samples = ["The capital city of France is Paris.", "Paris is France's capital."]
    res3 = p3.analyze("Paris is the capital of France.", samples)
    assert res3.consistency_failure_score < 0.50
    print(f"   -> Pillar 3 CF Score: {res3.consistency_failure_score} ({res3.reasoning})")

    # 5. Test Master Fusion & Pipeline
    print("\n[5/5] Testing Master Pipeline & H-Score Fusion...")
    pipeline = HallucinationDetectionPipeline()
    response_text = "Paris is the capital of France. The Eiffel Tower is located in Tokyo."
    report = pipeline.analyze_response(
        full_text=response_text,
        token_probabilities=[0.99, 0.95, 0.98, 0.94, 0.99, 0.95, 0.90, 0.88, 0.30, 0.15],
        evidence_items=evidence,
        sample_responses=["Paris is the capital of France. Eiffel Tower is in Paris."]
    )

    print(f"   -> Full Text: '{report.full_text}'")
    print(f"   -> Aggregate H-Score: {report.overall_h_score}")
    print(f"   -> Overall Risk Level: {report.overall_risk_level.value}")
    print(f"   -> Total Sentences Analyzed: {len(report.sentence_analyses)}")
    
    for s in report.sentence_analyses:
        print(f"      [Sentence {s.sentence_id}] Text: '{s.text}' | H-Score: {s.hallucination_score} | Risk: {s.risk_level.value} ({s.color_code})")

    assert len(report.sentence_analyses) == 2
    assert 0.0 <= report.overall_h_score <= 1.0

    print("\n==================================================")
    print(" MODULE 1 VERIFICATION PASSED SUCCESSFULLY! ")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
