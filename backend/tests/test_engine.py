import pytest
from app.core.engine.types import RiskLevel, EvidenceItem
from app.core.engine.pillar1_retrieval import Pillar1RetrievalEngine
from app.core.engine.pillar2_confidence import Pillar2ConfidenceEngine
from app.core.engine.pillar3_consistency import Pillar3ConsistencyEngine
from app.core.engine.fusion import FusionEngine
from app.core.engine.pipeline import HallucinationDetectionPipeline

def test_pillar1_retrieval_high_grounding(sample_evidence):
    engine = Pillar1RetrievalEngine()
    text = "Paris is the capital of France."
    result = engine.analyze(text, sample_evidence)

    assert result.factual_error_score < 0.2
    assert len(result.claims) >= 1
    assert "High factual grounding" in result.reasoning

def test_pillar1_retrieval_low_grounding():
    engine = Pillar1RetrievalEngine()
    text = "The Eiffel Tower is located in the center of downtown Tokyo."
    result = engine.analyze(text, [])

    assert result.factual_error_score == 1.0
    assert "Low factual grounding" in result.reasoning

def test_pillar2_confidence_entropy():
    engine = Pillar2ConfidenceEngine()
    # High confidence token (p=0.99) -> low entropy
    entropy_high_conf = engine.calculate_entropy(0.99)
    # Uncertain token (p=0.5) -> maximum binary entropy ~1.0
    entropy_low_conf = engine.calculate_entropy(0.50)

    assert entropy_high_conf < entropy_low_conf
    assert pytest.approx(entropy_low_conf, abs=0.05) == 1.0

def test_pillar2_confidence_tokens():
    engine = Pillar2ConfidenceEngine()
    tokens = ["The", "capital", "is", "Paris"]
    probs = [0.98, 0.95, 0.92, 0.99]
    result = engine.analyze(tokens, probs)

    assert result.avg_probability > 0.90
    assert result.confidence_gap_score < 0.2
    assert "High token confidence" in result.reasoning

def test_pillar3_consistency_evaluation():
    engine = Pillar3ConsistencyEngine()
    primary = "Paris is the capital city of France."
    samples = [
        "The capital of France is Paris.",
        "Paris serves as France's capital city.",
        "Tokyo is the capital of Japan." # Contradiction / outlier
    ]
    result = engine.analyze(primary, samples)

    assert len(result.pairwise_similarities) == 3
    assert result.consistency_failure_score > 0.0

def test_fusion_engine_h_score():
    fusion = FusionEngine(alpha=0.45, beta=0.30, gamma=0.25)
    # Perfect grounding, perfect confidence, perfect consistency -> H-Score = 0.0
    h_score_perfect = fusion.compute_h_score(fe=0.0, cg=0.0, cf=0.0)
    assert h_score_perfect == 0.0
    risk_perfect, color_perfect = fusion.determine_risk_level(h_score_perfect)
    assert risk_perfect == RiskLevel.VERIFIED
    assert color_perfect == "#10B981"

    # Worst case: FE=1.0, CG=1.0, CF=1.0 -> H-Score = 1.0
    h_score_worst = fusion.compute_h_score(fe=1.0, cg=1.0, cf=1.0)
    assert h_score_worst == 1.0
    risk_worst, color_worst = fusion.determine_risk_level(h_score_worst)
    assert risk_worst == RiskLevel.LIKELY_HALLUCINATED
    assert color_worst == "#EF4444"

def test_pipeline_end_to_end(sample_evidence):
    pipeline = HallucinationDetectionPipeline()
    response_text = "Paris is the capital of France. The Eiffel Tower is in Rome."
    
    report = pipeline.analyze_response(
        full_text=response_text,
        token_probabilities=[0.95, 0.90, 0.88, 0.92, 0.96, 0.90, 0.85, 0.88, 0.40, 0.20],
        evidence_items=sample_evidence,
        sample_responses=["Paris is the capital of France and Eiffel Tower is in Paris."]
    )

    assert report.full_text == response_text
    assert 0.0 <= report.overall_h_score <= 1.0
    assert len(report.sentence_analyses) == 2
    assert report.sentence_analyses[0].risk_level in [RiskLevel.VERIFIED, RiskLevel.NEEDS_VERIFICATION]
    assert report.weights_used["alpha_factual_error"] > 0
