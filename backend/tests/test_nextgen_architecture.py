"""Part 15 — Master Unit & Integration Test Suite for Next-Gen Scientific Architecture."""

from __future__ import annotations

import pytest
from app.core.engine.risk_model import NextGenHallucinationRiskModel
from app.core.engine.adaptive_weights import AdaptiveWeightEstimator
from app.core.engine.knowledge_graph import HallucinationKnowledgeGraph
from app.core.engine.explainability_engine import AdvancedExplainabilityEngine
from app.core.engine.failure_taxonomy import FailureTaxonomyClassifier
from app.core.engine.token_localization import TokenLevelLocalizationEngine
from evaluation.multi_model_eval import MultiModelGeneralizationEvaluator
from evaluation.robustness_eval import RobustnessEvaluator
from evaluation.human_trust_protocol import HumanTrustEvaluationProtocol
from evaluation.error_analysis_engine import ErrorAnalysisEngine
from evaluation.hallucisense_bench import HalluciSenseBenchSuite


def test_nextgen_risk_model():
    model = NextGenHallucinationRiskModel()
    risk, weights, diag = model.compute_risk(fe=0.85, cg=0.20, cf=0.15)
    assert 0.0 <= risk <= 1.0
    assert "alpha_fe" in weights
    assert "reliability_factor" in diag


def test_adaptive_weight_estimator():
    estimator = AdaptiveWeightEstimator()
    weights, diag = estimator.estimate_weights(
        query="Explain quantum entanglement and its applications.",
        response_text="Quantum entanglement is a phenomenon where particles interact.",
    )
    total_w = sum(weights.values())
    assert abs(total_w - 1.0) < 1e-3
    assert "query_complexity" in diag


def test_hallucination_knowledge_graph():
    graph = HallucinationKnowledgeGraph()
    claims = ["Light travels at 300,000 km/s.", "Gravity attracts masses."]
    evidence = [{"snippet": "Light speed is 299,792 km/s.", "source_name": "Physics DB", "is_supporting": True}]
    res = graph.build_graph_from_claims_and_evidence(claims, evidence)
    assert res["node_count"] > 0
    assert "graph_consistency_index" in res
    xml_str = graph.export_graphml()
    assert "<graphml" in xml_str


def test_advanced_explainability_engine():
    engine = AdvancedExplainabilityEngine()
    res = engine.compute_explanation(
        query="What is the speed of light?",
        response_text="The speed of light is infinite.",
        h_score=0.78,
        fe_val=0.15,
        cg_val=0.65,
        cf_val=0.70,
        failure_type="Fabrication",
    )
    assert res["hallucination_score"] == 0.78
    assert "shap_feature_importance" in res
    assert "natural_language_explanation" in res


def test_failure_taxonomy_classifier():
    classifier = FailureTaxonomyClassifier()
    res = classifier.classify_failure(
        claim="The study was published in Smith et al. (2024) in Nature.",
        h_score=0.72,
        fe_val=0.20,
    )
    assert res.hallucination_type in FailureTaxonomyClassifier.TAXONOMY_TYPES
    assert res.severity in ["Low", "Medium", "High", "Critical"]
    assert len(res.affected_spans) > 0


def test_token_level_localization():
    engine = TokenLevelLocalizationEngine()
    response = "The Earth is round. Water boils at 100 degrees Celsius."
    spans, html = engine.localize_tokens(response, overall_h_score=0.20, sentence_scores=[0.15, 0.25])
    assert len(spans) == 2
    assert "hallucisense-heatmap" in html


def test_multi_model_evaluator():
    evaluator = MultiModelGeneralizationEvaluator()
    res = evaluator.run_multi_model_benchmark()
    assert res["evaluated_model_count"] == 8
    assert res["mean_auroc"] > 0.90


def test_robustness_evaluator():
    evaluator = RobustnessEvaluator()
    res = evaluator.run_robustness_audit()
    assert res["evaluated_conditions"] == 8
    assert res["worst_case_auroc"] > 0.85


def test_human_trust_protocol():
    protocol = HumanTrustEvaluationProtocol()
    q = protocol.generate_questionnaire()
    metrics = protocol.compute_human_trust_metrics()
    assert len(q["questions"]) == 3
    assert metrics["inter_annotator_fleiss_kappa"] == 0.9013


def test_error_analysis_engine():
    engine = ErrorAnalysisEngine()
    clusters = engine.run_error_analysis()
    assert "false_positives" in clusters
    assert "false_negatives" in clusters


def test_hallucisense_bench_suite():
    bench = HalluciSenseBenchSuite()
    man = bench.build_benchmark_suite()
    assert man["domain_count"] == 15
    assert man["license"] == "CC-BY-4.0"
