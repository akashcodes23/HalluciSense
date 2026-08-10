"""Phase 4 Unit & Integration Tests for Temporal Adversarial Modality Resolution.

Verifies:
1. Protection of hypothetical conditionals ("If Brazil won the 2030 World Cup...").
2. Protection of future predictions ("Spain is predicted to win in 2030.").
3. Protection of negated facts ("Before 1969, no human had walked on the Moon.").
4. Detection of historical date mismatches ("George Washington elected in 2004.").
5. Sub-millisecond latency overhead over 1000 iterations.
6. 100% Deterministic evaluation across repeated calls.
"""

import statistics
import time
import pytest
from app.core.engine.pipeline import HallucinationDetectionPipeline
from app.core.engine.temporal import TemporalClaimEngine, TemporalStatus, EpistemicModality
from app.core.engine.types import EvidenceItem


def test_temporal_adversarial_hypothetical_conditional_protection():
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze(
        "If Brazil won the 2030 FIFA World Cup, they would celebrate their sixth title.",
        query="If Brazil won the 2030 World Cup, what would happen?",
    )
    assert float(report.overall_h_score) < 0.35
    assert str(report.overall_risk_level.value) == "VERIFIED"


def test_temporal_adversarial_future_prediction_protection():
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze(
        "Spain is predicted to win the 2030 FIFA World Cup.",
        query="Who is expected to win the 2030 World Cup?",
    )
    assert float(report.overall_h_score) < 0.35
    assert str(report.overall_risk_level.value) == "VERIFIED"


def test_temporal_adversarial_negated_claim_protection():
    pipeline = HallucinationDetectionPipeline()
    report = pipeline.analyze(
        "Before 1969, no human astronaut had ever walked on the lunar surface.",
        query="Did humans walk on the Moon before 1969?",
    )
    assert float(report.overall_h_score) < 0.35
    assert str(report.overall_risk_level.value) == "VERIFIED"


def test_temporal_adversarial_historical_date_mismatch_flagged():
    engine = TemporalClaimEngine()
    text = "George Washington was elected the first US President in 2004."
    evidence = [
        EvidenceItem(
            claim="George Washington election",
            snippet="George Washington was unanimously elected president in 1788.",
            source_name="Wikipedia",
            similarity_score=0.94,
        )
    ]
    res = engine.analyze_claim(text, query="When was George Washington elected?", evidence_items=evidence)
    assert res.temporal_status == TemporalStatus.DATE_MISMATCH
    assert res.temporal_inconsistency_score == 0.90


def test_temporal_adversarial_1000_iteration_latency():
    engine = TemporalClaimEngine()
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        engine.analyze_claim(
            text="If Brazil won the 2030 World Cup, they would celebrate.",
            query="If Brazil won in 2030?",
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = statistics.mean(latencies)
    assert mean_lat < 1.0, f"TemporalClaimEngine mean latency ({mean_lat:.4f} ms) exceeded 1.0 ms"


def test_temporal_adversarial_determinism_30_runs():
    engine = TemporalClaimEngine()
    results = []
    for _ in range(30):
        res = engine.analyze_claim(
            text="If Brazil won the 2030 World Cup, they would celebrate.",
            query="If Brazil won in 2030?",
        )
        results.append((res.temporal_status, res.temporal_inconsistency_score, res.modality, res.protected_from_temporal_penalty))

    assert len(set(results)) == 1, "TemporalClaimEngine returned non-deterministic results across identical calls!"
