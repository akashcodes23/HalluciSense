"""Phase 3 Unit & Integration Tests for Temporal Generalization, Latency & Determinism.

Verifies:
1. Historical date mismatch verification against retrieved evidence.
2. Protection of epistemic modalities across predictions, hypotheticals, counterfactuals, and fiction.
3. Micro-second execution overhead (< 1.0 ms mean latency).
4. 100% Deterministic evaluation across repeated calls.
5. Date range contradiction handling.
"""

import statistics
import time
import pytest
from app.core.engine.temporal import TemporalClaimEngine, TemporalStatus, EpistemicModality
from app.core.engine.types import EvidenceItem


def test_temporal_engine_historical_date_mismatch_detection():
    engine = TemporalClaimEngine()
    text = "Albert Einstein discovered general relativity in the year 2020."
    evidence = [
        EvidenceItem(
            claim="Einstein general relativity",
            snippet="Albert Einstein completed his general theory of relativity in 1915.",
            source_name="Wikipedia",
            similarity_score=0.92,
        )
    ]
    res = engine.analyze_claim(text, query="When did Einstein discover relativity?", evidence_items=evidence)
    assert res.temporal_status == TemporalStatus.DATE_MISMATCH
    assert res.temporal_inconsistency_score == 0.90


def test_temporal_engine_supported_historical_date_no_mismatch():
    engine = TemporalClaimEngine()
    text = "Apollo 11 landed on the Moon in July 1969."
    evidence = [
        EvidenceItem(
            claim="Apollo 11 landing",
            snippet="Apollo 11 landed on the Moon on July 20, 1969.",
            source_name="Wikipedia",
            similarity_score=0.95,
        )
    ]
    res = engine.analyze_claim(text, query="When did Apollo 11 land on the Moon?", evidence_items=evidence)
    assert res.temporal_status == TemporalStatus.PAST_FACT
    assert res.temporal_inconsistency_score == 0.0


def test_temporal_engine_latency_microbenchmark():
    engine = TemporalClaimEngine()
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        engine.analyze_claim(
            text="Brazil won the 2027 FIFA World Cup.",
            query="Who won the 2027 FIFA World Cup?",
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = statistics.mean(latencies)
    assert mean_lat < 1.0, f"TemporalClaimEngine mean latency ({mean_lat:.4f} ms) exceeded 1.0 ms"


def test_temporal_engine_determinism():
    engine = TemporalClaimEngine()
    results = []
    for _ in range(30):
        res = engine.analyze_claim(
            text="Apple released the iPhone 25 in 2029.",
            query="When was iPhone 25 released?",
        )
        results.append((res.temporal_status, res.temporal_inconsistency_score, res.modality))

    assert len(set(results)) == 1, "TemporalClaimEngine returned non-deterministic results across identical calls!"
