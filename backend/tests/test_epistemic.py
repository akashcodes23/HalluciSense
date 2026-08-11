"""Unit tests for Epistemic Modality Resolution Engine (Phase 6D)."""

import pytest
from app.core.engine.epistemic import EpistemicResolver, EpistemicFrame
from app.core.engine.temporal import EpistemicModality


@pytest.fixture
def resolver():
    return EpistemicResolver()


def test_asserted_fact_resolution(resolver):
    frame = resolver.resolve_frame("Apollo 11 landed on the Moon in 1969.")
    assert frame.modality == EpistemicModality.ASSERTED_FACT
    assert not frame.is_protected
    assert frame.confidence >= 0.90


def test_prediction_resolution(resolver):
    frame = resolver.resolve_frame("Artemis IV is targeted to launch in 2028.")
    assert frame.modality == EpistemicModality.PREDICTION
    assert frame.is_protected
    assert len(frame.trigger_spans) > 0


def test_hypothetical_resolution(resolver):
    frame = resolver.resolve_frame("What if commercial fusion power succeeds by 2038?")
    assert frame.modality == EpistemicModality.HYPOTHETICAL
    assert frame.is_protected
    assert "what if" in [t.lower() for t in frame.trigger_spans]


def test_counterfactual_resolution(resolver):
    frame = resolver.resolve_frame("If Candidate A had won the 2024 election, policies would have been different.")
    assert frame.modality == EpistemicModality.COUNTERFACTUAL
    assert frame.is_protected


def test_fictional_resolution(resolver):
    frame = resolver.resolve_frame("In the sci-fi novel, humans colonized Mars in 2045.")
    assert frame.modality == EpistemicModality.FICTIONAL
    assert frame.is_protected


def test_quoted_claim_resolution(resolver):
    frame = resolver.resolve_frame("The article falsely reported that the bridge collapsed in 2018.")
    assert frame.modality == EpistemicModality.QUOTED_CLAIM
    assert frame.is_protected
    assert frame.is_quoted or frame.is_negated


def test_negated_fact_resolution(resolver):
    frame = resolver.resolve_frame("There is no evidence that the city was destroyed in 1400.")
    assert frame.modality == EpistemicModality.NEGATED_FACT
    assert frame.is_protected
    assert frame.is_negated


def test_empty_string_handling(resolver):
    frame = resolver.resolve_frame("")
    assert frame.modality == EpistemicModality.UNKNOWN
    assert frame.confidence == 0.0
    assert not frame.is_protected


def test_query_response_independence(resolver):
    query_frame = resolver.resolve_frame("What if fusion power reaches scale in 2035?", is_query=True)
    resp_frame = resolver.resolve_frame("Fusion power reached commercial scale in 2025.", is_query=False)

    assert query_frame.modality == EpistemicModality.HYPOTHETICAL
    assert resp_frame.modality == EpistemicModality.ASSERTED_FACT
    assert not resp_frame.is_protected
