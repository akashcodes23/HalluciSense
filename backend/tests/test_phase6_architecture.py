"""Phase 6 architectural remediation tests.

These tests target general mechanisms rather than benchmark-specific strings.
The original Phase 5 70-case holdout remains untouched and is reserved for final evaluation.
"""

from app.core.engine.temporal import EpistemicModality, TemporalClaimEngine, TemporalStatus
from app.core.engine.types import EvidenceItem


def evidence(claim, snippet, score=0.9, supporting=True):
    return EvidenceItem(
        claim=claim,
        snippet=snippet,
        source_name="test-source",
        similarity_score=score,
        is_supporting=supporting,
    )


def test_query_modality_does_not_leak_into_asserted_response():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim(
        "Candidate A won the 2028 national election.",
        query="If Candidate A wins the 2028 election, what happens?",
    )
    assert result.query_modality in (EpistemicModality.CONDITIONAL, EpistemicModality.HYPOTHETICAL)
    assert result.response_modality == EpistemicModality.ASSERTED_FACT
    assert result.modality == EpistemicModality.FUTURE_FACT_ASSERTION
    assert result.temporal_inconsistency_score >= 0.90


def test_query_prediction_does_not_protect_future_asserted_response():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim(
        "The company released its 6G handset in 2030.",
        query="Will 6G handsets launch in 2030?",
    )
    assert result.query_modality == EpistemicModality.PREDICTION
    assert result.response_modality == EpistemicModality.ASSERTED_FACT
    assert result.temporal_status == TemporalStatus.FUTURE_IMPOSSIBLE_FACT


def test_global_evidence_year_consistency_ignores_background_years():
    engine = TemporalClaimEngine()
    items = [
        evidence(
            "The mountain was first climbed in 1953.",
            "Earlier expeditions attempted the mountain in 1924. The successful ascent occurred in 1953.",
        ),
        evidence(
            "The mountain was first climbed in 1953.",
            "The mountain was successfully summited in 1953 after several earlier attempts.",
        ),
    ]
    assert engine.verify_evidence_date_mismatch(
        "The mountain was first successfully summited in 1953.", items
    ) is None


def test_relational_temporal_claim_skips_naive_year_distance_penalty():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim(
        "Central banks raised rates following inflation spikes after 2021.",
        evidence_items=[
            evidence("Inflation rose after 2021.", "Inflation accelerated after 2021 before monetary policy tightened.")
        ],
    )
    assert result.temporal_status == TemporalStatus.TIME_RELATIVE
    assert result.temporal_inconsistency_score == 0.0


def test_intervening_words_prediction_marker_is_detected():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim(
        "The population is projected by the United Nations to reach 9.7 billion by 2050."
    )
    assert result.modality == EpistemicModality.PREDICTION
    assert result.protected_from_temporal_penalty is True


def test_fictional_universe_marker_is_structural_not_entity_specific():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim(
        "In the fictional Orion universe, the colony reaches Mars in 2084."
    )
    assert result.modality == EpistemicModality.FICTIONAL
    assert result.temporal_inconsistency_score == 0.0


def test_negated_fact_is_not_allowed_to_fabricate_factual_certainty():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim("The country did not declare independence in 1776.")
    assert result.modality == EpistemicModality.NEGATED_FACT
    assert result.protected_from_temporal_penalty is True


def test_no_year_event_relation_uses_retrieved_temporal_anchors():
    engine = TemporalClaimEngine()
    items = [
        evidence(
            "Western Roman Empire collapsed in 476.",
            "The Western Roman Empire collapsed in 476 CE.",
            score=0.95,
        ),
        evidence(
            "The European Renaissance began in the 14th century and continued into the 17th century.",
            "The Renaissance began in Italy in the 14th century and spread across Europe through the 16th century.",
            score=0.95,
        ),
    ]
    result = engine.analyze_claim(
        "The Western Roman Empire collapsed during the European Renaissance.",
        evidence_items=items,
    )
    assert result.temporal_inconsistency_score >= 0.90
    assert result.temporal_status == TemporalStatus.DATE_MISMATCH


def test_no_year_event_ordering_can_detect_anachronism():
    engine = TemporalClaimEngine()
    items = [
        evidence("Apollo 11 landed on the Moon in 1969.", "Apollo 11 achieved the first crewed lunar landing in 1969.", score=0.95),
        evidence("Smartphones became widespread after 2007.", "Modern smartphones became widespread after the introduction of the iPhone in 2007.", score=0.95),
    ]
    result = engine.analyze_claim(
        "The first manned moon landing happened after the widespread adoption of smartphones.",
        evidence_items=items,
    )
    assert result.temporal_inconsistency_score >= 0.90
    assert result.temporal_status == TemporalStatus.DATE_MISMATCH


def test_meta_claim_is_distinguished_from_direct_assertion():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim(
        "The article falsely reported that the moon landing occurred in 2015, which is untrue."
    )
    assert result.modality == EpistemicModality.QUOTED_CLAIM
    assert result.temporal_inconsistency_score == 0.0
