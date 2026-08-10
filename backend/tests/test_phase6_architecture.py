"""Phase 6 architectural regression tests.

These tests target the failure modes identified by the blind Phase 5 audit
without matching any Phase 5 holdout case verbatim.
"""

from app.core.engine.temporal import TemporalClaimEngine, TemporalStatus, EpistemicModality
from app.core.engine.pillar1_retrieval import EventTemporalAnchorResolver
from app.core.engine.types import EvidenceItem


def _evidence(snippet: str, claim: str = "") -> EvidenceItem:
    return EvidenceItem(
        snippet=snippet,
        claim=claim,
        similarity_score=0.90,
        source_name="phase6-test",
    )


def test_query_modality_does_not_leak_into_asserted_response():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim(
        "Candidate B won the national election in 2028.",
        query="If Candidate B wins the 2028 election, what happens next?",
    )
    assert result.modality == EpistemicModality.FUTURE_FACT_ASSERTION
    assert result.temporal_status == TemporalStatus.FUTURE_IMPOSSIBLE_FACT
    assert result.temporal_inconsistency_score >= 0.90
    assert result.protected_from_temporal_penalty is False


def test_prediction_phrase_with_intervening_words_is_protected():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim(
        "The agency projected global sea level to reach higher levels by 2050.",
        query="What is the 2050 projection?",
    )
    assert result.modality == EpistemicModality.PREDICTION
    assert result.temporal_inconsistency_score == 0.0
    assert result.protected_from_temporal_penalty is True


def test_meta_claim_is_not_treated_as_temporal_fact():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim(
        "The article falsely reported that the spacecraft landed on Mars in 2019.",
        query="What did the article report?",
    )
    assert result.modality == EpistemicModality.QUOTED_CLAIM
    assert result.temporal_inconsistency_score == 0.0
    assert result.protected_from_temporal_penalty is True


def test_global_evidence_year_match_prevents_background_year_false_positive():
    engine = TemporalClaimEngine()
    evidence = [
        _evidence("An earlier expedition occurred in 1924."),
        _evidence("The successful summit was completed in 1953."),
    ]
    result = engine.analyze_claim(
        "The successful summit occurred in 1953.",
        query="When did the successful summit occur?",
        evidence_items=evidence,
    )
    assert result.temporal_status != TemporalStatus.DATE_MISMATCH
    assert result.temporal_inconsistency_score == 0.0


def test_relational_language_bypasses_naive_year_difference():
    engine = TemporalClaimEngine()
    result = engine.analyze_claim(
        "The earlier event occurred before 1900, while the later event occurred in 1950.",
        query="Which event happened first?",
    )
    assert result.temporal_status == TemporalStatus.TIME_RELATIVE
    assert result.temporal_inconsistency_score == 0.0


def test_dynamic_event_anchor_resolver_can_flag_incompatible_ranges(monkeypatch):
    resolver = EventTemporalAnchorResolver()

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, params=None, **kwargs):
        if params.get("action") == "wbsearchentities":
            label = params["search"].lower()
            entity_id = "Q_FIRST" if "ancient empire" in label else "Q_LATER"
            return FakeResponse({"search": [{"id": entity_id, "label": params["search"]}]})

        entity_id = params["ids"]
        year = "+0400-01-01T00:00:00Z" if entity_id == "Q_FIRST" else "+1500-01-01T00:00:00Z"
        return FakeResponse({
            "entities": {
                entity_id: {
                    "labels": {"en": {"value": entity_id}},
                    "claims": {
                        "P585": [{"mainsnak": {"datavalue": {"value": {"time": year}}}}]
                    },
                }
            }
        })

    monkeypatch.setattr("app.core.engine.pillar1_retrieval.httpx.get", fake_get)
    score, reasoning, anchors = resolver.evaluate(
        "The Ancient Empire collapsed during the Later Civilization."
    )

    assert len(anchors) == 2
    assert score == 0.92
    assert "incompatible" in reasoning
