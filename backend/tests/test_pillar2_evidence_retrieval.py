"""
Unit tests for HalluciSense Pillar 2 — Module 10.3: Evidence Retrieval Layer.
"""

import pytest
from app.pillar2.evidence_retrieval.manager import EvidenceRetrievalManager
from app.pillar2.evidence_retrieval.providers import MockEvidenceProvider, WikipediaProvider
from app.pillar2.evidence_retrieval.schemas import RetrievalRequest


@pytest.fixture
def retrieval_manager():
    return EvidenceRetrievalManager()


def test_list_providers(retrieval_manager):
    providers = retrieval_manager.list_available_providers()
    assert "Wikipedia" in providers
    assert "Wikidata" in providers
    assert "CrossRef" in providers
    assert "Semantic Scholar" in providers
    assert "PubMed" in providers
    assert "GovData" in providers
    assert "MockProvider" in providers


@pytest.mark.asyncio
async def test_concurrent_retrieval_all_providers(retrieval_manager):
    req = RetrievalRequest(query="CRISPR gene editing", max_results_per_provider=2)
    res = await retrieval_manager.retrieve_evidence(req)
    assert res.query == "CRISPR gene editing"
    assert res.total_retrieved > 5
    assert len(res.providers_queried) >= 6
    assert res.total_latency_ms >= 0.0

    # Verify authority score presence
    for item in res.items:
        assert 0.0 <= item.authority_score <= 1.0
        assert item.snippet != ""
        assert item.source in res.providers_queried


@pytest.mark.asyncio
async def test_single_provider_retrieval(retrieval_manager):
    req = RetrievalRequest(
        query="Albert Einstein relativity",
        providers=["Wikipedia"],
        max_results_per_provider=2,
    )
    res = await retrieval_manager.retrieve_evidence(req)
    assert res.total_retrieved > 0
    assert res.providers_queried == ["Wikipedia"]
    assert all(item.source == "Wikipedia" for item in res.items)


@pytest.mark.asyncio
async def test_custom_provider_injection():
    mgr = EvidenceRetrievalManager(providers=[MockEvidenceProvider()])
    assert mgr.list_available_providers() == ["MockProvider"]
    req = RetrievalRequest(query="Test query")
    res = await mgr.retrieve_evidence(req)
    assert res.total_retrieved == 1
    assert res.items[0].source == "MockProvider"
