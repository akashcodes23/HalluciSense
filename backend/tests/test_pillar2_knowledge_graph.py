"""
Unit tests for HalluciSense Pillar 2 — Module 10.2: Entity + Relation Graph Builder.
"""

import pytest
from app.pillar2.claim_extraction.schemas import CharacterOffsets, ClaimType, ExtractedClaim
from app.pillar2.knowledge_graph.builder import EntityRelationGraphBuilder
from app.pillar2.knowledge_graph.schemas import EntityCategory


@pytest.fixture
def graph_builder():
    return EntityRelationGraphBuilder()


def test_empty_claims_graph(graph_builder):
    graph = graph_builder.build_graph([], graph_id="test_empty")
    assert graph.graph_id == "test_empty"
    assert graph.num_nodes == 0
    assert graph.num_edges == 0
    assert graph.density == 0.0


def test_graph_construction_from_claims(graph_builder):
    claim1 = ExtractedClaim(
        claim_id="claim_001",
        claim_text="Albert Einstein was born in Ulm in 1879.",
        claim_type=ClaimType.TEMPORAL,
        entities=["Albert Einstein", "Ulm"],
        dates=["1879"],
        numbers=[],
        relations=["born in"],
        character_offsets=CharacterOffsets(start=0, end=40),
    )
    claim2 = ExtractedClaim(
        claim_id="claim_002",
        claim_text="Einstein developed the Theory of Relativity.",
        claim_type=ClaimType.SCIENTIFIC,
        entities=["Einstein", "Relativity"],
        dates=[],
        numbers=[],
        relations=["developed"],
        character_offsets=CharacterOffsets(start=41, end=85),
    )

    graph = graph_builder.build_graph([claim1, claim2], graph_id="einstein_graph")
    assert graph.graph_id == "einstein_graph"
    assert graph.num_nodes >= 4
    assert graph.num_edges >= 2

    # Check categories
    categories = {node.category for node in graph.nodes}
    assert EntityCategory.DATE in categories
    assert EntityCategory.PERSON in categories or EntityCategory.LOCATION in categories


def test_adjacency_list(graph_builder):
    claim = ExtractedClaim(
        claim_id="claim_003",
        claim_text="NASA launched Apollo 11 in 1969.",
        claim_type=ClaimType.TEMPORAL,
        entities=["NASA", "Apollo 11"],
        dates=["1969"],
        numbers=[],
        relations=["launched"],
        character_offsets=CharacterOffsets(start=0, end=32),
    )
    graph = graph_builder.build_graph([claim])
    assert len(graph.nodes) >= 3
    assert len(graph.adjacency_list) == len(graph.nodes)
