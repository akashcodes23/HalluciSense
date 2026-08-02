"""Unit tests for Phase 6L.1B Contradiction Graph Builder."""

from __future__ import annotations

import pytest
from evaluation.phase6l.graph_builder import (
    build_contradiction_graph,
    extract_graph_topological_features,
)


def test_graph_builder_zero_edges():
    """Test 0 contradiction edges produces 0 density, 0 max_degree, 0 LCC ratio."""
    pairs = [
        {"claim_i_index": 0, "claim_j_index": 1, "c_max": 0.10},
        {"claim_i_index": 0, "claim_j_index": 2, "c_max": 0.05},
    ]
    res = extract_graph_topological_features(n_claims=3, evaluated_pairs=pairs)
    assert res["contradiction_graph_density"] == 0.0
    assert res["max_contradiction_degree"] == 0.0
    assert res["largest_contradictory_component_ratio"] == 0.0


def test_graph_builder_single_edge():
    """Test single edge on 3 nodes produces exact topological metrics."""
    pairs = [
        {"claim_i_index": 0, "claim_j_index": 1, "c_max": 0.85},
        {"claim_i_index": 0, "claim_j_index": 2, "c_max": 0.10},
        {"claim_i_index": 1, "claim_j_index": 2, "c_max": 0.05},
    ]
    res = extract_graph_topological_features(n_claims=3, evaluated_pairs=pairs)
    # 3 claims -> 3 possible edges. 1 edge -> density = 1/3
    assert pytest.approx(res["contradiction_graph_density"], abs=1e-3) == 0.3333
    # Max degree = 1 -> norm max deg = 1 / 2 = 0.50
    assert res["max_contradiction_degree"] == 0.50
    # LCC size = 2 -> LCC ratio = 2 / 3
    assert pytest.approx(res["largest_contradictory_component_ratio"], abs=1e-3) == 0.6667
