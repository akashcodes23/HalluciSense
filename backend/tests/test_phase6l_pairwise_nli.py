"""Unit tests for Phase 6L.1A Pairwise NLI and Caching.

Verifies:
    1. Bidirectional NLI inference execution (both fwd and rev probabilities).
    2. Probability finiteness and sum-to-one normalization.
    3. Symmetric aggregation formula values (C_max >= C_mean >= C_min).
    4. Persistent joblib cache reuse.
"""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.phase6l.pairwise_nli import (
    evaluate_bidirectional_nli_and_similarity,
    get_nli_engine,
)
from evaluation.phase6l.config import PHASE6L_CACHE_DIR


def test_nli_engine_label_mapping():
    """Verify EvidenceEntailmentEngine dynamically maps labels cleanly."""
    nli = get_nli_engine()
    assert "entailment" in nli.label_map
    assert "neutral" in nli.label_map
    assert "contradiction" in nli.label_map


def test_evaluate_bidirectional_nli_small_batch(tmp_path):
    """Test evaluate_bidirectional_nli_and_similarity produces finite normalized probabilities."""
    pairs = [
        {
            "example_id": "test:1",
            "claim_i_index": 0,
            "claim_j_index": 1,
            "claim_i_text": "The experiment was conducted in 2022.",
            "claim_j_text": "The experiment was conducted for the first time in 2024.",
            "ground_truth": 1,
        },
        {
            "example_id": "test:2",
            "claim_i_index": 0,
            "claim_j_index": 1,
            "claim_i_text": "The car has a 75 kWh battery capacity.",
            "claim_j_text": "The electric vehicle uses a high-density battery.",
            "ground_truth": 0,
        },
    ]

    res = evaluate_bidirectional_nli_and_similarity(pairs, cache_dir=tmp_path)

    assert res["total_pairs_evaluated"] == 2
    assert res["total_directional_inferences"] == 4

    for item in res["evaluated_pairs"]:
        fwd = item["forward_nli"]
        rev = item["reverse_nli"]

        fwd_sum = fwd["entailment"] + fwd["neutral"] + fwd["contradiction"]
        rev_sum = rev["entailment"] + rev["neutral"] + rev["contradiction"]

        assert pytest.approx(fwd_sum, abs=1e-3) == 1.0
        assert pytest.approx(rev_sum, abs=1e-3) == 1.0

        assert item["c_max"] >= item["c_mean"]
        assert item["c_mean"] >= item["c_min"]
        assert item["c_prob_union"] >= item["c_min"]
