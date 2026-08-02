"""Unit tests for Phase 6L.1A Claim Pairs Generation and Complexity Audit.

Verifies:
    1. Pair count formula correctness: M_r = n_r(n_r - 1)/2.
    2. No self-pairs (i != j).
    3. No duplicate unordered pairs (i < j).
    4. Deterministic DEV research subset sampling.
"""

from __future__ import annotations

import pytest
from evaluation.phase6l.claim_pairs import (
    generate_unordered_claim_pairs,
    extract_deterministic_dev_subset,
    audit_dev_pair_complexity,
)
from evaluation.phase6l.config import DEV_FEATURES_JSONL


def test_generate_unordered_claim_pairs_basic():
    """Test generate_unordered_claim_pairs produces exact n(n-1)/2 pairs with no self/duplicate pairs."""
    response = {
        "example_id": "test:1",
        "ground_truth": 1,
        "claim_details": [
            {"claim": "Claim A"},
            {"claim": "Claim B"},
            {"claim": "Claim C"},
            {"claim": "Claim D"},
        ],
    }

    pairs = generate_unordered_claim_pairs(response)

    # 4 claims -> 4 * 3 / 2 = 6 pairs
    assert len(pairs) == 6

    for p in pairs:
        assert p["claim_i_index"] < p["claim_j_index"]
        assert p["claim_i_text"] != p["claim_j_text"]


def test_generate_unordered_claim_pairs_single_or_zero():
    """Test 0 or 1 claim response produces 0 pairs."""
    r0 = {"example_id": "t0", "claim_details": []}
    r1 = {"example_id": "t1", "claim_details": [{"claim": "Single claim"}]}

    assert len(generate_unordered_claim_pairs(r0)) == 0
    assert len(generate_unordered_claim_pairs(r1)) == 0


def test_extract_deterministic_dev_subset():
    """Test extract_deterministic_dev_subset produces identical sampled subset across calls."""
    sub1 = extract_deterministic_dev_subset(DEV_FEATURES_JSONL, subset_size=50, seed=42)
    sub2 = extract_deterministic_dev_subset(DEV_FEATURES_JSONL, subset_size=50, seed=42)

    assert len(sub1) == 50
    assert len(sub2) == 50
    ids1 = [r["example_id"] for r in sub1]
    ids2 = [r["example_id"] for r in sub2]
    assert ids1 == ids2
