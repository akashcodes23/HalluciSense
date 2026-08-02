"""Unit tests for Phase 6L.1B Feature Extractor Pipeline & Schema Vector Contract."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.phase6l.config import STRUCTURAL_FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION
from evaluation.phase6l.feature_extractor import extract_structural_features_for_response


def test_feature_extractor_schema_contract():
    """Test extract_structural_features_for_response returns exactly 24 finite numeric features in exact order."""
    response_record = {
        "example_id": "test:24",
        "ground_truth": 1,
        "claim_details": [
            {"claim": "The experiment was conducted in 2022."},
            {"claim": "The experiment was conducted for the first time in 2024."},
        ],
    }

    pairs = [
        {
            "example_id": "test:24",
            "claim_i_index": 0,
            "claim_j_index": 1,
            "claim_i_text": "The experiment was conducted in 2022.",
            "claim_j_text": "The experiment was conducted for the first time in 2024.",
            "c_max": 0.85,
            "c_mean": 0.80,
            "e_ij": 0.05,
            "e_ji": 0.05,
            "embedding_cosine_similarity": 0.75,
        }
    ]

    res = extract_structural_features_for_response(response_record, pairs)

    assert res["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert len(res["features"]) == 24
    assert list(res["features"].keys()) == STRUCTURAL_FEATURE_COLUMNS

    for col in STRUCTURAL_FEATURE_COLUMNS:
        v = res["features"][col]
        assert isinstance(v, float)
        assert np.isfinite(v)


def test_feature_extractor_degenerate_response_n_less_than_2():
    """Test n=0 or n=1 claims returns all 24 features finite (0.0), zero pair count, zero NaN."""
    r_single = {
        "example_id": "test:single",
        "ground_truth": 0,
        "claim_details": [{"claim": "Single isolated claim"}],
    }

    res = extract_structural_features_for_response(r_single, [])

    assert res["num_claims"] == 1
    assert res["pair_count"] == 0
    assert len(res["features"]) == 24

    for col in STRUCTURAL_FEATURE_COLUMNS:
        v = res["features"][col]
        assert isinstance(v, float)
        assert np.isfinite(v)
