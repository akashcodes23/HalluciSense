"""Unit tests for Phase 6L.1B Temporal Consistency Extractor."""

from __future__ import annotations

import pytest
from evaluation.phase6l.temporal_extractor import extract_temporal_consistency_features


def test_temporal_extractor_year_conflict():
    """Test date conflict is detected for same entity/event with incompatible years."""
    claims = [
        "Company X was founded in 2010.",
        "Company X was founded in 2014.",
    ]
    res = extract_temporal_consistency_features(claims)
    assert res["temporal_conflict_count"] == 1.0
    assert len(res["explainability_records"]) == 1
    assert res["explainability_records"][0]["entity"] == "company x"


def test_temporal_extractor_same_date():
    """Test identical temporal dates do NOT trigger conflict."""
    claims = [
        "Company X was founded in 2010.",
        "The firm Company X was founded in 2010.",
    ]
    res = extract_temporal_consistency_features(claims)
    assert res["temporal_conflict_count"] == 0.0
