"""Unit tests for Phase 6L.1B Entity Consistency Extractor."""

from __future__ import annotations

import pytest
from evaluation.phase6l.entity_extractor import (
    extract_entity_mentions,
    extract_entity_consistency_features,
    normalize_entity_string,
)


def test_normalize_entity_string():
    """Test normalization removes punctuation, extra spaces, and lowercases text."""
    assert normalize_entity_string("  John Smith! ") == "john smith"
    assert normalize_entity_string("Company X, Inc.") == "company x inc"


def test_entity_extractor_conflict_detection():
    """Test entity conflict is detected for same entity with incompatible attributes."""
    claims = [
        "John was born in London.",
        "John was born in Paris.",
    ]
    res = extract_entity_consistency_features(claims)
    assert res["entity_conflict_count"] == 1.0
    assert len(res["explainability_records"]) == 1
    assert res["explainability_records"][0]["entity"] == "john"


def test_entity_extractor_different_entities():
    """Test different entities do NOT generate entity attribute conflicts."""
    claims = [
        "John lives in London.",
        "Mary lives in Paris.",
    ]
    res = extract_entity_consistency_features(claims)
    assert res["entity_conflict_count"] == 0.0
