"""Unit tests for Phase 6L.1B Numerical Consistency Extractor."""

from __future__ import annotations

import pytest
from evaluation.phase6l.numeric_extractor import (
    parse_numeric_value,
    extract_numeric_consistency_features,
)


def test_parse_numeric_value_magnitudes():
    """Test magnitude suffix scaling."""
    assert parse_numeric_value("1", "million") == 1e6
    assert parse_numeric_value("10", "M") == 1e7
    assert parse_numeric_value("500", "k") == 5e5


def test_numeric_extractor_same_context_conflict():
    """Test numeric conflict is detected for same semantic context with relative diff > 1%."""
    claims = [
        "The battery capacity is 75 kWh.",
        "The vehicle has a 52 kWh battery capacity.",
    ]
    res = extract_numeric_consistency_features(claims)
    assert res["numeric_conflict_count"] == 1.0
    assert res["max_numeric_disagreement"] > 0.30
    assert len(res["explainability_records"]) == 1


def test_numeric_extractor_different_contexts():
    """Test unrelated numbers in different contexts are NOT flagged as conflicts."""
    claims = [
        "Revenue was $10M.",
        "Employees numbered 500 people.",
    ]
    res = extract_numeric_consistency_features(claims)
    assert res["numeric_conflict_count"] == 0.0
