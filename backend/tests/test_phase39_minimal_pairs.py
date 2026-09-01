"""Phase 39 — Minimal Pair Discrimination & Semantic Separation Test Suite.

Evaluates minimal pair discrimination on key canonical factual mutations:
- Factual swaps (Paris vs Berlin)
- Entity swaps (Einstein vs Newton)
- Negations (boils vs does not boil)
- Temporal mutations (1947 vs 1958)
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.inference.semantic_nli import get_semantic_nli_adapter


@pytest.fixture(scope="module")
def adapter():
    return get_semantic_nli_adapter()


def test_paris_berlin_minimal_pair(adapter):
    """Test A01: Paris (true) vs Berlin (false) given France evidence."""
    ev = "Paris is the capital and most populous city of France."
    res_true = adapter.evaluate_pair(claim="The capital of France is Paris.", evidence=ev)
    res_false = adapter.evaluate_pair(claim="The capital of France is Berlin.", evidence=ev)

    assert res_true["label"] == "entailment"
    assert res_false["label"] == "contradiction"
    assert res_false["contradiction"] > res_true["contradiction"] + 0.50


def test_relativity_entity_swap(adapter):
    """Test B01: Einstein (true) vs Newton (false) given relativity evidence."""
    ev = "Albert Einstein developed the special and general theories of relativity."
    res_true = adapter.evaluate_pair(claim="Albert Einstein developed the theory of general relativity.", evidence=ev)
    res_false = adapter.evaluate_pair(claim="Isaac Newton developed the theory of general relativity.", evidence=ev)

    assert res_true["label"] == "entailment"
    assert res_false["label"] == "contradiction"
    assert res_false["contradiction"] > 0.80


def test_water_boiling_negation(adapter):
    """Test D01: Water boils (true) vs Water does not boil (false) given boiling point evidence."""
    ev = "At standard atmospheric pressure, the boiling point of pure water is 100 degrees Celsius."
    res_pos = adapter.evaluate_pair(claim="Water boils at 100 degrees Celsius at standard atmospheric pressure.", evidence=ev)
    res_neg = adapter.evaluate_pair(claim="Water does not boil at 100 degrees Celsius at standard atmospheric pressure.", evidence=ev)

    assert res_pos["label"] == "entailment"
    assert res_neg["label"] == "contradiction"
    assert res_neg["contradiction"] > 0.90


def test_india_independence_temporal(adapter):
    """Test E01: 1947 (true) vs 1958 (false) given independence evidence."""
    ev = "India gained independence from the United Kingdom on 15 August 1947."
    res_true = adapter.evaluate_pair(claim="India gained independence in 1947.", evidence=ev)
    res_false = adapter.evaluate_pair(claim="India gained independence in 1958.", evidence=ev)

    assert res_true["label"] == "entailment"
    assert res_false["label"] == "contradiction"
