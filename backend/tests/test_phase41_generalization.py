"""Phase 41 — Independent Generalization & Adversarial Unit Test Suite.

Verifies:
- Out-of-distribution domain evaluation stability
- Factual vs. Contradictory minimal pair separation
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.inference.semantic_nli import get_semantic_nli_adapter


def test_cross_domain_biology_generalization():
    """Verify semantic NLI evaluates biological taxonomy correctly."""
    adapter = get_semantic_nli_adapter()
    ev = "Mitochondria are double membrane-bound organelles found in most eukaryotic organisms."
    res_true = adapter.evaluate_pair(
        claim="Mitochondria are organelles found in eukaryotic organisms.",
        evidence=ev,
    )
    res_false = adapter.evaluate_pair(
        claim="Mitochondria are single-celled prokaryotic bacteria.",
        evidence=ev,
    )
    assert res_true["label"] == "entailment"
    assert res_false["label"] == "contradiction"
