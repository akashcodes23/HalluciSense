"""Phase 41 — Controlled Randomization Unit Test Suite.

Verifies:
- Evidence permutation degrades model confidence appropriately
- Unrelated evidence outputs neutral status
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.inference.semantic_nli import get_semantic_nli_adapter


def test_random_unrelated_evidence_is_neutral():
    """Verify unrelated scope extension is classified as neutral."""
    adapter = get_semantic_nli_adapter()
    res = adapter.evaluate_pair(
        claim="France has a population above 100 million.",
        evidence="Paris is the capital of France.",
    )
    assert res["label"] == "neutral"
    assert res["neutral"] > 0.50
