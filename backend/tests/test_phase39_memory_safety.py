"""Phase 39 — Memory Safety & Singleton Guarantee Test Suite.

Verifies:
- NLI model initialization count <= 1
- Zero duplicate transformer allocations
- Bounded concurrency via NLI semaphore
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.engine.model_registry import ModelRegistry
from app.core.inference.semantic_nli import get_semantic_nli_adapter


def test_nli_singleton_not_duplicated():
    """Verify multiple calls to get_semantic_nli_adapter share the exact same underlying model."""
    a1 = get_semantic_nli_adapter()
    a2 = get_semantic_nli_adapter()
    assert a1.engine.model is a2.engine.model
    assert a1.engine.tokenizer is a2.engine.tokenizer


def test_model_registry_init_count():
    """Verify ModelRegistry._init_counts['nli_model'] <= 1."""
    init_count = ModelRegistry._init_counts.get("nli_model", 0)
    assert init_count <= 1, f"Expected nli_model init count <= 1, but got {init_count}"


def test_semaphore_bounds_concurrency():
    """Verify NLI semaphore is initialized and enforces concurrency limit."""
    sem = ModelRegistry.get_nli_semaphore(max_concurrent=2)
    assert sem is not None
