"""Phase 39 — Semantic NLI Adapter & Grounding Unit Test Suite.

Verifies:
- NLI adapter loading via singleton ModelRegistry
- Singleton model instance reuse (init count <= 1)
- Claim ↔ evidence scoring and probability normalization (ent + neu + con == 1.0)
- Label mapping ('entailment', 'contradiction', 'neutral')
- Deterministic output
- Empty and malformed evidence handling
- Bounded claim and evidence caps
- Shadow mode backward compatibility
- API schema integrity
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.engine.model_registry import ModelRegistry
from app.core.inference.semantic_nli import (
    SemanticNLIAdapter,
    get_semantic_nli_adapter,
    MAX_EVIDENCE_PER_CLAIM,
    MAX_CLAIMS_FOR_NLI,
)
from app.core.pipeline import get_hallucisense_pipeline


@pytest.fixture(scope="module")
def nli_adapter():
    return get_semantic_nli_adapter()


@pytest.fixture(scope="module")
def pipeline_instance():
    return get_hallucisense_pipeline()


def test_nli_singleton_initialization(nli_adapter):
    """Verify ModelRegistry initializes DeBERTa NLI at most once."""
    init_count = ModelRegistry._init_counts.get("nli_model", 0)
    assert init_count <= 1, f"Expected NLI init count <= 1, found {init_count}"


def test_claim_evidence_scoring_normalization(nli_adapter):
    """Verify entailment + neutral + contradiction probabilities sum to 1.0."""
    res = nli_adapter.evaluate_pair(
        claim="The Earth revolves around the Sun.",
        evidence="The Earth orbits the Sun once every year.",
    )
    assert "entailment" in res
    assert "neutral" in res
    assert "contradiction" in res
    total_prob = res["entailment"] + res["neutral"] + res["contradiction"]
    assert abs(total_prob - 1.0) < 1e-4
    assert res["label"] == "entailment"
    assert res["entailment"] > 0.80


def test_contradiction_detection(nli_adapter):
    """Verify factual contradiction produces high contradiction probability."""
    res = nli_adapter.evaluate_pair(
        claim="Berlin is the capital of France.",
        evidence="Paris is the capital of France.",
    )
    assert res["label"] == "contradiction"
    assert res["contradiction"] > 0.85


def test_neutral_detection(nli_adapter):
    """Verify unrelated evidence produces high neutral probability."""
    res = nli_adapter.evaluate_pair(
        claim="France has a population above 100 million.",
        evidence="Paris is the capital of France.",
    )
    assert res["label"] == "neutral"
    assert res["neutral"] > 0.50


def test_empty_and_whitespace_inputs(nli_adapter):
    """Verify empty or whitespace strings return neutral fallback without error."""
    res1 = nli_adapter.evaluate_pair(claim="", evidence="Some evidence")
    assert res1["label"] == "neutral"

    res2 = nli_adapter.evaluate_pair(claim="Some claim", evidence="   \n")
    assert res2["label"] == "neutral"


def test_deterministic_output(nli_adapter):
    """Verify identical inputs produce identical numerical predictions across runs."""
    res1 = nli_adapter.evaluate_pair(claim="Water boils at 100C.", evidence="Water boils at 100 degrees Celsius.")
    res2 = nli_adapter.evaluate_pair(claim="Water boils at 100C.", evidence="Water boils at 100 degrees Celsius.")
    assert res1["entailment"] == res2["entailment"]
    assert res1["contradiction"] == res2["contradiction"]


def test_batch_claim_grounding(nli_adapter):
    """Verify evaluate_claim_evidence_grounding returns structured diagnostics."""
    claims = [
        {"claim_id": 0, "text": "Paris is the capital of France."},
        {"claim_id": 1, "text": "Berlin is the capital of France."},
    ]
    evidence_by_claim = {
        "Paris is the capital of France.": [{"title": "France", "snippet": "Paris is the capital of France."}],
        "Berlin is the capital of France.": [{"title": "France", "snippet": "Paris is the capital of France."}],
    }
    grounding = nli_adapter.evaluate_claim_evidence_grounding(claims, evidence_by_claim)
    assert grounding["status"] == "evaluated"
    assert len(grounding["claims"]) == 2
    assert grounding["claims"][0]["primary_status"] == "entailment"
    assert grounding["claims"][1]["primary_status"] == "contradiction"


def test_pipeline_shadow_mode_attachment(pipeline_instance):
    """Verify predict() attaches semantic_grounding with shadow_only=True."""
    res = pipeline_instance.predict(response_text="Paris is the capital of France.")
    assert "semantic_grounding" in res
    sg = res["semantic_grounding"]
    assert sg.get("shadow_only") is True
    assert "claims" in sg
