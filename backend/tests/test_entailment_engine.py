"""Unit tests for EvidenceEntailmentEngine and high-throughput classify_batch refactoring.
"""

import pytest
import torch
from app.core.engine.entailment import EvidenceEntailmentEngine


@pytest.fixture(scope="module")
def nli_engine():
    return EvidenceEntailmentEngine()


def test_label_map_caching(nli_engine):
    """Verify that id2label mapping is cached during initialization."""
    assert hasattr(nli_engine, "label_map")
    assert isinstance(nli_engine.label_map, dict)
    assert "entailment" in nli_engine.label_map
    assert "neutral" in nli_engine.label_map
    assert "contradiction" in nli_engine.label_map


def test_device_selection(nli_engine):
    """Verify device selection logic (MPS or CPU)."""
    if torch.backends.mps.is_available():
        assert nli_engine.device.type == "mps"
    else:
        assert nli_engine.device.type == "cpu"


def test_classify_single(nli_engine):
    """Test backward compatibility of single classify API."""
    claim = "The capital of France is Paris."
    evidence = "Paris is the capital and largest city of France."
    res = nli_engine.classify(claim=claim, evidence=evidence)

    assert isinstance(res, dict)
    assert "entailment" in res
    assert "neutral" in res
    assert "contradiction" in res
    assert isinstance(res["entailment"], float)
    assert isinstance(res["neutral"], float)
    assert isinstance(res["contradiction"], float)
    assert res["entailment"] > 0.5


def test_classify_batch_basic(nli_engine):
    """Test classify_batch with matching claims and evidences."""
    claims = [
        "The capital of France is Paris.",
        "Berlin is located in Japan.",
    ]
    evidences = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
    ]

    results = nli_engine.classify_batch(claims, evidences, batch_size=2)
    assert len(results) == 2

    # Pair 1: Entailment
    assert results[0]["entailment"] > 0.5
    # Pair 2: Contradiction
    assert results[1]["contradiction"] > 0.5


def test_classify_batch_length_mismatch(nli_engine):
    """Test that classify_batch raises ValueError when lengths mismatch."""
    claims = ["Claim 1", "Claim 2"]
    evidences = ["Evidence 1"]

    with pytest.raises(ValueError, match="length mismatch"):
        nli_engine.classify_batch(claims, evidences)


def test_classify_batch_empty_or_whitespace(nli_engine):
    """Test that empty or whitespace claims/evidences return default neutral dict."""
    claims = ["", "Valid claim", "  "]
    evidences = ["Valid evidence", "", "Valid evidence"]

    results = nli_engine.classify_batch(claims, evidences)
    assert len(results) == 3
    for res in results:
        assert res == {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}


def test_classify_vs_classify_batch_equivalence(nli_engine):
    """Test that classify() and classify_batch() produce identical numerical outputs."""
    claims = [
        "Water boils at 100 degrees Celsius.",
        "Dogs are mammals.",
    ]
    evidences = [
        "At standard atmospheric pressure, water boils at 100 °C.",
        "Dogs belong to the mammal class.",
    ]

    single_0 = nli_engine.classify(claims[0], evidences[0])
    single_1 = nli_engine.classify(claims[1], evidences[1])

    batch_res = nli_engine.classify_batch(claims, evidences, batch_size=32)

    assert pytest.approx(single_0["entailment"], abs=1e-5) == batch_res[0]["entailment"]
    assert pytest.approx(single_0["neutral"], abs=1e-5) == batch_res[0]["neutral"]
    assert pytest.approx(single_0["contradiction"], abs=1e-5) == batch_res[0]["contradiction"]

    assert pytest.approx(single_1["entailment"], abs=1e-5) == batch_res[1]["entailment"]
    assert pytest.approx(single_1["neutral"], abs=1e-5) == batch_res[1]["neutral"]
    assert pytest.approx(single_1["contradiction"], abs=1e-5) == batch_res[1]["contradiction"]
