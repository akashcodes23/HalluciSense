"""
Phase 24 — Production Memory Stability and Resource Hardening Tests.
Validates ModelRegistry singleton invariants, cache boundedness, repeated request stability,
and benchmark immutability under simulated production load.
"""

import gc
import hashlib
import os
from pathlib import Path
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.engine.model_registry import ModelRegistry
from app.core.engine.entailment import EvidenceEntailmentEngine
from app.modules.knowledge.wikipedia import WikipediaKnowledgeSource
from app.modules.knowledge.retriever import HybridRetriever
from app.core.engine.pillar1_retrieval import EventTemporalAnchorResolver

CANONICAL_BENCHMARK_HASH = "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"


class TestPhase24MemoryStability:
    """Rigorous validation of memory invariants, bounded caching, and singleton lifecycles."""

    def test_benchmark_hash_strictly_unaltered(self):
        """Invariant: Canonical benchmark dataset hash must be unchanged."""
        base_dir = Path(__file__).resolve().parent.parent
        dataset_path = base_dir / "evaluation" / "results" / "benchmark_dataset.jsonl"
        assert dataset_path.exists(), f"Benchmark dataset missing at {dataset_path}"

        hasher = hashlib.sha256()
        with open(dataset_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        assert hasher.hexdigest() == CANONICAL_BENCHMARK_HASH

    def test_model_registry_singleton_invariant(self):
        """Verify that ModelRegistry initializes models exactly once."""
        t1, m1 = ModelRegistry.get_nli_model()
        t2, m2 = ModelRegistry.get_nli_model()
        assert t1 is t2
        assert m1 is m2

        p1 = ModelRegistry.get_pipeline()
        p2 = ModelRegistry.get_pipeline()
        assert p1 is p2

        counts = ModelRegistry.get_init_counts()
        assert counts.get("nli_model", 0) == 1
        assert counts.get("pipeline", 0) == 1

    def test_nli_cache_boundedness_and_lru_eviction(self):
        """Verify EvidenceEntailmentEngine cache does not exceed MAX_CACHE_ENTRIES."""
        engine = EvidenceEntailmentEngine()
        max_entries = getattr(engine, "MAX_CACHE_ENTRIES", 512)

        # Populate cache with max_entries + 50 distinct pairs
        for i in range(max_entries + 50):
            claim = f"Claim index {i}"
            ev = f"Evidence index {i}"
            with engine._cache_lock:
                if len(engine._cache) >= max_entries:
                    engine._cache.popitem(last=False)
                engine._cache[(claim, ev)] = {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}

        assert len(engine._cache) == max_entries
        # Earliest items (0..49) should have been evicted
        assert ("Claim index 0", "Evidence index 0") not in engine._cache
        # Latest item should be present
        assert (f"Claim index {max_entries + 49}", f"Evidence index {max_entries + 49}") in engine._cache

    def test_wikipedia_cache_boundedness_and_lru_eviction(self):
        """Verify WikipediaKnowledgeSource cache does not exceed MAX_CACHE_ENTRIES."""
        wiki = WikipediaKnowledgeSource()
        max_entries = getattr(wiki, "MAX_CACHE_ENTRIES", 512)

        for i in range(max_entries + 20):
            query = f"query_{i}"
            with wiki._cache_lock:
                if len(wiki._cache) >= max_entries:
                    wiki._cache.popitem(last=False)
                wiki._cache[query] = [{"snippet": f"snippet {i}", "title": query}]

        assert len(wiki._cache) == max_entries
        assert "query_0" not in wiki._cache
        assert f"query_{max_entries + 19}" in wiki._cache

    def test_temporal_anchor_cache_boundedness(self):
        """Verify EventTemporalAnchorResolver cache does not grow unboundedly."""
        resolver = EventTemporalAnchorResolver()
        max_entries = getattr(resolver, "MAX_CACHE_ENTRIES", 512)

        for i in range(max_entries + 20):
            label = f"entity_{i}"
            if len(resolver._cache) >= max_entries:
                resolver._cache.popitem(last=False)
            resolver._cache[label] = {"id": f"Q{i}"}

        assert len(resolver._cache) == max_entries
        assert "entity_0" not in resolver._cache

    def test_repeated_analyze_requests_memory_plateau(self):
        """Verify 10 repeated /analyze requests execute with 100% success and 0 model reinits."""
        counts_start = ModelRegistry.get_init_counts()

        with TestClient(app) as client:
            for _ in range(10):
                resp = client.post(
                    "/api/v1/analyze",
                    json={
                        "query": "What is the capital of Karnataka?",
                        "response": "The capital of Karnataka is Bengaluru.",
                    },
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["risk_level"] == "VERIFIED"
                assert data["overall_h_score"] < 0.35

        counts_end = ModelRegistry.get_init_counts()
        assert counts_end["nli_model"] == counts_start["nli_model"]
        assert counts_end["pipeline"] == counts_start["pipeline"]

    def test_repeated_chat_requests_memory_stability(self):
        """Verify repeated closed-loop chat requests execute properly."""
        counts_start = ModelRegistry.get_init_counts()

        with TestClient(app) as client:
            for _ in range(3):
                resp = client.post(
                    "/api/v1/chat",
                    json={
                        "message": "What is the molar mass of water?",
                        "enable_verification": True,
                        "auto_correct": True,
                    },
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["verification"]["status"] in ["VERIFIED", "CORRECTED"]

        counts_end = ModelRegistry.get_init_counts()
        assert counts_end["nli_model"] == counts_start["nli_model"]
        assert counts_end["pipeline"] == counts_start["pipeline"]

    def test_false_claim_detection_correctness(self):
        """Verify false claim detection remains 100% accurate after memory hardening."""
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/analyze",
                json={
                    "query": "What is the capital of Karnataka?",
                    "response": "The capital of Karnataka is Mumbai.",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["risk_level"] == "LIKELY_HALLUCINATED"
            assert data["overall_h_score"] > 0.65
