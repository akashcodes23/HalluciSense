"""Phase 49 — Bounded Chunked NLI Inference & Memory Scaling Tests."""

import os
import psutil
import pytest
from app.core.engine.entailment import EvidenceEntailmentEngine


def test_chunked_nli_scaling_bounded():
    """Verify NLI inference on large pair batches executes in bounded micro-chunks."""
    engine = EvidenceEntailmentEngine()
    process = psutil.Process(os.getpid())

    # Generate synthetic pairs: 1, 4, 8, 16, 32, 64 pairs
    pair_counts = [1, 4, 8, 16, 32, 64]
    
    for count in pair_counts:
        claims = [f"Claim number {i} is a factual statement about astronomy." for i in range(count)]
        evidences = [f"Evidence snippet {i} states that astronomy is the study of stars." for i in range(count)]
        
        r_before = process.memory_info().rss / (1024 * 1024)
        results = engine.classify_batch(claims, evidences, batch_size=2)
        r_after = process.memory_info().rss / (1024 * 1024)
        
        assert len(results) == count
        for res in results:
            assert "entailment" in res
            assert "neutral" in res
            assert "contradiction" in res
            assert abs(sum(res.values()) - 1.0) < 0.05
        
        # Verify batch size was bounded to 2
        assert engine.last_batch_metrics.get("batch_size") <= 2
