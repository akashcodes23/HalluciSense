"""NLI Entailment Engine with one shared quantized ONNX model.

The production verifier must not load a second PyTorch copy of DeBERTa.
All P1/P3 NLI calls use the singleton supplied by ModelRegistry.
"""

import time
import threading
from collections import OrderedDict
from typing import Dict, List

import structlog

from app.core.engine.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)


class EvidenceEntailmentEngine:
    """NLI-based factual verification using the shared ONNX-int8 CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self.tokenizer, self.model = ModelRegistry.get_nli_model(model_name)

        # cross-encoder/nli-deberta-v3-small publishes this label ordering.
        # Keep the mapping explicit so P1/P3 do not depend on PyTorch config objects.
        self.label_map: Dict[str, int] = {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2,
        }

        self.last_batch_metrics = {
            "pairs": 0,
            "batches": 0,
            "batch_size": 8,
            "inference_ms": 0.0,
            "backend": "onnx-int8",
        }
        self.MAX_CACHE_ENTRIES = 256
        self._cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()

    def classify(self, claim: str, evidence: str) -> Dict[str, float]:
        return self.classify_batch([claim], [evidence])[0]

    def classify_batch(
        self,
        claims: List[str],
        evidences: List[str],
        batch_size: int = 8,
    ) -> List[Dict[str, float]]:
        if len(claims) != len(evidences):
            raise ValueError(f"Claims and evidences length mismatch: {len(claims)} vs {len(evidences)}")
        if not claims:
            self.last_batch_metrics = {
                "pairs": 0,
                "batches": 0,
                "batch_size": batch_size,
                "inference_ms": 0.0,
                "backend": "onnx-int8",
            }
            return []

        results = [{"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0} for _ in claims]
        uncached_indices: List[int] = []
        uncached_pairs: List[List[str]] = []

        for idx, (claim, evidence) in enumerate(zip(claims, evidences)):
            if not claim or not evidence or not claim.strip() or not evidence.strip():
                continue
            c_clean = claim.strip()
            e_clean = evidence.strip()
            cache_key = (c_clean, e_clean)
            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                results[idx] = dict(cached)
                continue
            uncached_indices.append(idx)
            # CrossEncoder convention: [sentence_a, sentence_b].
            # Preserve the historical P1 semantics where evidence is the premise
            # and claim is the hypothesis.
            uncached_pairs.append([e_clean, c_clean])

        if not uncached_indices:
            self.last_batch_metrics = {
                "pairs": len(claims),
                "batches": 0,
                "batch_size": batch_size,
                "inference_ms": 0.0,
                "backend": "onnx-int8",
            }
            return results

        t0 = time.perf_counter()
        semaphore = ModelRegistry.get_nli_semaphore(max_concurrent=1)
        num_batches = 0

        with semaphore:
            for start in range(0, len(uncached_pairs), max(1, min(batch_size, 8))):
                batch_pairs = uncached_pairs[start:start + max(1, min(batch_size, 8))]
                scores = self.model.predict(
                    batch_pairs,
                    batch_size=len(batch_pairs),
                    show_progress_bar=False,
                    apply_softmax=True,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                )
                num_batches += 1

                for offset, orig_idx in enumerate(uncached_indices[start:start + len(batch_pairs)]):
                    row = scores[offset]
                    pred = {
                        "contradiction": float(row[self.label_map["contradiction"]]),
                        "entailment": float(row[self.label_map["entailment"]]),
                        "neutral": float(row[self.label_map["neutral"]]),
                    }
                    results[orig_idx] = pred
                    cache_key = (
                        claims[orig_idx].strip(),
                        evidences[orig_idx].strip(),
                    )
                    with self._cache_lock:
                        if len(self._cache) >= self.MAX_CACHE_ENTRIES:
                            self._cache.popitem(last=False)
                        self._cache[cache_key] = pred

        import gc
        gc.collect()

        inference_ms = (time.perf_counter() - t0) * 1000.0
        self.last_batch_metrics = {
            "pairs": len(claims),
            "batches": num_batches,
            "batch_size": batch_size,
            "inference_ms": round(inference_ms, 2),
            "backend": "onnx-int8",
        }
        logger.info("nli_batch_completed", **self.last_batch_metrics)
        return results

    def predict_entailment_probabilities(self, premise: str, hypothesis: str):
        """Compatibility helper used by historical P3 code.

        Returns probabilities in the historical tuple order:
        entailment, contradiction, neutral.
        """
        result = self.classify(hypothesis, premise)
        return (
            result["entailment"],
            result["contradiction"],
            result["neutral"],
        )
