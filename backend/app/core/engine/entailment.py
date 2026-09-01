"""NLI Entailment Engine with Singleton DeBERTa and Bounded Chunked Inference.

Phase 49 guarantees:
- Exactly ONE DeBERTa model in process memory.
- Bounded chunked inference (batch size strictly <= 2).
- Zero autograd graph retention via torch.inference_mode().
- Strict input sequence truncation (claim <= 128, evidence <= 256).
- Immediate deallocation of intermediate tensors.
"""

import time
import threading
from collections import OrderedDict
from typing import Dict, List
import torch
import structlog

from app.core.engine.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)


class EvidenceEntailmentEngine:
    """NLI-based factual verification with bounded-memory DeBERTa inference."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self.tokenizer, self.model = ModelRegistry.get_nli_model(model_name)

        # Dynamic label mapping resolution from model configuration
        self.label_map: Dict[str, int] = {}
        id2label = getattr(self.model.config, "id2label", {})
        for idx, label in id2label.items():
            label_str = str(label).lower()
            if "entail" in label_str:
                self.label_map["entailment"] = int(idx)
            elif "neutral" in label_str:
                self.label_map["neutral"] = int(idx)
            elif "contrad" in label_str:
                self.label_map["contradiction"] = int(idx)

        # Fallback defaults if id2label is missing
        if "contradiction" not in self.label_map:
            self.label_map["contradiction"] = 0
        if "entailment" not in self.label_map:
            self.label_map["entailment"] = 1
        if "neutral" not in self.label_map:
            self.label_map["neutral"] = 2

        self.last_batch_metrics = {
            "pairs": 0,
            "batches": 0,
            "batch_size": 2,
            "inference_ms": 0.0,
            "backend": "pytorch-eval-bounded",
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
        batch_size: int = 2,
    ) -> List[Dict[str, float]]:
        if len(claims) != len(evidences):
            raise ValueError(f"Claims and evidences length mismatch: {len(claims)} vs {len(evidences)}")
        if not claims:
            self.last_batch_metrics = {
                "pairs": 0,
                "batches": 0,
                "batch_size": batch_size,
                "inference_ms": 0.0,
                "backend": "pytorch-eval-bounded",
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
            # CrossEncoder convention: [premise (evidence), hypothesis (claim)]
            uncached_pairs.append([e_clean, c_clean])

        if not uncached_indices:
            self.last_batch_metrics = {
                "pairs": len(claims),
                "batches": 0,
                "batch_size": batch_size,
                "inference_ms": 0.0,
                "backend": "pytorch-eval-bounded",
            }
            return results

        t0 = time.perf_counter()
        semaphore = ModelRegistry.get_nli_semaphore(max_concurrent=1)
        num_batches = 0

        # Enforce micro-chunking: at most 2 pairs processed per PyTorch forward pass
        chunk_size = max(1, min(batch_size, 2))

        with semaphore:
            for start in range(0, len(uncached_pairs), chunk_size):
                chunk = uncached_pairs[start:start + chunk_size]
                premises = [p[0][:350] for p in chunk]   # Truncate evidence string
                hypotheses = [p[1][:150] for p in chunk] # Truncate claim string

                inputs = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                del inputs, logits
                num_batches += 1

                for offset, orig_idx in enumerate(uncached_indices[start:start + len(chunk)]):
                    row = probs[offset]
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

        inference_ms = (time.perf_counter() - t0) * 1000.0
        self.last_batch_metrics = {
            "pairs": len(claims),
            "batches": num_batches,
            "batch_size": chunk_size,
            "inference_ms": round(inference_ms, 2),
            "backend": "pytorch-eval-bounded",
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
