"""NLI Entailment Engine with Singleton ModelRegistry and Bounded Concurrency."""

import time
from typing import Dict, List
import torch
import structlog

from app.core.engine.model_registry import ModelRegistry

logger = structlog.get_logger(__name__)


class EvidenceEntailmentEngine:
    """NLI-based factual verification engine with singleton DeBERTa inference."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self.tokenizer, self.model = ModelRegistry.get_nli_model(model_name)
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        try:
            self.model.to(self.device)
        except Exception:
            pass

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

        self.last_batch_metrics = {
            "pairs": 0,
            "batches": 0,
            "batch_size": 16,
            "inference_ms": 0.0,
        }
        self._cache: Dict[Tuple[str, str], Dict[str, float]] = {}

    def classify(self, claim: str, evidence: str) -> Dict[str, float]:
        return self.classify_batch([claim], [evidence])[0]

    def classify_batch(
        self,
        claims: List[str],
        evidences: List[str],
        batch_size: int = 16,
    ) -> List[Dict[str, float]]:
        if len(claims) != len(evidences):
            raise ValueError(f"Claims and evidences length mismatch: {len(claims)} vs {len(evidences)}")
        if not claims:
            self.last_batch_metrics = {"pairs": 0, "batches": 0, "batch_size": batch_size, "inference_ms": 0.0}
            return []

        results = [{"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0} for _ in claims]
        uncached_indices, uncached_evidences, uncached_claims = [], [], []

        for idx, (claim, evidence) in enumerate(zip(claims, evidences)):
            if claim and evidence and claim.strip() and evidence.strip():
                c_clean = claim.strip()
                e_clean = evidence.strip()
                cache_key = (c_clean, e_clean)
                if cache_key in self._cache:
                    results[idx] = dict(self._cache[cache_key])
                else:
                    uncached_indices.append(idx)
                    uncached_evidences.append(e_clean)
                    uncached_claims.append(c_clean)

        if not uncached_indices:
            self.last_batch_metrics = {"pairs": len(claims), "batches": 0, "batch_size": batch_size, "inference_ms": 0.0}
            return results

        ent_idx = self.label_map.get("entailment", 0)
        neu_idx = self.label_map.get("neutral", 1)
        con_idx = self.label_map.get("contradiction", 2)
        num_batches = 0
        t0 = time.perf_counter()

        semaphore = ModelRegistry.get_nli_semaphore()
        with semaphore:
            for b_start in range(0, len(uncached_indices), batch_size):
                b_end = min(b_start + batch_size, len(uncached_indices))
                inputs = self.tokenizer(
                    uncached_evidences[b_start:b_end],
                    uncached_claims[b_start:b_end],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                with torch.inference_mode():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                num_batches += 1
                for offset, orig_idx in enumerate(uncached_indices[b_start:b_end]):
                    row = probs[offset]
                    pred = {
                        "entailment": float(row[ent_idx]),
                        "neutral": float(row[neu_idx]),
                        "contradiction": float(row[con_idx]),
                    }
                    results[orig_idx] = pred
                    self._cache[(uncached_claims[b_start + offset], uncached_evidences[b_start + offset])] = pred

        inference_ms = (time.perf_counter() - t0) * 1000.0
        self.last_batch_metrics = {
            "pairs": len(claims),
            "batches": num_batches,
            "batch_size": batch_size,
            "inference_ms": round(inference_ms, 2),
        }
        logger.info("nli_batch_completed", **self.last_batch_metrics)
        return results
