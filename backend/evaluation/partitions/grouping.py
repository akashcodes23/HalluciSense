"""Dataset Grouping Strategy Module for HalluciSense Phase 6B.2.

Provides deterministic group key extraction for HaluBench, RAGTruth, and HaluEval
to prevent prompt/context/pair leakage across experimental partitions.
"""

hashlib_sha256 = None
import hashlib
from typing import Any, Dict

from evaluation.dataset import BenchmarkSample


class DatasetGroupExtractor:
    """Extracts deterministic logical group identifiers for benchmark samples."""

    @staticmethod
    def get_group_key(sample: BenchmarkSample) -> str:
        meta = sample.metadata or {}
        ds_name = str(meta.get("dataset", sample.category or "unknown")).lower()

        # 1. HaluEval: Group by task + base_id so paired correct/hallucinated items stay together
        if "halueval" in ds_name:
            task = str(meta.get("task", "qa")).lower()
            base_id = str(meta.get("base_id", sample.id)).lower()
            return f"halueval:{task}:{base_id}"

        # 2. RAGTruth: Group by source_id so all LLM responses for a source doc stay together
        if "ragtruth" in ds_name:
            source_id = str(meta.get("source_id", sample.id)).lower()
            return f"ragtruth:{source_id}"

        # 3. HaluBench: Group by source_ds + passage/question hash
        if "halubench" in ds_name:
            source_ds = str(meta.get("source_ds", "unknown")).lower()
            passage = str(meta.get("passage", "")).strip()
            question = str(meta.get("question", "")).strip()
            text_combo = f"{passage}:::{question}"
            text_hash = hashlib.sha256(text_combo.encode("utf-8")).hexdigest()[:16]
            return f"halubench:{source_ds}:{text_hash}"

        # Fallback for generic samples: hash of prompt
        prompt_hash = hashlib.sha256(sample.prompt.strip().encode("utf-8")).hexdigest()[:16]
        return f"{ds_name}:{prompt_hash}"
