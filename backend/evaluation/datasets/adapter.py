"""Independent Benchmark Dataset Adapter Architecture for HalluciSense Phase 6B.

Provides clean abstraction for loading, validating, and mapping independent labeled datasets
(JSON, JSONL, CSV) without modifying production inference or leaking labels into pipeline logic.
"""

from dataclasses import dataclass, field
import hashlib
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from evaluation.dataset import BenchmarkSample, DatasetLoader


# Canonical binary label mapping helper
DEFAULT_LABEL_MAP: Dict[Union[str, int, bool], int] = {
    # 0 = Factual / Non-hallucinated
    0: 0,
    "0": 0,
    False: 0,
    "false": 0,
    "factual": 0,
    "non-hallucinated": 0,
    "entailment": 0,
    "supported": 0,
    "correct": 0,
    # 1 = Hallucinated / Non-factual
    1: 1,
    "1": 1,
    True: 1,
    "true": 1,
    "hallucinated": 1,
    "hallucination": 1,
    "contradiction": 1,
    "refuted": 1,
    "incorrect": 1,
    "unsupported": 1,
}


@dataclass
class BenchmarkExample:
    """Standardized representation of a single benchmark sample."""

    example_id: str
    prompt: str
    response: str
    label: int  # 0 = factual, 1 = hallucinated
    category: str = "FACTUAL"
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    synthetic_test_fixture: bool = False

    def to_benchmark_sample(self) -> BenchmarkSample:
        meta = dict(self.metadata)
        if self.synthetic_test_fixture:
            meta["synthetic_test_fixture"] = True
        return BenchmarkSample(
            id=self.example_id,
            prompt=self.prompt,
            response=self.response,
            ground_truth_label=self.label,
            category=self.category,
            evidence=self.evidence,
            metadata=meta,
        )


@dataclass
class BenchmarkDataset:
    """Container holding validated benchmark examples and metadata."""

    dataset_name: str
    file_path: str
    checksum: str
    examples: List[BenchmarkExample]
    label_mapping_used: Dict[str, int]
    synthetic_test_fixture: bool = False

    @property
    def total_count(self) -> int:
        return len(self.examples)

    @property
    def factual_count(self) -> int:
        return sum(1 for e in self.examples if e.label == 0)

    @property
    def hallucinated_count(self) -> int:
        return sum(1 for e in self.examples if e.label == 1)

    def to_benchmark_samples(self) -> List[BenchmarkSample]:
        return [e.to_benchmark_sample() for e in self.examples]


class BenchmarkAdapter:
    """Adapter for loading and validating independent benchmark datasets."""

    @staticmethod
    def load_dataset(
        file_path: Union[str, Path],
        custom_label_map: Optional[Dict[Any, int]] = None,
        dataset_name: Optional[str] = None,
    ) -> BenchmarkDataset:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark dataset file not found: {path}")

        label_map = dict(DEFAULT_LABEL_MAP)
        if custom_label_map:
            for k, v in custom_label_map.items():
                label_map[k] = v
                if isinstance(k, str):
                    label_map[k.lower()] = v

        with open(path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        ext = path.suffix.lower()
        examples: List[BenchmarkExample] = []
        is_synthetic = False

        if ext in (".jsonl", ".ndjson"):
            examples, is_synthetic = BenchmarkAdapter._load_jsonl(path, label_map)
        elif ext == ".csv":
            examples, is_synthetic = BenchmarkAdapter._load_csv(path, label_map)
        elif ext == ".json":
            examples, is_synthetic = BenchmarkAdapter._load_json(path, label_map)
        else:
            raise ValueError(f"Unsupported benchmark file extension: '{ext}'. Use .jsonl, .json, or .csv.")

        name = dataset_name or path.stem

        mapping_summary = {str(k): v for k, v in label_map.items() if isinstance(k, (str, int, bool))}

        return BenchmarkDataset(
            dataset_name=name,
            file_path=str(path),
            checksum=checksum,
            examples=examples,
            label_mapping_used=mapping_summary,
            synthetic_test_fixture=is_synthetic,
        )

    @staticmethod
    def _parse_label(raw_label: Any, label_map: Dict[Any, int], line_ref: str) -> int:
        if raw_label in label_map:
            return label_map[raw_label]
        if isinstance(raw_label, str) and raw_label.lower() in label_map:
            return label_map[raw_label.lower()]
        raise ValueError(f"Unmapped ground-truth label '{raw_label}' on {line_ref}. Map it in custom_label_map.")

    @staticmethod
    def _load_jsonl(path: Path, label_map: Dict[Any, int]) -> Tuple[List[BenchmarkExample], bool]:
        examples = []
        seen_ids = set()
        is_synthetic = False

        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                data = json.loads(line)
                ex_id = str(data.get("id") or data.get("example_id") or data.get("sample_id") or f"sample_{idx}").strip()
                if ex_id in seen_ids:
                    raise ValueError(f"Duplicate example_id '{ex_id}' on line {idx} of {path.name}.")
                seen_ids.add(ex_id)

                raw_label = data.get("label", data.get("ground_truth_label", data.get("ground_truth")))
                label = BenchmarkAdapter._parse_label(raw_label, label_map, f"line {idx}")

                meta = data.get("metadata", {})
                if not isinstance(meta, dict):
                    meta = {}

                syn = bool(data.get("synthetic_test_fixture", meta.get("synthetic_test_fixture", False)))
                if syn or "dev_" in ex_id or "synthetic" in path.name.lower() or "fixture" in path.name.lower():
                    is_synthetic = True

                example = BenchmarkExample(
                    example_id=ex_id,
                    prompt=str(data.get("prompt", "")).strip(),
                    response=str(data.get("response", data.get("output", ""))).strip(),
                    label=label,
                    category=str(data.get("category", "FACTUAL")).strip(),
                    evidence=data.get("evidence", []),
                    metadata=meta,
                    synthetic_test_fixture=syn or is_synthetic,
                )
                examples.append(example)

        return examples, is_synthetic

    @staticmethod
    def _load_json(path: Path, label_map: Dict[Any, int]) -> Tuple[List[BenchmarkExample], bool]:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)

        if isinstance(content, dict) and "data" in content:
            records = content["data"]
        elif isinstance(content, list):
            records = content
        else:
            raise ValueError(f"JSON benchmark file {path.name} must contain an array or object with 'data' key.")

        examples = []
        seen_ids = set()
        is_synthetic = False

        for idx, data in enumerate(records, 1):
            ex_id = str(data.get("id") or data.get("example_id") or f"sample_{idx}").strip()
            if ex_id in seen_ids:
                raise ValueError(f"Duplicate example_id '{ex_id}' at index {idx} in {path.name}.")
            seen_ids.add(ex_id)

            raw_label = data.get("label", data.get("ground_truth_label"))
            label = BenchmarkAdapter._parse_label(raw_label, label_map, f"item {idx}")

            meta = data.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}

            syn = bool(data.get("synthetic_test_fixture", meta.get("synthetic_test_fixture", False)))
            if syn or "synthetic" in path.name.lower() or "fixture" in path.name.lower():
                is_synthetic = True

            example = BenchmarkExample(
                example_id=ex_id,
                prompt=str(data.get("prompt", "")).strip(),
                response=str(data.get("response", "")).strip(),
                label=label,
                category=str(data.get("category", "FACTUAL")).strip(),
                evidence=data.get("evidence", []),
                metadata=meta,
                synthetic_test_fixture=syn or is_synthetic,
            )
            examples.append(example)

        return examples, is_synthetic

    @staticmethod
    def _load_csv(path: Path, label_map: Dict[Any, int]) -> Tuple[List[BenchmarkExample], bool]:
        examples = []
        seen_ids = set()
        is_synthetic = False

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, 2):
                if not row:
                    continue
                ex_id = str(row.get("id") or row.get("example_id") or f"sample_{idx}").strip()
                if ex_id in seen_ids:
                    raise ValueError(f"Duplicate example_id '{ex_id}' at row {idx} of {path.name}.")
                seen_ids.add(ex_id)

                raw_label = row.get("label") or row.get("ground_truth_label") or row.get("ground_truth")
                label = BenchmarkAdapter._parse_label(raw_label, label_map, f"row {idx}")

                ev_raw = row.get("evidence", [])
                if isinstance(ev_raw, str) and ev_raw.strip():
                    try:
                        ev_raw = json.loads(ev_raw)
                    except Exception:
                        ev_raw = []

                meta_raw = row.get("metadata", {})
                if isinstance(meta_raw, str) and meta_raw.strip():
                    try:
                        meta_raw = json.loads(meta_raw)
                    except Exception:
                        meta_raw = {}

                syn = bool(row.get("synthetic_test_fixture", meta_raw.get("synthetic_test_fixture", False)))
                if syn or "synthetic" in path.name.lower() or "fixture" in path.name.lower():
                    is_synthetic = True

                example = BenchmarkExample(
                    example_id=ex_id,
                    prompt=str(row.get("prompt", "")).strip(),
                    response=str(row.get("response", "")).strip(),
                    label=label,
                    category=str(row.get("category", "FACTUAL")).strip(),
                    evidence=ev_raw if isinstance(ev_raw, list) else [],
                    metadata=meta_raw if isinstance(meta_raw, dict) else {},
                    synthetic_test_fixture=syn or is_synthetic,
                )
                examples.append(example)

        return examples, is_synthetic
