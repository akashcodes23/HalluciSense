"""Benchmark sample schema and dataset loader for HalluciSense Phase 6A evaluation.

Ensures deterministic loading, schema validation, duplicate detection, and network isolation.
"""

from enum import Enum
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
from pydantic import BaseModel, Field, field_validator


class SampleCategory(str, Enum):
    FACTUAL = "FACTUAL"
    ENTITY = "ENTITY"
    DATE = "DATE"
    NUMERIC = "NUMERIC"
    SCIENTIFIC = "SCIENTIFIC"
    GEOGRAPHIC = "GEOGRAPHIC"
    HISTORICAL = "HISTORICAL"
    MULTI_CLAIM = "MULTI_CLAIM"
    UNVERIFIABLE = "UNVERIFIABLE"


class BenchmarkSample(BaseModel):
    id: str = Field(..., description="Unique sample identifier")
    prompt: str = Field(..., description="Input prompt text")
    response: str = Field(..., description="LLM output response text")
    ground_truth_label: int = Field(
        ..., description="Binary ground truth: 0 = factual, 1 = hallucinated"
    )
    category: str = Field(
        default="FACTUAL",
        description="Sample category for error analysis",
    )
    evidence: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Optional ground-truth reference evidence documents",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional metadata key-value pairs",
    )

    @field_validator("ground_truth_label")
    @classmethod
    def validate_binary_label(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(
                f"ground_truth_label must be 0 (factual) or 1 (hallucinated), got {v}"
            )
        return v

    @field_validator("prompt", "response")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Prompt and response must be non-empty strings.")
        return v.strip()


class DatasetLoader:
    """Loads and validates benchmark datasets from JSONL or CSV files."""

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> List[BenchmarkSample]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        if path.suffix.lower() == ".jsonl":
            return DatasetLoader.load_from_jsonl(path)
        elif path.suffix.lower() == ".csv":
            return DatasetLoader.load_from_csv(path)
        else:
            raise ValueError(
                f"Unsupported dataset file extension '{path.suffix}'. Use .jsonl or .csv."
            )

    @staticmethod
    def load_from_jsonl(file_path: Union[str, Path]) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []
        seen_ids = set()

        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Malformed JSON on line {line_idx} of {path.name}: {e}"
                    ) from e

                sample = BenchmarkSample.model_validate(data)
                if sample.id in seen_ids:
                    raise ValueError(
                        f"Duplicate sample ID '{sample.id}' found on line {line_idx} of {path.name}."
                    )
                seen_ids.add(sample.id)
                samples.append(sample)

        if not samples:
            raise ValueError(f"Dataset file {path.name} contains no valid records.")

        return samples

    @staticmethod
    def load_from_csv(file_path: Union[str, Path]) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []
        seen_ids = set()

        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, 2):
                if not row:
                    continue
                # Parse evidence & metadata JSON fields if present as strings
                evidence_raw = row.get("evidence", [])
                if isinstance(evidence_raw, str) and evidence_raw.strip():
                    try:
                        evidence_raw = json.loads(evidence_raw)
                    except Exception:
                        evidence_raw = []
                elif not isinstance(evidence_raw, list):
                    evidence_raw = []

                metadata_raw = row.get("metadata", {})
                if isinstance(metadata_raw, str) and metadata_raw.strip():
                    try:
                        metadata_raw = json.loads(metadata_raw)
                    except Exception:
                        metadata_raw = {}
                elif not isinstance(metadata_raw, dict):
                    metadata_raw = {}

                data = {
                    "id": str(row.get("id", "")).strip(),
                    "prompt": str(row.get("prompt", "")).strip(),
                    "response": str(row.get("response", "")).strip(),
                    "ground_truth_label": int(row.get("ground_truth_label", 0)),
                    "category": str(row.get("category", "FACTUAL")).strip(),
                    "evidence": evidence_raw,
                    "metadata": metadata_raw,
                }

                sample = BenchmarkSample.model_validate(data)
                if sample.id in seen_ids:
                    raise ValueError(
                        f"Duplicate sample ID '{sample.id}' found on row {row_idx} of {path.name}."
                    )
                seen_ids.add(sample.id)
                samples.append(sample)

        if not samples:
            raise ValueError(f"CSV Dataset file {path.name} contains no valid records.")

        return samples
