"""Partition Loader and Final-Test Firewall for HalluciSense Phase 6B.2.

Enforces strict access control over the LOCKED_FINAL_TEST partition to prevent
accidental evaluation, parameter tuning, or data leakage during development.
"""

from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from evaluation.dataset import BenchmarkSample, DatasetLoader
from evaluation.partitions.partitioner import PartitionName


class EvaluationPurpose(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    FINAL_EVALUATION = "final_evaluation"


class LockedTestSetAccessError(ValueError):
    """Raised when calibration/development code attempts to access LOCKED_FINAL_TEST partition."""
    pass


class PartitionLoader:
    """Loads benchmark dataset partitions with strict purpose-based firewall enforcement."""

    PARTITION_DIR = Path("evaluation_data/partitions")

    @classmethod
    def load_partition(
        cls,
        dataset_name: str,
        partition: Union[PartitionName, str],
        purpose: Union[EvaluationPurpose, str],
        partition_dir: Optional[Union[str, Path]] = None,
    ) -> List[BenchmarkSample]:
        part_str = partition.value if isinstance(partition, PartitionName) else str(partition)
        purp_str = purpose.value if isinstance(purpose, EvaluationPurpose) else str(purpose)

        part_name = PartitionName(part_str.lower())
        purp_name = EvaluationPurpose(purp_str.lower())

        # FIREWALL ENFORCEMENT
        if part_name == PartitionName.LOCKED_FINAL_TEST and purp_name != EvaluationPurpose.FINAL_EVALUATION:
            raise LockedTestSetAccessError(
                f"FIREWALL DENIAL: LOCKED_FINAL_TEST partition cannot be loaded for purpose '{purp_name.value}'. "
                "Explicit EvaluationPurpose.FINAL_EVALUATION is required."
            )

        pdir = Path(partition_dir) if partition_dir else cls.PARTITION_DIR
        manifest_path = pdir / f"{dataset_name.lower()}_partitions.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Partition manifest not found at '{manifest_path}'. Run partition generation first."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assigned_ids = set(manifest.get("partitions", {}).get(part_name.value, []))
        if not assigned_ids:
            return []

        # Load processed dataset file
        processed_path = Path("evaluation_data") / manifest.get("processed_path", f"processed/{dataset_name.lower()}/benchmark.jsonl")
        all_samples = DatasetLoader.load_from_file(processed_path)

        partition_samples = [s for s in all_samples if s.id in assigned_ids]
        return partition_samples
