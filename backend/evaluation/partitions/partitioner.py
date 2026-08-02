"""Deterministic Group-Aware Partitioner for HalluciSense Phase 6B.2.

Partitioning strategy:
- Fixed project-level seed: HALLUCISENSE_PARTITION_SEED = 2026
- Ratios: 70% DEVELOPMENT, 15% VALIDATION, 15% LOCKED_FINAL_TEST
- SHA-256 group-based hashing ensures 0 cross-partition leakage of paired/source items.
"""

from enum import Enum
import hashlib
from typing import Any, Dict, List, Tuple

from evaluation.dataset import BenchmarkSample
from evaluation.partitions.grouping import DatasetGroupExtractor


HALLUCISENSE_PARTITION_SEED = 2026


class PartitionName(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    LOCKED_FINAL_TEST = "locked_final_test"


class GroupAwarePartitioner:
    """Partitions benchmark datasets deterministically based on logical group keys."""

    @staticmethod
    def assign_group_partition(
        group_key: str,
        seed: int = HALLUCISENSE_PARTITION_SEED,
        dev_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> PartitionName:
        hash_input = f"{seed}:{group_key}".encode("utf-8")
        hex_digest = hashlib.sha256(hash_input).hexdigest()
        val = int(hex_digest[:8], 16) / 0xFFFFFFFF

        if val < dev_ratio:
            return PartitionName.DEVELOPMENT
        elif val < (dev_ratio + val_ratio):
            return PartitionName.VALIDATION
        else:
            return PartitionName.LOCKED_FINAL_TEST

    @classmethod
    def partition_samples(
        cls,
        samples: List[BenchmarkSample],
        seed: int = HALLUCISENSE_PARTITION_SEED,
    ) -> Dict[PartitionName, List[BenchmarkSample]]:
        partitions: Dict[PartitionName, List[BenchmarkSample]] = {
            PartitionName.DEVELOPMENT: [],
            PartitionName.VALIDATION: [],
            PartitionName.LOCKED_FINAL_TEST: [],
        }

        group_map: Dict[str, PartitionName] = {}

        for sample in samples:
            gkey = DatasetGroupExtractor.get_group_key(sample)
            if gkey not in group_map:
                group_map[gkey] = cls.assign_group_partition(gkey, seed=seed)
            target_partition = group_map[gkey]
            partitions[target_partition].append(sample)

        return partitions
