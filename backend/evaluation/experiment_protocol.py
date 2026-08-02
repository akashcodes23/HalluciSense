"""Frozen Experiment Protocol Configuration and Fingerprint for HalluciSense Phase 6B.2.

Defines versioned experiment protocol parameters, partition seed, metric definitions,
locked-test access policies, and computes the combined experimental protocol fingerprint.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from evaluation.partitions.partitioner import HALLUCISENSE_PARTITION_SEED


PROTOCOL_VERSION = "1.0.0"
DATASET_ROOT = Path("evaluation_data")
PARTITION_DIR = DATASET_ROOT / "partitions"


class ExperimentProtocolConfig:
    """Versioned frozen experiment protocol configuration."""

    PROTOCOL_VERSION = PROTOCOL_VERSION
    PARTITION_SEED = HALLUCISENSE_PARTITION_SEED
    ADAPTER_VERSION = "1.0.0"
    PARTITION_ALGORITHM_VERSION = "1.0.0"

    DATASET_VERSIONS = {
        "HaluBench": "1.0.0",
        "RAGTruth": "1.0.0",
        "HaluEval": "1.0.0",
    }

    PARTITION_MANIFESTS = {
        "halubench": "partitions/halubench_partitions.json",
        "ragtruth": "partitions/ragtruth_partitions.json",
        "halueval": "partitions/halueval_partitions.json",
        "combined": "partitions/combined_partition_manifest.json",
    }

    LABEL_SEMANTICS = {
        0: "factual / non-hallucinated",
        1: "hallucinated",
    }

    LOCKED_TEST_POLICY = (
        "LOCKED_FINAL_TEST partition is strictly isolated. "
        "Loading requires explicit EvaluationPurpose.FINAL_EVALUATION. "
        "Any access during development, tuning, or calibration triggers LockedTestSetAccessError."
    )

    @classmethod
    def get_protocol_fingerprint(cls) -> str:
        """Computes combined experimental protocol reproducibility fingerprint."""
        checksums = {}

        for k, rel_path in cls.PARTITION_MANIFESTS.items():
            full_path = DATASET_ROOT / rel_path
            if full_path.exists():
                with open(full_path, "rb") as f:
                    checksums[rel_path] = hashlib.sha256(f.read()).hexdigest()
            else:
                checksums[rel_path] = "MISSING"

        fingerprint_data = {
            "protocol_version": cls.PROTOCOL_VERSION,
            "partition_seed": cls.PARTITION_SEED,
            "adapter_version": cls.ADAPTER_VERSION,
            "partition_algorithm_version": cls.PARTITION_ALGORITHM_VERSION,
            "dataset_versions": cls.DATASET_VERSIONS,
            "manifest_checksums": checksums,
        }

        fp_json = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fp_json.encode("utf-8")).hexdigest()


def main():
    print("=== HalluciSense Experiment Protocol Configuration ===")
    print(f"Protocol Version: {ExperimentProtocolConfig.PROTOCOL_VERSION}")
    print(f"Partition Seed: {ExperimentProtocolConfig.PARTITION_SEED}")
    print(f"Combined Protocol Fingerprint: {ExperimentProtocolConfig.get_protocol_fingerprint()}")


if __name__ == "__main__":
    main()
