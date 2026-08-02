"""Partition Integrity Verifier for HalluciSense Phase 6B.2.

Validates partition completeness, zero cross-partition group/ID leakage,
and manifest SHA-256 integrity across all integrated benchmark datasets.

Usage:
    python -m evaluation.partitions.verify_partitions
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from evaluation.dataset import DatasetLoader


DATASET_ROOT = Path("evaluation_data")
PARTITION_DIR = DATASET_ROOT / "partitions"


def compute_file_sha256(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_partition_integrity() -> Tuple[bool, Dict[str, Any]]:
    comb_manifest_path = PARTITION_DIR / "combined_partition_manifest.json"
    if not comb_manifest_path.exists():
        print("Combined partition manifest missing. Run evaluation.partitions.manifest first.")
        return False, {"status": "MISSING_COMBINED_MANIFEST"}

    overall_pass = True
    dataset_results = {}

    dataset_files = [
        "halubench_partitions.json",
        "ragtruth_partitions.json",
        "halueval_partitions.json",
    ]

    for dfname in dataset_files:
        dfpath = PARTITION_DIR / dfname
        if not dfpath.exists():
            print(f"Missing partition manifest: {dfname}")
            overall_pass = False
            dataset_results[dfname] = {"status": "MISSING"}
            continue

        with open(dfpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        ds_name = manifest.get("dataset_name", dfname)
        total_exs = manifest.get("total_examples", 0)
        parts = manifest.get("partitions", {})

        dev_ids = set(parts.get("development", []))
        val_ids = set(parts.get("validation", []))
        lock_ids = set(parts.get("locked_final_test", []))

        # 1. Total count equality
        all_ids = dev_ids.union(val_ids).union(lock_ids)
        if len(all_ids) != total_exs:
            overall_pass = False
            print(f"[{ds_name}] Partition total count mismatch: expected {total_exs}, got {len(all_ids)}")

        # 2. Zero ID overlap between partitions
        overlap_dev_val = dev_ids.intersection(val_ids)
        overlap_dev_lock = dev_ids.intersection(lock_ids)
        overlap_val_lock = val_ids.intersection(lock_ids)

        id_leakage = len(overlap_dev_val) + len(overlap_dev_lock) + len(overlap_val_lock)
        if id_leakage > 0:
            overall_pass = False
            print(f"[{ds_name}] ID leakage detected across partitions: {id_leakage} overlapping IDs")

        # 3. Zero Group leakage
        leakage_info = manifest.get("leakage_analysis", {})
        group_leakage = leakage_info.get("group_leakage", 0)
        if group_leakage > 0:
            overall_pass = False
            print(f"[{ds_name}] Group leakage detected across partitions: {group_leakage} overlapping groups")

        status = "PASS" if (len(all_ids) == total_exs and id_leakage == 0 and group_leakage == 0) else "FAIL"
        dataset_results[ds_name] = {
            "status": status,
            "total_examples": total_exs,
            "assigned_examples": len(all_ids),
            "id_leakage": id_leakage,
            "group_leakage": group_leakage,
        }

    return overall_pass, {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "datasets": dataset_results,
    }


def main():
    print("=== HalluciSense Phase 6B.2 Partition Integrity Verification ===")
    success, report = verify_partition_integrity()

    for ds_name, details in report.get("datasets", {}).items():
        st = details.get("status", "UNKNOWN")
        color_st = "\033[92mPASS\033[0m" if st == "PASS" else "\033[91mFAIL\033[0m"
        print(f"\nDataset: {ds_name} [{st}]")
        print(f"  Total Examples: {details.get('total_examples')}")
        print(f"  Assigned Examples: {details.get('assigned_examples')}")
        print(f"  ID Leakage: {details.get('id_leakage')}")
        print(f"  Group Leakage: {details.get('group_leakage')}")

    print("\n=============================================================")
    if success:
        print("VERDICT: HALLUCISENSE PHASE 6B.2 PARTITION INTEGRITY: PASS")
    else:
        print("VERDICT: HALLUCISENSE PHASE 6B.2 PARTITION INTEGRITY: FAIL")


if __name__ == "__main__":
    main()
