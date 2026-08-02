"""Partition Manifest Generator and Report Exporter for HalluciSense Phase 6B.2.

Executes deterministic partitioning over HaluBench, RAGTruth, and HaluEval,
writes dataset partition manifests and combined_partition_manifest.json under evaluation_data/partitions/,
and exports human/machine partition reports under evaluation_results/.

Usage:
    python -m evaluation.partitions.manifest
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluation.dataset import DatasetLoader, BenchmarkSample
from evaluation.partitions.grouping import DatasetGroupExtractor
from evaluation.partitions.partitioner import GroupAwarePartitioner, PartitionName, HALLUCISENSE_PARTITION_SEED


DATASET_ROOT = Path("evaluation_data")
PARTITION_DIR = DATASET_ROOT / "partitions"
MANIFEST_DIR = DATASET_ROOT / "manifests"
RESULTS_DIR = Path("evaluation_results")


def compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_file_sha256(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def analyze_partition_overlap(
    partition_map: Dict[PartitionName, List[BenchmarkSample]]
) -> Dict[str, Any]:
    """Analyzes exact and normalized response overlaps between partitions."""
    dev = partition_map[PartitionName.DEVELOPMENT]
    val = partition_map[PartitionName.VALIDATION]
    lock = partition_map[PartitionName.LOCKED_FINAL_TEST]

    dev_ids = set(s.id for s in dev)
    val_ids = set(s.id for s in val)
    lock_ids = set(s.id for s in lock)

    id_overlap_dev_val = len(dev_ids.intersection(val_ids))
    id_overlap_dev_lock = len(dev_ids.intersection(lock_ids))
    id_overlap_val_lock = len(val_ids.intersection(lock_ids))

    dev_groups = set(DatasetGroupExtractor.get_group_key(s) for s in dev)
    val_groups = set(DatasetGroupExtractor.get_group_key(s) for s in val)
    lock_groups = set(DatasetGroupExtractor.get_group_key(s) for s in lock)

    group_overlap_dev_val = len(dev_groups.intersection(val_groups))
    group_overlap_dev_lock = len(dev_groups.intersection(lock_groups))
    group_overlap_val_lock = len(val_groups.intersection(lock_groups))

    # Exact & normalized text response overlaps
    dev_norm_resps = set(s.response.strip().lower() for s in dev)
    val_norm_resps = set(s.response.strip().lower() for s in val)
    lock_norm_resps = set(s.response.strip().lower() for s in lock)

    resp_overlap_dev_val = len(dev_norm_resps.intersection(val_norm_resps))
    resp_overlap_dev_lock = len(dev_norm_resps.intersection(lock_norm_resps))
    resp_overlap_val_lock = len(val_norm_resps.intersection(lock_norm_resps))

    return {
        "id_leakage": id_overlap_dev_val + id_overlap_dev_lock + id_overlap_val_lock,
        "group_leakage": group_overlap_dev_val + group_overlap_dev_lock + group_overlap_val_lock,
        "exact_and_normalized_response_overlap": {
            "dev_val_overlap": resp_overlap_dev_val,
            "dev_lock_overlap": resp_overlap_dev_lock,
            "val_lock_overlap": resp_overlap_val_lock,
        },
    }


def generate_partition_manifest_for_dataset(
    dataset_name: str,
    processed_rel_path: str,
    source_manifest_sha256: str,
) -> Dict[str, Any]:
    proc_file = DATASET_ROOT / processed_rel_path
    samples = DatasetLoader.load_from_file(proc_file)

    partition_map = GroupAwarePartitioner.partition_samples(samples, seed=HALLUCISENSE_PARTITION_SEED)
    overlap_info = analyze_partition_overlap(partition_map)

    partitions_id_map = {
        pname.value: [s.id for s in samples_list]
        for pname, samples_list in partition_map.items()
    }

    partition_stats = {}
    for pname, samps in partition_map.items():
        total = len(samps)
        labels = Counter(s.ground_truth_label for s in samps)
        cats = Counter(s.category for s in samps)
        groups = set(DatasetGroupExtractor.get_group_key(s) for s in samps)

        partition_stats[pname.value] = {
            "total_samples": total,
            "factual_count_label_0": labels.get(0, 0),
            "hallucinated_count_label_1": labels.get(1, 0),
            "factual_percentage": round((labels.get(0, 0) / total * 100), 2) if total else 0.0,
            "hallucinated_percentage": round((labels.get(1, 0) / total * 100), 2) if total else 0.0,
            "unique_groups": len(groups),
            "category_distribution": dict(cats),
        }

    manifest = {
        "dataset_name": dataset_name,
        "processed_path": processed_rel_path,
        "source_manifest_sha256": source_manifest_sha256,
        "partition_algorithm_version": "1.0.0",
        "seed": HALLUCISENSE_PARTITION_SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_examples": len(samples),
        "partition_counts": {p.value: len(samps) for p, samps in partition_map.items()},
        "partition_statistics": partition_stats,
        "leakage_analysis": overlap_info,
        "partitions": partitions_id_map,
    }

    manifest_json = json.dumps(manifest, indent=2)
    manifest_sha = compute_sha256(manifest_json)
    manifest["manifest_sha256"] = manifest_sha

    out_file = PARTITION_DIR / f"{dataset_name.lower()}_partitions.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2))

    return manifest


def generate_all_partition_manifests() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    datasets = [
        ("HaluBench", "processed/halubench/benchmark.jsonl", "manifests/halubench.json"),
        ("RAGTruth", "processed/ragtruth/benchmark.jsonl", "manifests/ragtruth.json"),
        ("HaluEval", "processed/halueval/benchmark.jsonl", "manifests/halueval.json"),
    ]

    dataset_manifests = []
    combined_checksums = {}

    for name, proc_path, mf_path in datasets:
        src_sha = compute_file_sha256(DATASET_ROOT / mf_path)
        m = generate_partition_manifest_for_dataset(name, proc_path, src_sha)
        dataset_manifests.append(m)

        part_rel = f"partitions/{name.lower()}_partitions.json"
        part_sha = compute_file_sha256(DATASET_ROOT / part_rel)
        combined_checksums[part_rel] = part_sha

    combined_manifest = {
        "manifest_name": "HalluciSense Combined Partition Manifest",
        "partition_algorithm_version": "1.0.0",
        "seed": HALLUCISENSE_PARTITION_SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_benchmark_examples": sum(m["total_examples"] for m in dataset_manifests),
        "dataset_manifests": combined_checksums,
        "partition_summary": {
            "development_total": sum(m["partition_counts"]["development"] for m in dataset_manifests),
            "validation_total": sum(m["partition_counts"]["validation"] for m in dataset_manifests),
            "locked_final_test_total": sum(m["partition_counts"]["locked_final_test"] for m in dataset_manifests),
        },
    }

    comb_json = json.dumps(combined_manifest, indent=2)
    comb_sha = compute_sha256(comb_json)
    combined_manifest["combined_manifest_sha256"] = comb_sha

    comb_file = PARTITION_DIR / "combined_partition_manifest.json"
    with open(comb_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(combined_manifest, indent=2))

    return dataset_manifests, combined_manifest


def export_partition_reports(dataset_manifests: List[Dict[str, Any]], combined_manifest: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Machine-readable JSON report
    json_report_path = RESULTS_DIR / "phase6b2_partition_report.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "combined": combined_manifest,
            "datasets": dataset_manifests,
        }, f, indent=2)

    # 2. Human-readable Markdown report
    md = f"""# HalluciSense Phase 6B.2 — Partition & Leakage Control Report

## Executive Summary

Phase 6B.2 partition generation completed successfully using deterministic group-aware partitioning.
- **Fixed Partition Seed**: `{HALLUCISENSE_PARTITION_SEED}`
- **Total Canonical Examples**: `{combined_manifest['total_benchmark_examples']}`
- **Partition Ratios**:
  - **DEVELOPMENT**: `{combined_manifest['partition_summary']['development_total']}` samples
  - **VALIDATION**: `{combined_manifest['partition_summary']['validation_total']}` samples
  - **LOCKED_FINAL_TEST**: `{combined_manifest['partition_summary']['locked_final_test_total']}` samples

---

## Per-Dataset Partition Breakdown

"""
    for m in dataset_manifests:
        md += f"""### Dataset: {m['dataset_name']} (Total: {m['total_examples']})

- **Processed File**: `{m['processed_path']}`
- **Manifest SHA-256**: `{m['manifest_sha256']}`
- **Partitions**:
  - `DEVELOPMENT`: `{m['partition_counts']['development']}` samples (Factual: `{m['partition_statistics']['development']['factual_count_label_0']}`, Hallucinated: `{m['partition_statistics']['development']['hallucinated_count_label_1']}`)
  - `VALIDATION`: `{m['partition_counts']['validation']}` samples (Factual: `{m['partition_statistics']['validation']['factual_count_label_0']}`, Hallucinated: `{m['partition_statistics']['validation']['hallucinated_count_label_1']}`)
  - `LOCKED_FINAL_TEST`: `{m['partition_counts']['locked_final_test']}` samples (Factual: `{m['partition_statistics']['locked_final_test']['factual_count_label_0']}`, Hallucinated: `{m['partition_statistics']['locked_final_test']['hallucinated_count_label_1']}`)
- **Leakage Status**:
  - Shared Example IDs: `{m['leakage_analysis']['id_leakage']}`
  - Shared Group IDs: `{m['leakage_analysis']['group_leakage']}`

"""

    md_report_path = RESULTS_DIR / "phase6b2_partition_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md)


def main():
    print("=== Generating Phase 6B.2 Partition Manifests ===")
    dataset_manifests, combined_manifest = generate_all_partition_manifests()
    export_partition_reports(dataset_manifests, combined_manifest)
    print(f"Partition manifests successfully written to {PARTITION_DIR}/")
    print(f"Partition report generated under {RESULTS_DIR}/phase6b2_partition_report.md")


if __name__ == "__main__":
    main()
