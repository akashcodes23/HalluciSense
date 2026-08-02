"""Reproducible Acquisition CLI and Manifest Generator for HalluciSense Phase 6B.1.

Downloads canonical raw benchmark datasets (HaluBench, RAGTruth, HaluEval),
computes SHA-256 checksums, transforms raw records using dataset adapters,
generates normalized JSONL processed benchmarks, and updates machine-readable manifests.

Usage:
    python -m evaluation.datasets.acquire --dataset [halubench|ragtruth|halueval|all]
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
from typing import Any, Dict, List, Optional
import pandas as pd

from evaluation.datasets.halubench_adapter import HaluBenchAdapter
from evaluation.datasets.ragtruth_adapter import RAGTruthAdapter
from evaluation.datasets.halueval_adapter import HaluEvalAdapter
from evaluation.datasets.adapter import BenchmarkExample


DATASET_ROOT = Path("evaluation_data")
RAW_DIR = DATASET_ROOT / "raw"
PROCESSED_DIR = DATASET_ROOT / "processed"
MANIFEST_DIR = DATASET_ROOT / "manifests"


def compute_sha256(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def download_file(url: str, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {output_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
        f.write(resp.read())
    checksum = compute_sha256(output_path)
    print(f"Downloaded {output_path.name} (SHA-256: {checksum[:16]}...)")
    return checksum


def save_processed_jsonl(examples: List[BenchmarkExample], output_file: Path) -> str:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in examples:
            d = {
                "id": ex.example_id,
                "prompt": ex.prompt,
                "response": ex.response,
                "ground_truth_label": ex.label,
                "category": ex.category,
                "evidence": ex.evidence,
                "metadata": ex.metadata,
                "synthetic_test_fixture": ex.synthetic_test_fixture,
            }
            f.write(json.dumps(d) + "\n")
    return compute_sha256(output_file)


# =========================================================
# HALUBENCH ACQUISITION
# =========================================================

def acquire_halubench() -> Dict[str, Any]:
    url = "https://huggingface.co/datasets/PatronusAI/HaluBench/resolve/main/data/test-00000-of-00001.parquet"
    raw_file = RAW_DIR / "halubench" / "test-00000-of-00001.parquet"
    raw_sha = download_file(url, raw_file)

    examples = HaluBenchAdapter.load_from_parquet(raw_file)
    proc_file = PROCESSED_DIR / "halubench" / "benchmark.jsonl"
    proc_sha = save_processed_jsonl(examples, proc_file)

    factual_count = sum(1 for e in examples if e.label == 0)
    hallu_count = sum(1 for e in examples if e.label == 1)

    manifest = {
        "dataset_name": "HaluBench",
        "canonical_source": "PatronusAI/HaluBench (Hugging Face Datasets)",
        "source_type": "official_huggingface_repository",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC-BY-4.0",
        "license_source": "https://huggingface.co/datasets/PatronusAI/HaluBench",
        "raw_files": [str(raw_file.relative_to(DATASET_ROOT))],
        "checksums": {
            str(raw_file.relative_to(DATASET_ROOT)): raw_sha,
            str(proc_file.relative_to(DATASET_ROOT)): proc_sha,
        },
        "record_count_raw": len(examples),
        "record_count_processed": len(examples),
        "class_distribution": {
            "factual_count_label_0": factual_count,
            "hallucinated_count_label_1": hallu_count,
            "factual_percentage": round((factual_count / len(examples)) * 100, 2),
            "hallucinated_percentage": round((hallu_count / len(examples)) * 100, 2),
        },
        "adapter_version": "1.0.0",
        "label_mapping": {
            "PASS": 0,
            "FAIL": 1,
        },
        "processed_path": str(proc_file.relative_to(DATASET_ROOT)),
    }

    manifest_file = MANIFEST_DIR / "halubench.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# =========================================================
# RAGTRUTH ACQUISITION
# =========================================================

def acquire_ragtruth() -> Dict[str, Any]:
    url_resp = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"
    url_src = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"

    resp_file = RAW_DIR / "ragtruth" / "response.jsonl"
    src_file = RAW_DIR / "ragtruth" / "source_info.jsonl"

    resp_sha = download_file(url_resp, resp_file)
    src_sha = download_file(url_src, src_file)

    examples, span_stats = RAGTruthAdapter.load_from_jsonl(resp_file, src_file)
    proc_file = PROCESSED_DIR / "ragtruth" / "benchmark.jsonl"
    proc_sha = save_processed_jsonl(examples, proc_file)

    factual_count = sum(1 for e in examples if e.label == 0)
    hallu_count = sum(1 for e in examples if e.label == 1)

    manifest = {
        "dataset_name": "RAGTruth",
        "canonical_source": "ParticleMedia/RAGTruth (GitHub Repository)",
        "source_type": "official_github_repository",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license": "MIT",
        "license_source": "https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE",
        "raw_files": [
            str(resp_file.relative_to(DATASET_ROOT)),
            str(src_file.relative_to(DATASET_ROOT)),
        ],
        "checksums": {
            str(resp_file.relative_to(DATASET_ROOT)): resp_sha,
            str(src_file.relative_to(DATASET_ROOT)): src_sha,
            str(proc_file.relative_to(DATASET_ROOT)): proc_sha,
        },
        "record_count_raw": len(examples),
        "record_count_processed": len(examples),
        "class_distribution": {
            "factual_count_label_0": factual_count,
            "hallucinated_count_label_1": hallu_count,
            "factual_percentage": round((factual_count / len(examples)) * 100, 2),
            "hallucinated_percentage": round((hallu_count / len(examples)) * 100, 2),
        },
        "span_statistics": span_stats,
        "adapter_version": "1.0.0",
        "label_mapping": {
            "no_hallucination_spans": 0,
            "at_least_one_hallucination_span": 1,
        },
        "processed_path": str(proc_file.relative_to(DATASET_ROOT)),
    }

    manifest_file = MANIFEST_DIR / "ragtruth.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# =========================================================
# HALUEVAL ACQUISITION
# =========================================================

def acquire_halueval() -> Dict[str, Any]:
    tasks = ["qa", "dialogue", "summarization", "general"]
    base_url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data"

    all_examples: List[BenchmarkExample] = []
    raw_files = []
    checksums = {}
    task_counts = {}

    for task in tasks:
        filename = f"{task}_data.json"
        url = f"{base_url}/{filename}"
        raw_file = RAW_DIR / "halueval" / filename
        sha = download_file(url, raw_file)
        raw_files.append(str(raw_file.relative_to(DATASET_ROOT)))
        checksums[str(raw_file.relative_to(DATASET_ROOT))] = sha

        exs, stats = HaluEvalAdapter.load_from_json(raw_file, task_name=task)
        all_examples.extend(exs)
        task_counts[task] = stats["processed_examples"]

    proc_file = PROCESSED_DIR / "halueval" / "benchmark.jsonl"
    proc_sha = save_processed_jsonl(all_examples, proc_file)
    checksums[str(proc_file.relative_to(DATASET_ROOT))] = proc_sha

    factual_count = sum(1 for e in all_examples if e.label == 0)
    hallu_count = sum(1 for e in all_examples if e.label == 1)

    manifest = {
        "dataset_name": "HaluEval",
        "canonical_source": "RUCAIBox/HaluEval (GitHub Repository)",
        "source_type": "official_github_repository",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license": "MIT",
        "license_source": "https://github.com/RUCAIBox/HaluEval/blob/main/LICENSE",
        "raw_files": raw_files,
        "checksums": checksums,
        "record_count_raw": len(all_examples) // 2,
        "record_count_processed": len(all_examples),
        "task_distribution": task_counts,
        "class_distribution": {
            "factual_count_label_0": factual_count,
            "hallucinated_count_label_1": hallu_count,
            "factual_percentage": round((factual_count / len(all_examples)) * 100, 2),
            "hallucinated_percentage": round((hallu_count / len(all_examples)) * 100, 2),
        },
        "adapter_version": "1.0.0",
        "label_mapping": {
            "right_answer_response": 0,
            "hallucinated_answer_response": 1,
        },
        "processed_path": str(proc_file.relative_to(DATASET_ROOT)),
    }

    manifest_file = MANIFEST_DIR / "halueval.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def update_datasets_md(manifests: List[Dict[str, Any]]) -> None:
    """Generates human-readable provenance records in evaluation_data/DATASETS.md."""
    md = """# HalluciSense Phase 6B.1 — Benchmark Dataset Provenance & Integrity Record

This document records the provenance, licensing, checksums, and label mappings for the three real benchmark dataset families integrated into **HalluciSense**.

---
"""
    for m in manifests:
        md += f"""
## Dataset: {m['dataset_name']}

- **Canonical Source**: `{m['canonical_source']}` ({m['source_type']})
- **License**: `{m['license']}` ([License Source]({m['license_source']}))
- **Retrieval Timestamp**: `{m['retrieved_at']}`
- **Raw Records**: `{m['record_count_raw']}`
- **Processed Examples**: `{m['record_count_processed']}` (Factual: `{m['class_distribution']['factual_count_label_0']}`, Hallucinated: `{m['class_distribution']['hallucinated_count_label_1']}`)
- **Processed JSONL Path**: `{m['processed_path']}`
- **Label Mapping**:
"""
        for k, v in m["label_mapping"].items():
            md += f"  - `{k}` $\\rightarrow$ `{v}`\n"

        md += "- **SHA-256 Checksums**:\n"
        for fpath, sha in m["checksums"].items():
            md += f"  - `{fpath}`: `{sha}`\n"

        md += "\n---\n"

    with open(DATASET_ROOT / "DATASETS.md", "w", encoding="utf-8") as f:
        f.write(md)


def main():
    parser = argparse.ArgumentParser(description="Acquire and process Phase 6B.1 real benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["halubench", "ragtruth", "halueval", "all"],
        default="all",
        help="Dataset family to acquire and process",
    )
    args = parser.parse_args()

    manifests = []

    if args.dataset in ("halubench", "all"):
        print("\n=== Acquiring HaluBench ===")
        m_halu = acquire_halubench()
        manifests.append(m_halu)

    if args.dataset in ("ragtruth", "all"):
        print("\n=== Acquiring RAGTruth ===")
        m_rag = acquire_ragtruth()
        manifests.append(m_rag)

    if args.dataset in ("halueval", "all"):
        print("\n=== Acquiring HaluEval ===")
        m_heval = acquire_halueval()
        manifests.append(m_heval)

    if manifests:
        update_datasets_md(manifests)
        print(f"\nSuccessfully acquired and processed {len(manifests)} benchmark families.")
        print(f"Manifests generated under {MANIFEST_DIR}/")
        print(f"Provenance documentation saved to {DATASET_ROOT}/DATASETS.md")


if __name__ == "__main__":
    main()
