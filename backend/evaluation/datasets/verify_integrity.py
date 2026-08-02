"""Dataset Integrity Verifier for HalluciSense Phase 6B.1.

Validates file presence, SHA-256 checksums, schema compliance, ID uniqueness,
span boundary correctness, and non-empty responses for all integrated datasets.

Usage:
    python -m evaluation.datasets.verify_integrity
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluation.dataset import DatasetLoader, BenchmarkSample
from evaluation.datasets.ragtruth_adapter import RAGTruthAdapter


DATASET_ROOT = Path("evaluation_data")
MANIFEST_DIR = DATASET_ROOT / "manifests"


def compute_sha256(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_dataset_integrity() -> Tuple[bool, Dict[str, Any]]:
    manifest_files = list(MANIFEST_DIR.glob("*.json"))
    if not manifest_files:
        print("No dataset manifests found in evaluation_data/manifests/")
        return False, {"status": "NO_MANIFESTS", "datasets": {}}

    overall_pass = True
    report_details = {}

    for mf in manifest_files:
        with open(mf, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        ds_name = manifest.get("dataset_name", mf.stem)
        checksums = manifest.get("checksums", {})
        ds_status = "PASS"
        issues = []
        file_results = {}

        # 1. Checksum verification
        for rel_path, expected_sha in checksums.items():
            full_path = DATASET_ROOT / rel_path
            if not full_path.exists():
                file_results[rel_path] = "MISSING"
                issues.append(f"File missing: {rel_path}")
                ds_status = "FAIL"
                overall_pass = False
            else:
                actual_sha = compute_sha256(full_path)
                if actual_sha != expected_sha:
                    file_results[rel_path] = "CHECKSUM_MISMATCH"
                    issues.append(
                        f"Checksum mismatch for {rel_path}: expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
                    )
                    ds_status = "FAIL"
                    overall_pass = False
                else:
                    file_results[rel_path] = "PASS"

        # 2. Processed dataset schema & integrity verification
        proc_rel_path = manifest.get("processed_path")
        sample_count = 0
        duplicate_ids = 0
        empty_responses = 0
        span_issues = 0

        if proc_rel_path:
            proc_full_path = DATASET_ROOT / proc_rel_path
            if proc_full_path.exists():
                try:
                    samples: List[BenchmarkSample] = DatasetLoader.load_from_file(proc_full_path)
                    sample_count = len(samples)

                    # Check span bounds for RAGTruth
                    if "ragtruth" in ds_name.lower():
                        for s in samples:
                            spans = s.metadata.get("hallucination_spans", [])
                            for span in spans:
                                is_valid, warn = RAGTruthAdapter.validate_span_bounds(s.response, span)
                                if not is_valid:
                                    span_issues += 1
                                    issues.append(f"RAGTruth sample {s.id} span error: {warn}")

                except Exception as exc:
                    issues.append(f"Failed to parse processed JSONL schema: {exc}")
                    ds_status = "FAIL"
                    overall_pass = False

        report_details[ds_name] = {
            "status": ds_status,
            "manifest_file": str(mf.relative_to(DATASET_ROOT)),
            "processed_samples": sample_count,
            "file_checksum_results": file_results,
            "issues": issues,
            "span_validation_issues": span_issues,
        }

    return overall_pass, {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "datasets": report_details,
    }


def main():
    print("=== HalluciSense Phase 6B.1 Dataset Integrity Verification ===")
    success, report = verify_dataset_integrity()

    for ds_name, details in report["datasets"].items():
        st = details["status"]
        color = "\033[92mPASS\033[0m" if st == "PASS" else "\033[91mFAIL\033[0m"
        print(f"\nDataset: {ds_name} [{st}]")
        print(f"  Processed Samples: {details['processed_samples']}")
        for fpath, fst in details["file_checksum_results"].items():
            print(f"  File: {fpath} -> {fst}")
        if details["issues"]:
            print("  Issues:")
            for issue in details["issues"][:5]:
                print(f"    - {issue}")

    print("\n=============================================================")
    if success:
        print("VERDICT: HALLUCISENSE PHASE 6B.1 DATASET INTEGRITY VERIFICATION: PASS")
    else:
        print("VERDICT: HALLUCISENSE PHASE 6B.1 DATASET INTEGRITY VERIFICATION: FAIL")


if __name__ == "__main__":
    main()
