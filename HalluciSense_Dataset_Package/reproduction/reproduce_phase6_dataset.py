"""HalluciSense Phase 6A — Deterministic Benchmark Reproduction Script.

Recreates the canonical N=750 multi-domain benchmark dataset from deterministic
source templates, performs strict structural and cryptographic audits, and
verifies byte-level SHA-256 equivalence against the frozen reference.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from evaluation.benchmark_dataset.importer import generate_publication_benchmark_dataset
from evaluation.benchmark_dataset.dataset_schema import DOMAINS, BenchmarkExample

FROZEN_BENCHMARK_SHA256 = "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5"


def reproduce_benchmark_dataset(output_dir: Path | None = None) -> Dict[str, Any]:
    """Deterministically generates and validates the canonical N=750 benchmark."""
    # Step 1: Generate dataset with seed 42
    manager = generate_publication_benchmark_dataset(n_per_domain=50, seed=42)
    examples: List[BenchmarkExample] = manager.examples

    # Step 2: Validate total record count
    total_records = len(examples)
    records_pass = (total_records == 750)

    # Step 3: Validate domain counts
    domain_counts = {}
    for ex in examples:
        domain_counts[ex.domain] = domain_counts.get(ex.domain, 0) + 1
    domains_pass = (len(domain_counts) == 15 and all(c == 50 for c in domain_counts.values()))

    # Step 4: Validate class balance
    factual_count = sum(1 for ex in examples if ex.ground_truth == 0)
    hallucinated_count = sum(1 for ex in examples if ex.ground_truth == 1)
    class_balance_pass = (factual_count == 375 and hallucinated_count == 375)

    # Step 5: Validate schema and required fields
    malformed_count = 0
    for ex in examples:
        if not ex.id or not ex.question or not ex.response or ex.ground_truth not in (0, 1) or ex.domain not in DOMAINS:
            malformed_count += 1
    schema_pass = (malformed_count == 0)

    # Step 6: Validate unique IDs
    ids = [ex.id for ex in examples]
    duplicate_count = len(ids) - len(set(ids))
    duplicates_pass = (duplicate_count == 0)

    # Step 7: Serialize to canonical JSONL lines and compute SHA-256
    jsonl_lines = [json.dumps(ex.to_dict(), ensure_ascii=False) for ex in examples]
    jsonl_content = "\n".join(jsonl_lines) + "\n"
    computed_sha256 = hashlib.sha256(jsonl_content.encode("utf-8")).hexdigest()

    sha_pass = (computed_sha256 == FROZEN_BENCHMARK_SHA256)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "benchmark_dataset.jsonl"
        out_file.write_text(jsonl_content, encoding="utf-8")

    return {
        "records_pass": records_pass,
        "total_records": total_records,
        "domains_pass": domains_pass,
        "domain_count": len(domain_counts),
        "class_balance_pass": class_balance_pass,
        "factual_count": factual_count,
        "hallucinated_count": hallucinated_count,
        "schema_pass": schema_pass,
        "malformed_count": malformed_count,
        "duplicates_pass": duplicates_pass,
        "duplicate_count": duplicate_count,
        "sha_pass": sha_pass,
        "computed_sha256": computed_sha256,
        "expected_sha256": FROZEN_BENCHMARK_SHA256,
    }


def main():
    res = reproduce_benchmark_dataset()

    print("\nDATASET REPRODUCTION CHECK")
    print("==========================")
    print(f"Records: {'PASS' if res['records_pass'] else 'FAIL'} ({res['total_records']}/750)")
    print(f"Domains: {'PASS' if res['domains_pass'] else 'FAIL'} ({res['domain_count']}/15)")
    print(f"Class balance: {'PASS' if res['class_balance_pass'] else 'FAIL'} ({res['factual_count']} factual / {res['hallucinated_count']} hallucinated)")
    print(f"Schema: {'PASS' if res['schema_pass'] else 'FAIL'} ({res['malformed_count']} malformed)")
    print(f"Duplicate check: {'PASS' if res['duplicates_pass'] else 'FAIL'} ({res['duplicate_count']} duplicates)")
    print(f"SHA-256: {'PASS' if res['sha_pass'] else 'FAIL'} ({res['computed_sha256']})")

    all_passed = all([
        res["records_pass"],
        res["domains_pass"],
        res["class_balance_pass"],
        res["schema_pass"],
        res["duplicates_pass"],
        res["sha_pass"],
    ])

    if all_passed:
        print("\nAll verification gates PASSED. The canonical dataset is 100% reproducible.")
        sys.exit(0)
    else:
        print("\nVerification gates FAILED. Hash or schema mismatch detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
