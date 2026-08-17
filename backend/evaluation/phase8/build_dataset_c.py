"""Phase 8 — Dataset C Builder: Controlled Hallucination Injection Benchmark.

Derives a controlled hallucination dataset from factual Phase 6 records (GT=0).
Applies 10 rule-based corruption types to factual responses to generate
known-hallucinated variants with deterministic ground truth GT=1.

Every corruption is rule-based and transparent — no LLM generation required.
Ground truth is self-evident from the transformation itself.

Output: backend/reports/phase8/controlled_hallucination_dataset.jsonl
"""

from __future__ import annotations

import json
import re
import time
import random
import hashlib
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BACKEND_DIR / "evaluation" / "results" / "benchmark_dataset.jsonl"
PHASE8_DIR = BACKEND_DIR / "reports" / "phase8"

RANDOM_SEED = 42
N_PER_TYPE = 30  # 30 samples per corruption type → 300 total
CORRUPTION_TYPES = [
    "ENTITY_SUBSTITUTION",
    "NUMERIC_SUBSTITUTION",
    "DATE_SUBSTITUTION",
    "TEMPORAL_ERROR",
    "LOCATION_SUBSTITUTION",
    "PERSON_SUBSTITUTION",
    "CAUSAL_REVERSAL",
    "CONTRADICTION",
    "PARTIAL_CLAIM_CORRUPTION",
    "MULTI_CLAIM_CORRUPTION",
]

# Severity levels: 1=minor, 2=moderate, 3=major, 4=critical
CORRUPTION_SEVERITY = {
    "ENTITY_SUBSTITUTION": 2,
    "NUMERIC_SUBSTITUTION": 2,
    "DATE_SUBSTITUTION": 2,
    "TEMPORAL_ERROR": 2,
    "LOCATION_SUBSTITUTION": 2,
    "PERSON_SUBSTITUTION": 2,
    "CAUSAL_REVERSAL": 3,
    "CONTRADICTION": 4,
    "PARTIAL_CLAIM_CORRUPTION": 1,
    "MULTI_CLAIM_CORRUPTION": 3,
}

# ── Substitution tables (deterministic, well-known facts) ──────────────────
ENTITY_SUBS = [
    ("France", "Germany"), ("Germany", "France"),
    ("United States", "Canada"), ("London", "Manchester"),
    ("Paris", "Berlin"), ("Berlin", "Vienna"),
    ("Wikipedia", "Britannica"), ("Nobel", "Pulitzer"),
    ("Atlantic", "Pacific"), ("Pacific", "Indian"),
    ("Amazon", "Nile"), ("Everest", "K2"),
]
NUMERIC_SUBS = [
    (r"\b(\d{1,3})\b", lambda m: str(int(m.group(1)) + 11)),
    (r"\b100\b", "89"), (r"\b50\b", "37"),
    (r"\b(\d+)%", lambda m: str(int(m.group(1)) + 15) + "%"),
]
DATE_SUBS = [
    ("1969", "1973"), ("1945", "1941"), ("2001", "1998"),
    ("1776", "1789"), ("1865", "1870"), ("1919", "1924"),
    ("1989", "1991"), ("1903", "1908"), ("1861", "1866"),
    ("1492", "1502"),
]
TEMPORAL_SUBS = [
    ("first", "last"), ("last", "first"),
    ("before", "after"), ("after", "before"),
    ("earlier", "later"), ("later", "earlier"),
    ("began", "ended"), ("ended", "began"),
    ("preceded", "followed"), ("followed", "preceded"),
]
LOCATION_SUBS = [
    ("North", "South"), ("South", "North"),
    ("East", "West"), ("West", "East"),
    ("Northern", "Southern"), ("Southern", "Northern"),
    ("Eastern", "Western"), ("Western", "Eastern"),
    ("Asia", "Africa"), ("Africa", "Europe"),
    ("Europe", "Asia"), ("America", "Australia"),
]
PERSON_SUBS = [
    ("Einstein", "Newton"), ("Newton", "Einstein"),
    ("Shakespeare", "Milton"), ("Darwin", "Mendel"),
    ("Lincoln", "Jefferson"), ("Jefferson", "Washington"),
    ("Churchill", "Roosevelt"), ("Napoleon", "Wellington"),
    ("Curie", "Pasteur"), ("Tesla", "Edison"),
]


def apply_entity_sub(text: str, rng: random.Random) -> tuple[str, str]:
    for orig, repl in rng.sample(ENTITY_SUBS, len(ENTITY_SUBS)):
        if orig.lower() in text.lower():
            corrupted = re.sub(re.escape(orig), repl, text, count=1, flags=re.IGNORECASE)
            if corrupted != text:
                return corrupted, f"Entity '{orig}' replaced with '{repl}'"
    return _fallback_corruption(text, rng)


def apply_numeric_sub(text: str, rng: random.Random) -> tuple[str, str]:
    nums = re.findall(r'\b\d{2,5}\b', text)
    if nums:
        n = rng.choice(nums)
        orig_val = int(n)
        new_val = orig_val + rng.choice([7, 11, 13, -7, -11])
        corrupted = text.replace(n, str(new_val), 1)
        return corrupted, f"Numeric '{n}' replaced with '{new_val}'"
    return _fallback_corruption(text, rng)


def apply_date_sub(text: str, rng: random.Random) -> tuple[str, str]:
    for orig, repl in rng.sample(DATE_SUBS, len(DATE_SUBS)):
        if orig in text:
            return text.replace(orig, repl, 1), f"Date '{orig}' replaced with '{repl}'"
    years = re.findall(r'\b(1[89]\d\d|20[0-2]\d)\b', text)
    if years:
        yr = rng.choice(years)
        new_yr = str(int(yr) + rng.choice([3, 5, 7, -3, -5]))
        return text.replace(yr, new_yr, 1), f"Year '{yr}' replaced with '{new_yr}'"
    return _fallback_corruption(text, rng)


def apply_temporal_sub(text: str, rng: random.Random) -> tuple[str, str]:
    for orig, repl in rng.sample(TEMPORAL_SUBS, len(TEMPORAL_SUBS)):
        if orig.lower() in text.lower():
            corrupted = re.sub(r'\b' + re.escape(orig) + r'\b', repl, text, count=1, flags=re.IGNORECASE)
            if corrupted != text:
                return corrupted, f"Temporal term '{orig}' replaced with '{repl}'"
    return _fallback_corruption(text, rng)


def apply_location_sub(text: str, rng: random.Random) -> tuple[str, str]:
    for orig, repl in rng.sample(LOCATION_SUBS, len(LOCATION_SUBS)):
        if orig.lower() in text.lower():
            corrupted = re.sub(r'\b' + re.escape(orig) + r'\b', repl, text, count=1, flags=re.IGNORECASE)
            if corrupted != text:
                return corrupted, f"Location term '{orig}' replaced with '{repl}'"
    return _fallback_corruption(text, rng)


def apply_person_sub(text: str, rng: random.Random) -> tuple[str, str]:
    for orig, repl in rng.sample(PERSON_SUBS, len(PERSON_SUBS)):
        if orig.lower() in text.lower():
            corrupted = re.sub(r'\b' + re.escape(orig) + r'\b', repl, text, count=1, flags=re.IGNORECASE)
            if corrupted != text:
                return corrupted, f"Person '{orig}' replaced with '{repl}'"
    return _fallback_corruption(text, rng)


def apply_causal_reversal(text: str, rng: random.Random) -> tuple[str, str]:
    patterns = [
        (r'\bbecause\b', 'despite'), (r'\bdue to\b', 'regardless of'),
        (r'\btherefore\b', 'however'), (r'\bconsequently\b', 'unexpectedly'),
        (r'\bsince\b', 'although'), (r'\bso that\b', 'even though'),
    ]
    for pat, repl in rng.sample(patterns, len(patterns)):
        corrupted = re.sub(pat, repl, text, count=1, flags=re.IGNORECASE)
        if corrupted != text:
            return corrupted, f"Causal connector '{pat}' reversed to '{repl}'"
    # Inject causal inversion directly
    sentences = text.split('. ')
    if len(sentences) >= 2:
        sentences[0] = sentences[0] + " — but this is incorrect"
        return '. '.join(sentences), "Causal inversion injected"
    return _fallback_corruption(text, rng)


def apply_contradiction(text: str, rng: random.Random) -> tuple[str, str]:
    negations = [
        (r'\bis\b', 'is not'), (r'\bwas\b', 'was not'),
        (r'\bare\b', 'are not'), (r'\bwere\b', 'were not'),
        (r'\bcan\b', 'cannot'), (r'\bhas\b', 'has never'),
        (r'\bhave\b', 'have never'), (r'\bdid\b', 'did not'),
        (r'\bwill\b', 'will not'),
    ]
    for pat, repl in rng.sample(negations, len(negations)):
        corrupted = re.sub(r'\b' + pat.strip(r'\b') + r'\b', repl, text, count=1, flags=re.IGNORECASE)
        if corrupted != text:
            return corrupted, f"Direct contradiction introduced: '{pat}' → '{repl}'"
    return _fallback_corruption(text, rng)


def apply_partial_claim(text: str, rng: random.Random) -> tuple[str, str]:
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) >= 2:
        # Corrupt second sentence only
        corrupted_sent = sentences[1] + " (this part is incorrect)"
        corrupted = '. '.join([sentences[0], corrupted_sent] + sentences[2:])
        return corrupted, "Second claim corrupted while first remains factual"
    words = text.split()
    if len(words) > 4:
        mid = len(words) // 2
        words[mid] = "incorrectly"
        return ' '.join(words), "Partial word-level corruption injected at midpoint"
    return _fallback_corruption(text, rng)


def apply_multi_claim(text: str, rng: random.Random) -> tuple[str, str]:
    """Corrupt both entity and numeric in the same response."""
    step1, _ = apply_entity_sub(text, rng)
    step2, _ = apply_numeric_sub(step1, rng)
    if step2 != text:
        return step2, "Multiple corruption: entity + numeric substitution applied"
    step3, _ = apply_date_sub(step2, rng)
    if step3 != text:
        return step3, "Multiple corruption: entity + date substitution applied"
    return _fallback_corruption(text, rng)


def _fallback_corruption(text: str, rng: random.Random) -> tuple[str, str]:
    """Generic fallback: negate the first verb or append contradictory phrase."""
    return text + " However, this claim is factually incorrect.", "Generic negation appended as fallback"


CORRUPTION_FNS = {
    "ENTITY_SUBSTITUTION": apply_entity_sub,
    "NUMERIC_SUBSTITUTION": apply_numeric_sub,
    "DATE_SUBSTITUTION": apply_date_sub,
    "TEMPORAL_ERROR": apply_temporal_sub,
    "LOCATION_SUBSTITUTION": apply_location_sub,
    "PERSON_SUBSTITUTION": apply_person_sub,
    "CAUSAL_REVERSAL": apply_causal_reversal,
    "CONTRADICTION": apply_contradiction,
    "PARTIAL_CLAIM_CORRUPTION": apply_partial_claim,
    "MULTI_CLAIM_CORRUPTION": apply_multi_claim,
}


def main():
    PHASE8_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    print("Loading Phase 6 canonical dataset (factual records only)…")
    factual_records = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["ground_truth"] == 0:
                factual_records.append(d)

    print(f"  Found {len(factual_records)} factual records for corruption pool.")

    records = []
    record_id = 0

    for corruption_type in CORRUPTION_TYPES:
        corrupt_fn = CORRUPTION_FNS[corruption_type]
        severity = CORRUPTION_SEVERITY[corruption_type]
        pool = rng.sample(factual_records, min(N_PER_TYPE, len(factual_records)))

        generated_count = 0
        for src in pool:
            orig_response = src["response"]
            corrupted_response, corruption_detail = corrupt_fn(orig_response, rng)

            # Verify the corruption actually changed the text
            is_actually_corrupted = (corrupted_response != orig_response)

            record_id += 1
            records.append({
                "sample_id": f"phase8_c_{record_id:04d}",
                "source_sample_id": src["id"],
                "domain": src["domain"],
                "difficulty": src["difficulty"],
                "corruption_type": corruption_type,
                "corruption_severity": severity,
                "severity_label": {1: "MINOR", 2: "MODERATE", 3: "MAJOR", 4: "CRITICAL"}[severity],
                "is_actually_corrupted": is_actually_corrupted,
                "corruption_detail": corruption_detail,
                # Dual representation
                "original_factual_response": orig_response,
                "corrupted_response": corrupted_response,
                # Ground truth is SELF-EVIDENT from the transformation
                "ground_truth": 1,  # Always hallucinated
                "ground_truth_method": "rule_based_transformation_self_evident",
                "ground_truth_confidence": 1.0 if is_actually_corrupted else 0.5,
                "ground_truth_reason": (
                    f"Controlled {corruption_type} injection. {corruption_detail}. "
                    "Ground truth is self-evident from the deterministic transformation."
                ),
                "query": src["question"],
                "evidence_passages": src.get("evidence_passages", [])[:2],
                "evidence_source": "Phase6_canonical_benchmark",
            })
            generated_count += 1

        print(f"  {corruption_type}: {generated_count} samples generated")

    # Write output
    out_path = PHASE8_DIR / "controlled_hallucination_dataset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dataset_hash = hashlib.sha256(out_path.read_bytes()).hexdigest()
    actually_corrupted = sum(1 for r in records if r["is_actually_corrupted"])

    manifest = {
        "dataset_name": "Phase8_Dataset_C_ControlledHallucinationInjection",
        "version": "1.0.0",
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": "Phase6_canonical_benchmark (factual records only)",
        "source_sha256": "dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5",
        "output_sha256": dataset_hash,
        "generation_script": "backend/evaluation/phase8/build_dataset_c.py",
        "random_seed": RANDOM_SEED,
        "total_samples": len(records),
        "actually_corrupted_count": actually_corrupted,
        "fallback_corruption_count": len(records) - actually_corrupted,
        "corruption_types": CORRUPTION_TYPES,
        "samples_per_type": N_PER_TYPE,
        "severity_scale": CORRUPTION_SEVERITY,
        "all_ground_truth": 1,
        "ground_truth_method": "rule_based_transformation_self_evident",
        "schema_version": "phase8_v1",
        "note": (
            "All corruptions are deterministic rule-based transformations. "
            "Ground truth (GT=1 hallucinated) is self-evident from the transformation. "
            "No LLM was used to generate or annotate this dataset. "
            "No HalluciSense H-score was used as ground truth."
        )
    }

    (PHASE8_DIR / "controlled_hallucination_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\nDataset C complete: {len(records)} records written to {out_path}")
    print(f"  SHA-256: {dataset_hash}")
    print(f"  Actually corrupted (text changed): {actually_corrupted} / {len(records)}")
    print(f"  Fallback only: {len(records) - actually_corrupted}")


if __name__ == "__main__":
    main()
