"""
HalluciSense Phase 9 — Step 7: Production Packaging
=====================================================
Model registry, versioned metadata, input validator, API schema,
model card, latency/memory benchmarks. Deployable model bundle.

FROZEN FIREWALL: No models, scalers, or thresholds are modified.
"""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PHASE6K = ROOT / "evaluation_results" / "phase6k"
FINAL_MODEL_DIR = PHASE6K / "final_model"
PHASE6I = ROOT / "evaluation_results" / "phase6i"
PUB = PHASE6K / "publication"
BUNDLE_DIR = PUB / "step7_production_bundle"
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "mean_entailment",
    "max_entailment",
    "mean_contradiction",
    "min_support_margin",
    "num_claims",
]
OPERATING_THRESHOLD = 0.56
MODEL_VERSION = "1.0.0"
MODEL_NAME = "hallucisense-pillar1-logistic"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> None:
    print("=" * 70)
    print("HalluciSense Phase 9 — Step 7: Production Packaging")
    print("=" * 70)
    t0 = time.time()

    model = joblib.load(FINAL_MODEL_DIR / "pillar1_logistic_model.joblib")
    scaler = joblib.load(FINAL_MODEL_DIR / "robust_scaler.joblib")

    model_sha = sha256_file(FINAL_MODEL_DIR / "pillar1_logistic_model.joblib")
    scaler_sha = sha256_file(FINAL_MODEL_DIR / "robust_scaler.joblib")

    # Load VAL for benchmarks
    val_rows = []
    with open(PHASE6I / "claim_evidence_features_validation.jsonl") as f:
        for line in f:
            obj = json.loads(line.strip()) if line.strip() else {}
            val_rows.append([obj.get(fn, float("nan")) for fn in FEATURE_NAMES])
    X_val = np.array(val_rows, dtype=np.float64)

    # ── 1. Model Registry ─────────────────────────────────────────────────────
    print("\n[1/6] Building model registry...")
    registry = {
        "registry_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "phase": "6K",
            "verdict": "PILLAR 1 VALIDATED WITH LIMITATIONS",
            "type": "binary_classifier",
            "algorithm": "LogisticRegression",
            "solver": "liblinear",
            "penalty": "l2",
            "C": 1.0,
            "random_state": 42,
            "artifacts": {
                "classifier": {
                    "path": str(FINAL_MODEL_DIR / "pillar1_logistic_model.joblib"),
                    "sha256": model_sha,
                    "format": "joblib",
                },
                "scaler": {
                    "path": str(FINAL_MODEL_DIR / "robust_scaler.joblib"),
                    "sha256": scaler_sha,
                    "format": "joblib",
                    "type": "RobustScaler",
                },
            },
            "feature_schema": {
                "feature_names": FEATURE_NAMES,
                "feature_count": len(FEATURE_NAMES),
                "input_dtype": "float64",
            },
            "inference": {
                "operating_threshold": OPERATING_THRESHOLD,
                "output_classes": {0: "GROUNDED", 1: "HALLUCINATED"},
                "output_probabilities": True,
            },
            "performance": {
                "val_roc_auc": 0.6902,
                "val_pr_auc": 0.6311,
                "val_brier_score": 0.2332,
                "val_f1_at_056": 0.6618,
                "val_mcc_at_056": 0.3587,
                "val_accuracy_at_056": 0.6803,
                "val_n_samples": 3500,
                "dev_n_samples": 58002,
            },
            "training_metadata": {
                "created_utc": "2026-08-03T04:22:00Z",
                "framework": "scikit-learn",
                "scaler": "RobustScaler",
                "feature_set": "SET_D_DECOLLINEARIZED_DISCRIMINATIVE",
                "cross_validation": "5-fold stratified",
                "stability_gate": "32/32 PASS",
            },
        },
    }

    registry_out = BUNDLE_DIR / "model_registry.json"
    with open(registry_out, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"  Registry → {registry_out}")

    # ── 2. Input Validator ────────────────────────────────────────────────────
    print("\n[2/6] Generating input validator...")
    # Compute feature bounds from VAL (permissive validation)
    feature_bounds = {}
    for i, fn in enumerate(FEATURE_NAMES):
        col = X_val[:, i]
        feature_bounds[fn] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
        }

    validator_code = '''"""
HalluciSense Pillar-1 Input Validator
======================================
Validates feature vectors before Pillar-1 inference.
Auto-generated by Phase 9 Step 7. DO NOT EDIT MANUALLY.
Generated: {timestamp}
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np


FEATURE_NAMES = {feature_names}

# Feature value bounds observed in the validated VAL partition
FEATURE_BOUNDS = {feature_bounds}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    feature_vector: Optional[np.ndarray] = None


def validate_features(
    mean_entailment: float,
    max_entailment: float,
    mean_contradiction: float,
    min_support_margin: float,
    num_claims: float,
) -> ValidationResult:
    """
    Validate input features for Pillar-1 inference.

    Parameters
    ----------
    All five Pillar-1 NLI features (see FEATURE_NAMES).

    Returns
    -------
    ValidationResult with valid=True if all checks pass.
    """
    errors = []
    warnings = []
    values = [mean_entailment, max_entailment, mean_contradiction,
              min_support_margin, num_claims]

    for fname, val in zip(FEATURE_NAMES, values):
        # Type check
        if not isinstance(val, (int, float)):
            errors.append(f"{{fname}}: expected numeric, got {{type(val).__name__}}")
            continue

        # Finiteness check
        if not np.isfinite(val):
            errors.append(f"{{fname}}: non-finite value ({{val}})")
            continue

        # Range bounds (warn only — do not reject)
        bounds = FEATURE_BOUNDS[fname]
        if val < bounds["min"] - 3 * bounds["std"]:
            warnings.append(f"{{fname}}={{val:.4f}} is far below observed min {{bounds['min']:.4f}}")
        elif val > bounds["max"] + 3 * bounds["std"]:
            warnings.append(f"{{fname}}={{val:.4f}} is far above observed max {{bounds['max']:.4f}}")

    # Semantic constraints
    if np.isfinite(mean_entailment) and np.isfinite(max_entailment):
        if max_entailment < mean_entailment - 1e-6:
            errors.append(
                f"max_entailment ({{max_entailment:.4f}}) < mean_entailment ({{mean_entailment:.4f}}): "
                "max must be >= mean"
            )

    if np.isfinite(num_claims) and num_claims < 0:
        errors.append(f"num_claims={{num_claims}}: must be non-negative")

    if np.isfinite(mean_entailment) and not (0.0 <= mean_entailment <= 1.0):
        warnings.append(f"mean_entailment={{mean_entailment:.4f}} outside expected [0, 1]")

    if np.isfinite(mean_contradiction) and not (0.0 <= mean_contradiction <= 1.0):
        warnings.append(f"mean_contradiction={{mean_contradiction:.4f}} outside expected [0, 1]")

    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    feature_vector = np.array(values, dtype=np.float64)
    return ValidationResult(valid=True, errors=[], warnings=warnings,
                             feature_vector=feature_vector)
'''.format(
        timestamp=datetime.now(timezone.utc).isoformat(),
        feature_names=repr(FEATURE_NAMES),
        feature_bounds=repr(feature_bounds),
    )

    validator_out = BUNDLE_DIR / "input_validator.py"
    with open(validator_out, "w") as f:
        f.write(validator_code)
    print(f"  Validator → {validator_out}")

    # ── 3. API Schema ─────────────────────────────────────────────────────────
    print("\n[3/6] Generating OpenAPI schema snippet...")
    api_schema = {
        "openapi": "3.1.0",
        "info": {
            "title": "HalluciSense Pillar-1 Inference API",
            "version": MODEL_VERSION,
            "description": (
                "Production inference API for the frozen HalluciSense Pillar-1 "
                "logistic regression hallucination detector."
            ),
        },
        "paths": {
            "/api/v1/pillar1/predict": {
                "post": {
                    "summary": "Predict hallucination probability",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": FEATURE_NAMES,
                                    "properties": {
                                        "mean_entailment": {"type": "number", "minimum": 0, "maximum": 1,
                                                            "description": "Mean NLI entailment score"},
                                        "max_entailment": {"type": "number", "minimum": 0, "maximum": 1,
                                                           "description": "Max NLI entailment score"},
                                        "mean_contradiction": {"type": "number", "minimum": 0, "maximum": 1,
                                                               "description": "Mean NLI contradiction score"},
                                        "min_support_margin": {"type": "number", "minimum": -1, "maximum": 1,
                                                               "description": "Minimum support margin"},
                                        "num_claims": {"type": "number", "minimum": 0,
                                                       "description": "Number of atomic claims"},
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "prediction": {"type": "string", "enum": ["GROUNDED", "HALLUCINATED"]},
                                            "probability": {"type": "number", "minimum": 0, "maximum": 1},
                                            "threshold": {"type": "number"},
                                            "margin": {"type": "number"},
                                            "confidence_category": {
                                                "type": "string",
                                                "enum": ["Very High", "High", "Moderate", "Low"],
                                            },
                                            "model_version": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            }
        },
    }

    api_out = BUNDLE_DIR / "inference_api_schema.json"
    with open(api_out, "w") as f:
        json.dump(api_schema, f, indent=2)
    print(f"  API Schema → {api_out}")

    # ── 4. Latency Benchmark ──────────────────────────────────────────────────
    print("\n[4/6] Latency benchmark (1000 predictions, single-threaded)...")
    X_bench = scaler.transform(X_val[:1000])  # already validated
    # Warmup
    for _ in range(10):
        _ = model.predict_proba(X_bench[:10])

    latencies_ms = []
    for i in range(1000):
        x_single = X_bench[i:i+1]
        t_start = time.perf_counter()
        _ = model.predict_proba(x_single)
        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000)

    lat_arr = np.array(latencies_ms)
    latency_results = {
        "n_predictions": 1000,
        "mode": "single_prediction",
        "median_ms": float(np.median(lat_arr)),
        "mean_ms": float(lat_arr.mean()),
        "std_ms": float(lat_arr.std()),
        "p50_ms": float(np.percentile(lat_arr, 50)),
        "p90_ms": float(np.percentile(lat_arr, 90)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
        "max_ms": float(lat_arr.max()),
        "min_ms": float(lat_arr.min()),
        "throughput_qps": float(1000.0 / lat_arr.sum() * 1000),
    }
    print(f"  P50={latency_results['p50_ms']:.3f}ms | P95={latency_results['p95_ms']:.3f}ms | "
          f"P99={latency_results['p99_ms']:.3f}ms")

    # Batch latency
    t_batch_start = time.perf_counter()
    _ = model.predict_proba(X_val_scaled := scaler.transform(X_val))
    t_batch_end = time.perf_counter()
    batch_ms = (t_batch_end - t_batch_start) * 1000
    latency_results["batch_3500_ms"] = round(batch_ms, 3)
    latency_results["batch_throughput_qps"] = round(3500 / (batch_ms / 1000), 1)
    print(f"  Batch 3500: {batch_ms:.2f}ms ({latency_results['batch_throughput_qps']:.0f} QPS)")

    # ── 5. Memory Benchmark ───────────────────────────────────────────────────
    print("\n[5/6] Memory benchmark...")
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    # Simulate full inference pipeline
    model2 = joblib.load(FINAL_MODEL_DIR / "pillar1_logistic_model.joblib")
    scaler2 = joblib.load(FINAL_MODEL_DIR / "robust_scaler.joblib")
    X_mem = scaler2.transform(X_val)
    _ = model2.predict_proba(X_mem)

    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap2.compare_to(snap1, "lineno")
    total_kb = sum(s.size_diff for s in stats) / 1024
    peak_kb = sum(s.size for s in stats[:10]) / 1024

    memory_results = {
        "total_allocated_kb": round(total_kb, 2),
        "top_frames_kb": round(peak_kb, 2),
        "inference_batch_size": 3500,
        "model_size_bytes": (FINAL_MODEL_DIR / "pillar1_logistic_model.joblib").stat().st_size,
        "scaler_size_bytes": (FINAL_MODEL_DIR / "robust_scaler.joblib").stat().st_size,
        "total_artifact_size_bytes": (
            (FINAL_MODEL_DIR / "pillar1_logistic_model.joblib").stat().st_size +
            (FINAL_MODEL_DIR / "robust_scaler.joblib").stat().st_size
        ),
    }
    print(f"  Total allocated: {total_kb:.1f} KB | "
          f"Artifact size: {memory_results['total_artifact_size_bytes']} bytes")

    # ── 6. Model Card ─────────────────────────────────────────────────────────
    print("\n[6/6] Writing model card...")
    model_card = f"""---
model_name: {MODEL_NAME}
version: {MODEL_VERSION}
language: en
license: cc-by-nc-4.0
tags:
  - hallucination-detection
  - NLI
  - logistic-regression
  - RAG
  - evaluation
---

# HalluciSense Pillar-1: Logistic Regression Hallucination Detector

## Model Summary

HalluciSense Pillar-1 is a 5-feature logistic regression classifier trained to detect
hallucinations in RAG (Retrieval-Augmented Generation) system outputs using NLI-based
claim-evidence alignment signals.

**Model Version**: {MODEL_VERSION}  
**Algorithm**: LogisticRegression (liblinear, L2, C=1.0)  
**Validation Verdict**: PILLAR 1 VALIDATED WITH LIMITATIONS  
**Generated**: {datetime.now(timezone.utc).isoformat()}

## Performance (Held-Out VAL — 3,500 samples)

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.6902 |
| PR-AUC (AP) | 0.6311 |
| F1 (τ=0.56) | 0.6618 |
| MCC (τ=0.56) | 0.3587 |
| Accuracy (τ=0.56) | 0.6803 |
| Brier Score | 0.2332 |
| ECE (10-bin) | see publication |

## Input Features

| Feature | Description | Range |
| --- | --- | --- |
| `mean_entailment` | Mean NLI entailment score across claims | [0, 1] |
| `max_entailment` | Maximum NLI entailment score | [0, 1] |
| `mean_contradiction` | Mean NLI contradiction score | [0, 1] |
| `min_support_margin` | Minimum support margin | [-1, 1] |
| `num_claims` | Number of atomic claims | ≥ 0 |

## Preprocessing

Input features must be passed through the paired **RobustScaler** before inference.
See `robust_scaler.joblib` in the artifact directory.

## Operating Threshold

Default threshold: **0.56** (optimized for balanced F1/MCC on DEV set).  
For higher recall, threshold can be lowered to 0.50 (see threshold analysis).

## Limitations & Known Issues

- ROC-AUC of 0.69 indicates moderate discrimination — above baseline but not publication-perfect.
- The model relies on NLI signals only; factual grounding beyond textual entailment is not captured.
- Performance may degrade on out-of-distribution domains not represented in HaluBench/HaluEval/RAGTruth.
- `num_claims` feature has relatively low importance (see permutation importance analysis).
- Logit values are bounded [-3, 3]; probability outputs are well-constrained.

## Intended Use

- Automated hallucination detection in RAG pipelines.
- Screening of LLM outputs for downstream verification.
- Research benchmark for claim-level NLI alignment methods.

## Out-of-Scope Uses

- High-stakes medical/legal decision-making without human review.
- Generative model replacement or factual grounding.

## Training Data

- **Development set**: 58,002 samples (HaluBench + HaluEval + RAGTruth)
- **Validation set**: 3,500 held-out samples (protocol-locked before training)
- **NLI model**: `cross-encoder/nli-deberta-v3-small`

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| `pillar1_logistic_model.joblib` | `{model_sha}` |
| `robust_scaler.joblib` | `{scaler_sha}` |

## Citation

If you use this model, please cite the HalluciSense paper (forthcoming, Elsevier).
"""

    model_card_out = BUNDLE_DIR / "MODEL_CARD.md"
    with open(model_card_out, "w") as f:
        f.write(model_card)
    print(f"  Model Card → {model_card_out}")

    elapsed = time.time() - t0

    # Final bundle report
    bundle_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "9_step7_production_packaging",
        "model_registry": registry,
        "latency_benchmark": latency_results,
        "memory_benchmark": memory_results,
        "artifacts_produced": [
            str(registry_out), str(validator_out),
            str(api_out), str(model_card_out),
        ],
        "elapsed_seconds": round(elapsed, 2),
    }

    bundle_report_out = PUB / "step7_production_report.json"
    with open(bundle_report_out, "w") as f:
        json.dump(bundle_report, f, indent=2)

    # Markdown report
    md_lines = [
        "# Phase 9 — Step 7: Production Packaging",
        "",
        f"**Generated**: {bundle_report['generated_at_utc']}",
        "",
        "## 1. Artifacts Produced",
        "",
        "| Artifact | Description |",
        "| --- | --- |",
        "| `model_registry.json` | Versioned model registry with SHA-256 hashes |",
        "| `input_validator.py` | Feature validator with semantic and range checks |",
        "| `inference_api_schema.json` | OpenAPI 3.1 schema for /predict endpoint |",
        "| `MODEL_CARD.md` | HuggingFace-format model card |",
        "",
        "## 2. Latency Benchmark (1000 single-predictions)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| P50 | {latency_results['p50_ms']:.3f} ms |",
        f"| P95 | {latency_results['p95_ms']:.3f} ms |",
        f"| P99 | {latency_results['p99_ms']:.3f} ms |",
        f"| Single-prediction throughput | {latency_results['throughput_qps']:.0f} QPS |",
        f"| Batch 3500 | {latency_results['batch_3500_ms']:.1f} ms "
        f"({latency_results['batch_throughput_qps']:.0f} QPS) |",
        "",
        "## 3. Memory Profile (3500-sample batch)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Model artifact size | {memory_results['model_size_bytes']} bytes |",
        f"| Scaler artifact size | {memory_results['scaler_size_bytes']} bytes |",
        f"| Total artifact footprint | {memory_results['total_artifact_size_bytes']} bytes |",
        f"| Inference allocated | {memory_results['total_allocated_kb']:.1f} KB |",
        "",
        "## 4. Model Integrity",
        "",
        f"| Artifact | SHA-256 |",
        f"| --- | --- |",
        f"| `pillar1_logistic_model.joblib` | `{model_sha[:32]}…` |",
        f"| `robust_scaler.joblib` | `{scaler_sha[:32]}…` |",
    ]

    md_out = PUB / "step7_production_report.md"
    with open(md_out, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  MD  → {md_out}")

    print(f"\n✅ Step 7 complete in {elapsed:.1f}s")
    print(f"   P95 latency: {latency_results['p95_ms']:.3f}ms | "
          f"Artifact: {memory_results['total_artifact_size_bytes']} bytes")


if __name__ == "__main__":
    run()
