# Phase 6C: Dataset Inventory & Split Audit

**Generated**: 2026-08-10
**Git SHA**: cbe4de7f72b7c874727e1025acf348219136ed60
**Machine-readable version**: phase6c_dataset_inventory.json

---

## Dataset Role Definitions

| Role | Description |
|:---|:---|
| `FINAL_TEST` | Never used during Phase 6 development. Primary publication evaluation set. |
| `VALIDATION` | Constructed for Phase 6 evaluation; used as validation, not for tuning. |
| `DEVELOPMENT_VALIDATION` | Used during Phase 6 development. NOT suitable as final test. |

**Principle**: No dataset used for tuning or system improvement may be reported as a final test set.

---

## Dataset 1: HaluBench (PatronusAI)

| Property | Value |
|:---|:---|
| Source | PatronusAI/HaluBench (HuggingFace) |
| N | 100 |
| Positive (hallucinated) | 100 (100.0%) |
| Negative (factual) | 0 (0.0%) |
| sha256_prefix | 20e2101e396dfcb2 |
| Role | FINAL_TEST |
| Used in Phase 6 development | NO |
| Task type | Hallucination detection (QA) |
| Domains | Mixed QA |

**⚠️ CRITICAL CAVEAT**: HaluBench is **100% hallucinated** (no factual examples). Accuracy, specificity, and FPR are **undefined** when evaluated on HaluBench in isolation. Only combined evaluation with other datasets (where negatives exist) produces meaningful accuracy/specificity metrics.

---

## Dataset 2: RAGTruth (ParticleMedia)

| Property | Value |
|:---|:---|
| Source | ParticleMedia/RAGTruth (HuggingFace) |
| N | 300 |
| Positive (hallucinated) | 98 (32.7%) |
| Negative (factual) | 202 (67.3%) |
| sha256_prefix | 22b72af4e7a12879 |
| Role | FINAL_TEST |
| Used in Phase 6 development | NO |
| Task type | RAG hallucination detection |
| Domains | News, financial, general QA |

**⚠️ CAVEAT**: Class imbalance (32.7% / 67.3%). Raw Accuracy is upward-biased toward the majority class. Report Balanced Accuracy and MCC alongside Accuracy.

---

## Dataset 3: HaluEval (RUCAIBox)

| Property | Value |
|:---|:---|
| Source | RUCAIBox/HaluEval (HuggingFace) |
| N | 150 |
| Positive (hallucinated) | 75 (50.0%) |
| Negative (factual) | 75 (50.0%) |
| sha256_prefix | 4d0b220f1e6799bb |
| Role | FINAL_TEST |
| Used in Phase 6 development | NO |
| Task type | Hallucination detection (QA, dialogue, summarization) |
| Domains | General QA, dialogue, summarization |

**✅ BEST SINGLE-DATASET CANDIDATE**: Balanced (50/50). Raw Accuracy is an unbiased estimator here.

---

## Dataset 4: Phase 6 Unseen Benchmark (Internal)

| Property | Value |
|:---|:---|
| Source | HalluciSense internal (constructed for Phase 6) |
| N | 105 |
| Positive (hallucinated) | 35 (~33.3%) |
| Negative (factual) | 70 (~66.7%) |
| sha256_prefix | internal |
| Role | VALIDATION |
| Used in Phase 6 development | NO (constructed after Phase 6 design) |
| Task type | Temporal hallucination detection |
| Domains | 15 domains (history, astronomy, medicine, etc.) |
| Temporal categories | 15 categories |

**⚠️ IMPORTANT ACCURACY DISCREPANCY**:  
- `phase6_architectural_evaluation.md` reports **Accuracy=89.52%** → this is a pre-freeze, temporal-engine-only evaluation on an earlier system version. NOT the authoritative final evaluation.
- `phase6_unseen_benchmark.json` reports **Accuracy=53.33%** → this is the authoritative frozen full-pipeline evaluation.

**Authoritative result**: 53.33% (full pipeline, frozen system, commit cbe4de7).

---

## Dataset 5: Phase 5 Holdout (Internal)

| Property | Value |
|:---|:---|
| Source | HalluciSense internal (Phase 5 blind holdout) |
| N | 70 |
| Positive | ~24 (~34%) |
| Negative | ~46 (~66%) |
| Role | DEVELOPMENT_VALIDATION |
| Used in Phase 6 development | **YES** — used as evaluation set during Phase 6 architecture iterations |

**❌ DO NOT USE AS FINAL TEST**: This dataset was used during Phase 6 development. It is NOT independent of the Phase 6 system and MUST NOT be reported as a final held-out test result.

Phase 6 achieved 88.57% on this set — this is a **development validation** result, not a publication-quality final evaluation.

---

## Combined External Dataset (Phase 6B/6C Primary Evaluation)

| Property | Value |
|:---|:---|
| Datasets | HaluBench + RAGTruth + HaluEval |
| N | 550 |
| Positive | 273 (49.6%) |
| Negative | 277 (50.4%) |

**Note**: Near-balanced by coincidence from dataset mixing. Per-dataset class distributions vary significantly (see above). Report per-dataset breakdowns when possible.

---

## Data Use Summary

```
Phase 6C Evaluation Design:
  
  FINAL TEST (publication):     HaluBench + RAGTruth + HaluEval (N=550)
                                Phase 6 Unseen Benchmark (N=105)
  
  DEVELOPMENT VALIDATION:       Phase 5 Holdout (N=70)  ← NOT for final test
  
  Production tuning data:       NONE (weights/thresholds frozen)
```

---

## Split Independence Assertion

No examples from any FINAL_TEST dataset were used to:
- Design the architecture of Phase 6
- Tune fusion weights (α=0.40, β=0.30, γ=0.30)
- Set risk thresholds
- Select temporal pattern regexes
- Adjust detection parameters

The system architecture was frozen at commit cbe4de7 **before** the external datasets were used for evaluation.
