# HalluciSense Canonical Benchmark Dataset (Phase 6 Freeze)

## 1. Dataset Overview

The **HalluciSense Canonical Benchmark Dataset** is a frozen, multi-domain diagnostic evaluation suite specifically constructed to measure large language model hallucination detection performance across 15 distinct research disciplines.

* **Benchmark Name**: HalluciSense Multi-Domain Canonical Benchmark
* **Version**: `1.0.0-phase6-freeze`
* **Total Sample Count ($N$)**: **750 Claims**
* **Domains**: **15 Research Domains** (50 samples per domain)
* **Ground Truth Distribution**: **375 Factual (Label 0) / 375 Hallucinated (Label 1)** (Exact 50/50 balance)
* **Canonical Path**: `backend/evaluation/results/benchmark_dataset.jsonl`
* **Canonical SHA-256**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`

---

## 2. Distinction: Canonical Benchmark vs. Official Public Datasets

> [!IMPORTANT]
> **Scientific Transparency Notice**:
> This dataset is **not** a raw direct download of external public corpora (such as TruthfulQA, HaluEval, or FEVER).
> It is a **locally curated, deterministically generated multi-domain diagnostic suite** authored in `backend/evaluation/benchmark_dataset/importer.py` that incorporates fact-checking patterns, misconception challenges, and scientific verification themes inspired by established literature.

---

## 3. Domain Distribution ($N=50$ per domain)

| # | Domain | Sample Count | Factual / Hallucinated | Difficulty Tiers |
|---|---|---|---|---|
| 1 | **General Knowledge** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 2 | **Medicine** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 3 | **Law** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 4 | **Finance** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 5 | **History** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 6 | **Science** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 7 | **Computer Science** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 8 | **Physics** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 9 | **Biology** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 10 | **Chemistry** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 11 | **News** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 12 | **Mathematics** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 13 | **Geography** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 14 | **Politics** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| 15 | **Literature** | 50 | 25 / 25 | 17 Easy, 17 Med, 16 Hard |
| **Total** | **15 Domains** | **750** | **375 / 375** | **255 Easy, 255 Med, 240 Hard** |

---

## 4. Dataset Schema (`BenchmarkExample`)

Each record in `benchmark_dataset.jsonl` follows this strict JSON schema:

```json
{
  "id": "gen_0001",
  "question": "What is the capital of France?",
  "response": "Paris is the capital of France.",
  "ground_truth": 0,
  "domain": "General Knowledge",
  "difficulty": "easy",
  "source": "HalluciSense-Benchmark-v1",
  "llm_name": "GPT-4",
  "label": "factual",
  "claims": [
    "Paris is the capital of France."
  ],
  "evidence_passages": [],
  "metadata": {}
}
```

### Field Definitions:
* `id` (`string`): Unique claim identifier formatted as `{domain_prefix}_{counter}`.
* `question` (`string`): The natural-language query or prompt.
* `response` (`string`): The target model response under test.
* `ground_truth` (`integer`): Binary label where `0 = Factual / Supported` and `1 = Hallucinated / Contradicted`.
* `domain` (`string`): One of the 15 canonical domain names.
* `difficulty` (`string`): `easy`, `medium`, or `hard`.
* `label` (`string`): Human-readable label (`"factual"` or `"hallucinated"`).
* `claims` (`array[string]`): Segmented atomic factual assertions extracted from the response.

---

## 5. Construction & Preprocessing Pipeline

```
Domain Definition (15 Canonical Research Domains)
                       ↓
Curated Fact-Check Query & Response Alignment (Factual vs Contradicted Pairs)
                       ↓
Deterministic Multi-Domain Expansion (50 per domain, Seed = 42)
                       ↓
Schema Validation (100% Validated against BenchmarkExample dataclass)
                       ↓
Deduplication Audit (0 Duplicates across 750 samples)
                       ↓
Cryptographic Hash Freeze (SHA-256 Verified)
```

---

## 6. How to Reproduce the Dataset

To deterministically recreate the canonical benchmark and verify cryptographic parity against the frozen file:

```bash
PYTHONPATH=backend ./venv/bin/python backend/evaluation/reproduce_phase6_dataset.py
```

Expected output:
```text
DATASET REPRODUCTION CHECK
==========================
Records: PASS (750/750)
Domains: PASS (15/15)
Class balance: PASS (375/375)
Schema: PASS (0 malformed)
Duplicate check: PASS (0 duplicates)
SHA-256: PASS (dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5)
```

---

## 7. Known Limitations & Research Context

1. **Signal Availability**: In offline static text benchmarking, token logprobs ($P_2$) are omitted from the raw JSONL records. Therefore, offline evaluation operates primarily via **Pillar 1 (Evidence Grounding)**.
2. **Deterministic Synthetics**: Because the 750 claims are generated from curated deterministic templates, the benchmark acts as a structured unit & diagnostic regression suite rather than an open-ended conversational corpus.
3. **Licensing**: Released under the **MIT License**. Reference literature methodologies are attributed to their respective authors (Li et al., 2023; Lin et al., 2022; Thorne et al., 2018).
