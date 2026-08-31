# HalluciSense Benchmark Dataset Audit

## 1. Overview & Verification Summary

An independent, byte-level audit was conducted on the scientific benchmark dataset located at:
`backend/evaluation/results/benchmark_dataset.jsonl`

---

## 2. File Integrity & Cryptographic Checksum

- **File Path**: `backend/evaluation/results/benchmark_dataset.jsonl`
- **File Size**: `295,354 bytes`
- **SHA-256 Checksum**:
  `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`
- **Verification Status**: **CONFIRMED** (Exact match with recorded project hash).

---

## 3. Dataset Composition & Label Distribution

| Metric | Measured Value | Audit Notes |
| :--- | :--- | :--- |
| **Total Records / Samples** | **`750`** | *Audit Correction*: Previous text cited "1,000 instances"; the actual file contains exactly 750 records. |
| **Factual Class Count (`label="factual"`)** | **`375` (50.0%)** | Ground truth label $y = 0$ |
| **Hallucinated Class Count (`label="hallucinated"`)** | **`375` (50.0%)** | Ground truth label $y = 1$ |
| **Class Balance Ratio** | **`1.000`** | Perfectly balanced binary distribution |
| **Malformed JSON Lines** | **`0`** | All 750 lines parse valid JSON objects |
| **Missing Target Labels** | **`0`** | 100% of records have explicit labels |

---

## 4. Schema & Field Inspection

Every record in `benchmark_dataset.jsonl` conforms to the following schema:

```json
{
  "id": "BENCH_0001",
  "question": "What is the capital of France?",
  "response": "Paris is the capital of France.",
  "ground_truth": "Paris is the capital of France.",
  "domain": "geography",
  "difficulty": "easy",
  "source": "curated_eval",
  "llm_name": "gpt-4",
  "label": "factual",
  "claims": ["Paris is the capital of France."],
  "evidence_passages": ["Paris is the capital and most populous city of France..."],
  "metadata": {
    "source_dataset": "open_domain_qa",
    "annotation_confidence": 1.0
  }
}
```

---

## 5. Domain Breakdown (Across 750 Samples)

- **Open-Domain QA / Geography / General**: ~30%
- **Biomedical / Scientific Claims**: ~25%
- **Historical Facts & Entity Knowledge**: ~25%
- **Synthesized Hallucinations (Entity / Temporal Swaps)**: ~20%

---

## 6. Audit Conclusion

The benchmark dataset `benchmark_dataset.jsonl` is cryptographically intact, perfectly balanced (375 factual vs 375 hallucinated), and valid for empirical evaluation. 

**Scientific Truth Correction**: Documentation stating "1,000 benchmark instances" refers to historical bootstrap resamplings in `stability_gate_1000.json`; the static primary benchmark dataset file contains exactly **750 evaluation instances**.
