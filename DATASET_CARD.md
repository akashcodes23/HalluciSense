# HalluciSense Benchmark Dataset Card (Phase 22)

## Dataset Overview
- **Dataset Name**: HalluciSense Public Research Benchmark Suite (Phase 22)
- **Total Samples**: $N = 750$ claims across 15 research domains.
- **Integrated Public Datasets**: HaluEval, TruthfulQA, FreshQA, FEVER, SciFact, HoVer, VitaminC, FActScore, BEGIN, XSumFaith, PubHealth, PubMedQA, MedQA.
- **License**: Creative Commons Attribution 4.0 / MIT / Apache-2.0.

---

## Benchmark Domain Distribution

| Research Domain | Sample Count | Primary Dataset Source | Task Type |
| :--- | :---: | :--- | :--- |
| **General Knowledge** | 50 | HaluEval / TruthfulQA | QA Hallucination Verification |
| **Medicine** | 50 | PubHealth / PubMedQA | Biomedical Fact Checking |
| **Law** | 50 | Legal-FEVER | Legal Precedent Verification |
| **Finance** | 50 | FinFact | Financial Statement Audit |
| **History** | 50 | FactKG / FEVER | Historical Event Verification |
| **Science** | 50 | SciFact | Scientific Paper Verification |
| **Computer Science** | 50 | CS-Benchmark | Algorithm & Complexity Audit |
| **Physics** | 50 | SciFact-Physics | Physical Law Verification |
| **Biology** | 50 | PubMedQA | Biological Pathway Checking |
| **Chemistry** | 50 | SciFact-Chem | Chemical Formula Verification |
| **News** | 50 | FreshQA / XSumFaith | Temporal News Faithfulness |
| **Mathematics** | 50 | Math-Fact | Mathematical Derivation Check |
| **Geography** | 50 | FEVER-Geo | Spatial Location Verification |
| **Politics** | 50 | PolitiFact | Governance Claim Verification |
| **Literature** | 50 | Lit-Bench | Literary Citation Check |

---

## Data Schema (`BenchmarkExample`)
Each JSONL record contains:
```json
{
  "id": "gen_0001",
  "question": "What is the capital of France?",
  "response": "Paris is the capital of France.",
  "ground_truth": 0,
  "label": "factual",
  "domain": "General Knowledge",
  "difficulty": "medium",
  "source": "HaluEval/FactKG/Benchmark",
  "llm_name": "GPT-4",
  "claims": ["Paris is the capital of France."],
  "evidence_passages": ["Paris is the capital and most populous city of France."]
}
```
