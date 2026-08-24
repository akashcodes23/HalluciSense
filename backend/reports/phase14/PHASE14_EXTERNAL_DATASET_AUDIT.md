# Phase 14 — External Dataset Strategy & Audit

## 1. Executive Summary
To rigorously validate the external generalizability of HalluciSense beyond its internally constructed canonical benchmark, we selected five authoritative, peer-reviewed public benchmarks covering diverse domains, tasks, and generator LLMs.

---

## 2. Selected External Datasets

| Dataset | Primary Citation | License | Sample Size ($N$) | Primary Domain | Generator Architectures |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TruthfulQA** | Lin et al. (ACL 2022) | Apache-2.0 | $200$ | Multi-domain Misconceptions | GPT-3, GPT-4, LLaMA |
| **HaluEval** | Li et al. (EMNLP 2023) | MIT | $200$ | Open-domain QA & Dialogue | ChatGPT, InstructGPT |
| **FEVER** | Thorne et al. (NAACL 2018) | CC BY-SA 4.0 | $200$ | Encyclopedia Fact Extraction | Human Annotators & Perturbations |
| **RAGTruth** | Yue et al. (ACL 2024) | MIT | $150$ | RAG Retrieval & Longform | GPT-4, LLaMA-2, Mistral-7B |
| **BioASQ-FactCheck**| Krithara et al. (2023) | Open Access | $100$ | Medicine & Clinical Science | PubMed / Clinical-LLM |

---

## 3. Strict Zero-Tuning Protocol
For all five external benchmarks:
1. **Zero Weight Retuning:** Fixed base weights ($\alpha=0.40, \beta=0.30, \gamma=0.30$) were applied without dataset-specific tuning.
2. **Zero Calibration Refitting:** Platt scaling parameters ($a=1.82, b=-0.45$) fitted on the internal training split were applied out-of-the-box.
3. **Zero Rule Modification:** No dataset-specific heuristics, keywords, or token biases were introduced.
4. **Frozen Configuration File:** Stored in [`backend/evaluation/phase14/phase14_external_frozen_config.json`](file:///Users/akashgpatil/major_project/backend/evaluation/phase14/phase14_external_frozen_config.json).
