# Phase 17 — Literature Search & Provenance Log

## 1. Literature Search Strategy & Protocol
- **Databases Queried:** ACL Anthology, IEEE Xplore, ScienceDirect / Elsevier, arXiv CS.CL / CS.AI, NeurIPS / ICLR Proceedings.
- **Search Queries:**
  1. `"hallucination detection" AND "retrieval augmented"`
  2. `"multi-signal fusion" AND "fact checking"`
  3. `"selective prediction" AND "large language models"`
  4. `"confidence calibration" AND "hallucination"`
  5. `"hallucination correction" AND "re-verification"`

---

## 2. Structured Inclusion & Exclusion Log

| Search Query | Venue / Source | Candidate System | Decision | Inclusion Justification / Exclusion Reason |
| :--- | :--- | :--- | :---: | :--- |
| `black-box hallucination` | EMNLP 2023 | SelfCheckGPT | **INCLUDED** | Seminal baseline for sample consistency without logprobs. |
| `atomic factuality` | EMNLP 2023 | FActScore | **INCLUDED** | Standard atomic claim decomposition methodology. |
| `efficient fact-checking` | EMNLP 2024 | MiniCheck | **INCLUDED** | High-efficiency DeBERTa/RoBERTa NLI cross-encoder baseline. |
| `iterative verification` | ACL 2024 | Chain-of-Verification | **INCLUDED** | Frontier baseline for multi-step question verification. |
| `semantic entropy` | ICLR 2023 | Semantic Uncertainty | **INCLUDED** | Foundation for cluster-based semantic variance in Pillar 3. |
| `selective classification` | NeurIPS 2017 | Geifman & El-Yaniv | **INCLUDED** | Mathematical basis for risk-coverage curves and AURC. |
| `probability calibration` | Platt 1999 | Platt Scaling | **INCLUDED** | Established standard for monotonic sigmoidal logit calibration. |
| `LLM-as-a-judge` | NeurIPS 2023 | G-Eval / MT-Bench | **EXCLUDED** | Prompt-based subjective judge; non-deterministic and lacks epistemic calibration. |
| `proprietary closed API` | Commercial | Proprietary Enterprise Evaluators | **EXCLUDED** | Non-reproducible private training data and unverified weights. |
