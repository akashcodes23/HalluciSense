# Phase 18 — Retrieval Contamination & Evidence Leakage Audit

## 1. High-Priority Investigation Objective
To investigate whether external retrieval from Wikipedia REST endpoints, FAISS passage stores, or local BM25 indices could inadvertently expose the hallucination verifier to label-bearing metadata, circular prompt references, or test-set contamination.

---

## 2. Granular Investigation Findings

| Audit Dimension | Investigation Method | Empirical Finding | Contamination Risk |
| :--- | :--- | :--- | :---: |
| **Evidence Corpus Provenance** | Inspected Wikipedia dump / REST endpoint queries | Retrieval queries are constructed strictly from generated claim strings (e.g. *"The speed of light in vacuum is 300,000 km/s"*); no ground-truth annotations are passed. | **LOW RISK** |
| **Retrieval Timing & Execution** | Pipeline execution trace analysis | Evidence retrieval occurs *dynamically at inference runtime* per claim. | **LOW RISK** |
| **Label-Bearing Metadata** | HTML / JSON parser inspection | Parsers extract raw text extracts only (\texttt{extract} field); all external categories, edit histories, and dispute tags are stripped. | **LOW RISK** |
| **Circular Prompt Leakage** | Exact string match between benchmark questions and retrieved articles | Benchmark questions are derived from natural scientific facts; retrieved articles are canonical encyclopedia entries. | **LOW RISK** |
| **Retrieval Cache Contamination** | Cache inspection in `backend/` | Local cache is partitioned strictly by hash of query text; zero label or partition metadata is cached. | **LOW RISK** |
| **Negative Claim Entailment Behavior** | Evaluated NLI behavior on counterfactual / hallucinated claims | For false claims (e.g. *"Water is $H_3O$"*), retrieval fetches the canonical article (*"Water"*), and DeBERTa-v3 correctly outputs **CONTRADICTION**. | **LOW RISK** |

---

## 3. Overall Retrieval Contamination Classification
### Verdict: `LOW RISK`

**Justification:**
Evidence retrieval behaves strictly as an open-domain factual grounding lookup. The pipeline does not store or retrieve benchmark-specific labels, and counterfactual claims retrieve factual consensus passages that trigger NLI contradiction rather than false agreement.
