# Phase 16 — Data Integrity & Provenance Lock

## 1. Canonical Benchmark Hash Invariant
The canonical benchmark dataset file has remained strictly frozen and unaltered throughout all phases:
- **File Path:** [`backend/evaluation/results/benchmark_dataset.jsonl`](file:///Users/akashgpatil/major_project/backend/evaluation/results/benchmark_dataset.jsonl)
- **SHA-256 Hash:** `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`
- **Verification Status:** **CONFIRMED & UNCHANGED**

---

## 2. Partitioning & Leakage Audit

| Integrity Check | Procedure | Observed Result | Status |
| :--- | :--- | :---: | :---: |
| **Exact Duplicate Check** | Pairwise $(Q+R)$ normalized string hashing | 0 duplicates across 750 records | **PASS** |
| **Near-Duplicate Overlap** | 3-gram Jaccard coefficient sweep | Group-stratified partition in `phase13_split_manifest.json` | **PASS** |
| **Label Leakage** | Code inspection of engine inputs | No ground truth labels flow into inference | **PASS** |
| **Threshold Leakage** | Parameter fitting audit | Thresholds fitted exclusively on Train/Val splits | **PASS** |
| **Calibration Leakage** | Logistic parameter fitting audit | Platt parameters fitted exclusively on Training partition | **PASS** |
| **External Contamination** | Cross-benchmark overlap audit | 0 overlap between internal dataset and 5 external datasets | **PASS** |

---

## 3. Implementation vs Architecture Clarification
- **Scientific Architecture:** The three-pillar concept, Mode A, Mode B, Platt calibration, and closed-loop repair remain 100% architecturally preserved as designed.
- **Implementation Hardening:** In Phase 15, `compute_adaptive_h_score` in [`backend/app/core/engine/fusion.py`](file:///Users/akashgpatil/major_project/backend/app/core/engine/fusion.py) was enhanced to accept `fe: Optional[float] = None` and compute dynamic $m_{\text{FE}} \in \{0, 1\}$ rather than assuming $m_{\text{FE}}=1$ unconditionally.
- **Wording Invariant:** Manuscript text will explicitly state: *"No new architectural component was introduced; existing adaptive fusion mechanisms were formalized and hardened for general signal missingness."*
