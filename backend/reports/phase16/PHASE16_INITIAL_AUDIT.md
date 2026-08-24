# Phase 16 — Initial System & Manuscript Evidence Audit

## 1. Executive Summary
This initial audit reviews the state of the HalluciSense repository and evaluation artifacts at the threshold of Phase 16. The goal is to identify all methodological vulnerabilities, baseline ambiguities, statistical reporting anomalies, and potential reviewer attacks before finalizing the manuscript evidence package.

---

## 2. Current Architecture & Research Invariants
- **Claim-Level Decomposition:** Input LLM responses are parsed into atomic constituent claims.
- **Three-Pillar Multi-Signal Verification:**
  * **Pillar 1 ($\text{FE}$):** External Evidence Grounding via BM25 + FAISS + DeBERTa-v3 NLI + Symbolic numeric/unit/causal checks.
  * **Pillar 2 ($\text{CG}$):** Predictive Uncertainty via token-level entropy and confidence gap.
  * **Pillar 3 ($\text{CF}$):** Semantic Consistency via sentence transformer embeddings and claim-aligned cross-comparison.
- **Fusion Formulations:**
  * Canonical Mode A: $H_{\text{canonical}} = \alpha \cdot \text{FE} + \beta \cdot \text{CG} + \gamma \cdot \text{CF}$ ($\alpha=0.40, \beta=0.30, \gamma=0.30$).
  * Availability-Aware Mode B: $H_{\text{adaptive}} = \frac{\sum m_i r_i w_i S_i}{\sum m_i r_i w_i}$ with indicator mask $\mathbf{m} \in \{0, 1\}^3$ and reliability vector $\mathbf{r} \in (0, 1]^3$.
- **Downstream Reliability & Safety:**
  * Platt Probability Calibration ($a=1.82, b=-0.45$).
  * Selective Abstention Gate (`INSUFFICIENT_EVIDENCE`, `ABSTAIN`).
  * Closed-Loop Repair with independent downstream re-verification gate ($H_{\text{post}} < 0.20$).
  * ModelRegistry singleton architecture bounding PyTorch FP32 memory to $\le 1.2\text{ GB}$ peak.

---

## 3. Current Scientific Claims & Experimental Baselines
- **Main Internal Results:** Canonical benchmark $N=750$ (SHA-256: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`), held-out test AUROC = `1.0000`, Platt ECE = `0.0937`.
- **External Generalization:** 5 peer-reviewed external datasets ($N=850$): TruthfulQA, HaluEval, FEVER, RAGTruth, BioASQ. Combined AUROC = `0.9964` (95% CI: `[0.9938, 0.9985]`).
- **Baselines Evaluated:** Single Pillars ($P_1, P_2, P_3$), Fixed Fusion, Adaptive Fusion, Adaptive+Calibration, Adaptive+Abstention, Full Pipeline, and 4 literature baselines (SelfCheckGPT, MiniCheck, FActScore, CoVe).

---

## 4. Current Statistical Procedures & Identified Methodological Vulnerabilities

### A. Cohen's $d$ Anomaly on Paired AUROC Differences
- **Identified Issue:** Phase 14 / 15 reported Cohen's $d$ values of `25.69` and `33.52` for $\Delta\text{AUROC}$.
- **Root Cause Analysis:** These values were derived by taking the mean bootstrap delta divided by the bootstrap distribution standard error ($SE_{\text{boot}}$), effectively measuring a $z$-score of the bootstrap mean rather than standard per-sample standardized effect size ($d = \frac{\bar{D}}{s_D}$).
- **Phase 16 Remediation:** In Phase 16, we explicitly separate:
  1. Paired per-instance prediction score difference effect size: $d = \frac{\bar{H}_{\text{fixed}} - \bar{H}_{\text{adaptive}}}{s_D}$.
  2. Nonparametric paired bootstrap difference: $\Delta\text{AUROC}$ with empirical 95% percentile confidence intervals.
  3. Paired Wilcoxon signed-rank and McNemar tests for paired classification decisions.

### B. Baseline Comparability Classification
- **Identified Issue:** Published literature baselines (SelfCheckGPT, MiniCheck, FActScore, CoVe) must be explicitly categorized as *"Reported from original literature"* to ensure skeptical reviewers do not mistake them for identical local reimplementations.
- **Phase 16 Remediation:** Create a strict baseline registry distinguishing (A) Directly Reproduced, (B) Reproduced under modified protocol, (C) Reported from original literature, and (D) Not directly comparable.

### C. Selective Abstention Wording
- **Identified Issue:** Phrases like "100% precision @ 80% coverage" must avoid implying unconditional perfection.
- **Phase 16 Remediation:** Adopt precise wording: *"On the evaluated test population, the retained 80% coverage subset exhibited zero observed classification errors under the pre-selected validation threshold."*

### D. Closed-Loop Metric Denominators
- **Identified Issue:** Denominators for CSR (Correction Success Rate), RPR (Reverification Pass Rate), and CIHR (Correction-Induced Hallucination Rate) must be explicitly documented at claim-level vs response-level.

---

## 5. Potential Reviewer Attack Vectors & Mitigations

| Reviewer Attack Vector | Severity | Mitigation & Evidence Locked in Phase 16 |
| :--- | :--- | :--- |
| **1. "How do you know the high AUROC isn't dataset artifact?"** | HIGH | Falsification audit with 9 trivial/permutation baselines (label scramble, claim length, domain-only, etc.). |
| **2. "Did you tune thresholds on the test set?"** | CRITICAL | Explicit provenance graph showing all thresholds/calibration parameters were fitted exclusively on Train/Val. |
| **3. "Are external literature comparisons direct?"** | HIGH | Baseline registry clearly tagging literature reported metrics vs native HalluciSense evaluations. |
| **4. "Why is Cohen's d so large?"** | HIGH | Methodological audit clarifying per-instance vs bootstrap distribution effect sizes. |
| **5. "What if external Wikipedia retrieval is unavailable?"** | MEDIUM | Signal mask `[0, 1, 1]` offline evaluation documented honestly with a $-8.8\%$ AUROC disclosure. |
| **6. "Can an independent researcher reproduce this?"** | CRITICAL | Clean-room reproduction script, frozen seeds, complete machine-readable manifests. |

---

## 6. Files & Modules Targeted for Phase 16 Execution
- `backend/evaluation/phase16/`: Baseline registry, statistical audit runner, falsification runner, master gate runner.
- `backend/reports/phase16/`: All 8 comprehensive analytical reports, 13 paper-grade CSV tables, 10 publication figures.
- `backend/tests/`: `test_phase16_scientific_gate.py` verifying all gate invariants.
