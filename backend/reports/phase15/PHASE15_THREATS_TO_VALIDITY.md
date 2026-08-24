# Phase 15 — Methodological Threats to Validity

## 1. Internal Validity
- **Threshold Selection Provenance:** All decision boundaries ($0.20, 0.35, 0.50, 0.65$), Platt scaling parameters ($a=1.82, b=-0.45$), and abstention margins ($0.08$) were fitted exclusively on the internal development partition ($60\%$ Train, $20\%$ Validation). Held-out test partitions ($N=150$) and external benchmarks ($N=850$) had zero exposure during parameter fitting.
- **Data Contamination:** Exact string matching and 3-gram Jaccard sweeps confirmed 0 exact duplicate claims and 0 label leaks into the model inference graph.
- **Reference Corpus Grounding:** External retrieval naturally accesses Wikipedia articles containing factual reference truths (e.g. speed of light, chemical formulas). This is standard for open-domain QA (FEVER, FActScore) but limits verification when external references are unavailable or paywalled.

---

## 2. External Validity
- **Domain Coverage:** Evaluated across 6 core domains: Physics, Chemistry, Biology, Medicine, Mathematics, and General Knowledge. While natural and formal sciences are well covered, legal case law and highly localized regulatory domains require domain-specific knowledge corpus integration.
- **Language Coverage:** Current experiments are conducted exclusively in English. Multilingual generalization remains a research avenue.
- **Generator Portability:** Evaluated on leading frontier models (GPT-4, Gemini-1.5, Claude-3.5, LLaMA-3). Extreme output idiosyncratic formatting in smaller open-source models ($\le 3\text{B}$) may require specialized claim segmentation tuning.

---

## 3. Construct Validity
- **Definition of Hallucination:** Hallucination is operationalized as factual inconsistency, numerical scale error, negation flip, causal reversal, or unsupported claim relative to verifiable consensus evidence. Stylistic bias, subjectivity, or conversational tone are not measured as hallucinations.
- **H-Score Interpretation:** The continuous score $H \in [0, 1]$ represents calibrated posterior risk of hallucination; it does not measure grammatical fluency or creative quality.

---

## 4. Statistical Validity
- **Sample Size:** Evaluated on $N=750$ canonical benchmark claims and $N=850$ external benchmark instances ($N=1,600$ total empirical points).
- **Confidence Bounds:** Nonparametric empirical bootstrap ($B=500$) resamples were used for all metric confidence intervals and paired difference tests.
- **Multiple Comparisons:** Bonferroni / False Discovery Rate adjustments were observed when reporting significance across multiple ablation conditions.

---

## 5. Operational & Systems Limitations
- **Wikipedia / REST API Latency:** External retrieval ($P_1$) accounts for $\sim 65\%$ of end-to-end latency ($1203\text{ ms}$ total). In network-isolated environments, offline mode ($P_2 + P_3$) must be utilized with a $-8.8\%$ AUROC tradeoff.
- **Long-Context Coreference:** Complex multi-hop pronouns across $>10$ conversational turns require upstream coreference resolution.
- **Arbitrary Precision Symbolic Calculations:** Symbolic unit and numeric parsers cover standard SI units and scientific exponents, but do not replace specialized Computer Algebra Systems (CAS) for abstract symbolic integrals.
