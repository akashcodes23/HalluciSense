# Phase 18 — Reviewer #1 Evaluation Report

**Reviewer Identity:** Senior Machine Learning & Fact-Checking Researcher  
**Manuscript Title:** *HalluciSense: An Availability-Aware, Calibrated Multi-Signal Verification Framework with Selective Abstention and Reverification-Gated Repair for Large Language Models*  
**Recommendation:** **MINOR REVISION**  
**Reviewer Confidence:** 5 / 5 (Expert)

---

## 1. Summary of the Work
The manuscript presents HalluciSense, a multi-signal hallucination verification framework combining evidence grounding (BM25+FAISS+DeBERTa-v3 NLI), predictive confidence (token logprob entropy), and semantic consistency (sentence transformer embeddings across alternate completions). The authors' core contribution is an *availability-aware adaptive fusion mechanism* using dynamic indicator masks $\mathbf{m} \in \{0, 1\}^3$ and reliability vectors $\mathbf{r}$, which renormalizes weights when verifier modalities are missing. The authors also integrate Platt probability calibration, selective prediction (risk-coverage gating), and closed-loop repair with downstream re-verification. Evaluations are reported on an internal benchmark ($N=750$) and 5 external datasets ($N=850$).

---

## 2. Key Strengths
1. **Practical Formulation of Signal Availability:** Formalizing verifier missingness via dynamic indicator masks $\mathbf{m}$ addresses a critical real-world limitation of black-box commercial LLM APIs.
2. **Methodological Completeness:** The paper provides a thorough end-to-end pipeline linking detection $\to$ probability calibration $\to$ selective abstention $\to$ closed-loop repair $\to$ independent re-verification.
3. **Rigorous Statistical Disclosures:** The authors explicitly separate per-sample paired effect sizes (Cohen's $d = 1.42$) from bootstrap distribution metrics and report nonparametric bootstrap 95% confidence intervals throughout.
4. **Reproducibility Architecture:** The codebase enforces singleton memory constraints ($\le 1.2\text{ GB}$ peak RAM), provides machine-readable manifests, and maintains an immutable benchmark dataset hash.

---

## 3. Major Concerns
1. **Potential Retrieval Grounding Over-Reliance:** The reported AUROC on external benchmarks ($0.9964$) is exceptionally high. Although trivial feature baselines perform near random chance ($0.5120$), the authors must explicitly discuss whether the dense Wikipedia retrieval index provides an overly favorable grounding advantage on standard open-domain benchmarks.
2. **Ablation Interpretation under Single Modalities:** Under single-pillar operation (e.g. Mask $[1, 0, 0]$), the adaptive formulation naturally simplifies to $1.0 \times \text{FE}$. The authors should clearly explain that the $+0.2380$ $\Delta\text{AUROC}$ gain over fixed fusion reflects the elimination of zero-imputation penalty rather than a novel representation transformation.
3. **Abstention Cost in High-Stakes Environments:** Achieving $0.0\%$ empirical selective risk requires abstaining on $20\%$ of queries. The authors must emphasize the operational cost of this rejection rate in automated production pipelines.

---

## 4. Minor Concerns
1. In Section 4.2, ensure Eq. (2) explicitly documents that $\sum m_i \ge 1$ is required to prevent a zero-denominator condition.
2. In Table 4, ensure that citation footnotes for literature baselines are prominently displayed in the main text.
3. Clarify the execution time profile: retrieval constitutes $\sim 65\%$ ($780\text{ ms}$) of the end-to-end latency ($1205\text{ ms}$).

---

## 5. Novelty Assessment: `MODERATE TO STRONG NOVELTY`
The mathematical formulation of availability-aware dynamic masking with reliability modulation is novel and well-grounded. The pipeline integration (calibration + selective abstention + reverified repair) represents a strong system contribution.

---

## 6. Methodological & Statistical Soundness: `EXCELLENT`
The statistical evaluation is thorough: 500-iteration paired bootstrap intervals, paired Wilcoxon tests, and proper Cohen's $d = 1.42$ reporting are methodologically sound.

---

## 7. Recommendation: `MINOR REVISION`
Acceptable for publication in an Elsevier AI journal following minor clarifications regarding retrieval dependency and single-modality scaling.
