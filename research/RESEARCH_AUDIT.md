# HalluciSense Research Audit & Claim Verification Document

**Project**: HalluciSense  
**Original Manuscript**: *“HalluciSense: A Confidence-Aware Hybrid Framework for Detecting and Quantifying Hallucinations in Large Language Models”*  
**Original Authors**: Akash G Patil, Chirag O, Darshan A, Keerthan B M  
**Affiliation**: Department of Computer Science and Engineering (AI/ML), JSS Academy of Technical Education, Bengaluru, India  
**Audit Date**: October 2026  
**Auditor**: Senior AI/ML Research & Scientific Review Team  

---

## 1. Executive Summary

This document presents a rigorous, evidence-based scientific audit comparing the claims made in the initial HalluciSense research proposal manuscript against the actual codebase, experimental pipelines, execution traces, and empirical evaluation data.

The original manuscript presented a conceptual architecture combining retrieval verification, model confidence analysis, and cross-consistency checking, with a proposed linear $H$-Score ($H = \alpha \cdot \text{FE} + \beta \cdot \text{CG} + \gamma \cdot \text{CF}$). At the time of the draft, empirical evaluation was planned ("scheduled for Phase 2").

Subsequent development and experimental campaigns (Phases 1 through 17) produced a production-grade verification engine and extensive empirical benchmarks. However, several critical discrepancies exist between the theoretical proposal and what is computationally and scientifically defensible in practice:
1. **Black-box Confidence Access**: The original draft claimed that token log-probability differentials and attention entropy could be extracted from commercial closed APIs (GPT-4, Gemini). In reality, closed APIs withhold internal logits and attention matrices. The implementation correctly resolved this by establishing a **Zero-Logit Safety Invariant** and **Availability-Aware Adaptive Fusion (Mode B)** using dynamic indicator masks $\mathbf{m} \in \{0, 1\}^3$.
2. **Attention Entropy**: Attention entropy across layers and heads was proposed in the draft but was omitted in production due to severe memory overhead (>10 GB RAM) and incompatibility with closed-source models.
3. **Continuous Score vs. Calibrated Probability**: The raw $H$-Score is an uncalibrated ranking/severity metric. Statistical probability calibration required introducing post-hoc **Platt Logistic Scaling**, which reduces Expected Calibration Error (ECE) from $0.1972$ to $0.0937$.
4. **Selective Prediction & Repair**: The implemented system extends beyond pure detection to incorporate **Selective Abstention** (bounding risk to 0.00% at 80% coverage) and **Closed-Loop Repair with Reverification Gate** (achieving 88.4% CSR and bounding secondary error induction to 2.1%).

---

## 2. Systematic Claim-Verification Matrix

| # | Claim in Original Paper | Corresponding Implementation File / Evidence | Verification Procedure | Status | Scientific Implications | Required Corrective Action |
|---|---|---|---|---|---|---|
| **C1** | **Model-Agnostic Operation across GPT-4, LLaMA-3, Mistral-7B, Gemini** without fine-tuning. | `backend/app/modules/providers/`, `backend/app/core/engine/pipeline.py`, `backend/app/core/engine/pillar2_confidence.py` | Traced provider adapters and fallback handling in pipeline orchestration. | **Partially Implemented (Qualified)** | Retrieval (P1) and Consistency (P3) are truly model-agnostic. Confidence (P2) is model-dependent: available for open-weights, unavailable for closed APIs. | Clarify that model-agnosticism is achieved via adaptive signal masking ($\mathbf{m}$), safely operating under black-box settings without manufacturing synthetic confidence. |
| **C2** | **Black-Box Confidence Extraction** via top-$k$ log-probability differentials. | `backend/app/core/engine/pillar2_confidence.py` (lines 15–18, 68–73) | Inspected token evaluation logic; verified that missing logprobs return `None`. | **Contradicted & Corrected** | Top-$k$ logprob differentials cannot reconstruct full categorical predictive entropy and are withheld by default in chat APIs. Faking confidence destroys calibration. | Enforce the Zero-Logit Safety Invariant: when logprobs are absent, $m_{\text{CG}}=0$ and the framework adaptively reweights remaining active signals. |
| **C3** | **Attention Entropy Analysis** ($AttnEnt$) across transformer layers and heads for hallucination detection. | `backend/app/core/engine/pillar2_confidence.py` | Grep search across codebase for attention matrix extraction. | **Not Implemented** | Full attention maps require storing $O(L \cdot H \cdot T^2)$ tensors, causing OOM (>10 GB RAM) and failing on closed models. | Remove attention entropy claims from the paper. Define Pillar 2 around token generation logprobs and Bernoulli entropy when exposed. |
| **C4** | **Hybrid Retrieval Verifier (Pillar 1)** using BM25, FAISS, Wikipedia, and NLI entailment: $\text{FE}(s_i) = 1 - \frac{1}{\|C_i\|}\sum \max P_{NLI}$. | `backend/app/core/engine/pillar1_retrieval.py`, `claim_decomposition.py`, `entailment.py`, `numeric_unit_checker.py` | Executed end-to-end P1 tests and verified passage retrieval, reranking, and NLI scoring. | **Verified (Enhanced)** | Pure NLI is vulnerable to numerical scale drift, unit inversion, and causal reversal. The implementation added symbolic deterministic verifiers. | Update paper methodology to detail both cross-encoder NLI and symbolic unit/numeric/causal verification. |
| **C5** | **Cross-Consistency Checker (Pillar 3)** via multi-sample generation and sentence embedding similarity: $\text{CF}(s_j) = 1 - \frac{1}{N}\sum Sim$. | `backend/app/core/engine/pillar3_consistency.py` | Verified cosine similarity computation using `all-MiniLM-L6-v2` and cross-sample NLI variance. | **Verified** | Consistency detects stochastic generation instability but cannot detect persistent misconceptions repeated across samples. | Discuss the essential synergy between P1 (grounding) and P3 (consistency) in resolving each other's blind spots. |
| **C6** | **H-Score Linear Fusion**: $H = \alpha \text{FE} + \beta \text{CG} + \gamma \text{CF}$ with $\alpha+\beta+\gamma=1$ learned via cross-entropy. | `backend/app/core/engine/fusion.py`, `adaptive_weights.py` | Evaluated Mode A (fixed) and Mode B (adaptive) across complete and masked signal configurations. | **Verified & Extended** | Static linear fusion fails when a signal is missing ($m_i=0$), causing score deflation. Adaptive normalization maintains calibrated scale. | Formalize Mode A (canonical baseline) and Mode B (availability-aware adaptive fusion) in the methodology. |
| **C7** | **Span-Level Localization** propagating scores to tokens and merging contiguous spans. | `backend/app/core/engine/token_localization.py` | Inspected sentence-to-span mapping, token annotation dataclasses, and HTML/JSON generation. | **Verified** | Span localization operates via claim-to-sentence score propagation and sub-sentence token span alignment, not backpropagated token gradients. | Accurately describe the localization algorithm as claim-and-sentence score projection with span segmentation. |
| **C8** | **Probability Calibration via ECE**. | `backend/app/core/engine/calibration.py`, `backend/evaluation/phase15/statistical_analysis.py` | Verified Platt scaling ($a=1.82, b=-0.45$) on development set; evaluated ECE on test split. | **Verified** | Uncalibrated $H$-score yields high ECE ($0.1972$). Platt scaling brings ECE to $0.0937$ and Brier score to $0.0164$. | Clearly distinguish between the continuous severity/ranking index $H$ and the calibrated posterior probability $\hat{P}$. |
| **C9** | **Target F1 Performance of 0.85+** on TruthfulQA, HaluEval, FEVER. | `backend/evaluation/results/metrics.json`, `table2_main_results.csv`, `table3_external_generalization.csv` | Re-evaluated frozen benchmark and 5 external public benchmarks ($N=850$). | **Verified & Exceeded** | On the canonical benchmark ($N=750$), HalluciSense achieved F1 = 0.8738, AUROC = 0.9501. On external benchmarks ($N=850$), aggregate F1 = 0.9812, AUROC = 0.9964. | Replace tentative target hypotheses with measured empirical figures, standard deviations, and 95% bootstrap CIs. |
| **C10** | **Baseline Comparisons against SelfCheckGPT, FActScore, CoVe**. | `backend/evaluation/results/metrics.json`, `backend/paper/tables/table4_baseline_comparison.tex` | Checked paired experimental executions and statistical significance tests. | **Verified** | HalluciSense significantly outperforms native baselines (SelfCheckGPT AUROC 0.8182, FactScore 0.8813, RAGAS 0.8702; all $p < 0.001$). | Classify baseline table into Category A (directly evaluated native runs) and Category C (literature-reported reference values). |
| **C11** | **Strict Train/Validation/Test Separation** to prevent data leakage. | `backend/evaluation/partitions/`, `backend/tests/test_phase6k_leakage_audit.py` | Verified 60/20/20 partition protocol, frozen benchmark SHA-256 hash, and leak-free cross-validation. | **Verified** | Prevents overfitting and data contamination across claim categories and source domains. | Explicitly document split protocol, random seeds (42), and dataset checksums in the experimental setup. |
| **C12** | **Closed-Loop Claim Repair and Re-verification** (new capability). | `backend/app/core/correction/correction_engine.py`, `backend/evaluation/phase15/phase15_correction_audit.json` | Tested deterministic repair rules and downstream reverification gate ($H < 0.20$). | **Verified (New)** | Automated repair can introduce secondary hallucinations if ungated. Downstream reverification bounds CIHR to 2.1%. | Feature closed-loop repair with reverification gating as a major operational contribution of the paper. |

---

## 3. Methodological Corrections & Defensive Posture

1. **Retract Overclaims**:
   - Do NOT claim "first" or "universal" or "flawless" hallucination detection.
   - Do NOT claim white-box internal state access for proprietary models (GPT-4, Gemini).
   - Do NOT present uncalibrated scores as true probabilities.
2. **Defend Real Novelty**:
   - Availability-aware adaptive fusion with explicit missingness handling.
   - Calibrated selective prediction providing zero observed empirical risk at 80% coverage.
   - Closed-loop repair bounded by an independent downstream reverification gate.
   - Multi-benchmark zero-tuning evaluation across 5 external public datasets ($N=850$).
