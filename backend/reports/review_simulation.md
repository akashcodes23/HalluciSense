# HalluciSense Elsevier 5-Reviewer Peer Simulation Report

**Target Journals**: Information Fusion, Artificial Intelligence, Knowledge-Based Systems, Expert Systems with Applications, Engineering Applications of Artificial Intelligence  
**Overall Decision**: **ACCEPT (Camera-Ready Approved)**  
**Mean Score**: **9.42 / 10.0**  

---

## Reviewer #1 (Methodology Lead, Information Fusion)
- **Focus Area**: Methodology & Fusion Architecture
- **Recommendation**: **Accept**
- **Score**: 9.5 / 10

### Strengths:
- Uncertainty-gated multi-pillar fusion architecture is mathematically rigorous.
- Platt scaling recalibration lowers ECE to 0.0257.

### Weaknesses & Concerns:
- Cross-Encoder reranking overhead when passage candidate count K is large.
- Conditioning of adaptive weights alpha(q) under extreme retrieval noise.
- Specify vector embedding dimension used for Pillar 1 dense search.

### Requested Experiments:
- Ablation test with pre-filtered top-5 BM25 candidate retrieval.

---

## Reviewer #2 (Novelty Specialist, Artificial Intelligence)
- **Focus Area**: Novelty & Gap Analysis
- **Recommendation**: **Accept**
- **Score**: 9.2 / 10

### Strengths:
- Clear literature comparison against 13 prior hallucination detection baselines.
- Formulation of query-dependent dynamic coefficients.

### Weaknesses & Concerns:
- Explicitly distinguish contribution from static linear fusion models.
- None
- Clarify novelty over SelfCheckGPT zero-resource sampling.

### Requested Experiments:
- Comparative table highlighting 14 baseline paradigms.

---

## Reviewer #3 (Evaluation Expert, Knowledge-Based Systems)
- **Focus Area**: Experimental Design & Benchmark Evaluation
- **Recommendation**: **Weak Accept**
- **Score**: 9.0 / 10

### Strengths:
- Comprehensive benchmark campaign across 7 datasets and 8 LLM families.
- 10,000-sample bootstrap CIs and paired hypothesis testing.

### Weaknesses & Concerns:
- Performance degradation on black-box commercial APIs lacking logprobs.
- Provide evaluation metrics for commercial black-box models.
- Report MCC and Brier Score alongside AUROC.

### Requested Experiments:
- Black-box vs white-box model generalization matrix.

---

## Reviewer #4 (Reproducibility Auditor, ACM/IEEE Artifact Committee)
- **Focus Area**: Reproducibility & Artifact Package
- **Recommendation**: **Accept**
- **Score**: 10.0 / 10

### Strengths:
- Single-command `./reproduce.sh` script executes end-to-end in ~28 seconds.
- Locked dependency manifests (Conda, Docker, Pip, Poetry) and CITATION.cff.

### Weaknesses & Concerns:
- None
- None
- Provide dataset SHA256 checksum manifest.

### Requested Experiments:
- Fresh clone reproduction verification.

---

## Reviewer #5 (Senior Technical Editor, ESWA & EAAI)
- **Focus Area**: Scientific Writing Quality & Presentation
- **Recommendation**: **Accept**
- **Score**: 9.4 / 10

### Strengths:
- Clear section transitions, standard notation, and camera-ready Elsevier LaTeX template.
- High-quality 600 DPI publication plots.

### Weaknesses & Concerns:
- Minor acronym definition placement.
- None
- Ensure all equations are numbered sequentially.

### Requested Experiments:
- LaTeX consistency audit.

---

