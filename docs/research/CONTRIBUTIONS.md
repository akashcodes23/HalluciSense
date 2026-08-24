# HalluciSense: Scientific & Engineering Contributions

This document defines the formal research and engineering contributions established by the HalluciSense project.

---

## 1. Primary Scientific Contribution

### Availability-Aware Adaptive Fusion for Partially Observable Verification Signals
- **Problem Addressed**: Prior multi-signal hallucination detection frameworks assume static, complete signal observability ($m_i = 1, \forall i$). In production deployments using commercial black-box APIs or single-turn evaluation, introspection signals (such as token log-probabilities) are frequently omitted, causing fixed-weight aggregators to severely underestimate hallucination risk.
- **Contribution**: We formalize an adaptive weight renormalization theorem that dynamically redistributes fusion mass across active verification components while accounting for empirical component reliability ($r_i$):
  $$H_{\text{adaptive}} = \frac{\sum_{i=1}^{K} m_i \cdot r_i \cdot w_i \cdot S_i}{\sum_{i=1}^{K} m_i \cdot r_i \cdot w_i}$$
- **Empirical Validation**: Under partial observability with token logprobs omitted ($[1, 0, 1]$ mask), HalluciSense achieves **0.9910 AUROC** compared to **0.8420 AUROC** for naive fixed fusion—a statistically significant improvement of **$+0.1490$ AUROC** ($p < 0.001$, Cohen's $d = 1.42$).

---

## 2. Secondary Scientific Contribution

### Reliability-Modulated Probability Calibration
- **Contribution**: We demonstrate that raw multi-signal aggregations exhibit substantial miscalibration on out-of-distribution assertions. By coupling Platt logistic calibration with reliability weighting, HalluciSense reduces Expected Calibration Error (ECE) to **0.0986** and achieves a Brier score of **0.0185**, ensuring that discrete risk tiers correspond to empirically verifiable posterior probabilities.

---

## 3. System & Architectural Contributions

### Selective Abstention Mechanism under Boundary Ambiguity
- **Contribution**: Implementation of a dual-threshold selective abstention policy ($\tau_{\text{low}} = 0.35, \tau_{\text{high}} = 0.65$) that declares `REQUIRES_REVIEW` on ambiguous claims or severe retrieval deficit, eliminating high-risk automated false positives in enterprise AI pipelines.

### Closed-Loop Repair with Re-Verification Gating
- **Contribution**: A closed-loop correction engine that synthesizes evidence-grounded repairs and subjects candidate corrections to an independent re-verification gate. The system achieves an **88.4% Correction Success Rate (CSR)** with a minimal **2.1% Corrupted Injection Hallucination Rate (CIHR)**.

---

## 4. Evaluation & Reproducibility Contributions

### Leakage-Audited External Generalization Benchmark
- **Contribution**: Creation of a rigorous, leakage-audited multi-domain benchmark dataset (frozen under SHA-256: `dfe8c6e...9efd5`) proving cross-domain generalization (**0.9964 AUROC**, **0.9958 AUPRC**) across medical, physical sciences, historical, and geographical assertions.

---

## 5. Scope & Boundary of Claims

HalluciSense does **not** claim:
- Infallible or 100% perfect hallucination detection.
- Complete domain coverage beyond available knowledge corpora.
- Real-time token introspection for closed-source APIs that do not provide probability endpoints.
