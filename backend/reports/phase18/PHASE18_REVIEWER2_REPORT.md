# Phase 18 — Reviewer #2 Adversarial Novelty Attack

**Reviewer Identity:** Senior Novelty & Prior-Art Reviewer  
**Focus:** Hostile Critique of Novelty Claims against Prior Literature  
**Recommendation:** **MINOR REVISION (Novelty Scoped & Defensible)**

---

## 1. Hostile Novelty Challenge: "Is Adaptive Fusion Genuinely Novel?"
*Reviewer's Opening Critique:*
> "The authors claim novelty for 'Availability-Aware Adaptive Fusion' (Eq. 2). However, dynamic re-weighting under missing features is a well-studied paradigm in multimodal machine learning (missing-modality imputation), late-fusion ensemble weighting, and mixture-of-experts (MoE). Why should re-scaling weights by $\frac{m_i r_i w_i}{\sum m_i r_i w_i}$ be considered a fundamental scientific contribution rather than standard missing-data renormalization?"

---

## 2. Granular Prior-Art Comparison Table

| Prior Paradigm / System | Canonical Citation | Core Formulation | Overlap with HalluciSense | Fundamental Difference | Threat Level to Novelty | Residual HalluciSense Contribution |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Missing-Modality Late Fusion** | Baltrušaitis et al. (IEEE TPAMI 2018, DOI: 10.1109/TPAMI.2017.2699988) | Multi-sensor late fusion with missing channel indicators | Dynamic indicator gating $\mathbf{m}$ | Operates on continuous perceptual signals; HalluciSense targets discrete LLM fact checking and non-manufactured logits | **MEDIUM** | Adapting missingness formalization to black-box LLM API constraints ($m_{\text{CG}}=0$). |
| **Mixture of Experts (MoE)** | Shazeer et al. (ICLR 2017, arXiv:1701.06538) | Parameterized gating network $\text{Softmax}(H \cdot W_g)$ | Input-dependent weighting | MoE requires end-to-end differentiable backprop; HalluciSense executes zero-training epistemic reliability weighting | **LOW** | Training-free empirical reliability modulation for factuality scoring. |
| **Ensemble Reliability Weighting** | Lakshminarayanan et al. (NeurIPS 2017, NeurIPS:9ef2ed4b) | Variance-weighted deep ensembles | Weighting by predictive variance | Treats homogeneous model checkpoints; HalluciSense unifies heterogeneous modalities (dense retrieval + token logprobs + NLI) | **MEDIUM** | Heterogeneous modality hybridization across symbolic, token, and semantic channels. |
| **Selective Classification** | Geifman & El-Yaniv (NeurIPS 2017, NeurIPS:4a27cea7) | Risk-coverage curves and threshold selection | Selective abstention mechanism | General classification formulation; HalluciSense couples selective rejection with closed-loop claim repair | **LOW** | Direct integration of dual-criteria abstention with downstream symbolic claim repair. |
| **Post-Hoc Fact Checking** | Min et al. (EMNLP 2023, DOI: 10.18653/v1/2023.emnlp-main.741) | Static atomic claim precision scoring | Atomic claim decomposition | FActScore assumes static retrieval; fails completely when retrieval or logprobs are missing | **LOW** | HalluciSense provides dynamic multi-signal recovery when single signals fail. |

---

## 3. Reviewer #2 Evaluation Verdict by Claim
- **Contribution N1 (Availability-Aware Adaptive Fusion):** `NOVEL IN LLM VERIFICATION CONTEXT`. While missing-modality fusion exists in multimodal computer vision, its mathematical adaptation to black-box LLM API constraints with zero-logit safety is novel and highly practical.
- **Contribution N2 (Reliability Modulation):** `PARTIALLY NOVEL`. Reliability weighting is known in statistical sensor fusion, but its specific grounding-entropy-semantic formulation is original.
- **Contribution N3 (Zero-Logit Safety Contract):** `SYSTEM INVARIANT (NOT AN ALGORITHMIC NOVELTY)`. Properly classified as a system design principle.
- **Contribution N4 (Selective Abstention):** `APPLICATION NOVELTY`. Adapts Geifman & El-Yaniv selective prediction to LLM factuality risk scores.
- **Contribution N5 (Reverification-Gated Repair):** `MODERATE NOVELTY`. The closed-loop repair with independent reverification thresholding is an effective safety gate.
- **Contribution N6 (Unified Integration):** `STRONG INTEGRATION CONTRIBUTION`.

---

## 4. Conclusion
The authors' conservative framing in [`PHASE16_NOVELTY_POSITIONING.md`](file:///Users/akashgpatil/major_project/backend/reports/phase16/PHASE16_NOVELTY_POSITIONING.md) and [`novelty_matrix.csv`](file:///Users/akashgpatil/major_project/backend/paper/literature/novelty_matrix.csv) correctly refrains from claiming "first-ever" discovery, accurately attributing prior art while positioning HalluciSense as an availability-aware, reliability-weighted verification framework. Novelty is scientifically defensible.
