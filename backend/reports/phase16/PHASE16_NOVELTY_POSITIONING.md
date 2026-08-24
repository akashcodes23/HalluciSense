# Phase 16 — Scientific Novelty & Literature Positioning

## 1. Conservative Proposed Central Novelty Statement
> *"HalluciSense investigates an availability-aware, reliability-weighted multi-signal fusion framework for hallucination verification that maintains calibrated risk estimation under heterogeneous verifier availability, coupled with selective abstention and closed-loop correction followed by independent re-verification."*

---

## 2. Granular Novelty Breakdown (N1 to N6)

### Contribution N1: Availability-Aware Multi-Signal Adaptive Fusion with Dynamic Signal Masks
- **Prior Art:** Existing systems either assume static weight vectors (e.g. fixed linear combinations) or fail completely when internal token logprobs/multiple stochastic samples are unavailable.
- **HalluciSense Novelty:** Formulates dynamic indicator masking $\mathbf{m} \in \{0, 1\}^3$ that renormalizes verification weights dynamically without synthetic logit substitution.
- **Classification:** **`STRONG NOVELTY`**
- **Safe Wording:** *"An availability-aware adaptive fusion mechanism that dynamically renormalizes verification weights without synthetic logit substitution."*

### Contribution N2: Reliability-Modulated Fusion of Heterogeneous Verification Signals
- **Prior Art:** Uniform weighting or heuristic rule sets across heterogeneous signal sources.
- **HalluciSense Novelty:** Modulates active signals by empirical confidence reliability metrics $r_i$ (e.g. retrieval passage margin, token calibration stability, and semantic variance).
- **Classification:** **`STRONG NOVELTY`**
- **Safe Wording:** *"Reliability-modulated weighting combining retrieval density, token entropy stability, and cross-sample agreement."*

### Contribution N3: Zero-Logit-Safe Behavior for Black-Box LLM Verification
- **Prior Art:** Systems either manufacture dummy logprobs or throw runtime errors when providers omit logprobs.
- **HalluciSense Novelty:** Enforces a strict non-manufacturing safety contract ensuring $m_{\text{CG}} = 0$ leaves confidence unasserted.
- **Classification:** **`SYSTEM-LEVEL NOVELTY`**
- **Safe Wording:** *"A strict non-manufacturing safety contract ensuring missing provider logprobs remain unavailable."*

### Contribution N4: Selective Abstention Integrated Directly into Verification
- **Prior Art:** Isolated post-hoc binary rejection curves.
- **HalluciSense Novelty:** Dual-criteria rejection gate triggering on severe retrieval deficit ($S_{\text{evidence}} < 0.40$) or boundary epistemic ambiguity ($|H - 0.40| < 0.08$).
- **Classification:** **`MODERATE NOVELTY`**
- **Safe Wording:** *"A dual-criteria rejection gate triggering on retrieval deficit or boundary epistemic ambiguity."*

### Contribution N5: Closed-Loop Correction Followed by Independent Re-Verification
- **Prior Art:** Unverified LLM prompt-based rewrites that risk introducing secondary hallucinations.
- **HalluciSense Novelty:** Deterministic symbolic repair coupled with an independent downstream re-verification gate ($H_{\text{post}} < 0.20$), achieving $\text{CIHR} \le 2.1\%$.
- **Classification:** **`MODERATE NOVELTY`**
- **Safe Wording:** *"An independent downstream reverification gate that rejects candidate corrections if post-repair H-score exceeds 0.20."*

### Contribution N6: Unified Architecture Integrating Detection, Calibration, Abstention, Correction, and Memory Safety
- **Prior Art:** Disjoint scripts for detection and repair.
- **HalluciSense Novelty:** Unified end-to-end framework with ModelRegistry singleton memory bounds ($\le 1.2\text{ GB}$ peak) and single-worker thread safety.
- **Classification:** **`INTEGRATION CONTRIBUTION`**
- **Safe Wording:** *"A unified open-domain verification architecture integrating multi-pillar signals, calibrated risk estimation, selective abstention, and closed-loop repair."*
