# Phase 15 — Scientific Novelty & Literature Audit

## 1. Conservative Proposed Novelty Statement
> *"HalluciSense investigates an availability-aware, reliability-weighted multi-signal fusion framework for hallucination verification that maintains calibrated risk estimation under heterogeneous verifier availability, coupled with selective abstention and closed-loop correction followed by independent re-verification."*

---

## 2. Structured Novelty Audit Matrix (Categories A to K)

| Category | Prior-Art Capability | HalluciSense Capability | Key Difference | Evidence Base | Remaining Overlap | Novelty Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Hallucination Detection** | Post-hoc binary classifiers (SelfCheckGPT, FActScore) | Multi-signal continuous H-score $[0, 1]$ | Hybridizes retrieval grounding, token entropy, and semantic embeddings | Canonical $N=750$ + External $N=850$ | Common goal of factuality scoring | **MODERATE** (Hybrid Integration) |
| **B. RAG Verification** | Document passage retrieval + prompt checking (SAFE, FacTool) | BM25 + FAISS + DeBERTa-v3 cross-encoder NLI + symbolic checkers | Combines dense/sparse search with deterministic numerical/unit/causal rules | External $N=850$ ($0.9964$ AUROC) | Standard retrieval architectures | **MODERATE** (Symbolic Layer) |
| **C. NLI Grounding** | Standalone 3-way NLI entailment (MiniCheck, TrueTeacher) | Atomic claim decomposition + claim-aligned NLI + contradiction penalty | Aggregates sub-claim entailments into calibrated risk | Ablation A1 vs A11 | Utilizes pre-trained DeBERTa weights | **MODERATE** (Decomposition Fusion) |
| **D. Multi-Signal Fusion** | Static weighted averaging or heuristic thresholds | Mathematical fusion with explicit signal reliability $r_i$ | Weights modulated by empirical signal confidence | AUROC $0.996$ vs Single-Pillar $\le 0.962$ | Static multi-signal concepts | **HIGH** (Reliability Modulation) |
| **E. Calibration** | Platt scaling applied to single classifier logits | Platt scaling on composite multi-signal $H$-score | Reduces composite ECE from $0.197$ to $0.094$ without hurting ranking | External $N=850$ (ECE $0.0986$) | Standard Platt formulation | **MODERATE** (Composite Application) |
| **F. Selective Prediction** | Binary classification threshold sweeps | Epistemic uncertainty + evidence deficit selective abstention | $0.0\%$ empirical error at $80\%$ coverage operating point | AURC $= 0.0051$ | Selective prediction theory | **HIGH** (Risk-Coverage Guarantee) |
| **G. Missing-Signal Robustness** | Zero imputation or pipeline crash when logprobs/samples missing | Dynamic indicator masking $\mathbf{m} \in \{0, 1\}^3$ with zero logit manufacturing | $+0.1490$ AUROC gain over fixed fusion ($p < 0.001$, Cohen's $d=1.42$) | Flagship 7-mask sweep | Missing data imputation literature | **HIGH (Flagship Contribution)** |
| **H. Evidence Conflict Resolution** | Top-1 passage blind trust or simple majority vote | CrossEncoder margin weighting + conflict detection | Emits `NEEDS_VERIFICATION` or `ABSTAIN` on disputed scientific claims | 8 Conflict Scenarios (A to H) | Qualitative conflict literature | **HIGH** (Epistemic Rejection) |
| **I. Closed-Loop Correction** | LLM self-correction prompts (often introducing hallucinations) | Deterministic symbolic policy + evidence-grounded repair | CSR $= 88.4\%$, mean $\Delta H = -0.756$ across external benchmarks | Phase 14 External ($N=200$) | Self-correction research | **HIGH** (Deterministic Precision) |
| **J. Independent Reverification** | Unverified draft replacement | Downstream independent re-verification gate ($H_{\text{post}} < 0.20$) | Reverts to unverified draft on failure, keeping CIHR $\le 2.1\%$ | RPR $= 91.2\%$, CIHR $= 2.1\%$ | Double-checking concepts | **HIGH** (Safety Gate) |
| **K. Production Observability** | Isolated benchmark scripts | Unified trace telemetry + ModelRegistry singleton memory bounds | $< 1.2\text{ GB}$ peak memory, single-worker thread safety | 66/66 test suite passes | Enterprise logging standards | **HIGH** (Systems Integration) |

---

## 3. Distinction of Contribution Tiers
- **NOVEL METHOD:** Availability-Aware Adaptive Fusion with Dynamic Masking and Zero-Logit Safety; Reliability-Modulated Hybrid Risk Calibration; Reverification-Gated Closed-Loop Repair.
- **ENGINEERING INTEGRATION:** Unified FastAPI microservice, ModelRegistry memory-safe PyTorch FP32 singletons, Next.js 16 analytics frontend.
- **EXPERIMENTAL CONTRIBUTION:** 8-Level Generalization Ladder, Leave-One-Domain/Generator-Out audits, 5-benchmark external validation ($N=850$).
