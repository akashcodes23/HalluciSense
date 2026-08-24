# HalluciSense: Technical Presentation Slide Deck

**Title**: HalluciSense — An Availability-Aware Multi-Signal Framework for Detecting, Calibrating, Abstaining on, and Correcting LLM Hallucinations  
**Presenter**: Final Year Project / Research Group Presentation  
**Target Duration**: 15–20 minutes (15 Slides)

---

### Slide 1: Title & Overview
- **Header**: HalluciSense
- **Subtitle**: Confidence-Aware Multi-Pillar Verification & Closed-Loop Repair Framework for LLMs
- **Core Message**: Bridging empirical ML calibration with real-time enterprise AI safety.
- **Presenter Information**: Research Lead & Engineering Team

---

### Slide 2: Problem Statement & Motivation
- **The Core Issue**: LLMs generate fluent, authoritative, yet factually erroneous statements (hallucinations).
- **The Stakes**: Misinformation in biomedical diagnostics, financial forecasting, and legal analysis carries extreme risk.
- **The Diagnostic Challenge**: Hallucinations often mimic correct statistical patterns, evading simple token heuristics.

---

### Slide 3: Limitations of Prior Art
- **Pure Retrieval (RAG)**: Prone to retrieval noise, context window saturation, and corpus omission gaps.
- **White-Box Uncertainty**: Relies on token log-probabilities which are unavailable in commercial APIs (GPT-4, Claude).
- **Self-Consistency**: High computational cost ($5\times$ sampling latency) and vulnerable to shared model biases.
- **Static Heuristics**: Naive weighted sum treat missing verification signals as zero risk, skewing output probabilities.

---

### Slide 4: The HalluciSense Vision & Solution
- **Multi-Signal Decomposition**: Decoupling detection into three orthogonal signals (Grounding, Confidence, Consistency).
- **Availability-Awareness**: Dynamic weight renormalization based on measured signal presence ($m_i$) and reliability ($r_i$).
- **Statistical Calibration**: Platt scaling transformation converting raw scores into true posterior probabilities.
- **Closed-Loop Action**: Evidence-grounded symbolic/neural repair validated by an independent re-verification gate.

---

### Slide 5: System Architecture Overview
- **Visual Diagram**: Complete pipeline dataflow from query input to auditable output.
- **Key Modules**:
  1. Atomic Proposition Claim Parser
  2. Three Independent Verification Pillars
  3. Availability-Aware Adaptive Fusion Engine
  4. Calibration & Selective Abstention Gate
  5. Closed-Loop Correction & Re-Verification

---

### Slide 6: Pillar 1 — Evidence Grounding (FE)
- **Objective**: Quantify factual divergence against authoritative knowledge corpora.
- **Pipeline**:
  - BM25 sparse keyword index + FAISS dense embedding search (`all-MiniLM-L6-v2`).
  - Cross-Encoder Natural Language Inference (`cross-encoder/nli-deberta-v3-small`).
- **Mathematical Form**: $S_1 = P(\text{Contradiction}) + \frac{1}{2} P(\text{Neutral}) \in [0, 1]$.
- **Base Weight / Reliability**: $w_1 = 0.45, r_1 = 0.95$.

---

### Slide 7: Pillar 2 — Confidence Estimation (CG)
- **Objective**: Capture internal token-level epistemic uncertainty.
- **Mechanisms**:
  - Shannon Entropy: $H(p) = -\sum_{i} p_i \log p_i$.
  - Sequence-level minimum token probability and confidence gap quantification.
- **Availability Gating**: $m_2 = 1$ if white-box logprobs accessible; $m_2 = 0$ for black-box APIs.
- **Base Weight / Reliability**: $w_2 = 0.30, r_2 = 0.85$.

---

### Slide 8: Pillar 3 — Consistency Reasoning (CF)
- **Objective**: Detect stochastic variance across temperature-perturbed candidate generations.
- **Mechanisms**:
  - Generation of $k \in [3, 5]$ stochastic candidates.
  - Pairwise semantic embedding cosine dissimilarity via Sentence-BERT.
- **Availability Gating**: $m_3 = 1$ in multi-sample generation mode; $m_3 = 0$ in single-response inspection.
- **Base Weight / Reliability**: $w_3 = 0.25, r_3 = 0.80$.

---

### Slide 9: Availability-Aware Adaptive Fusion
- **Full Mode A**: $H = \alpha \cdot \text{FE} + \beta \cdot \text{CG} + \gamma \cdot \text{CF} \quad (\alpha=0.45, \beta=0.30, \gamma=0.25)$
- **Adaptive Mode B**:
  $$H_{\text{adaptive}} = \frac{\sum_{i=1}^{3} m_i \cdot r_i \cdot w_i \cdot S_i}{\sum_{i=1}^{3} m_i \cdot r_i \cdot w_i}$$
- **Zero-Signal Invariant**: If $\sum m_i = 0$, explicitly returns `status="FAILED"`, `h_score=None` (never assumes zero risk).

---

### Slide 10: Calibration & Selective Abstention
- **Platt Scaling Calibration**:
  $$P(\text{Hallucination}=1 \mid S) = \frac{1}{1 + \exp(-(a \cdot S + b))}$$
- **Expected Calibration Error (ECE)**: Reduced to $0.0986$ (Brier score $0.0185$).
- **Selective Abstention Policy**: Abstains when $0.35 \le H \le 0.65$ or during severe retrieval deficit, preventing high-risk automated decision-making.

---

### Slide 11: Closed-Loop Correction & Re-Verification
- **Repair Engine**: Extracts contradictory claim spans and generates evidence-grounded candidate edits.
- **Re-Verification Gate**: Subject candidate repairs to an independent downstream verification pass.
- **Key Metrics**:
  - **Correction Success Rate (CSR)**: $88.4\%$
  - **Repair Precision Rate (RPR)**: $91.2\%$
  - **Corrupted Injection Hallucination Rate (CIHR)**: $2.1\%$

---

### Slide 12: Empirical Experimental Validation
- **Held-Out External Benchmark Results**:
  - **External AUROC**: **0.9964**
  - **External AUPRC**: **0.9958**
  - **Adaptive Mask $[1, 0, 1]$ AUROC**: **0.9910** vs Fixed Mask **0.8420** ($+0.1490$ gain, Cohen's $d = 1.42$).
- **Strict Data Isolation**: Zero test-set leakage, audited via SHA-256 frozen manifest.

---

### Slide 13: Production System & Live Deployment
- **Live Infrastructure**: Railway cloud deployment (`https://hallucisense-production.up.railway.app`).
- **Engineering Optimizations**:
  - `ModelRegistry` singleton architecture (622 MB RSS footprint).
  - Wikipedia HTTP persistent connection pooling (`urllib3` pool maxsize = 10).
  - LRU NLI pair caching (100x speedup on repeated claims).
  - In-memory token-bucket rate limiting (100 req/min).

---

### Slide 14: System Limitations & Failure Modes
- **Corpus Dependence**: Verification precision bounded by retrieval index coverage.
- **Language Scope**: English-only benchmark evaluation in current release.
- **Symbolic Parser Edge Cases**: Complex nested clauses can challenge proposition splitting.
- **Latency Trade-Off**: Multi-sample mode introduces 15–30s generation latency.

---

### Slide 15: Conclusion & Future Directions
- **Summary of Contributions**:
  1. Availability-aware adaptive fusion addressing missing signals in black-box LLMs.
  2. Calibrated, abstention-gated risk estimation with auditable open-source traces.
  3. Closed-loop repair with validated re-verification gating.
- **Open Source Repository**: `https://github.com/akashcodes23/HalluciSense`
- **Thank You & Q&A**
