# Phase 14 — Initial System Audit & Architectural Baseline

## 1. System Architecture & Component Mapping
HalluciSense implements an open-domain hybrid hallucination verification architecture that detects, quantifies, localizes, explains, corrects, and reverifies factual errors in Large Language Model responses.

```
                    LLM RESPONSE
                          │
                          ▼
                   CLAIM EXTRACTION
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
       PILLAR 1        PILLAR 2        PILLAR 3
       Grounding      Confidence      Consistency
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
              AVAILABILITY-AWARE
                ADAPTIVE FUSION
                          │
                          ▼
                      H-SCORE
                          │
                          ▼
                    CALIBRATION
                          │
                          ▼
              SELECTIVE ABSTENTION
                          │
                          ▼
                 CLAIM LOCALIZATION
                          │
                          ▼
                 CORRECTION ENGINE
                          │
                          ▼
             INDEPENDENT RE-VERIFICATION
                          │
                          ▼
                    FINAL RESULT
```

---

## 2. Core Mathematical Equations

### A. Canonical Tri-Pillar Formulation (Mode A)
When all three pillars are available:
$$H_{\text{canonical}} = \alpha \cdot \text{FE} + \beta \cdot \text{CG} + \gamma \cdot \text{CF}$$
Subject to: $\alpha + \beta + \gamma = 1.0$, with learned baseline weights:
- $\alpha = 0.40$ (External Evidence Grounding)
- $\beta = 0.30$ (Predictive Token Uncertainty)
- $\gamma = 0.30$ (Semantic Consistency)

### B. Availability-Aware Adaptive Fusion (Mode B)
Under real-world signal missingness (e.g. black-box APIs lacking token logprobs or single-turn prompts lacking alternate samples):
$$H_{\text{adaptive}} = \frac{\sum_{i=1}^3 m_i \cdot r_i \cdot w_i \cdot S_i}{\sum_{i=1}^3 m_i \cdot r_i \cdot w_i}$$
where:
- $\mathbf{S} = [\text{FE}, \text{CG}, \text{CF}]^T \in [0, 1]^3$
- $\mathbf{m} = [m_{\text{FE}}, m_{\text{CG}}, m_{\text{CF}}]^T \in \{0, 1\}^3$ is the indicator vector of genuinely available verification signals.
- $\mathbf{r} = [r_{\text{FE}}, r_{\text{CG}}, r_{\text{CF}}]^T \in (0, 1]^3$ is the empirical signal reliability vector.
- $\mathbf{w} = [\alpha, \beta, \gamma]^T$ are baseline importance coefficients.
- **Invariant:** When $m_i = 0$, signal $S_i$ is excluded from both numerator and denominator with zero synthetic logit manufacturing.

---

## 3. Probability Calibration & Decision Boundaries
- **Platt Logistic Scaling:** $P(Y=1 \mid H) = \frac{1}{1 + \exp(-(a \cdot \text{logit}(H) + b))}$ with $a = 1.82, b = -0.45$.
- **Decision Tiers:**
  * `VERIFIED` ($H < 0.20$)
  * `LOW_RISK` ($0.20 \le H < 0.35$)
  * `NEEDS_VERIFICATION` ($0.35 \le H < 0.50$)
  * `MODERATE_RISK` ($0.50 \le H < 0.65$)
  * `LIKELY_HALLUCINATED` ($H \ge 0.65$)
- **Selective Abstention:**
  * `INSUFFICIENT_EVIDENCE`: Triggers when $S_{\text{evidence}} < 0.40$ and epistemic uncertainty $> 0.85$.
  * `ABSTAIN`: Triggers when $|H - 0.40| < 0.08$ with high epistemic uncertainty ($> 0.75$).

---

## 4. Production Models & Memory Invariants
- **NLI Model:** `cross-encoder/nli-deberta-v3-small` (Singleton in [`backend/app/core/engine/model_registry.py`](file:///Users/akashgpatil/major_project/backend/app/core/engine/model_registry.py)).
- **Sentence Embeddings:** `all-MiniLM-L6-v2`.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (Lazy singleton).
- **Execution Mode:** PyTorch FP32 with `torch.inference_mode()`, single-worker bounded memory ($\sim 1.1\text{ GB}$ peak).
- **Canonical Benchmark Hash:** `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`.
