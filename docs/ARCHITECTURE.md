# HalluciSense Architecture Specification

## 1. System Overview
**HalluciSense** is a model-agnostic hybrid framework for detecting, quantifying, localizing, explaining, correcting, and re-verifying hallucinations in Large Language Models.

```
                            USER QUERY Q + LLM RESPONSE R
                                          │
                                          ▼
                            CLAIM / SENTENCE DECOMPOSITION
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
          PILLAR 1                    PILLAR 2                    PILLAR 3
      Evidence Grounding         Predictive Confidence       Semantic Consistency
      (Hybrid: Dense + BM25)     (White-Box Logprobs /       (Multi-Sample Paraphrase
      (NLI: DeBERTa-v3)          Black-Box Entropy)          + Claim-Aligned NLI)
              │                           │                           │
              ▼ (FE)                      ▼ (CG)                      ▼ (CF)
      [m₁ ∈ {0,1}, r₁]            [m₂ ∈ {0,1}, r₂]            [m₃ ∈ {0,1}, r₃]
              └───────────────────────────┬───────────────────────────┘
                                          │
                                          ▼
                        AVAILABILITY-AWARE ADAPTIVE FUSION
                       H_adaptive = Σ(mᵢ·wᵢ·Sᵢ) / Σ(mᵢ·wᵢ)
                       H_canonical = α·FE + β·CG + γ·CF
                                          │
                                          ▼
                             CALIBRATION & ABSTENTION
                         (Platt Scaling / Isotonic / ECE)
                     Decision: VERIFIED | NEEDS_REVIEW | ...
                                          │
                                          ▼
                             SPAN & TOKEN LOCALIZATION
                                          │
                                          ▼
                          CLOSED-LOOP REPAIR & REVERIFICATION
```

## 2. The Three Scientific Pillars

### Pillar 1: Evidence Grounding ($\text{FE}$)
- **Objective:** Evaluate factual consistency against external reference corpora.
- **Components:**
  - Claim Extraction: Atomic factual proposition decomposition.
  - Multi-Hop Retrieval: Batched Wikipedia API + internal FAISS vector store + BM25 keyword matching.
  - Cross-Encoder Re-ranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - Natural Language Inference (NLI): `cross-encoder/nli-deberta-v3-small`.
  - Symbolic Checks: Numeric unit scaling, negation polarity detection, and causal directionality verification.

### Pillar 2: Predictive Confidence ($\text{CG}$)
- **Objective:** Quantify model intrinsic uncertainty when token-level log probabilities are accessible.
- **Formulation:** Binary entropy $H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$ and Confidence Gap $\text{CG} = 0.7 \cdot (1 - \bar{p}) + 0.3 \cdot (\text{uncertain\_fraction})$.
- **Missing Data Invariant:** Black-box API models without logprobs are flagged as `available=False` with zero synthetic logit manufacturing.

### Pillar 3: Semantic Consistency ($\text{CF}$)
- **Objective:** Evaluate self-consistency across stochastic generations.
- **Formulation:** Sentence embeddings (`all-MiniLM-L6-v2`) cosine similarities combined with claim-aligned NLI cross-comparison to detect explicit mutual contradictions.
