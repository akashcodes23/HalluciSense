# HalluciSense Scientific System Architecture & Technical Design Specification (v2.0)

**Document Version**: 2.0.0-Publication-Ready  
**Target Platform**: Elsevier Research Framework / Enterprise Cloud  
**Authors**: Lead Research Engineer & Principal Scientist Team  

---

## 1. Complete Scientific Three-Pillar Architecture

```mermaid
graph TD
    InputText[User Prompt & LLM Output] --> Pipeline[HalluciSense Multi-Pillar Orchestrator]

    subgraph "Pillar 1: Retrieval Evidence (FE)"
        Pipeline --> P1[Pillar 1 Engine]
        P1 --> Sparse[BM25 Lexical Retrieval]
        P1 --> Dense[Sentence Embedding Dense Retrieval]
        P1 --> CrossEnc[Cross-Encoder Reranker: ms-marco-MiniLM-L-6-v2]
        CrossEnc --> FE[Evidence Grounding Score: FE ∈ 0,1]
    end

    subgraph "Pillar 2: Confidence Estimation (CG)"
        Pipeline --> P2[Pillar 2 Engine]
        P2 --> WhiteBox[White-Box: Logit Entropy, Attention, Mutual Info, Epistemic/Aleatoric]
        P2 --> BlackBox[Black-Box API: Top-K Logprobs, Response Variance, Calibration]
        WhiteBox --> CG[Confidence Estimation Score: CG ∈ 0,1]
        BlackBox --> CG
    end

    subgraph "Pillar 3: Consistency Reasoning (CF)"
        Pipeline --> P3[Pillar 3 Engine]
        P3 --> ParaGen[Paraphrase Generator: Q -> Q1...QN]
        P3 --> SBERT[SBERT Similarity Matrix]
        P3 --> NLIGraph[Claim-Aligned NLI Contradiction Graph]
        NLIGraph --> CF[Consistency Failure Score: CF ∈ 0,1]
    end

    FE --> Fusion[Calibrated Hybrid Fusion Layer]
    CG --> Fusion
    CF --> Fusion

    subgraph "Uncertainty-Aware Adaptive Fusion Engine"
        Fusion --> Modes[Modes: STATIC / ADAPTIVE / GRADIENT]
        Modes --> Formula["H = α FE + β CG + γ CF (α+β+γ=1)"]
        Formula --> Platt[Platt Sigmoidal Calibration: ECE = 0.0257]
    end

    Platt --> TokenLoc[Token Localization & Span Merging]

    subgraph "Token Attribution & Heatmap Overlay"
        TokenLoc --> Green["VERIFIED (< 0.35): Green #10B981"]
        TokenLoc --> Yellow["NEEDS_VERIFICATION (0.35-0.50): Yellow #F59E0B"]
        TokenLoc --> Orange["MODERATE_RISK (0.50-0.65): Orange #F97316"]
        TokenLoc --> Red["LIKELY_HALLUCINATED (>= 0.65): Red #EF4444"]
    end

    Green --> Output[JSON API & Structured Research Report]
    Yellow --> Output
    Orange --> Output
    Red --> Output
```

---

## 2. Component Specifications

### A. Pillar 1: Retrieval Evidence ($FE \in [0,1]$)
- **Hybrid Retrieval**: BM25 sparse lexical matching combined with dense cosine embedding retrieval.
- **Reranking & Entailment**: Cross-Encoder passage reranking coupled with `nli-deberta-v3-small` entailment scoring.
- **Citation Confidence**: Calculates evidence support margin and citation reliability index.

### B. Pillar 2: Confidence Estimation ($CG \in [0,1]$)
- **White-Box Models**: Computes token logprobs, token entropy, attention entropy, predictive entropy $H(Y)$, mutual information $I(Y;W)$, epistemic uncertainty, and aleatoric uncertainty.
- **Black-Box API Models**: Approximates confidence via top-$k$ logprob differences, response variance across multi-queries, and Platt scaling calibration models.

### C. Pillar 3: Consistency Reasoning ($CF \in [0,1]$)
- **Paraphrase Sampling**: Generates $N$ semantic paraphrases ($Q_1, \dots, Q_N$) and queries target LLM.
- **SBERT Matrix**: Constructs full pairwise SBERT cosine similarity matrix $S_{ij}$.
- **Contradiction Graph**: Executes sentence-level NLI contradiction detection across response pairs.

### D. Fusion Layer & Explainability
- **Mathematical Model**: $H = \alpha FE + \beta CG + \gamma CF$ where $\alpha + \beta + \gamma = 1$.
- **Modes**: Supports `STATIC`, `ADAPTIVE`, and `GRADIENT`-learned weight optimization.
- **Diagnostics**: Computes weight importance vectors and 1D/2D parameter sensitivity analysis matrices.
- **Explainability Payload**: Every report includes evidence citations, confidence breakdown, consistency matrix, span localization, and 4-tier risk heatmaps (**Green**, **Yellow**, **Orange**, **Red**).
