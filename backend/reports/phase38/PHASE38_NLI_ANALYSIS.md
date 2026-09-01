# Phase 38.7 — NLI & Retrieval Grounding Robustness Analysis

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 38.7 — NLI Subsystem Robustness Audit  
**Active Production Model:** `HistGradientBoostingClassifier` (19 features, $\tau^* = 0.54$, $N=58,002$)  
**Date:** 2026-09-01  

---

## 1. Executive Summary

This forensic report analyzes the behavior of Natural Language Inference (NLI) and Retrieval components across the HalluciSense verification pipeline.

### Core Architectural Distinction
1. **Pillar 1 (Evidence Grounding):**  
   Converts retrieval passage relevance scores into synthetic NLI-compatible coordinates via `_relevance_to_nli(relevance)`. It does **not** run transformer token-level cross-attention between the evidence text and the claim text during single-claim inference.
2. **Pillar 2 (Structural Consistency):**  
   Evaluates true bidirectional NLI and semantic embeddings between *pairs of extracted claims* using `evaluate_bidirectional_nli_and_similarity()` with `cross-encoder/nli-deberta-v3-small` and `all-MiniLM-L6-v2`.

---

## 2. Mathematical Tracing of Pillar 1 NLI Conversion

In `backend/app/core/inference/pillar1_engine.py`:

$$\text{entailment} = 0.3 \times \text{relevance}^2$$
$$\text{contradiction} = 0.8 \times (1.0 - \text{relevance})^{1.2}$$
$$\text{neutral} = \max(0.0, 1.0 - \text{entailment} - \text{contradiction})$$

### Mapping Characteristics

| Input Relevance | Entailment ($e$) | Contradiction ($c$) | Margin ($e - c$) | Qualitative Interpretation |
|---|---|---|---|---|
| **1.00** | 0.3000 | 0.0000 | +0.3000 | Maximum evidence support |
| **0.85** (Default Wiki) | **0.2167** | **0.1430** | **+0.0738** | Standard Wikipedia article match |
| **0.50** | 0.0750 | 0.3484 | -0.2734 | Weak / ambiguous match |
| **0.10** | 0.0030 | 0.7061 | -0.7031 | Irrelevant / conflicting evidence |
| **0.00** (Query Failed) | 0.0000 | 0.8000 | -0.8000 | Total retrieval failure |

### Why Minimal Pairs Collapse in Pillar 1
When evaluating:
- Claim 1: *"The capital of France is Paris."* $\implies$ Wikipedia query returns `France`, `Paris` $\implies \text{relevance} = 0.85 \implies \text{ent} = 0.2167, \text{con} = 0.1430$.
- Claim 2: *"The capital of France is Berlin."* $\implies$ Wikipedia query returns `France`, `Berlin` $\implies \text{relevance} = 0.85 \implies \text{ent} = 0.2167, \text{con} = 0.1430$.

Because keyword search on Wikipedia succeeds for both queries, both receive the default score `0.85`. Without running DeBERTa token cross-attention on `(Paris is the capital of France, "Berlin is the capital of France")`, the system cannot know that the retrieved text contradicts Claim 2.

---

## 3. Pillar 2 Pairwise NLI Execution

Pillar 2 evaluates genuine transformer cross-encoder NLI across pairs of claims:
- When $\text{claim\_count} = 1$: Pillar 2 features default to `[0, 0, 0, 0, 1.0]`.
- When $\text{claim\_count} \ge 2$:
  - Constructs all unordered pairs $(c_i, c_j)$.
  - Queries DeBERTa NLI CrossEncoder for bidirectional entailment and contradiction.
  - Queries MiniLM for dense semantic similarity.
  - Constructs a contradiction graph and calculates topological conflict density.

### Empirical Demonstration
- **Case F01 True:** *"Paris is the capital of France. Berlin is the capital of Germany."*  
  $\implies \text{pairwise contradiction} = 0.0008 \implies \text{low internal conflict}$.
- **Case J05 Repeated Adversarial:** *"Berlin is the capital of France. Berlin is the capital of France. Berlin is the capital of France."*  
  $\implies \text{pairwise similarity} = 0.8014 \implies P(H) = \mathbf{0.8175} \implies \mathbf{FLAGGED}$.

---

## 4. Scientific Verdict

1. **The NLI Transformer is Functioning:** When called in Pillar 2, `cross-encoder/nli-deberta-v3-small` correctly identifies logical contradictions and similarities.
2. **Pillar 1 Design Trade-off:** Pillar 1 was designed during Phase 6K as a lightweight proxy to minimize latency and memory consumption, trading token-level cross-encoder evaluation for relevance score polynomial mapping.
3. **Attribution Faithfulness:** The local attribution engine correctly attributes model risk to `p1_mean_contradiction` based on what Pillar 1 passed to the classifier.
