# Phase 38.3 — Feature Representation Collapse Analysis

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 38.3 — Forensic Feature Representation Collapse Analysis  
**Active Production Model:** `HistGradientBoostingClassifier` (19 features, $\tau^* = 0.54$, $N=58,002$)  
**Dataset:** 162 evaluated adversarial cases across 10 distinct categories  
**Date:** 2026-09-01  

---

## 1. Executive Summary & Core Finding

A mathematical and architectural evaluation of 60 minimal pairs across Categories A through F (Factual Minimal Pairs, Entity Swaps, Numerical Mutations, Negations, Temporal Mutations, and Multi-Claim Structural Pairs) was conducted on the complete production inference pipeline.

### Quantitative Summary Metrics

| Metric | Measured Value | Scientific Interpretation |
|---|---|---|
| **Total Minimal Pairs Evaluated** | **60 pairs (120 cases)** | Comprehensive cross-domain coverage |
| **Identical Representation Pairs ($L_2 = 0.0$)** | **55 pairs (91.7%)** | Complete collapse of 19-feature vector |
| **Near-Identical Pairs ($L_2 \le 0.01$)** | **55 pairs (91.7%)** | Zero discrimination across atomic modifications |
| **Representation Discrimination Rate** | **8.3%** | Only 5 pairs produced distinguishable vectors |
| **Probability Separation Mean** | **0.0000** (for single-claim pairs) | Identical $P(H)$ for true vs. false statements |
| **Verdict Separation Rate** | **0.0%** | Neither truth nor falsehood crossed $\tau^* = 0.54$ |

---

## 2. Root Cause of Representation Collapse

The forensic audit traced the representation collapse to an exact architectural bottleneck in the interaction between `HybridRetriever` and `Pillar1Engine`:

1. **Hardcoded Retrieval Relevance Assignment:**  
   In `backend/app/modules/knowledge/retriever.py` (lines 45–47):
   ```python
   for claim in clean_claims:
       for item in wiki_by_claim.get(claim, []):
           evidence = dict(item)
           evidence["claim"] = claim
           if "similarity_score" not in evidence:
               evidence["similarity_score"] = 0.85
           all_evidence.append(evidence)
   ```
   Whenever Wikipedia returns search results, all passages are assigned a static default `similarity_score = 0.85`.

2. **Bypass of Cross-Encoder NLI Evaluation:**  
   In `backend/app/core/inference/pillar1_engine.py`:  
   Pillar 1 does **not** evaluate the DeBERTa-v3 cross-encoder model on the pair `(evidence_snippet, claim_text)`.  
   Instead, it applies a static deterministic mathematical transformation function `_relevance_to_nli(relevance)`:
   ```python
   entailment = 0.3 * (relevance ** 2)              # 0.3 * 0.85^2 = 0.21675
   contradiction = 0.8 * ((1.0 - relevance) ** 1.2) # 0.8 * 0.15^1.2 = 0.0811 (normalized to 0.1430)
   margin = entailment - contradiction              # 0.07376
   ```

3. **Single-Claim Invariance in Pillar 2:**  
   When an input contains a single sentence ($\text{claim\_count} = 1$), Pillar 2 pairwise contradiction is structurally 0.0 (`p2_0..p2_4 = [0, 0, 0, 0, 1.0]`).

4. **Resulting 19-Feature Collapse:**  
   Consequently, every single-sentence factual assertion (whether true or false) produces the **identical vector**:
   $$X = [0.2167, 0.2167, 0.1430, 0.0738, 1.0, 0, 0, 0, 0, 1.0, 0.4879, 0.4341, -0.0483, -0.2650, 0.0538, 0.4610, 0.4879, 0.4341, 1.1239]$$
   and the identical prediction:
   $$P(H) = 0.2973$$

---

## 3. Minimal-Pair Representation Collapse Table

| Pair ID | True Statement | False / Mutated Statement | Category | $L_2$ Distance | $P(H)_{\text{true}}$ | $P(H)_{\text{false}}$ | Separation Status |
|---|---|---|---|---|---|---|---|
| **A01** | *"The capital of France is Paris."* | *"The capital of France is Berlin."* | Minimal Pair | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **A02** | *"Oxygen has an atomic number of 8."* | *"Oxygen has an atomic number of 9."* | Minimal Pair | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **A03** | *"Mount Everest is highest on Earth."* | *"K2 is highest on Earth."* | Minimal Pair | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **B01** | *"Einstein developed relativity."* | *"Newton developed relativity."* | Entity Swap | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **B02** | *"Tokyo is populous in Japan."* | *"Kyoto is populous in Japan."* | Entity Swap | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **B03** | *"Shakespeare wrote Hamlet."* | *"Dickens wrote Hamlet."* | Entity Swap | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **C01** | *"12 multiplied by 8 equals 96."* | *"12 multiplied by 8 equals 95."* | Numerical | **0.0091** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **C02** | *"Speed of light is 299792458 m/s."* | *"Speed of light is 299792459 m/s."* | Numerical | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **D01** | *"Water boils at 100°C."* | *"Water does not boil at 100°C."* | Negation | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **D02** | *"Earth revolves around the Sun."* | *"Earth does not revolve around the Sun."*| Negation | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **E01** | *"India gained independence in 1947."* | *"India gained independence in 1958."*| Temporal | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **E02** | *"Apollo 11 landing occurred in 1969."*| *"Apollo 11 landing occurred in 1984."*| Temporal | **0.0000** | 0.2973 | 0.2973 | ❌ COLLAPSED |
| **F01** | *"Paris capital France. Berlin Germany."* | *"Paris capital France. Berlin France."* | Multi-claim | **0.0000** | 0.3546 | 0.3546 | ❌ COLLAPSED |

---

## 4. Exceptions: Where Feature Discrimination Occurred

Feature separation ($L_2 > 0.01$) occurred exclusively in cases where:
1. **Wikipedia Retrieval Failed Entirely:**
   - Case G01 (*"Ancient subterranean civilization..."*): $\text{failed\_queries} = 1 \implies \text{relevance} = 0.0834, \text{margin} = -0.2422 \implies L_2 = 0.4178$.
2. **Repeated Adversarial Sentence Structure:**
   - Case J05 (*"Berlin is capital of France. Berlin is capital of France. Berlin is capital of France."*):
     Extracted 3 identical claims $\implies \text{pairwise similarity} = 0.8014, P(H) = 0.8175 \implies \mathbf{FLAGGED}$.
3. **Empty Retrieval on Obscure Query:**
   - Case J08 (*"Recent epistemological paradigms..."*): Wikipedia query returned 0 pages $\implies P(H) = 0.6653 \implies \mathbf{FLAGGED}$.

---

## 5. Architectural Implications

- **The Classifier is Not Broken:** When given distinct feature vectors (e.g. Case J05 with $P(H) = 0.8175$ or Case J08 with $P(H) = 0.6653$), `HistGradientBoostingClassifier` correctly flags high-risk hallucinations.
- **The Attribution Engine is Faithful:** The local attribution engine accurately describes what the classifier saw; it reports that `p1_mean_contradiction` is the primary driver because that is the exact value present in $X$.
- **The True Bottleneck is Upstream Feature Extraction:** The static mapping of retrieval similarity to NLI features without cross-encoder evaluation causes single-claim inputs to collapse into identical feature representations regardless of semantic truth value.
