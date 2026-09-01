# Phase 39.1 — Forensic Architecture Baseline Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 39 — Semantic Evidence Grounding  
**Active Production Model:** `HistGradientBoostingClassifier` (19 features, $\tau^* = 0.54$, $N=58,002$)  
**Date:** 2026-09-01  

---

## 1. End-to-End Inference Trace & Subsystem Inventory

Tracing execution from `response_text` to verdict and local counterfactual explanation:

```
response_text
  │
  ▼
[1] Claim Extraction (`extract_claims`)
    - Abbreviation protection (21 patterns)
    - Sentence segmentation → `List[{"claim_id": int, "text": str}]`
  │
  ├──► [2] Pillar 1 Engine (`pillar1_engine.py`)
  │        - Retrieves evidence via `HybridRetriever.get_evidence(claim_text)`
  │        - Returns list of passages with text snippets (`p["snippet"]`, `p["title"]`)
  │        - CURRENT PRODUCTION PROXY: Maps relevance score via `_relevance_to_nli(relevance)`
  │        - Constructs 5 features: `[mean_ent, max_ent, mean_con, min_margin, num_claims]`
  │        - LogisticRegression Base Model → `prob_p1`
  │
  ├──► [3] Pillar 2 Engine (`pillar2_engine.py`)
  │        - If `claim_count >= 2`: Generates pairwise combinations
  │        - Evaluates pairwise NLI (`cross-encoder/nli-deberta-v3-small`) & similarity (`all-MiniLM-L6-v2`)
  │        - Constructs 5 features: `[max_con, mean_con, max_sim, frac_con, num_claims]`
  │        - LogisticRegression Base Model → `prob_p2`
  │
  ▼
[4] Meta Fusion Vector Assembly (19 features)
    - 5 Pillar 1 features + 5 Pillar 2 features + 4 Base logits + 5 Meta interaction signals
  │
  ▼
[5] RobustScaler Transformation (`preprocessing.joblib`)
    - $X_{\text{scaled}} = \text{scaler.transform}(X_{\text{raw}})$
  │
  ▼
[6] Hybrid Meta Classifier (`HistGradientBoostingClassifier`)
    - $P(H) = \text{clf.predict\_proba}(X_{\text{scaled}})[0, 1]$
    - Verdict: $P(H) \ge 0.54$
  │
  ▼
[7] Local Counterfactual Attribution (`local_attribution.py`)
    - 21 deterministic model evaluations against `RobustScaler.center_`
    - $a_i = P(H \mid X) - P(H \mid X_i)$
    - Interaction gap: $\mathcal{I}(X) = [P(X) - P(\text{baseline})] - \sum a_i$
  │
  ▼
[8] Response Serialization & UI Rendering
```

---

## 2. Evidence Grounding Gap Traced to Exact Lines

| Component | File & Line | Current Behavior | Target Behavior in Phase 39 |
|---|---|---|---|
| **Retrieval Relevance** | `backend/app/modules/knowledge/retriever.py` (L45-47) | Assigns default `0.85` similarity score when no dense match | Preserve retrieval passages containing full snippet texts |
| **Pillar 1 NLI Proxy** | `backend/app/core/inference/pillar1_engine.py` (L25-70, L120-122) | Applies synthetic polynomial `_relevance_to_nli(0.85)` | Run genuine claim $\leftrightarrow$ evidence DeBERTa cross-attention |
| **Model Reusability** | `backend/app/core/engine/model_registry.py` (L47-64) | Singleton `ModelRegistry.get_nli_model()` already manages DeBERTa | Reuse existing singleton instance across Pillar 1 & Pillar 2 |
| **Inference Concurrency** | `backend/app/core/engine/model_registry.py` (L39-44) | `_nli_semaphore` bounds concurrent transformer inferences to 2 | Enforce strict semaphore usage to prevent RAM spikes |

---

## 3. Singleton & Model Sharing Architecture

- `ModelRegistry` in `backend/app/core/engine/model_registry.py` maintains strict singleton references with double-checked locking (`_lock = threading.RLock()`).
- `EvidenceEntailmentEngine` in `backend/app/core/engine/entailment.py` wraps `ModelRegistry.get_nli_model()` with an internal LRU cache (512 entries) and batched tokenization (`batch_size=16`, `max_length=512`).
- **Guarantee:** There is exactly **ONE** DeBERTa model loaded in memory (~280 MB), eliminating any risk of duplicate transformer allocation.
