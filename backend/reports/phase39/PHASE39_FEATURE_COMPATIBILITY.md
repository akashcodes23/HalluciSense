# Phase 39.12 — Production Feature Compatibility Assessment

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 39.12 — Formal Feature Compatibility & Substitution Assessment  
**Active Production Model:** `HistGradientBoostingClassifier` (19 features, $\tau^* = 0.54$, $N=58,002$)  
**Date:** 2026-09-01  

---

## 1. Feature Schema Alignment Analysis

The 5 Pillar-1 features defined during Phase 6K and locked for the hybrid classifier are:

| Feature Index | Feature Name | Historical Proxy Generator | Semantic NLI Adapter Generator | Semantic Compatibility |
|---|---|---|---|---|
| `[0]` | `p1_mean_entailment` | `_relevance_to_nli(rel)[0]` (quadratic polynomial) | Mean DeBERTa cross-encoder entailment score across evidence passages | ✅ Exact Match (0.0 to 1.0) |
| `[1]` | `p1_max_entailment` | Peak proxy entailment score | Peak DeBERTa cross-encoder entailment score across evidence passages | ✅ Exact Match (0.0 to 1.0) |
| `[2]` | `p1_mean_contradiction` | `_relevance_to_nli(rel)[1]` (superlinear decay) | Mean DeBERTa cross-encoder contradiction score across evidence passages | ✅ Exact Match (0.0 to 1.0) |
| `[3]` | `p1_min_support_margin` | $\text{max\_ent} - \text{mean\_con}$ | $\text{max\_ent} - \text{mean\_con}$ from real NLI distributions | ✅ Exact Match (-1.0 to 1.0) |
| `[4]` | `p1_num_claims` | Float claim count | Float claim count | ✅ Exact Match ($1.0$ to $\infty$) |

---

## 2. Compatibility Assessment

- **Option A (No Schema Change Needed):** The new semantic NLI adapter produces values that share the exact physical units, theoretical bounds, and semantic meanings as the original training schema.
- **Option B (Preservation of 19-Feature Dimensionality):** Substituting real NLI outputs into Pillar 1 preserves the 19-dimensional input shape $X \in \mathbb{R}^{19}$ without modifying `preprocessing.joblib` (`RobustScaler`, $N=19$) or `hybrid_meta_classifier.joblib`.
- **Option C (Shadow vs. Active Control):** Controlled via `HALLUCISENSE_SEMANTIC_NLI_MODE`:
  - `shadow` (Default): Downstream classifier evaluates proxy features for 100% legacy baseline invariance, while returning full semantic NLI grounding diagnostics in the API response.
  - `active`: Downstream classifier receives true semantic NLI features, eliminating representation collapse on single-claim minimal pairs.

---

## 3. Decision & Scientific Policy

1. **Production Default Remains Safe:** `HALLUCISENSE_SEMANTIC_NLI_MODE=shadow` ensures zero surprise regressions or uncalibrated shifts in production probability during initial deployment.
2. **Explainability Surfaces Full Trace:** Both `/predict` and `/explain` expose the rich semantic evidence trace directly to users and frontend consumers.
3. **No Retraining Required:** The frozen `HistGradientBoostingClassifier` requires 0 weight modifications.
