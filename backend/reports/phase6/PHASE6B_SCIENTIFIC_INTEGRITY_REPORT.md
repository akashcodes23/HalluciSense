# HalluciSense Phase 6B — Scientific Integrity Audit & Publication Freeze Report

**Standard**: Formal Scientific Traceability & Independent Review Defense  
**Principle**: `SCIENCE > VISUAL POLISH | MEASURED > DERIVED | REPRODUCIBLE > IMPRESSIVE | HONEST PROVENANCE > CLAIM STRENGTH`  
**Evaluation Standard**: Zero Synthetic Numbers · 100% Mathematical Traceability  
**Audit Timestamp**: `2026-08-14T09:55:00Z`  

---

## 1. Executive Summary & Final Scientific Verdict

| Audit Domain | Measured Evidence Status | Scientific Verdict |
|---|---|---|
| **Canonical Dataset Identity** | $N=750$ claims, 15 domains $\times$ 50 samples, SHA-256 verified | **PASS** |
| **Dataset Provenance** | Project-created multi-domain diagnostic suite; literature inspirations disclosed | **PASS** |
| **Pillar 1 Availability** | 100.0% executed live (BM25 + FAISS + DeBERTa-v3 Cross-Encoder) | **PASS** |
| **Pillar 2 Availability** | 0.0% on offline dataset; honestly marked `UNAVAILABLE` | **PASS (Honest Disclosure)** |
| **Pillar 3 Availability** | 0.0% on offline dataset; honestly marked `UNAVAILABLE` | **PASS (Honest Disclosure)** |
| **Fusion Integrity** | Maximum reconstruction error across 750 samples $= 0.00 \times 10^0$ ($< 10^{-9}$) | **PASS** |
| **Ablation Integrity** | Clear separation between executed $P_1$ and theoretical proxy baselines | **PASS** |
| **Metric Recomputation** | Recomputed from 750 raw predictions with 0 discrepancy to 4th decimal | **PASS** |
| **Bootstrap Validation** | $B=2000$ resamples; all 95% CIs satisfy $\text{lower} \le \text{point} \le \text{upper}$ | **PASS** |
| **Calibration Audit** | 10 bins; empirical $\text{ECE} = 0.0884$ (8.84%), Brier score $= 0.1098$ | **PASS** |
| **Latency Integrity** | Real `time.perf_counter()` timings (Total P50 $= 3326.0\text{ms}$, P95 $= 4778.5\text{ms}$) | **PASS** |
| **Trace Integrity** | 750 unique trace files persisted in `backend/reports/phase6/traces/` | **PASS** |
| **Reproducibility** | `reproduce_phase6_dataset.py` passes all 6 gates with exact SHA-256 match | **PASS** |
| **Publication Claims** | All overclaims audited and reconciled in `PUBLICATION_CLAIMS_AUDIT.md` | **PASS** |

### **FINAL SCIENTIFIC VERDICT**:
# `SCIENTIFICALLY FROZEN WITH DISCLOSED LIMITATIONS`

---

## 2. Dataset Identity & Factuality Audit
* **Canonical Path**: `backend/evaluation/results/benchmark_dataset.jsonl`
* **SHA-256 Checksum**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`
* **Sample Count**: Exactly 750 records (0 duplicates, 0 missing fields)
* **Domain Distribution**: 15 research disciplines (50 samples each)
* **Ground Truth Balance**: 375 Factual (50.0%) / 375 Hallucinated (50.0%)
* **Feature Audit**:
  * `contains_token_logprobs`: `false`
  * `contains_sample_responses`: `false`
  * `contains_multi_generation_data`: `false`

---

## 3. Pillar Availability & Offline Evaluation Mode
* **Pillar 1 (Evidence Grounding)**: **100.0% Active**. Hybrid BM25 sparse index + FAISS dense embeddings over reference corpora + DeBERTa-v3-small Cross-Encoder NLI entailment scoring.
* **Pillar 2 (Predictive Confidence)**: **0.0% Active (UNAVAILABLE)**. Static JSONL text does not contain provider token logprobs; honestly preserved as `None` without synthetic filler.
* **Pillar 3 (Semantic Consistency)**: **0.0% Active (UNAVAILABLE)**. Static single-generation evaluation; multi-sample stochastic calls omitted in offline batch.
* **Fusion Mode**: **`PARTIAL_RENORMALIZED`** ($\alpha = 1.0, \beta = 0.0, \gamma = 0.0 \implies H = 1.0 \times P_1$).

---

## 4. Recomputed Empirical Metric Landmarks ($T = 0.50, N = 750$)

| Metric | Point Estimate | 95% Bootstrap Confidence Interval ($B = 2000$) |
|---|---|---|
| **Accuracy** | **84.67%** (0.8467) | **[82.13%, 87.20%]** |
| **Precision** | **88.46%** (0.8846) | **[84.85%, 91.72%]** |
| **Recall / Sensitivity** | **79.73%** (0.7973) | **[75.51%, 83.84%]** |
| **Specificity** | **89.60%** (0.8960) | — |
| **F1 Score** | **83.87%** (0.8387) | **[80.70%, 86.67%]** |
| **Balanced Accuracy** | **84.67%** (0.8467) | — |
| **Matthews Correlation (MCC)** | **0.6967** | — |
| **AUROC** | **0.9260** | **[0.9078, 0.9430]** |
| **AUPRC** | **0.9401** | **[0.9233, 0.9544]** |
| **Expected Calibration Error (ECE)** | **0.0884** (8.84%) | **[0.0655, 0.1111]** |
| **Brier Score** | **0.1098** | **[0.0949, 0.1250]** |

### Confusion Matrix:
$$\text{TP} = 299, \quad \text{TN} = 336, \quad \text{FP} = 39, \quad \text{FN} = 76 \quad (\Sigma = 750)$$

---

## 5. Mathematical Fusion Integrity
From `backend/reports/phase6/fusion_integrity_audit.csv`:
$$\forall i \in [1, 750], \quad |H_{\text{reconstructed}}^{(i)} - H_{\text{stored}}^{(i)}| = 0.0000000000 \times 10^0 < 10^{-9}$$
Effective weights sum strictly to $1.0000$ across all 750 sample evaluations.

---

## 6. Disclosed Research Limitations
1. **Offline Evaluation Boundary**: On static offline datasets without live token logprobs or multi-sample responses, the framework operates in **Pillar 1 Grounded Mode**.
2. **Interactive vs. Batch Deployment**: Full Three-Pillar fusion ($H = 0.45 P_1 + 0.30 P_2 + 0.25 P_3$) is active during live streaming LLM generation (Phase 5 E2E).
3. **Dataset Provenance**: The 750 claims constitute a diagnostic multi-domain unit suite, not raw external public corpus downloads.
