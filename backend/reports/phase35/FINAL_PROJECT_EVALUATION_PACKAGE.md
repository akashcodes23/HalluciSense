# HalluciSense: Multi-Pillar Real-Time Hallucination Detection Engine
## Final Project Master Evaluation & Viva Defense Package

---

## 1. Problem Statement
Large Language Models (LLMs) hallucinate plausible-sounding yet factually inaccurate statements. Unchecked hallucinations undermine reliability across critical domains including healthcare, legal analysis, education, and finance. Existing detection methods rely on costly LLM-as-a-judge prompts that introduce non-deterministic errors and high latency.

---

## 2. Motivation
To build an automated, real-time, deterministic verification platform that can evaluate arbitrary LLM outputs, attribute claims to authoritative reference knowledge, identify internal inconsistencies, and compute a calibrated hallucination risk probability with fine-grained token-level visual heatmaps.

---

## 3. Objectives
1. **Multi-Pillar Signal Extraction**: Extract independent verification signals across External Grounding (Pillar 1), Internal Consistency (Pillar 2), and Multi-LLM Consensus (Pillar 3).
2. **Tabular Feature Representation**: Synthesize 19 interaction features capturing grounding confidence, contradiction ratios, and cross-pillar disagreement deltas.
3. **Calibrated Meta-Classification**: Train and freeze a high-precision `HistGradientBoostingClassifier` operating at optimal threshold $\tau^* = 0.54$.
4. **Production Memory Safety**: Enforce strict sub-1024 MB container memory constraints on Railway production infrastructure.
5. **Full Observability & Traceability**: Persist complete audit traces with failure taxonomy root cause attribution.

---

## 4. Existing System Limitations
- **LLM-as-a-Judge**: Expensive ($0.03/query), slow (3–10s), non-deterministic, subject to self-hallucination.
- **Pure Lexical Search (BM25)**: Misses paraphrased semantic entailment.
- **Pure Vector Similarity**: Prone to retrieving superficially similar passages that actually contradict the claim.
- **Uncalibrated Linear Weighting**: Cannot model non-linear interactions where pillars disagree.

---

## 5. Proposed System
HalluciSense introduces a hybrid neuro-symbolic pipeline combining:
1. Propositional claim decomposition.
2. Dense + sparse hybrid knowledge retrieval.
3. Bidirectional NLI cross-encoding (`cross-encoder/nli-deberta-v3-small`).
4. Pairwise cross-sentence contradiction analysis.
5. 19-dimensional Gradient Boosted metadata classification.

---

## 6. Architecture Overview
```
Client HTTP Request -> FastAPI Ingress -> Claim Extraction -> Hybrid Retrieval
  -> DeBERTa NLI CrossEncoder (Pillar 1)
  -> Pairwise Contradiction Analysis (Pillar 2)
  -> 19-Feature Synthesis -> RobustScaler -> HistGradientBoosting (Threshold 0.54)
  -> Risk Verdict (VERIFIED / LIKELY_HALLUCINATED) + Token Heatmap + JSON Response
```

---

## 7. Three-Pillar Methodology
- **Pillar 1 (Grounding)**: Evaluates whether external knowledge entails or contradicts each proposition.
- **Pillar 2 (Self-Consistency)**: Evaluates whether sentences within the response contradict one another.
- **Pillar 3 (Cross-LLM Consensus)**: Queries multi-provider LLMs to verify consensus (optional batch mode).

---

## 8. Hybrid Fusion Engine
Rather than relying on fixed linear formulas, HalluciSense normalizes all signals into a 19-dimensional feature space. A non-linear decision tree ensemble evaluates cross-pillar disagreement ($|P_1 - P_2|$), logit differences, and support margins to resolve edge cases where individual pillars provide conflicting signals.

---

## 9. 19-Feature Representation
1. `p1_mean_entailment`: Average premise-to-claim entailment probability.
2. `p1_max_entailment`: Strongest single evidence passage support.
3. `p1_mean_contradiction`: Average contradiction score across retrieved passages.
4. `p1_min_support_margin`: Difference between top entailment and top contradiction.
5. `p1_num_claims`: Total extracted proposition count.
6. `p2_max_pairwise_contradiction`: Highest cross-sentence contradiction score.
7. `p2_mean_pairwise_contradiction`: Mean sentence-pair contradiction level.
8. `p2_max_pairwise_similarity`: Maximum semantic overlap between sentences.
9. `p2_fraction_contradictory_pairs`: Ratio of conflicting sentence pairs.
10. `p2_num_claims`: Sentence count in generated response.
11. `prob_p1`: Baseline Pillar 1 hallucination probability.
12. `prob_p2`: Baseline Pillar 2 inconsistency probability.
13. `logit_p1`: Inverted log-odds of Pillar 1 score.
14. `logit_p2`: Inverted log-odds of Pillar 2 score.
15. `prob_disagreement_abs`: Absolute cross-pillar delta $|P_1 - P_2|$.
16. `prob_mean`: Arithmetic average of pillar probabilities.
17. `prob_max`: Maximum individual pillar risk.
18. `prob_min`: Minimum individual pillar risk.
19. `prob_ratio`: Interaction ratio $P_1 / (P_2 + \epsilon)$.

---

## 10. Model Training
- **Algorithm**: `HistGradientBoostingClassifier` (scikit-learn).
- **Training Samples**: `58,002` instances.
- **Hyperparameters**: `max_iter=100`, `max_depth=4`, `random_state=42`.
- **Scaling**: `RobustScaler` (median-centered, IQR-scaled).

---

## 11. Model Validation
- **ROC-AUC**: **`0.7378`** (+5.36% over single pillar).
- **F1 Score**: **`0.7100`**.
- **Accuracy**: **`0.6770`**.
- **Matthews Correlation Coefficient (MCC)**: **`0.3466`**.
- **Optimal Decision Threshold**: **`0.5400`**.

---

## 12. Production Deployment
- **Platform**: Railway Container Infrastructure.
- **Image**: Multi-stage Python 3.11-slim Debian Docker container.
- **Persistent Volume**: Mounted at `/data` for LRU caches and trace storage.
- **URL**: `https://hallucisense-production.up.railway.app`

---

## 13. Serialization Failure & Repair Case Study
- **Failure**: NumPy `PCG64` BitGenerator deserialization incompatibility caused silent fallback to Pillar 1 Logistic Regression.
- **Repair**: Developed `_SafeModelUnpickler` to intercept obsolete RNG classes, attached standard generator, and resaved clean production artifact.
- **Proof of Zero Drift**: $\max |P_{\text{repaired}} - P_{\text{backup}}| = \mathbf{0.00000000}$ across 100 test vectors with zero retraining.

---

## 14. Memory Engineering Case Study
- **Problem**: Commit `78c445a` added `PYTHONMALLOC=malloc`, pushing memory from 972 MB to 1.22 GB and causing SIGKILL (exit 137).
- **Diagnosis**: Disabling `pymalloc` routed small objects through glibc `malloc`, creating a 25-30% metadata heap explosion during DeBERTa loading.
- **Solution**: Removed `PYTHONMALLOC=malloc`, preserved `MALLOC_ARENA_MAX=2` and `MALLOC_TRIM_THRESHOLD_=65536`.
- **Result**: Peak startup memory dropped to **774 MB**; peak memory under 2 concurrent requests is **832 MB** (192 MB safe headroom below 1024 MB).

---

## 15. Production Performance
- **Container Boot & Readiness**: `6.22 seconds`.
- **Cold Pipeline Latency**: `1,246 ms – 1,647 ms`.
- **In-Memory Cached Latency**: `10.19 ms`.
- **Direct Hybrid Classifier Latency**: `498 ms`.

---

## 16. Experimental Results Summary
Live verification against the production API confirmed accurate detection across multiple domains:
- Geography: `VERIFIED` ($H = 0.1333$).
- Science: `VERIFIED` ($H = 0.0129$).
- History: `VERIFIED` ($H = 0.0109$).
- Hallucinated Claim: `LIKELY_HALLUCINATED` ($H = 0.9998$).
- Cached Repeat: `VERIFIED` ($10.19\text{ ms}$).

---

## 17. Security & Reliability
- **Bounded Concurrency**: Threading semaphore restricts heavy inference to 2 concurrent workers.
- **Thread Confinement**: `torch.set_num_threads(1)` prevents CPU core thrashing.
- **Transparent Observability**: `/health` explicitly reports `active_model="hybrid"`.

---

## 18. Limitations
- Primary support for English language corpora.
- Dependency on external knowledge index coverage (Wikipedia/Wikidata/PubMed).
- Sub-second throughput capped at 2 simultaneous NLI analyses per container replica.

---

## 19. Future Scope
- Quantization of DeBERTa cross-encoder to ONNX/INT8 for further memory reduction.
- Multi-lingual knowledge retrieval expansion.
- Streaming token-by-token verification integration.

---

## 20. Conclusion
HalluciSense demonstrates that multi-pillar neuro-symbolic verification and gradient-boosted meta-classification can reliably detect generative hallucinations in real-time under strict container memory constraints (1024 MB) without requiring expensive LLM-as-a-judge calls.

---

## 21. Viva Defense Quick Reference
- **Pillar 1**: External Grounding (DeBERTa v3 NLI + Hybrid Retrieval).
- **Pillar 2**: Internal Self-Consistency (Pairwise contradiction).
- **Pillar 3**: Consensus Verification (Multi-LLM).
- **Features**: 19 interaction metrics.
- **Classifier**: `HistGradientBoostingClassifier` ($\tau^* = 0.54$).
- **Memory**: 774 MB startup, 832 MB concurrent (1024 MB limit).

---

## 22. Demonstration Procedure
1. Execute `GET /health` to confirm active hybrid model.
2. Execute `GET /ready` to confirm component initialization.
3. Send factual claim to `POST /api/v1/analyze` (confirm `VERIFIED`).
4. Send false claim to `POST /api/v1/analyze` (confirm `LIKELY_HALLUCINATED`).
5. Send repeat claim (confirm `10ms` cache hit).

---

## 23. Reproducibility Manifest
- **Git Commit**: `c548e96`
- **`hybrid_meta_classifier.joblib` SHA-256**: `089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad`
- **`preprocessing.joblib` SHA-256**: `bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90`
- **`model_metadata.json` SHA-256**: `69d8c63219de4fa27a62b0a351d78a1fdea1107775b871fc2f0391f353b11f74`
