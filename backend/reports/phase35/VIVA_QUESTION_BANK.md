# HalluciSense Viva Comprehensive Question Bank (50+ Questions)

This comprehensive question bank is organized into 20 thematic sections (A–T) with both concise viva answers and deep technical elaborations.

---

## Section A: Project Fundamentals

### Q1: What is HalluciSense in one sentence?
- **Short Viva Answer**: HalluciSense is a production-grade hallucination detection platform that integrates multi-pillar semantic verification signals into a frozen 19-feature Gradient Boosted Decision Tree ensemble.
- **Deeper Technical Answer**: HalluciSense decomposes LLM responses into atomic factual claims, extracts external grounding signals via hybrid retrieval and DeBERTa NLI cross-encoding, evaluates internal pairwise semantic consistency, and feeds a 19-dimensional interaction vector to a calibrated `HistGradientBoostingClassifier` operating at decision threshold $\tau^* = 0.54$.

### Q2: What exact problem does HalluciSense solve?
- **Short Viva Answer**: It detects factual inaccuracies, confabulations, and internal contradictions in LLM outputs in real-time without requiring ground-truth answer keys.
- **Deeper Technical Answer**: Generative LLMs frequently hallucinate plausible-sounding falsehoods due to probabilistic token sampling and knowledge cutoff. HalluciSense provides external reference verification and internal consistency checking with token-level visual heatmaps and root cause failure attribution.

---

## Section B: Problem Statement & Taxonomy

### Q3: How does HalluciSense categorize different types of hallucinations?
- **Short Viva Answer**: It classifies them into intrinsic contradictions, entity linking failures, temporal inconsistencies, numeric errors, and reference absence.
- **Deeper Technical Answer**: HalluciSense implements a formal Root Cause Failure Taxonomy. When an H-score exceeds threshold, the `FusionEngine` analyzes sub-pillar signals to identify whether the error stems from *Entity Linking Failure* (wrong attributes assigned to an entity), *Premise Contradiction* (direct conflict with authoritative knowledge), or *Knowledge Base Absence* (unverifiable claim).

---

## Section C: System Architecture

### Q4: Explain the high-level architecture of HalluciSense.
- **Short Viva Answer**: A FastAPI service routing requests to a lazy master pipeline proxy that executes Pillar 1, Pillar 2, and the Hybrid meta-classifier.
- **Deeper Technical Answer**: Inbound HTTP requests hit `app.modules.verification.router`, which assigns a unique `trace_id`. The request enters `HallucinationDetectionPipeline`, which coordinates `Pillar1RetrievalGroundingEngine` (dense FAISS + BM25 + DeBERTa NLI) and `Pillar2SelfConsistencyEngine`. The resulting 19 features are normalized by `RobustScaler` and classified by `HistGradientBoostingClassifier`.

---

## Section D: Three-Pillar Methodology

### Q5: What are the Three Pillars in HalluciSense?
- **Short Viva Answer**: Pillar 1 is External Evidence Grounding; Pillar 2 is Internal Self-Consistency; Pillar 3 is Multi-LLM Consensus Verification.
- **Deeper Technical Answer**: 
  - **Pillar 1**: Compares claims against external knowledge bases using NLI entailment.
  - **Pillar 2**: Evaluates semantic agreement across sentences within the response without external search.
  - **Pillar 3**: Queries auxiliary LLM providers to verify claim consensus across distinct model families.

---

## Section E: NLI & Natural Language Inference

### Q6: Which NLI model is used and why?
- **Short Viva Answer**: `cross-encoder/nli-deberta-v3-small` because it provides high cross-attention entailment accuracy with a compact memory footprint (~500 MB).
- **Deeper Technical Answer**: Cross-encoders pass both the premise (evidence) and hypothesis (claim) simultaneously through all transformer layers, capturing bidirectional token interactions far better than bi-encoders while remaining small enough to run reliably on CPU within a 1024 MB container limit.

---

## Section F: Retrieval Architecture

### Q7: How does Pillar 1 retrieve evidence?
- **Short Viva Answer**: It uses a hybrid retrieval pipeline combining dense FAISS embeddings, sparse BM25 keyword matching, and external knowledge APIs.
- **Deeper Technical Answer**: Claims are embedded via `SentenceTransformer` and searched against an in-memory FAISS index while simultaneously queried via BM25 and real-time Wikipedia/Wikidata REST APIs. Results are deduplicated, scored, and the top passages are passed to the NLI cross-encoder.

---

## Section G: Pairwise Contradiction Detection

### Q8: How does Pillar 2 detect self-contradictions?
- **Short Viva Answer**: By forming all sentence pairs within the generated response and computing bidirectional semantic contradiction probabilities.
- **Deeper Technical Answer**: If an LLM generates sentence $S_1$ and sentence $S_2$, Pillar 2 computes similarity and contradiction metrics across $(S_i, S_j)$, aggregating `p2_max_pairwise_contradiction`, `p2_mean_pairwise_contradiction`, and `p2_fraction_contradictory_pairs`.

---

## Section H: 19-Feature Representation

### Q9: Name the main groups of features in the 19-feature vector.
- **Short Viva Answer**: 5 Pillar 1 grounding features, 5 Pillar 2 consistency features, 4 individual probability/logit features, and 5 cross-pillar interaction features.
- **Deeper Technical Answer**:
  1. *P1 Grounding*: `p1_mean_entailment`, `p1_max_entailment`, `p1_mean_contradiction`, `p1_min_support_margin`, `p1_num_claims`.
  2. *P2 Consistency*: `p2_max_pairwise_contradiction`, `p2_mean_pairwise_contradiction`, `p2_max_pairwise_similarity`, `p2_fraction_contradictory_pairs`, `p2_num_claims`.
  3. *Base Probabilities*: `prob_p1`, `prob_p2`, `logit_p1`, `logit_p2`.
  4. *Interaction Signals*: `prob_disagreement_abs` ($|P_1 - P_2|$), `prob_mean`, `prob_max`, `prob_min`, `prob_ratio` ($P_1 / P_2$).

---

## Section I: Hybrid Fusion

### Q10: Why use a meta-classifier rather than simple weighted averaging?
- **Short Viva Answer**: Non-linear tree ensembles capture complex cross-pillar interactions that simple linear weighted averages cannot represent.
- **Deeper Technical Answer**: If Pillar 1 has high entailment but Pillar 2 shows an internal contradiction, or if probability disagreement is high ($|P_1 - P_2| > 0.5$), a linear sum creates ambiguity. A gradient-boosted tree splits on cross-pillar disagreement features to yield a +5.36% higher ROC-AUC (0.7378 vs 0.7012).

---

## Section J: HistGradientBoostingClassifier

### Q11: Why select `HistGradientBoostingClassifier` over standard XGBoost or Random Forests?
- **Short Viva Answer**: It offers native histogram binning for fast sub-millisecond tabular inference, native NaN handling, and zero external binary dependencies outside scikit-learn.
- **Deeper Technical Answer**: `HistGradientBoostingClassifier` discretizes continuous features into integer bins (max 256 bins), reducing memory consumption during tree construction and inference to just 218 KB, avoiding heavyweight external C++ library dependencies like `xgboost` or `lightgbm`.

---

## Section K: RobustScaler Preprocessing

### Q12: Why use `RobustScaler` instead of `StandardScaler`?
- **Short Viva Answer**: `RobustScaler` scales features using the median and interquartile range (IQR), preventing extreme outlier ratios or logits from distorting the feature space.
- **Deeper Technical Answer**: In NLP claim extraction, metrics like `prob_ratio` ($P_1/P_2$) and logit differences can exhibit heavy-tailed distributions and extreme spikes. `RobustScaler` centers by median and scales by IQR $[Q_1, Q_3]$, ensuring stable numerical inputs without being biased by extreme values.

---

## Section L: Decision Threshold ($\tau^* = 0.54$)

### Q13: Why is the decision threshold set to 0.54 instead of 0.50?
- **Short Viva Answer**: Threshold 0.54 was derived via Youden's Index on development data to minimize false positives caused by imperfect retrieval coverage.
- **Deeper Technical Answer**: When external knowledge bases lack a highly specific obscure fact, raw grounding models can show slightly elevated hallucination probabilities on factual statements. Calibrating the operating threshold to $\tau^* = 0.54$ optimizes the trade-off between sensitivity and precision, yielding an empirical F1 score of 0.7100.

---

## Section M: Model Validation & Scientific Evidence

### Q14: How many samples were used to train the Hybrid model?
- **Short Viva Answer**: 58,002 balanced multi-domain instances.
- **Deeper Technical Answer**: The development dataset (`phase6m`) contains 58,002 instances across factual and hallucinated classes, evaluated using 5-fold cross-validation and out-of-fold generalization checks.

---

## Section N: Serialization Failure & Repair

### Q15: What caused the production serialization failure and how was it fixed?
- **Short Viva Answer**: An obsolete NumPy `PCG64` BitGenerator reference caused standard `joblib.load()` to fail; it was repaired using a surrogate `_SafeModelUnpickler` that preserved 100% of the model weights.
- **Deeper Technical Answer**: In inference mode, random number generators are unused. A custom unpickler intercepted the obsolete class lookup, attached a standard modern generator, and re-saved the clean artifact. Mathematical evaluation proved a maximum probability difference of $0.00000000$ against the original backup.

---

## Section O: Production Deployment & Railway

### Q16: How is HalluciSense hosted and configured?
- **Short Viva Answer**: As a Docker container on Railway production servers under a 1024 MB memory limit.
- **Deeper Technical Answer**: The container runs Debian-based Python 3.11 with FastAPI, Uvicorn, and persistent volume storage mounted at `/data` for traces and caches.

---

## Section P: Memory Optimization

### Q17: Why did commit `78c445a` crash and how was it resolved?
- **Short Viva Answer**: Setting `PYTHONMALLOC=malloc` disabled Python's `pymalloc`, causing glibc metadata overhead to push memory from 972 MB to 1.22 GB; removing it dropped memory to 774 MB.
- **Deeper Technical Answer**: Glibc `malloc()` adds 8–16 bytes of header per allocation. When loading DeBERTa, hundreds of thousands of small Python objects were instantiated, inflating heap RSS by ~277 MB and triggering Linux kernel OOM (exit 137). Restoring `pymalloc` eliminated the regression and created 250 MB of headroom.

---

## Section Q: Concurrency & Threading

### Q18: Why are PyTorch CPU threads limited to 1?
- **Short Viva Answer**: Multi-threaded PyTorch spawns thread-local memory pools per core that exceed the 1024 MB container limit.
- **Deeper Technical Answer**: In containerized single-process environments, multi-threaded PyTorch operators multiply temporary tensor buffers across vCPUs. Setting `torch.set_num_threads(1)` bounds allocation memory while maintaining sub-second inference.

---

## Section R: Caching & Latency

### Q19: What caching mechanisms exist in HalluciSense?
- **Short Viva Answer**: An in-memory LRU cache stores claim embeddings, retrieval results, and NLI verification scores.
- **Deeper Technical Answer**: When identical queries or repeated claims are evaluated, the pipeline bypasses external REST API calls and matrix multiplications, reducing response latency from ~1,500 ms to ~10 ms.

---

## Section S: Limitations

### Q20: What are the main limitations of HalluciSense?
- **Short Viva Answer**: English language focus, dependency on external search API availability, and memory capping under extreme concurrency.
- **Deeper Technical Answer**: HalluciSense relies on authoritative knowledge coverage (Wikipedia/Wikidata/PubMed); obscure private domain facts may yield false positives if absent from indices. Concurrency is capped at 2 simultaneous NLI analyses to respect the 1024 MB ceiling.

---

## Section T: "Why This Design?" — High-Impact Viva Defenses

### Q21: Why not just ask another LLM (e.g. GPT-4) to check if the text is a hallucination?
- **Answer**: LLM-as-a-judge approaches are non-deterministic, suffer from their own hallucinations, have high per-call API cost, and lack verifiable attribution citations. HalluciSense provides deterministic, token-level mathematical attribution with sub-second latencies and verifiable knowledge passages.

### Q22: Why use a frozen model rather than continuously retraining in production?
- **Answer**: In mission-critical verification, reproducibility is paramount. A frozen model with audited SHA-256 hashes guarantees deterministic, non-drifting compliance across all evaluation runs.

### Q23: Why is explicit active-model telemetry in `/health` necessary?
- **Answer**: To prevent silent fallback masking. If a heavy ML artifact fails to load, a naive system might return 200 OK using degraded heuristics without alerting operators. Explicit telemetry (`active_model="hybrid"`) guarantees full observability.
