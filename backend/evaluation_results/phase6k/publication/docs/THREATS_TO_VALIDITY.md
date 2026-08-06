# HalluciSense Pillar-1: Threats to Validity

*Generated: 2026-08-03T04:49:02.142488+00:00*  
*Phase: 6K (Frozen)*

---

## 1. Internal Validity

### Training-Validation Leakage
**Threat**: Data contamination between DEV and VAL may inflate metrics.  
**Mitigation**: SHA-256 sample fingerprinting confirmed 0% leakage.

### Label Noise
**Threat**: Ground truth labels in HaluBench/HaluEval may be incorrect.  
**Mitigation**: Multiple annotator consensus labels used where available; annotation quality not independently verified.

### Threshold Optimization Bias
**Threat**: The 0.56 threshold was selected on DEV, which may overfit.  
**Mitigation**: Protocol-locked VAL evaluation uses the same threshold; VAL was not used for threshold selection.

### Multiple Comparisons
**Threat**: Evaluating multiple candidate feature sets increases false discovery risk.  
**Mitigation**: Final candidate selected by predefined DEV criteria; VAL evaluated only once.

## 2. External Validity

### Domain Generalizability
**Threat**: Model trained on HaluBench/HaluEval/RAGTruth may not generalize to other domains.  
**Mitigation**: Multi-benchmark training provides partial coverage; OOD evaluation is future work.

### LLM Dependency
**Threat**: Hallucination patterns depend on the generating LLM; model may not transfer across LLMs.  
**Mitigation**: Benchmarks include outputs from diverse LLMs.

### Evidence Quality
**Threat**: Retrieval quality varies; poor retrieval can produce misleading NLI scores.  
**Mitigation**: Features are computed on actual retrieved passages, reflecting real retrieval quality.

## 3. Construct Validity

### Hallucination Definition
**Threat**: "Hallucination" is operationalized differently across benchmarks.  
**Mitigation**: All three benchmarks frame hallucination as claim-level unsupportedness by retrieved evidence.

### NLI as a Proxy
**Threat**: NLI entailment/contradiction scores are imperfect proxies for factual grounding.  
**Mitigation**: NLI models are validated on established benchmarks; limitations documented in §6 of limitations.

## 4. Conclusion Validity

### Statistical Power
**Threat**: 3,500 VAL samples may provide insufficient power for fine-grained analysis.  
**Mitigation**: Bootstrap CIs computed; effect sizes reported alongside p-values.

### Generalization of Results
**Threat**: Reported metrics reflect a single VAL partition evaluation.  
**Mitigation**: Bootstrap CI bounds provide uncertainty estimates around the point estimate.
