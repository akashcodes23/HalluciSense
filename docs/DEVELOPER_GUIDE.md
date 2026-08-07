# HalluciSense Developer & Architecture Guide

## System Architecture

HalluciSense implements a modular, three-pillar architecture for hallucination detection:

1. **Pillar 1: Evidence Grounding Engine (`Pillar1RetrievalEngine`)**
   - Retrieves passages from Wikipedia/Wikidata via `HybridRetriever`.
   - Reranks using CrossEncoder `ms-marco-MiniLM-L-6-v2`.
   - Computes factual error (FE) using `DeBERTa-v3-small` NLI.

2. **Pillar 2: Confidence Engine (`Pillar2ConfidenceEngine`)**
   - Computes sequence entropy and token logprob calibration.

3. **Pillar 3: Semantic Consistency Engine (`Pillar3ConsistencyEngine`)**
   - Evaluates paraphrase consistency via SBERT embeddings and NLI contradiction graphs.

4. **Adaptive Fusion Engine (`FusionEngine`)**
   - Blends pillar metrics into a calibrated overall hallucination score $H \in [0, 1]$.

---

## Adding a Custom Knowledge Provider

Inherit from `BaseKnowledgeSource` in `app/modules/knowledge/`:
```python
class CustomProvider(BaseKnowledgeSource):
    def retrieve(self, query: str) -> List[dict]:
        # Return [{"source_name": "...", "source_url": "...", "snippet": "..."}]
        pass
```
