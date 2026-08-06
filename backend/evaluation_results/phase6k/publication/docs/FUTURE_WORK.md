# HalluciSense Pillar-1: Future Work

*Generated: 2026-08-03T04:49:02.142488+00:00*  
*Phase: 6K (Frozen)*

---

## 1. Pillar 2: Semantic Similarity Signals

Pillar-1 covers NLI-based evidence alignment. Pillar-2 should add:
- **Dense retrieval similarity**: FAISS-based cosine similarity between claim embeddings and evidence
- **BM25 lexical overlap**: Term-level evidence recall
- **Entity coverage**: Named entity overlap between claims and evidence
- **Expected performance gain**: +0.03 to +0.07 ROC-AUC over Pillar-1 alone

## 2. Pillar 3: Factual Grounding (Knowledge-Based)

Pillar-3 should address hallucinations not detectable via evidence alignment:
- **Knowledge graph consistency**: Wikidata triple verification for factual claims
- **Entity disambiguation**: Cross-reference named entities to authoritative sources
- **Temporal reasoning**: Detect temporally inconsistent claims

## 3. Hybrid Fusion

A linear or learned fusion of Pillar-1 + Pillar-2 + Pillar-3 signals:
- **Stacking**: Train a meta-classifier on Pillar probability outputs
- **Weighted ensemble**: Grid-search optimal Pillar weights on DEV
- **Expected performance**: Potentially +0.08 to +0.12 AUC over Pillar-1

## 4. Model Upgrades (Pillar-1 Only)

Within the Pillar-1 scope:
- **Claim-level prediction**: Predict hallucination per-claim rather than per-response
- **Attention-weighted aggregation**: Weight NLI scores by claim salience
- **Calibrated ensemble**: Aggregate multiple NLI model outputs
- **Stronger NLI model**: Evaluate `cross-encoder/nli-deberta-v3-large`

## 5. Calibration

- **Temperature scaling**: Evaluate as a post-hoc calibration method
- **Isotonic regression**: Production calibration wrapper if ECE > 0.05
- **Dataset-specific thresholds**: Per-domain threshold optimization

## 6. Domain Adaptation

- Extend evaluation to medical (MedHallu), legal, and scientific domains
- Fine-tune NLI model on domain-specific claim-evidence pairs
- Evaluate cross-lingual transferability

## 7. Reproducibility Infrastructure

- Docker image for end-to-end pipeline reproduction
- GitHub Actions CI for automated benchmark regression testing
- Hugging Face Hub model card and artifact hosting

## 8. Publication Roadmap

Target venues for the HalluciSense paper:
1. **Primary**: Elsevier Expert Systems with Applications (confirmed scope)
2. **Secondary**: ACL findings (NLP track), EMNLP (Evaluation track)
3. **Workshop**: HalEval @ ACL (hallucination evaluation workshop)
