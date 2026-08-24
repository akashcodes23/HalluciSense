# HalluciSense: Research & Engineering Limitations

In accordance with rigorous scientific reporting standards, this document details the acknowledged limitations, boundaries of validity, and known trade-offs of the HalluciSense system.

---

## 1. Corpus and Retrieval Dependencies
- **Knowledge Base Coverage**: Pillar 1 verification accuracy is strictly bounded by the completeness, freshness, and accuracy of the indexing corpus (e.g., Wikipedia dump or enterprise documentation). If ground truth facts are absent or corrupted within the retrieval index, the system may register a retrieval deficit and abstain or misclassify.
- **Dynamic Real-Time World State**: Events occurring after the indexing timestamp cannot be verified via static corpus retrieval without external live search integration.

---

## 2. Linguistic & Domain Scope
- **Language Scope**: The current benchmark validation and cross-encoder NLI models (`cross-encoder/nli-deberta-v3-small`) were evaluated primarily on English-language propositions. Multilingual generalization requires cross-lingual embedding models (e.g., XLM-RoBERTa).
- **Subject-Specific Formats**: Highly specialized symbolic reasoning domains (e.g., formal mathematical proofs, low-level binary code disassembly) require tailored symbolic checkers beyond general NLI entailment.

---

## 3. Computational Complexity & Latency
- **Multi-Sample Overhead**: Operating in full Mode A with Pillar 3 active requires generating $k \in [3, 5]$ stochastic candidate responses, introducing a 15–30s latency overhead and proportional token costs from the LLM provider.
- **Single-Turn Default**: For latency-sensitive production environments, HalluciSense defaults to single-turn verification (P1 active, P2/P3 adaptive fallback), achieving ~2.5s end-to-end response times.

---

## 4. Atomic Proposition Decomposition Edge Cases
- **Complex Syntactic Structures**: Sentences with deeply nested sub-clauses, counterfactual antecedents, or colloquial metaphors can occasionally be over-fragmented or under-parsed by heuristic claim segmenters, impacting fine-grained token heatmaps.

---

## 5. Token Log-Probability Accessibility
- **Black-Box API Constraints**: As commercial providers (OpenAI, Anthropic) frequently restrict full vocabulary log-probability matrices or apply aggressive top-logprob filters, Pillar 2 must frequently rely on the adaptive renormalization fallback mode ($m_2 = 0$).

---

## 6. Selective Coverage Trade-off
- **Abstention Policy**: HalluciSense prioritizes precision over raw coverage. For claims with borderline scores ($0.35 \le H \le 0.65$), the system returns `REQUIRES_REVIEW` rather than making an automated binary decision. Users requiring 100% automated decision coverage must configure customized decision thresholds.
