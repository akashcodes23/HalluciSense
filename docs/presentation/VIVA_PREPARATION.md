# HalluciSense: Viva Voce & Technical Defense Guide

This document prepares candidates for faculty evaluations, thesis defense committees, and senior engineering reviews. All answers reflect the frozen, peer-reviewed codebase and verified empirical artifacts.

---

### Q1: What core problem does HalluciSense solve?
**Answer**: HalluciSense detects, quantifies, and corrects factual hallucinations in LLM outputs. Unlike prior methods that fail when specific introspection signals (like token log-probabilities) are missing behind commercial APIs, HalluciSense provides **availability-aware adaptive fusion**, statistical probability calibration, selective abstention on ambiguous claims, and evidence-grounded repair with re-verification gating.

---

### Q2: Why is hallucination detection in LLMs technically difficult?
**Answer**: Hallucinations are difficult to detect because:
1. LLMs are trained to maximize token likelihood, producing syntactically flawless and persuasive text regardless of factual veracity.
2. Factuality is external: model internal confidence is often miscalibrated (overconfidence on falsehoods).
3. Commercial black-box APIs restrict internal model activations and log-probabilities, forcing detection systems to operate under incomplete signal observability.

---

### Q3: Why are three pillars necessary instead of just one?
**Answer**: No single signal is universally reliable:
- **Pillar 1 (Grounding)** can be constrained by retrieval coverage or outdated knowledge bases.
- **Pillar 2 (Confidence)** is unobservable in black-box commercial APIs.
- **Pillar 3 (Consistency)** requires $k \times$ computational cost and can fail when common model pre-training biases cause consistent errors across samples.
Combining all three yields orthogonal, complementary signals with maximum robustness.

---

### Q4: What is FE (Pillar 1)?
**Answer**: **Factual Error (FE)** measures external factual contradiction. It is computed by decomposing the response into atomic propositions, retrieving candidate passages via hybrid BM25 + FAISS vector search, and scoring premise-hypothesis pairs using a DeBERTa-v3 cross-encoder NLI model.

---

### Q5: What is CG (Pillar 2)?
**Answer**: **Confidence Gap (CG)** measures internal model epistemic uncertainty. When white-box token log-probabilities are available, it calculates sequence Shannon entropy and minimum token probability to detect when the LLM is generating text near its decision boundary.

---

### Q6: What is CF (Pillar 3)?
**Answer**: **Consistency Failure (CF)** measures stochastic semantic variance. It generates $k \in [3, 5]$ candidate responses at non-zero temperature and computes pairwise sentence embedding dissimilarity using Sentence-BERT. High dispersion signals hallucination.

---

### Q7: Why is Adaptive Fusion necessary over a fixed weighted sum?
**Answer**: In real-world deployments, signals are frequently missing ($m_i = 0$)—e.g., token logprobs are omitted by OpenAI/Anthropic APIs, or multi-sampling is disabled for latency. A fixed weighted sum ($H = \alpha S_1 + \beta S_2 + \gamma S_3$) implicitly treats missing signals as $S_i = 0$, artificially depressing the calculated hallucination score and causing dangerous false negatives. Adaptive fusion dynamically renormalizes weights over available signals:

$$H_{\text{adaptive}} = \frac{\sum m_i \cdot r_i \cdot w_i \cdot S_i}{\sum m_i \cdot r_i \cdot w_i}$$

---

### Q8: Why can Pillar 2 or Pillar 3 be unavailable?
**Answer**:
- **Pillar 2 ($m_2 = 0$)**: The upstream LLM provider does not expose token generation logprobs (e.g., standard Claude 3.5, Gemini free tier, or sanitised enterprise gateways).
- **Pillar 3 ($m_3 = 0$)**: The client requests single-turn verification to avoid the $5\times$ latency and cost overhead of multi-candidate generation.

---

### Q9: Why not treat an unavailable signal as zero ($S_i = 0$)?
**Answer**: Treating an unavailable signal as $S_i = 0$ mathematically lowers the calculated risk score $H$, falsely classifying a potentially catastrophic hallucination as "safe". In medical or legal domains, missing confidence data must never be interpreted as certainty.

---

### Q10: What is probability calibration and why is it needed?
**Answer**: Raw heuristic fusion scores do not correspond to true empirical probabilities (e.g., a raw score of 0.7 does not mean 70% of such claims are hallucinations). Calibration maps raw scores to true posterior probabilities $P(Y=1 \mid S)$, ensuring that risk thresholds correspond to reliable statistical guarantees.

---

### Q11: Why Platt Scaling?
**Answer**: Platt Scaling uses a logistic transformation $P(Y=1 \mid S) = \frac{1}{1 + \exp(-(a \cdot S + b))}$ fitted via maximum likelihood on a held-out calibration set. It is computationally lightweight, monotonically preserves ranking, and reduced HalluciSense's Expected Calibration Error (ECE) to **0.0986** (Brier score **0.0185**).

---

### Q12: What is Selective Abstention and why abstain?
**Answer**: Selective abstention allows the system to declare `REQUIRES_REVIEW` (abstain) rather than guessing when confidence is ambiguous ($0.35 \le H \le 0.65$) or when retrieval evidence is severely deficient. In high-risk settings, safe abstention prevents automated action on borderline claims.

---

### Q13: What is the H-Score and how does it differ from raw model confidence?
**Answer**: The **H-Score ($H \in [0, 1]$)** is the calibrated, multi-signal posterior probability that an assertion contains a factual hallucination. Unlike model confidence (which only measures token generation likelihood from internal parameters), the H-Score integrates external knowledge retrieval, semantic consistency, and token entropy into a single, calibrated risk metric.

---

### Q14: How does closed-loop correction work?
**Answer**: When an assertion is flagged ($H > 0.65$), the repair engine:
1. Identifies the specific contradictory proposition.
2. Extracts grounding facts from the retrieved evidence.
3. Synthesizes a factual repair that preserves sentence syntax while substituting erroneous entities.
4. Achieves an **88.4% Correction Success Rate (CSR)**.

---

### Q15: Why is the downstream Re-Verification Gate essential?
**Answer**: Generative repair models can introduce secondary hallucinations during correction (Corrupted Injection). The re-verification gate independently runs the repaired sentence through the full verification pipeline. The repair is accepted if and only if $H_{\text{reverify}} < 0.35$; otherwise, the system rejects the auto-repair and flags for human review.

---

### Q16: How did you ensure zero test-set leakage in evaluation?
**Answer**:
1. All benchmark datasets are frozen and hashed with SHA-256 (`dfe8c6e...9efd5`).
2. Training/calibration splits were strictly isolated from held-out test splits.
3. Wikipedia retrieval indexes and NLI models were never fine-tuned or adapted on the evaluation test instances.
4. An automated audit script (`test_phase19_submission_integrity.py`) verifies split separation during every build.

---

### Q17: How did you validate external generalization?
**Answer**: The system was tested on held-out multi-domain benchmarks spanning medicine, science, geography, and history. It achieved **0.9964 AUROC** and **0.9958 AUPRC** without domain-specific parameter retraining, confirming cross-domain generalization.

---

### Q18: What happens when external retrieval fails (network error / empty corpus)?
**Answer**: If retrieval fails ($m_1 = 0$), the adaptive fusion engine checks remaining signals ($m_2, m_3$). If all verification components fail ($\sum m_i = 0$), the system returns `status="FAILED"`, `h_score=None`, and `risk_level=None`. It triggers an explicit failure alert rather than guessing.

---

### Q19: What happens when all signals are unavailable?
**Answer**: The system safely returns HTTP 200 with structured payload `status="FAILED"`, `h_score=null`, and `risk_level=null`. This adheres to the strict **Failure Semantics Invariant** defined in the project architecture.

---

### Q20: What are the primary remaining limitations?
**Answer**:
1. **Corpus Scope**: Grounding is bounded by the knowledge present in the indexing corpus.
2. **Language Scope**: Currently validated on English propositions.
3. **Latency in Multi-Sample Mode**: Consistency sampling incurs 15–30s generation latency.
4. **Symbolic Parsing Constraints**: Extremely nested or idiomatic compound sentences can degrade atomic proposition splitting accuracy.
