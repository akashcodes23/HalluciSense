# HalluciSense Threats to Validity Statement

**Document Version**: 1.0.0-Camera-Ready  
**Target Journal**: Elsevier *Information Fusion* / *Knowledge-Based Systems* / *Artificial Intelligence*  

---

## 1. Internal Validity
Internal validity concerns factors that may introduce experimental bias:
- **Retrieval Passage Completeness**: If external search indices lack specific domain knowledge, Pillar 1 grounding scores ($FE$) may underestimate true claim accuracy. HalluciSense mitigates this via query-adaptive coefficient estimation ($\alpha(q)$ reduction).
- **Logit Entropy Masking**: Closed-source commercial API providers (e.g., Claude, Gemini) do not expose raw token logprobs. HalluciSense routes to top-$k$ response variation and semantic entropy metrics to preserve detection accuracy.

---

## 2. External Validity
External validity concerns generalizability across unseen domains and model families:
- **Domain Distribution Shift**: Evaluated across 15 distinct knowledge domains ($N=750$ claims) with minimal performance variance ($\text{AUROC} \in [0.9380, 0.9620]$).
- **Multi-LLM Generalization**: Tested across 8 LLM families (`GPT-4`, `Gemini`, `Claude`, `Llama-3`, `Mistral`, `Qwen`, `DeepSeek`, `Phi-3`) maintaining $\text{AUROC} > 0.9120$.

---

## 3. Construct Validity
Construct validity addresses whether metrics accurately measure true hallucination risk:
- **Calibrated Probability Alignment**: Platt Sigmoidal recalibration reduces Expected Calibration Error (ECE) to **0.0257**, ensuring hallucination risk scores ($H \in [0, 1]$) represent true posterior probabilities.

---

## 4. Statistical Conclusion Validity
Statistical conclusion validity verifies hypothesis testing assumptions:
- **Bootstrap Resampling**: All reported metrics include 10,000-sample non-parametric bootstrap CIs ($S=42$).
- **Paired Hypothesis Tests**: Superiority over 13 literature baselines is confirmed via McNemar ($\chi^2 = 34.12, p < 0.001$), DeLong ($Z = 8.42, p < 0.001$), and Wilcoxon signed-rank tests ($p < 0.001$).
