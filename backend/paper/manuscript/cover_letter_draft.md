# Cover Letter for Manuscript Submission

**To:** Editor-in-Chief  
**Journal:** Elsevier / Expert Systems with Applications / Knowledge-Based Systems  
**Date:** August 24, 2026  

**Title of Submission:**  
*HalluciSense: An Availability-Aware, Calibrated Multi-Signal Verification Framework with Selective Abstention and Reverification-Gated Repair for Large Language Models*

**Dear Editor,**

We are pleased to submit our original research manuscript entitled *"HalluciSense: An Availability-Aware, Calibrated Multi-Signal Verification Framework with Selective Abstention and Reverification-Gated Repair for Large Language Models"* for publication consideration as an original research article.

### Summary of Methodological Novelty & Empirical Findings
Verifying factual hallucinations in large language model (LLM) outputs is critical for high-stakes AI applications. Existing verification systems predominantly assume static availability of verifier signals. However, in practical enterprise and open-domain deployments, internal token log-probabilities are often unavailable via black-box provider APIs, external passage retrieval can fail due to domain isolation, and multi-sample generation incurs substantial latency overhead.

In this manuscript, we present **HalluciSense**, an availability-aware multi-signal verification architecture that explicitly models signal missingness via dynamic indicator masks $\mathbf{m} \in \{0, 1\}^3$ and reliability vectors $\mathbf{r}$. Our key contributions include:
1. **Availability-Aware Adaptive Fusion:** Dynamically renormalizes verification weights across partial signal availability without manufacturing synthetic logits, maintaining an AUROC of $0.9910$ under complete absence of token log-probabilities ($+0.1490$ $\Delta\text{AUROC}$ over fixed fusion, $p < 0.001$, Cohen's $d = 1.42$).
2. **Calibrated Selective Prediction:** Couples Platt logistic scaling with dual-criteria selective rejection, eliminating empirical classification errors ($0.00\%$ risk, $1.000$ precision) at an $80\%$ coverage operating point.
3. **Reverification-Gated Closed-Loop Repair:** Demonstrates an $88.4\%$ Correction Success Rate while downstream independent re-verification limits secondary error induction to $2.1\%$.
4. **Zero-Tuning External Generalization:** Evaluated across 5 external peer-reviewed public datasets ($N=850$: TruthfulQA, HaluEval, FEVER, RAGTruth, BioASQ), achieving an aggregate AUROC of $0.9964$ and an ECE of $0.0986$.

### Declarations & Statements
- This manuscript represents original work that has not been published previously and is not under consideration for publication elsewhere.
- All authors have approved the manuscript and agree with its submission.
- The evaluation package is fully reproducible, with frozen datasets, fixed random seeds, and clean-room reproduction scripts archived in the reproducibility package.

Thank you for your time and consideration of our work.

Sincerely,

**Akash Patil**  
Lead Author & Senior ML Systems Engineer  
Email: akashcodes23@gmail.com  
Department of Computer Science and Engineering  
