# 5-Minute Live Demonstration Script

**Project**: HalluciSense — Availability-Aware Multi-Signal Hallucination Verification  
**Presenter Guide**: Follow this exact timeline for faculty evaluations, technical interviews, and project presentations.

---

### `0:00 – 0:30` | The Problem
> *"Good morning. Large Language Models are increasingly integrated into mission-critical workflows across healthcare, law, and engineering. However, LLMs suffer from a well-known vulnerability: hallucinations. They generate statements that are syntactically fluent and persuasive, but factually false. In high-stakes environments, unverified AI outputs present severe liability and safety risks."*

---

### `0:30 – 1:00` | Why Conventional Approaches Fall Short
> *"Existing detection approaches generally take one of three narrow paths: pure retrieval, token probability introspection, or multi-sample consistency. But in the real world, these individual approaches break down. White-box token logprobs are inaccessible behind commercial APIs like Claude or GPT-4. Multi-sampling multiplies inference cost and latency by 5x. And naive heuristic aggregators treat missing signals as zero risk, leading to dangerous false confidence. We need a system that detects hallucinations across varied API constraints without making false assumptions."*

---

### `1:00 – 1:45` | The HalluciSense Architecture
> *"This is HalluciSense. Rather than relying on a single fallible mechanism, HalluciSense implements a 3-pillar framework:*
> 1. *Pillar 1: Evidence Grounding via hybrid BM25 and FAISS retrieval paired with DeBERTa-v3 cross-encoder NLI.*
> 2. *Pillar 2: Confidence Estimation via token entropy when white-box access is available.*
> 3. *Pillar 3: Consistency Reasoning via stochastic semantic dispersion.*
> 
> *Crucially, our Availability-Aware Adaptive Fusion Engine dynamically renormalizes weights based on binary signal presence ($m_i$) and empirical component reliability ($r_i$), followed by Platt-scaled calibration and closed-loop repair."*

---

### `1:45 – 2:30` | Scenario 1: Verifying a Factual Statement
*(Navigate to the `/verify` tab on the live dashboard)*
> *"Let's test this live on our production deployment. I will enter a factual statement:*
> **Question**: *'What is the capital of Karnataka?'*  
> **Response**: *'The capital of Karnataka is Bengaluru.'*
> 
> *I click Verify. In under 3 seconds, the pipeline retrieves authoritative Wikipedia passages, runs cross-encoder entailment, and outputs a Calibrated Hallucination Score of 13.3%, safely assigning a green 'VERIFIED' risk tier."*

---

### `2:30 – 3:15` | Scenario 2: Catching a Corrupted Entity Hallucination
*(Modify the response in the `/verify` input)*
> *"Now let's evaluate a subtle entity substitution hallucination:*
> **Response**: *'The capital of Karnataka is Mumbai.'*
> 
> *We execute verification. Immediately, HalluciSense flags the response with a 99.1% H-Score and classifies it as 'LIKELY_HALLUCINATED'. Notice the Root Cause Taxonomy: it specifically identifies an 'Entity Linking Failure', and the token risk heatmap highlights 'Mumbai' in red."*

---

### `3:15 – 3:45` | Inspecting Retrieved Evidence
*(Scroll down to the Claim-Level Analysis and Evidence cards)*
> *"HalluciSense is completely transparent. Here in the Claim Analysis card, we see the extracted atomic proposition. Below it, the system displays the retrieved authoritative evidence snippets. The NLI model detected an explicit contradiction between the claim that Mumbai is Karnataka's capital and the retrieved state legislature records naming Bengaluru."*

---

### `3:45 – 4:15` | Inspecting Distributed Execution Traces
*(Click 'View traces' or navigate to `/traces`)*
> *"Let's look under the hood at the execution trace for this query. The waterfall view breaks down every microsecond: input validation, atomic decomposition, hybrid BM25/FAISS retrieval, and cross-encoder inference. We track end-to-end latency with OpenTelemetry trace headers, ensuring production-grade observability."*

---

### `4:15 – 4:35` | Adaptive Fusion & Signal Availability Handling
*(Highlight the Fusion Decomposition panel)*
> *"Notice the Fusion Decomposition panel. Because we evaluated a single static text input without token logprobs, Pillar 2 and Pillar 3 were unavailable ($m_2=0, m_3=0$). Instead of treating missing signals as zero risk, the adaptive engine renormalized the effective weight to Pillar 1 ($\alpha=1.0$), ensuring the hallucination was not artificially masked."*

---

### `4:35 – 5:00` | Research Contribution & Conclusion
*(Navigate to `/scientific`)*
> *"Finally, here in our Scientific Lab, we display the held-out benchmark evaluations supporting our research manuscript: an external AUROC of 0.9964, expected calibration error of 0.0986, and an empirical $+0.1490$ AUROC advantage over naive fixed fusion under partial observability. HalluciSense bridges theoretical ML calibration with production-ready AI safety. Thank you, and I welcome any questions."*
