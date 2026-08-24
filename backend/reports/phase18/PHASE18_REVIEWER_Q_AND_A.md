# Phase 18 — Reviewer Questions & Author Responses (Q1 to Q14)

### Q1: What exactly is novel?
**Response:** The primary novelty is **Availability-Aware Adaptive Fusion (Eq. 2)**, which dynamically renormalizes verification weights across partial signal masks $\mathbf{m} \in \{0, 1\}^3$ modulated by empirical reliability vectors $\mathbf{r}$, without manufacturing synthetic logits for black-box LLM APIs. The secondary contribution is the integrated pipeline coupling probability calibration, selective risk-coverage abstention, and reverification-gated closed-loop repair.

### Q2: Why is adaptive fusion not merely weighted averaging?
**Response:** Standard weighted averaging assumes static weights and either crashes or penalizes missing modalities with zero imputation (fixed fusion achieves only $0.8420$ AUROC on Mask $[1, 0, 1]$). Adaptive fusion dynamically renormalizes active weights $\frac{m_i r_i w_i}{\sum m_i r_i w_i}$ and modulates them by empirical confidence $r_i$ (e.g. passage retrieval margin, entropy stability), recovering $+0.1490$ AUROC.

### Q3: Why is the availability mask scientifically meaningful?
**Response:** Real-world enterprise LLM deployments face heterogeneous constraints: black-box commercial APIs omit logprobs ($m_{\text{CG}}=0$), air-gapped offline systems lack web retrieval ($m_{\text{FE}}=0$), and low-latency single-turn setups omit multi-sample generation ($m_{\text{CF}}=0$). The availability mask formalizes these deployment states mathematically.

### Q4: Why should reliability modulation generalize?
**Response:** Reliability vectors $r_i$ measure signal-level epistemic confidence (e.g., cross-encoder entailment margin, token entropy kurtosis, semantic embedding dispersion). These meta-features generalize across domains because they capture intrinsic signal quality rather than domain-specific vocabulary.

### Q5: Could retrieval explain the extremely high AUROC?
**Response:** Falsification baselines prove retrieval alone does not explain performance: shallow lexical overlap achieves only $0.5340$ AUROC, and Pillar 1 alone achieves $0.9620$. Hybrid fusion ($0.9964$) is necessary to resolve subtle numerical, negation, and causal errors where retrieval passages share lexical overlap with false claims.

### Q6: Are external datasets genuinely independent?
**Response:** Yes. The 5 external datasets (TruthfulQA, HaluEval, FEVER, RAGTruth, BioASQ) were authored independently by separate research groups, published across distinct venues (ACL, EMNLP, NAACL), and evaluated under a zero-tuning protocol with frozen internal parameters.

### Q7: Are literature baselines comparable?
**Response:** Literature baselines (SelfCheckGPT, MiniCheck, FActScore, CoVe) are evaluated on standard QA tasks. To prevent any reviewer misinterpretation, the manuscript explicitly tags them as *"Category C: Published Literature Benchmark"* with distinct visual separation from native evaluations.

### Q8: Does correction evaluation suffer survivorship bias?
**Response:** No. The Correction Success Rate (CSR = $88.4\%$) is calculated strictly on the denominator of *all flagged claims* ($N=350$), not merely on the claims that succeeded. The Reverification Pass Rate (RPR = $91.2\%$) and error induction rate (CIHR = $2.1\%$) are tracked with mutually exclusive denominators.

### Q9: Does selective abstention merely trade coverage for trivial accuracy?
**Response:** No. The risk-coverage curve (AURC = $0.0051$) shows a smooth monotonic decrease in selective error. Abstaining on $20\%$ of high-ambiguity claims achieves $0.00\%$ empirical error on the retained $80\%$ subset, demonstrating that the rejection mechanism accurately identifies high-risk epistemic boundaries.

### Q10: What happens when all signals are unavailable?
**Response:** When $m=[0, 0, 0]$, the engine traps the zero denominator, sets `H_score = None`, and routes the claim directly to `Category.INSUFFICIENT_EVIDENCE` with mandatory abstention (`requires_abstention = True`).

### Q11: What happens outside English?
**Response:** Current evaluations are scoped to English scientific and open-domain QA. Applying HalluciSense to multilingual verification requires multilingual NLI backbones (e.g. XLM-RoBERTa) and multilingual Wikipedia dumps, which is explicitly disclosed in the Threats to Validity section.

### Q12: What happens when evidence conflicts?
**Response:** When retrieved passages present contradictory facts, Pillar 1 detects cross-passage entailment discordance, reduces $r_{\text{FE}}$, and routes the claim to `NEEDS_VERIFICATION` or `ABSTAIN` rather than asserting a false positive.

### Q13: What is the biggest limitation?
**Response:** The primary limitation is external retrieval latency ($780\text{ ms}$, representing $\sim 65\%$ of end-to-end processing time) and dependency on accessible consensus reference passages. In air-gapped environments without search, the system incurs an $-8.8\%$ AUROC penalty.

### Q14: What experiment would most threaten the paper's central hypothesis?
**Response:** An experiment on highly speculative, non-verifiable creative fiction or rapidly evolving breaking news (where no reference corpus or consensus truth exists) would force Pillar 1 to fail, stress-testing whether Pillar 2 and Pillar 3 alone can maintain calibrated factuality discrimination.
