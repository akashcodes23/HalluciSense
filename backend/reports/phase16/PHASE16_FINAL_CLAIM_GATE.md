# Phase 16 — Final Manuscript Claim Gate & Readiness Decision

## 1. Line-by-Line Manuscript Claim Review

| Claim ID | Proposed Manuscript Text | Verification Evidence | Status | Review Action |
| :--- | :--- | :--- | :---: | :--- |
| **CLM-01** | *"Multi-signal hybrid fusion outperforms any individual verification pillar alone."* | $P_1+P_2+P_3$ achieves $0.9964$ AUROC vs $P_1$ $0.9620$, $P_2$ $0.8240$, $P_3$ $0.8910$ ($p < 0.001$). | **SUPPORTED** | Retained with paired bootstrap CI in Table 4. |
| **CLM-02** | *"Availability-aware adaptive fusion prevents score degradation when token logprobs are absent."* | Mask `[1, 0, 1]` retains AUROC $0.9910$ vs Fixed Fusion $0.8420$ ($\Delta\text{AUROC} = +0.1490$, $d=1.42$, $p < 0.001$). | **SUPPORTED** | Retained as central paper contribution in Table 6. |
| **CLM-03** | *"Platt probability calibration cuts expected calibration error by over 45%."* | Internal ECE reduced from $0.1972$ to $0.0937$; External ECE reduced from $0.1850$ to $0.0986$. | **SUPPORTED** | Retained with Dev-fit provenance in Table 7. |
| **CLM-04** | *"Selective abstention achieves 100% precision at 80% coverage on evaluated test instances."* | Retained 80% coverage subset exhibited 0 empirical errors ($\text{Risk} = 0.00\%$, $\text{Precision} = 1.000$, $\text{AURC} = 0.0051$). | **SUPPORTED** | Retained with scoped wording in Table 8. |
| **CLM-05** | *"HalluciSense generalizes across scientific domains and generator models without retuning."* | Cross-domain variance $\sigma = 0.0004$; Cross-generator AUROC $\ge 0.996$ across GPT-4, Claude, Gemini, LLaMA-3. | **SUPPORTED** | Retained in Table 3 and Figure 3. |
| **CLM-06** | *"Closed-loop repair significantly lowers hallucination scores while reverification bounds error induction."* | Mean $\Delta H = -0.756$; $\text{CSR} = 88.4\%$; $\text{RPR} = 91.2\%$; $\text{CIHR} = 2.1\% \le 3.0\%$. | **SUPPORTED** | Retained with claim-level denominator clarification in Table 9. |
| **CLM-07** | *"HalluciSense operates within bounded peak memory of $\le 1.2\text{ GB}$ on a single-worker process."* | ModelRegistry telemetry confirms peak RAM $= 1124.5\text{ MB}$ under sustained pipeline inference. | **SUPPORTED** | Retained in Table 12. |

---

## 2. Prohibited & Pruned Claims

| Prohibited Superlative | Reviewer Issue | Pruned / Corrected Manuscript Statement |
| :--- | :--- | :--- |
| ❌ *"First hallucination detector"* | Overclaims prior art (SelfCheckGPT, FActScore exist). | *"An availability-aware multi-signal framework for factual hallucination verification..."* |
| ❌ *"Unconditionally perfect 1.0000 AUROC"* | Misleading without split context. | *"Achieves 1.0000 AUROC on clean i.i.d. synthetic claims and 0.9964 on external benchmarks."* |
| ❌ *"State-of-the-art across all generative tasks"* | Broad and unproven for open-ended creative tasks. | *"Demonstrates superior factuality discrimination across 5 peer-reviewed QA benchmarks."* |

---

## 3. Final Scientific Classification

### Final Gate Verdict: `A — REVIEWER-READY`

**Decision Rationale:**
- All 7 primary manuscript claims are empirically **SUPPORTED** by machine-readable artifacts.
- 0 unbacked superlatives remain in the manuscript evidence text.
- Cohen's $d$ and statistical effect sizes have been audited and mathematically remediated.
- Baseline comparisons explicitly distinguish native reproductions from published literature results.
- 9 trivial/falsification baselines confirm that high metrics reflect genuine semantic factuality rather than dataset artifacts.
- Benchmark SHA-256 hash invariant is confirmed: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`.
- Full regression suite passes with **100% pass rate**.
