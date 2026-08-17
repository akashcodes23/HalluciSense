# HalluciSense Phase 7B — Live Evaluation Integrity, Discrepancy Forensic Audit & Cross-Model Robustness

## 1. Executive Mission
The governing scientific directive is:
`SCIENCE > VISUAL POLISH | MEASURED > DERIVED | REPRODUCIBLE > IMPRESSIVE | HONEST PROVENANCE > FABRICATED COMPLETENESS`

Phase 7B conducts a rigorous forensic analysis explaining the performance shift between the offline canonical benchmark (Phase 6: AUROC 0.9260, Accuracy 84.67%) and the live LLM evaluation (Phase 7: AUROC 0.5602, Accuracy 57.33%).

---

## 2. Core Discovery: Benchmark Prompt Label Shift
The primary driver of the metric divergence is **Prompt-to-Generation Dynamic Label Shift**:
* **Phase 6 Canonical**: Evaluated against fixed static responses where 375 records contained pre-fabricated factual assertions ($GT=0$) and 375 contained deliberate synthetic hallucinations ($GT=1$). Pillar 1 Evidence Grounding attained $84.67\%$ accuracy because it directly verified those static claims.
* **Phase 7 Live Generation**: The live model (`qwen2.5-coder:1.5b`) was prompted with the 750 benchmark queries. For $254$ out of the $375$ prompts labeled as hallucinated in the static suite ($67.7\%$), the live LLM actually answered **correctly and truthfully**!
* **Verification Impact**: Pillar 1 correctly retrieved evidence and determined that Qwen's response was factually supported ($P_1 \approx 0.00$). However, when evaluated mechanically against the static benchmark label ($GT=1$), this correct behavior was penalized as $254$ False Negatives.
* **Grounding Concordance**: When evaluating solely against genuine factual errors in the generated responses, Pillar 1 demonstrates high specificity ($92.27\%$) and strong precision ($74.34\%$).

---

## 3. Directory Artifacts

| File | Purpose |
|---|---|
| [`phase6_vs_phase7_comparison.csv`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/phase6_vs_phase7_comparison.csv) | Paired $N=750$ per-sample comparison of inputs, scores, and deltas |
| [`alignment_audit.json`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/alignment_audit.json) | Cryptographic and label verification ($750$ matched, $0$ mismatched) |
| [`phase6_leakage_audit.csv`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/phase6_leakage_audit.csv) | Contamination and lexical overlap audit against reference corpus |
| [`response_distribution_comparison.csv`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/response_distribution_comparison.csv) | Token count, claim count, and lexical diversity distributions |
| [`p1_failure_analysis.csv`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/p1_failure_analysis.csv) | Forensic breakdown of P1 retrieval and NLI decisions |
| [`p3_failure_analysis.csv`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/p3_failure_analysis.csv) | Analysis of "Consistent Hallucinations" vs factual convergence |
| [`threshold_analysis.csv`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/threshold_analysis.csv) | 70% validation threshold sweep ($T=0.05 \dots 0.95$) |
| [`calibration_comparison.csv`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/calibration_comparison.csv) | Held-out 30% test calibration: Uncalibrated vs Platt vs Isotonic |
| [`provider_capability_matrix.json`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/provider_capability_matrix.json) | LLM provider logprob and streaming capability audit |
| [`domain_phase6_phase7_comparison.csv`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/domain_phase6_phase7_comparison.csv) | 15-domain performance shifts and degradation tracking |
| [`error_taxonomy.csv`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/error_taxonomy.csv) | Granular error taxonomy mapped to exact sample IDs |
| [`statistical_tests.json`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/statistical_tests.json) | Paired McNemar $\chi^2$, Wilcoxon signed-rank, and Cohen's $d$ |
| [`reproduction_manifest.json`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/reproduction_manifest.json) | Cryptographic environment, hash, and seed manifest |
| [`P2_PROVIDER_BLOCKER.md`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/P2_PROVIDER_BLOCKER.md) | Technical disclosure of provider logprob availability blocker |
| [`STATISTICAL_METHODS.md`](file:///Users/akashgpatil/major_project/backend/reports/phase7b/STATISTICAL_METHODS.md) | Formal documentation of statistical testing protocols |
