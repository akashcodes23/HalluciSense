# HalluciSense Phase 7B — Scientific Claims Audit

| Claim / Finding | Empirical Evidence | Status |
|---|---|---|
| *"Phase 6 and Phase 7 were joined with 100% sample alignment"* | Verified in `alignment_audit.json` (750 matched, 0 mismatches) | **VERIFIED** |
| *"Phase 6 factual responses exhibit higher lexical overlap with evidence than live generations"* | Overlap 0.72 (Phase 6) vs 0.41 (Phase 7) in `phase6_leakage_summary.json` | **VERIFIED** |
| *"P1 discrepancy is primarily driven by Qwen generating factual answers to hallucinated prompts"* | Documented in `p1_failure_analysis.csv` (254 / 375 cases) | **VERIFIED** |
| *"P3 self-consistency increases live evaluation precision from 64.17% to 74.34%"* | Measured in `ablation_comparison.csv` and `metrics.json` | **VERIFIED** |
| *"P2 Confidence was honestly marked unavailable for the local Ollama endpoint"* | Documented in `P2_PROVIDER_BLOCKER.md`; zero synthetic logprobs | **VERIFIED** |
| *"Threshold sweep on held-out validation identified optimal T = 0.35"* | Evaluated on 70% validation split in `threshold_analysis.csv` | **VERIFIED** |
| *"Phase 6 and Phase 7 artifacts remain untouched"* | Verified by cryptographic hash comparison in `reproduction_manifest.json` | **VERIFIED** |
