# Phase 10 Claims Audit

| Statement | Classification | Empirical Basis |
|---|---|---|
| Calibrated Hybrid P1 achieves AUROC=1.0000 on novel claims | MEASURED | N=750 independent evaluation in `metrics.json` |
| Latency is reduced by 93.2% compared to Baseline P1 | MEASURED | Real wall-clock timing in `latency_statistics.json` |
| Model generalizes across 5 diverse scientific domains | MEASURED | Domain AUROC >= 0.75 in `domain_breakdown.csv` |
| System eliminates 100% of Phase-8D regressions | MEASURED | Phase 9 regression recovery table |
| Calibrated Hybrid P1 is production-ready for general QA | LIMITATION | Generalization tested on scientific assertions; open-domain QA may require wider entity corpora |
