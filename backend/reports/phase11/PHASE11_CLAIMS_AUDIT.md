# Phase 11 Claims Audit

| Statement | Classification | Empirical Basis |
|---|---|---|
| Closed-loop chat repairs 100% of evaluated numerical and unit errors | MEASURED | N=30 acceptance suite in `phase11_results.csv` |
| Re-verification gate enforces zero unverified repairs released | MEASURED | 100% re-verification pass rate in `phase11_summary.json` |
| System preserves 100% of true scientific assertions | MEASURED | 0.0% false correction rate across true controls |
| Response latency remains under 250ms for complete closed loop | MEASURED | Real wall-clock timing: mean=118.34ms, P95=132.36ms |
| Closed-loop chat provides clinical diagnostic advice | LIMITATION | Scientific assertions only; does not replace medical professional judgment |
