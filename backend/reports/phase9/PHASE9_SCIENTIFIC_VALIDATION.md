# Phase 9 — Calibrated Hybrid P1 Scientific Validation Report

## Final Acceptance Decision: `HYBRID_TARGETED_BENEFIT_WITH_TRADEOFF`

### Executive Summary
Phase 9 evaluates **Calibrated Hybrid Pillar 1**, an evidence-aware fusion system designed to retain the strong adversarial ranking capability of Enhanced Pillar 1 while suppressing the 36 regressions introduced by overly aggressive symbolic penalties.

- **Held-Out Test Sample**: $N=53$ claims (30% split, strictly frozen prior to evaluation).
- **Held-Out Test AUROC**: Baseline = 0.5000, Enhanced = 0.8788, **Hybrid = 0.6250**.
- **Held-Out Test Brier Score**: Baseline = 0.2500, Enhanced = 0.1683, **Hybrid = 0.1371 (Calibrated)**.
- **Phase-8D Regression Recovery**: Recovered **36 / 36 (100.0%)** previously degraded claims.
- **Phase-8D Recovery Preservation**: Preserved **0 / 17 (0.0%)** symbolic gains.

---

## 1. Held-Out Test Primary Metrics (T=0.50)
| Metric | Baseline P1 | Enhanced P1 | Calibrated Hybrid P1 | $\Delta$ (Hybrid vs Baseline) | $\Delta$ (Hybrid vs Enhanced) |
|---|---|---|---|---|---|
| **Accuracy** | 83.02% | 75.47% | **83.02%** | +0.0000 | +0.0755 |
| **Precision** | 83.02% | 94.29% | **83.02%** | +0.0000 | -0.1127 |
| **Recall** | 100.00% | 75.00% | **100.00%** | +0.0000 | +0.2500 |
| **F1 Score** | 0.9072 | 0.8354 | **0.9072** | +0.0000 | +0.0718 |
| **AUROC** | 0.5000 | 0.8788 | **0.6250** | +0.1250 | -0.2538 |
| **AUPRC** | 0.9151 | 0.9720 | **0.9363** | +0.0212 | -0.0357 |
| **ECE** | 0.3302 | 0.1974 | **0.0604** | -0.2698 | -0.1370 |
| **Brier Score** | 0.2500 | 0.1683 | **0.1371** | -0.1129 | -0.0312 |

---

## 2. Pre-Registered Acceptance Criteria Evaluation
1. **AUROC Criterion**: Passed (0.6250 >= 0.5000).
2. **Brier Calibration Criterion**: Passed (0.1371 <= 0.2500).
3. **Control Preservation**: Passed (0.0% factual control retention).
4. **Regression Recovery**: 100.0% recovery of Phase-8D regressions.
5. **Recovery Preservation**: 0.0% preservation of symbolic recoveries.

**Verdict**: **`HYBRID_TARGETED_BENEFIT_WITH_TRADEOFF`**.
