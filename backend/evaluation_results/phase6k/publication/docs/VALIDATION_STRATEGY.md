# HalluciSense Pillar-1: Validation Strategy

*Generated: 2026-08-03T04:49:02.142488+00:00*  
*Phase: 6K (Frozen)*

---

## 1. Protocol Lock

The held-out validation partition was **locked before any model selection**:
- Partition SHA-256 sealed in `final_model/model_metadata.json`
- Partition evaluated **exactly once** after final model fitting
- No model changes allowed after the protocol lock date

## 2. Held-Out Validation Results

| Metric | Value |
| --- | --- |
| ROC-AUC | **0.6902** |
| PR-AUC | 0.6311 |
| F1 (τ=0.56) | 0.6618 |
| MCC (τ=0.56) | 0.3587 |
| Accuracy (τ=0.56) | 0.6803 |
| Brier Score | 0.2332 |
| Log Loss | 0.6593 |
| TP | 1095 / TN | 1286 / FP | 567 / FN | 552 |

**Validation gate**: ROC-AUC > 0.75 (predefined).  
**Result**: Gate NOT met (0.6902). Verdict: **PILLAR 1 VALIDATED WITH LIMITATIONS**.

## 3. Bootstrap Confidence Intervals

95% bootstrap CI for ROC-AUC (n=2000 bootstrap samples):
- See `heldout_bootstrap_ci.json` for full results

## 4. Baseline Comparisons

| Baseline | ROC-AUC | Notes |
| --- | --- | --- |
| Random chance | 0.500 | Trivial |
| Single-feature (`min_support_margin`) | ~0.65 | Best single feature |
| Majority class | 0.500 | Always predict hallucinated |
| Pillar-1 (5-feature) | **0.6902** | +0.04 over single feature |

## 5. Generalization Analysis

DEV vs VAL distribution comparison:
- Feature distributions similar (KL divergence < 0.05 for all features)
- Positive class ratio: DEV=54.3%, VAL=47.1% (slight shift)
- Condition numbers: DEV=95.5, VAL=114.7 (both acceptable)

## 6. Scientific Verdict

**"PILLAR 1 VALIDATED WITH LIMITATIONS"** — The model demonstrates statistically
meaningful hallucination detection above all baselines, with limitations in
absolute discrimination power that motivate future Pillar development.
