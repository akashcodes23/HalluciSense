# HalluciSense Model Card — Candidate 5 Hybrid Model

## Model Overview
- **Architecture**: `HistGradientBoostingClassifier(max_iter=100, max_depth=4)`
- **Preprocessing**: `RobustScaler`
- **Feature Dimension**: 19 hybrid features (`SET_A_FULL_HYBRID`)
- **Operating Threshold**: $\tau^* = 0.54$

## Performance Summary
- **DEV 5-Fold 3-Repeat CV**: $\text{ROC-AUC} = 0.7267$, $\text{MCC} = 0.3370$, $\text{ECE} = 0.0066$
- **Held-Out VAL ($N=12,483$)**: $\text{ROC-AUC} = 0.6558$, $\text{PR-AUC} = 0.6733$, $\text{MCC} = 0.1945$
- **Statistical Superiority**: Outperforms Pillar 1 alone by $+0.0299$ ROC-AUC ($p < 10^{-15}$).
