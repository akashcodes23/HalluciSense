# Phase 9 — Step 6: Research Deliverables

**Generated**: 2026-08-03T04:48:46.758334+00:00

## Performance Summary (VAL — 3,500 samples)

| Metric | Pillar-1 | Single Feature | Random Chance |
| --- | --- | --- | --- |
| ROC-AUC | **0.6902** | 0.6303 | 0.5000 |
| PR-AUC (AP) | **0.6317** | 0.6004 | 0.4706 |
| F1 (@0.56) | **0.6618** | 0.6160 | 0.6400 |
| MCC (@0.56) | **0.3587** | 0.1284 | 0.0000 |
| Brier Score | **0.2332** | 0.3149 | 0.2491 |

## Figures Generated (300 DPI)

| File | Description |
| --- | --- |
| `step6_roc_curve.png` | ROC curve with operating point and baselines |
| `step6_pr_curve.png` | PR curve with operating point and no-skill baseline |
| `step6_confusion_matrix.png` | Normalized + raw confusion matrices |
| `step6_threshold_analysis.png` | F1/MCC/Precision/Recall vs threshold |
| `step6_coefficient_table.png` | Publication-quality coefficient table |
| `step6_statistical_comparison.png` | Statistical comparison table |
| `step6_roc_pr_combined.png` | Combined ROC + PR panel |
| `step6_metrics_summary.png` | Key metrics bar chart vs baseline |

## LaTeX Table

Coefficient table exported to `step6_coefficient_table.tex` for direct inclusion in IEEE manuscript.