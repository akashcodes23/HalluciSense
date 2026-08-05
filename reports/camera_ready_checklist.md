# Camera-Ready Publication Submission Checklist (Phase 23)

## Conference Target: IEEE / ACL / EMNLP / NAACL / COLING

### 1. Paper Format & Formatting
- [x] Template: IEEEtran / ACL 2-column LaTeX format (`paper/paper.tex`).
- [x] Abstract: Compliant with 200-word limit.
- [x] Figures: High-resolution vector PDF, SVG, and 300 DPI PNG graphics embedded in `paper/figures/`.
- [x] Tables: Formatted using `booktabs` (`\toprule`, `\midrule`, `\bottomrule`).

### 2. Experimental Verification
- [x] All 21 metrics computed directly from actual predictions in `evaluation/results/predictions.csv`.
- [x] 10,000 Bootstrap 95% CIs reported for AUROC, F1 Score, and MCC.
- [x] Hypothesis testing ($p < 0.001$ McNemar, DeLong, Wilcoxon, Permutation test) and Cohen's $d / Cliff's Delta effect sizes verified.

### 3. Open Source & Reproducibility
- [x] Single-command master script `python run_all_experiments.py` verified.
- [x] `MODEL_CARD.md` and `DATASET_CARD.md` available in repository root.
- [x] Environment reproducibility manifest `experiment_config.json` and `environment.yaml` generated.
- [x] CITATION.cff and LICENSE attached.
