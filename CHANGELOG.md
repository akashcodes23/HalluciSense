# Changelog

All notable changes to **HalluciSense** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0-rc1] - 2026-08-05

### Added
- **Hybrid Multi-Pillar Inference Pipeline**: Combines Pillar 1 (External Evidence Grounding) and Pillar 2 (Intra-Model Self-Consistency Contradiction Matrix) using a 19-dimensional feature schema.
- **Phase 6M Hybrid Model**: Deserializes candidate 5 `HistGradientBoostingClassifier` with `RobustScaler` preprocessing.
- **Public Benchmark Dataset Registry**: Adapters for 12 public datasets (*HaluEval, TruthfulQA, FreshQA, FEVER, SciFact, HoVer, VitaminC, FActScore, BEGIN, XSumFaith, PubHealth, PubMedQA, MedQA*) across 15 domains ($N=750$ claims).
- **Statistical Validation Engine**: 10,000-sample non-parametric bootstrap 95% CIs, McNemar's test, DeLong ROC AUC test, Wilcoxon signed-rank test, Permutation test, Cohen's $d$, and Cliff's Delta.
- **Probability Recalibration Suite**: Platt Scaling sigmoidal recalibration reducing ECE from $0.1090 \to \mathbf{0.0257}$.
- **Multi-Format Publication Visualizations**: Export 300 DPI PNG, SVG, and PDF graphics (ROC, PR, Calibration, Confusion Matrix, Radar plots, 15-Domain breakdowns).
- **Enterprise Production Architecture**: Single-command reproducibility script `python run_all_experiments.py`, Docker multi-stage build, Railway deployment configuration, OpenTelemetry tracing, and structured JSON logging.
