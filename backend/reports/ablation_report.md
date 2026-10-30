# Phase 22.8 — 9-Variant Component Ablation Study Report

## Component Ablation Results

| Configuration Variant | Accuracy | F1 Score | AUROC | AUPRC | MCC | Delta AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Model (Production)** | 0.9787 | 0.9786 | 0.9988 | 0.9988 | 0.9574 | Base |
| No CrossEncoder | 0.9067 | 0.9077 | 0.9676 | 0.9703 | 0.8135 | -0.0312 |
| No NLI | 0.8587 | 0.8594 | 0.9484 | 0.9507 | 0.7174 | -0.0504 |
| No Hybrid Fusion | 0.8093 | 0.8091 | 0.8926 | 0.8920 | 0.6187 | -0.1062 |
| No Graph | 0.9627 | 0.9625 | 0.9947 | 0.9949 | 0.9254 | -0.0041 |
| No Calibration | 0.9733 | 0.9733 | 0.9965 | 0.9967 | 0.9467 | -0.0023 |
| No Claim Extraction | 0.8787 | 0.8775 | 0.9517 | 0.9509 | 0.7575 | -0.0471 |
| No Evidence Ranking | 0.9253 | 0.9255 | 0.9794 | 0.9791 | 0.8507 | -0.0194 |
| No SHAP | 0.9733 | 0.9730 | 0.9988 | 0.9988 | 0.9469 | +0.0000 |
