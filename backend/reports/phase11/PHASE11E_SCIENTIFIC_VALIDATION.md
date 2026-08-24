# Phase 11E — Scientific Validation & Benchmark Report

## 1. Domain Performance Breakdown

| Model Candidate | Physics AUROC | Chemistry AUROC | Biology AUROC | Medicine AUROC | Mathematics AUROC |
|---|:---:|:---:|:---:|:---:|:---:|
| **DeBERTa-v3-Small (Control)** | `0.6132` | `0.6047` | `0.7692` | `0.8707` | `0.5737` |
| **DeBERTa-v3-XSmall** | `0.7853` | `0.61` | `0.7329` | `0.7179` | `0.6688` |
| **DistilRoBERTa-NLI** | `0.6191` | `0.61` | `0.5972` | `0.7628` | `0.5705` |
| **DistilBERT-MNLI** | `0.4979` | `0.5801` | `0.6378` | `0.7382` | `0.4728` |

---

## 2. Adversarial Evaluation Summary

| Model Candidate | Phase 8A AUROC | Phase 8A F1 | Phase 8A Accuracy |
|---|:---:|:---:|:---:|
| **DeBERTa-v3-Small (Control)** | `0.8288` | `0.8958` | `0.8286` |
| **DeBERTa-v3-XSmall** | `0.8717` | `0.9072` | `0.8457` |
| **DistilRoBERTa-NLI** | `0.7604` | `0.8611` | `0.7714` |
| **DistilBERT-MNLI** | `0.8047` | `0.8837` | `0.8` |
