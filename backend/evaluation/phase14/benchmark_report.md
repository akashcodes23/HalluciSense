# Phase 14 — Large Scale Multi-Domain Benchmark Report

## Overview
Evaluation performed across 15 domains (N=750 total samples) comparing HalluciSense against 8 baselines.

## Benchmark Performance Table

| Model | Accuracy | F1 Score | AUROC | AUPRC | MCC | ECE | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SelfCheckGPT** | 0.7360 | 0.7402 | 0.8182 | 0.8183 | 0.4722 | 0.0383 | 299.0 |
| **RAGAS** | 0.7853 | 0.7901 | 0.8702 | 0.8476 | 0.5713 | 0.0888 | 301.9 |
| **AlignScore** | 0.7867 | 0.7872 | 0.8687 | 0.8681 | 0.5733 | 0.0777 | 299.2 |
| **TRUE** | 0.7147 | 0.7177 | 0.7994 | 0.8084 | 0.4294 | 0.0368 | 302.7 |
| **FactScore** | 0.7573 | 0.7605 | 0.8636 | 0.8630 | 0.5148 | 0.0665 | 303.9 |
| **Pure Retrieval** | 0.6547 | 0.6523 | 0.7196 | 0.7089 | 0.3094 | 0.0491 | 297.9 |
| **Pure CrossEncoder** | 0.7053 | 0.7081 | 0.7722 | 0.7780 | 0.4107 | 0.0432 | 296.9 |
| **Pure NLI** | 0.7347 | 0.7364 | 0.8023 | 0.8031 | 0.4694 | 0.0388 | 299.8 |
| **HalluciSense** | 0.8760 | 0.8738 | 0.9501 | 0.9492 | 0.7525 | 0.1090 | 150.8 |

## Domain Breakdown Analysis

| Domain | Total Samples | Correct | False Positives | False Negatives | Domain Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **General Knowledge** | 50 | 45 | 3 | 2 | 90.00% |
| **Medicine** | 50 | 41 | 3 | 6 | 82.00% |
| **Law** | 50 | 41 | 4 | 5 | 82.00% |
| **Finance** | 50 | 46 | 2 | 2 | 92.00% |
| **Science** | 50 | 47 | 2 | 1 | 94.00% |
| **History** | 50 | 44 | 2 | 4 | 88.00% |
| **Computer Science** | 50 | 42 | 5 | 3 | 84.00% |
| **Mathematics** | 50 | 41 | 4 | 5 | 82.00% |
| **News** | 50 | 44 | 2 | 4 | 88.00% |
| **Geography** | 50 | 46 | 0 | 4 | 92.00% |
| **Politics** | 50 | 44 | 2 | 4 | 88.00% |
| **Biology** | 50 | 45 | 4 | 1 | 90.00% |
| **Chemistry** | 50 | 44 | 3 | 3 | 88.00% |
| **Physics** | 50 | 43 | 2 | 5 | 86.00% |
| **Literature** | 50 | 44 | 2 | 4 | 88.00% |
