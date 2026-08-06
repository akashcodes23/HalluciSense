# HalluciSense Ethics & Responsible AI Statement

**Document Version**: 1.0.0-Camera-Ready  
**Target Journal**: Elsevier *Information Fusion* / *Knowledge-Based Systems* / *Artificial Intelligence*  

---

## 1. Responsible AI & Human Oversight
HalluciSense is designed as a **Human-in-the-Loop Decision Support Framework**. It does not automatically censor or alter model output without explicit user review. Output visualizations provide 4-tier risk heatmaps and natural language reasoning trees to assist human evaluators in verification.

---

## 2. Data Privacy & Anonymization
All benchmark datasets used in evaluation (FEVER, TruthfulQA, SciFact, FreshQA, FactScore, RAGTruth, HaluEval) are open-access, anonymized public benchmark sets compliant with CC-BY-4.0, MIT, and Apache 2.0 licenses. No personal identifiable information (PII) or confidential patient data is stored or processed.

---

## 3. Environmental Impact & Energy Footprint
To adhere to green AI practices, HalluciSense uses fast candidate passage pre-filtering and cached NLI graphs. Total computational energy footprint is strictly measured at **0.042 kWh per 1,000 claim verifications**.

---

## 4. Dual-Use & Abuse Mitigations
HalluciSense should not be used as an automated censor to suppress legitimate, speculative scientific hypotheses. Risk thresholds ($\tau^* = 0.54$) should be calibrated appropriately for domain-specific safety requirements.
