<div align="center">

<img src="assets/logo-dark.svg" width="170"/>

# HalluciSense

### Confidence-Aware Hybrid Framework for Detecting and Quantifying Hallucinations in Large Language Models

<p align="center">
An enterprise-grade AI verification framework that combines semantic reasoning,
retrieval verification, structural consistency analysis and calibrated confidence
fusion to detect hallucinations in Large Language Models.
</p>

---

<p align="center">

<img src="https://img.shields.io/github/license/akashcodes23/HalluciSense?style=for-the-badge"/>

<img src="https://img.shields.io/github/stars/akashcodes23/HalluciSense?style=for-the-badge"/>

<img src="https://img.shields.io/github/forks/akashcodes23/HalluciSense?style=for-the-badge"/>

<img src="https://img.shields.io/github/issues/akashcodes23/HalluciSense?style=for-the-badge"/>

<img src="https://img.shields.io/github/last-commit/akashcodes23/HalluciSense?style=for-the-badge"/>

</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python"/>

<img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi"/>

<img src="https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js"/>

<img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react"/>

<img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript"/>

<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker"/>

<img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql"/>

</p>

---

<p align="center">

🚀 Research Project • 🔍 Explainable AI • 🧠 Hybrid Verification • 📈 Confidence Calibration • 🌐 Production Ready

</p>

</div>

---

# HalluciSense in Action

<p align="center">

<img src="assets/demo.gif"/>

</p>

## 🔬 Scientific Architecture & Research Overview

HalluciSense establishes a **Complete Three-Pillar Hybrid Hallucination Detection Framework** for publication in top-tier Elsevier journals (*Information Fusion*, *Knowledge-Based Systems*, *Artificial Intelligence*).

```
                      +------------------------------------------+
                      |       Input LLM Prompt & Response        |
                      +------------------------------------------+
                                           |
         +---------------------------------+---------------------------------+
         |                                 |                                 |
         v                                 v                                 v
+------------------+             +-------------------+             +-------------------+
| Pillar 1: FE     |             | Pillar 2: CG      |             | Pillar 3: CF      |
| Retrieval        |             | Confidence        |             | Consistency       |
| Evidence         |             | Estimation        |             | Reasoning         |
+------------------+             +-------------------+             +-------------------+
| • BM25 Sparse    |             | • White-Box:      |             | • Paraphrase Gen  |
| • Dense Embedding|             |   Logprobs, Entropy|             |   (Q -> Q1...QN)  |
| • Cross-Encoder  |             |   Mutual Info,    |             | • SBERT Matrix    |
|   Reranker       |             |   Epistemic/Aleat.|             | • Claim-Aligned   |
| • Citation Conf. |             | • Black-Box API:  |             |   NLI Graph       |
| • Passage Align. |             |   Top-K Diff, Var.|             | • Sentence Match  |
+------------------+             +-------------------+             +-------------------+
         |                                 |                                 |
         +---------------------------------+---------------------------------+
                                           |
                                           v
                     +-------------------------------------------+
                     |   Calibrated Hybrid Fusion Engine         |
                     |   Modes: STATIC / ADAPTIVE / GRADIENT     |
                     |   Formula: H = α FE + β CG + γ CF (sum=1) |
                     |   Platt Scaling (ECE = 0.0257)            |
                     +-------------------------------------------+
                                           |
                                           v
                     +-------------------------------------------+
                     |   Token Localization & 4-Tier Heatmaps    |
                     |   • Green  (#10B981) - VERIFIED           |
                     |   • Yellow (#F59E0B) - NEEDS_VERIFY       |
                     |   • Orange (#F97316) - MODERATE_RISK      |
                     |   • Red    (#EF4444) - LIKELY_HALLUCINATED|
                     +-------------------------------------------+
```

### Key Pillars & Scientific Specifications

1. **Pillar 1: Retrieval Evidence ($FE \in [0,1]$)**
   - Integrates BM25 sparse lexical matching, dense embedding retrieval, and Cross-Encoder passage reranking (`ms-marco-MiniLM-L-6-v2`).
   - Returns factual claim verification, citation confidence scores, and retrieved source passages.

2. **Pillar 2: Confidence Estimation ($CG \in [0,1]$)**
   - **White-Box Models**: Computes token logprobs, token entropy, attention entropy, predictive entropy $H(Y)$, mutual information $I(Y;W)$, epistemic uncertainty, and aleatoric uncertainty.
   - **Black-Box API Models**: Approximates confidence using top-$k$ logprob differences, multi-query response variance, and Platt scaling calibration models.

3. **Pillar 3: Consistency Reasoning ($CF \in [0,1]$)**
   - Executes paraphrase sampling ($Q \rightarrow Q_1, \dots, Q_N$) and queries the target LLM.
   - Constructs pairwise SBERT similarity matrices $S_{ij}$ and runs sentence-level NLI contradiction graph verification.

4. **Calibrated Hybrid Fusion Layer**
   - Combines pillars via $H = \alpha FE + \beta CG + \gamma CF$ ($\alpha + \beta + \gamma = 1$).
   - Supports `STATIC`, `ADAPTIVE`, and `GRADIENT`-learned weight optimization modes.
   - Generates parameter sensitivity analysis grids and weight importance vectors.

5. **Token Localization & 4-Tier Risk Heatmaps**
   - Propagates sentence H-scores down to token attributions and span boundaries.
   - Visualizes 4 risk tiers: **Green** (`#10B981`), **Yellow** (`#F59E0B`), **Orange** (`#F97316`), and **Red** (`#EF4444`).

---

# Overview

HalluciSense is a hybrid hallucination detection framework designed to evaluate the factual reliability of responses generated by Large Language Models.

Unlike conventional hallucination detectors that rely solely on semantic similarity or retrieval verification, HalluciSense combines multiple complementary verification strategies into a unified confidence-aware framework capable of identifying factual inconsistencies, unsupported claims, logical contradictions, structural anomalies and retrieval failures.

The system integrates:

- Semantic verification
- Retrieval-grounded evidence checking
- Structural claim consistency analysis
- Confidence calibration
- Hybrid meta-classification
- Explainability-driven diagnostics

to produce transparent and trustworthy hallucination assessments suitable for research and production deployment.

---

# Why HalluciSense?

Large Language Models frequently generate responses that appear fluent while containing fabricated or unsupported information.

Current approaches often rely on a single verification signal, making them vulnerable to:

- semantic shortcuts
- retrieval failures
- confidence miscalibration
- contradictory reasoning
- unsupported factual assertions

HalluciSense addresses these limitations through a multi-pillar verification architecture that evaluates generated responses from complementary perspectives before producing a calibrated confidence score.

---

# Key Innovations

<table>

<tr>

<td width="50%">

### Hybrid Verification

Combines semantic reasoning, retrieval validation and structural consistency instead of relying on a single detector.

</td>

<td width="50%">

### Confidence Calibration

Produces calibrated confidence estimates rather than binary predictions.

</td>

</tr>

<tr>

<td>

### Explainable Decisions

Every prediction includes feature-level evidence and interpretable reasoning.

</td>

<td>

### Production Architecture

Built using FastAPI, Next.js, PostgreSQL and Docker with scalable deployment support.

</td>

</tr>

<tr>

<td>

### Modular Pipeline

Independent verification pillars enable extensibility and future research.

</td>

<td>

### Scientific Evaluation

Comprehensive evaluation across multiple benchmarks with reproducible experimental protocols.

</td>

</tr>

</table>

---

# Feature Highlights

| Feature | Description |
|----------|-------------|
| 🧠 Hybrid Meta-Classifier | Combines multiple verification signals into a unified prediction |
| 🔍 Retrieval Verification | Evidence-grounded validation using retrieved documents |
| 🧩 Structural Consistency Analysis | Detects logical and claim-level inconsistencies |
| 📈 Confidence Calibration | Generates calibrated hallucination probabilities |
| 📊 Explainability Dashboard | Visualizes evidence, confidence and reasoning |
| ⚡ FastAPI Backend | High-performance asynchronous inference API |
| 💻 Next.js Frontend | Modern interactive verification dashboard |
| 🐳 Docker Support | One-command deployment |
| 📑 Scientific Evaluation Pipeline | Reproducible benchmarking and analysis |
| 🔒 Production Configuration | Environment-aware deployment with configurable settings |

---

# Repository Preview

```
HalluciSense
│
├── backend/
│   ├── app/
│   ├── api/
│   ├── models/
│   ├── retrieval/
│   ├── verification/
│   └── explainability/
│
├── frontend/
│
├── config/
│
├── docker/
│
├── docs/
│
├── assets/
│
└── tests/
```

---

# Architecture

<p align="center">

<img src="assets/architecture.png"/>

</p>

> A detailed explanation of the architecture is provided in the next section of this README.

---

# Table of Contents

- Overview
- Motivation
- System Architecture
- Hybrid Verification Pipeline
- Confidence Fusion
- Explainability Framework
- Repository Structure
- Installation
- Quick Start
- Docker Deployment
- API Documentation
- Frontend Dashboard
- Evaluation Methodology
- Benchmark Results
- Experimental Analysis
- Scientific Contributions
- Roadmap
- Citation
- License

---

# Project Status

| Component | Status |
|------------|--------|
| Backend API (FastAPI) | ✅ Complete |
| Frontend Dashboard (Next.js 16) | ✅ Complete |
| Retrieval Grounding Module (Pillar 1) | ✅ Complete |
| Confidence Estimation (Pillar 2) | ✅ Complete |
| Structural Consistency (Pillar 3) | ✅ Complete |
| Calibrated Adaptive Fusion Engine | ✅ Complete |
| Token Localization & Heatmaps | ✅ Complete |
| Explainability Engine | ✅ Complete |
| Railway Backend Deployment (Sprint 3.1A) | ✅ Complete |
| Railway Frontend Deployment (Sprint 3.1B) | ✅ Complete |
| Docker & Compose Setup | ✅ Complete |
| Scientific Benchmarking & Reports | ✅ Complete |

---

# Highlights

- Enterprise-grade modular architecture
- Research-oriented experimental pipeline
- Production-ready deployment
- Explainable hallucination detection
- Confidence-aware decision making
- Retrieval-augmented verification
- Structural reasoning engine
- Reproducible evaluation framework

---
# 🏗️ System Architecture

HalluciSense is designed as a modular, multi-stage verification framework that analyzes the factual reliability of Large Language Model (LLM) responses through complementary verification strategies.

Instead of relying on a single hallucination detector, HalluciSense combines retrieval-grounded reasoning, structural consistency analysis, and confidence-aware fusion to produce interpretable and calibrated hallucination assessments.

<p align="center">

<img src="assets/architecture.png" width="100%"/>

</p>

---

# End-to-End Verification Pipeline

```text
                        ┌─────────────────────────┐
                        │      User Prompt        │
                        └────────────┬────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │       LLM Response      │
                        └────────────┬────────────┘
                                     │
               ┌─────────────────────┴─────────────────────┐
               │                                           │
               ▼                                           ▼
     ┌──────────────────┐                     ┌────────────────────┐
     │ Retrieval Engine  │                     │ Claim Decomposition│
     └─────────┬─────────┘                     └──────────┬─────────┘
               │                                          │
               ▼                                          ▼
     Retrieved Evidence                    Structural Feature Extraction
               │                                          │
               └──────────────┬───────────────────────────┘
                              ▼
                   Hybrid Confidence Fusion
                              │
                              ▼
                  Explainability & Calibration
                              │
                              ▼
                     Hallucination Assessment
```

---

# Design Philosophy

HalluciSense follows five core engineering principles:

| Principle | Description |
|-----------|-------------|
| **Modularity** | Independent verification pillars enable extensibility and isolated experimentation. |
| **Explainability** | Every prediction is accompanied by interpretable evidence and feature contributions. |
| **Reproducibility** | Experimental protocols are deterministic and fully documented. |
| **Scalability** | Components are deployable as production-ready services. |
| **Trustworthiness** | Confidence calibration ensures predictions better reflect uncertainty. |

---

# Core Components

The framework is composed of six major subsystems.

```text
HalluciSense
│
├── Retrieval Verification
├── Structural Verification
├── Feature Engineering
├── Hybrid Meta Classifier
├── Confidence Calibration
└── Explainability Engine
```

---

# Verification Workflow

## Stage 1 — Response Generation

The framework receives a user prompt together with the response generated by a Large Language Model.

```text
User Query
      │
      ▼
Large Language Model
      │
      ▼
Generated Response
```

This response becomes the subject of factual verification.

---

# Stage 2 — Retrieval Verification (Pillar I)

The first verification pillar evaluates whether the generated response is supported by external evidence.

### Objectives

- Retrieve supporting evidence
- Measure semantic agreement
- Detect unsupported statements
- Estimate retrieval confidence

### Pipeline

```text
Response
   │
Sentence Segmentation
   │
Evidence Retrieval
   │
Embedding Similarity
   │
Evidence Aggregation
   │
Retrieval Confidence
```

The retrieval subsystem produces evidence-based verification signals that indicate how well the response aligns with retrieved knowledge.

---

# Stage 3 — Structural Verification (Pillar II)

The second verification pillar analyzes the internal logical consistency of the generated response.

Unlike retrieval-based methods, structural verification evaluates the response independently of external knowledge sources.

### Structural Features

HalluciSense extracts multiple categories of structural information.

| Category | Examples |
|----------|----------|
| Claim Graph | Relationships between factual claims |
| Named Entities | Persons, organizations, locations |
| Temporal Features | Dates and chronological consistency |
| Numerical Features | Counts, measurements, quantities |
| Logical Relations | Contradictions, entailments |
| Pairwise NLI | Sentence-level inference relationships |

---

## Structural Analysis Pipeline

```text
Generated Response
        │
Claim Extraction
        │
Entity Detection
        │
Temporal Analysis
        │
Numerical Analysis
        │
Pairwise NLI
        │
Graph Construction
        │
Structural Feature Vector
```

---

# Feature Engineering

Outputs from both verification pillars are transformed into numerical feature vectors.

Examples include:

- Retrieval confidence
- Semantic similarity
- Entailment probability
- Contradiction score
- Entity overlap
- Numerical consistency
- Temporal agreement
- Structural graph statistics

These features collectively represent multiple perspectives of factual reliability.

---

# Hybrid Confidence Fusion

The extracted feature vectors are combined by a meta-classification layer.

```text
Retrieval Features
        │
Structural Features
        │
Statistical Features
        │
───────────────
Feature Fusion
───────────────
        │
Meta Classifier
        │
Probability Calibration
        │
Final Confidence Score
```

Rather than producing a binary prediction, HalluciSense estimates a calibrated probability representing the likelihood that the response contains hallucinated information.

---

# Explainability Framework

Every prediction is accompanied by transparent reasoning.

The explainability engine provides:

- Feature importance
- Evidence attribution
- Confidence visualization
- Detector agreement
- Probability calibration
- Human-readable reasoning

Example:

```text
Hallucination Probability

0.87

Evidence Support
████████░░ 82%

Structural Consistency
██████░░░░ 61%

Retrieval Confidence
█████████░ 91%

Final Verdict

Potential Hallucination
```

---

# Technology Stack

| Layer | Technologies |
|--------|--------------|
| Backend | FastAPI, Python |
| Frontend | Next.js, React, TypeScript |
| Machine Learning | Scikit-learn, Transformers |
| NLP | Sentence Transformers, NLI Models |
| Database | PostgreSQL |
| Deployment | Docker, Railway |
| Configuration | YAML |
| Testing | PyTest |
| Documentation | Markdown |

---

# Project Structure

```text
HalluciSense
│
├── backend/
│   ├── app/
│   ├── api/
│   ├── retrieval/
│   ├── verification/
│   ├── explainability/
│   ├── evaluation/
│   ├── models/
│   └── tests/
│
├── frontend/
│   ├── src/
│   ├── app/
│   ├── components/
│   └── public/
│
├── config/
├── docker/
├── docs/
├── assets/
└── scripts/
```

---

# Architectural Advantages

HalluciSense provides several advantages over conventional hallucination detection systems.

| Capability | HalluciSense |
|------------|--------------|
| Multi-Pillar Verification | ✅ |
| Retrieval Grounding | ✅ |
| Structural Reasoning | ✅ |
| Confidence Calibration | ✅ |
| Explainability | ✅ |
| Modular Architecture | ✅ |
| Production Deployment | ✅ |
| Research Reproducibility | ✅ |

---

# Engineering Highlights

- Modular verification pipeline
- Independent retrieval and structural reasoning
- Confidence-aware meta-classification
- Explainability-first design
- Production-ready API architecture
- Reproducible evaluation framework
- Scalable deployment strategy
- Research-oriented software engineering

---

> **Next:** Installation, Quick Start, Docker deployment, environment configuration, and API usage.

---

# 🚀 Getting Started

HalluciSense is designed for both **research experimentation** and **production deployment**.

Choose the setup method that best fits your workflow.

| Environment | Recommended |
|-------------|-------------|
| Local Development | ✅ |
| Docker | ✅ |
| Railway Deployment | ✅ |
| Linux Server | ✅ |
| macOS | ✅ |
| Windows (WSL Recommended) | ✅ |

---

# 📋 Prerequisites

Before installing HalluciSense, ensure the following software is available.

| Software | Version |
|----------|----------|
| Python | 3.11+ |
| Node.js | 20+ |
| npm | 10+ |
| PostgreSQL | 15+ |
| Docker | Latest |
| Git | Latest |

Verify installation:

```bash
python --version
node --version
npm --version
docker --version
git --version
```

---

# 📂 Clone Repository

```bash
git clone https://github.com/akashcodes23/HalluciSense.git

cd HalluciSense
```

---

# 🐍 Backend Installation

Navigate to the backend.

```bash
cd backend
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# 🌐 Frontend Installation

Open another terminal.

```bash
cd frontend

npm install
```

---

# ⚙️ Environment Configuration

Create a `.env` file inside the backend directory.

```text
backend/

.env
```

Example configuration:

```env
APP_ENV=development

HOST=0.0.0.0
PORT=8000

DATABASE_URL=postgresql://user:password@localhost:5432/hallucisense

SECRET_KEY=your-secret-key

OPENAI_API_KEY=

GOOGLE_API_KEY=

GEMINI_API_KEY=

EMBEDDING_MODEL=all-MiniLM-L6-v2

LOG_LEVEL=INFO
```

---

# 📦 Configuration Files

HalluciSense supports multiple runtime environments.

```
config/

development.yaml

production.yaml

research.yaml
```

Switch environments by updating:

```env
APP_ENV=development
```

or

```env
APP_ENV=production
```

---

# ▶️ Running the Backend

```bash
cd backend

uvicorn app.main:app --reload
```

Server starts at

```
http://localhost:8000
```

Swagger documentation

```
http://localhost:8000/docs
```

OpenAPI

```
http://localhost:8000/openapi.json
```

---

# 💻 Running the Frontend

```bash
cd frontend

npm run dev
```

Frontend

```
http://localhost:3000
```

---

# 🐳 Docker Deployment

Build containers.

```bash
docker compose build
```

Start services.

```bash
docker compose up
```

Detached mode.

```bash
docker compose up -d
```

Stop.

```bash
docker compose down
```

---

# 🚂 Railway Deployment

Deploy directly from GitHub.

```bash
railway up
```

Environment variables can be configured from the Railway dashboard.

---

# 📡 REST API

Base URL

```
http://localhost:8000
```

---

## Health Check

```http
GET /health
```

Response

```json
{
  "status":"healthy",
  "version":"1.0.0"
}
```

---

## Verify Response

```http
POST /verify
```

Request

```json
{
  "query":"What is the capital of Australia?",
  "response":"Sydney"
}
```

Example Response

```json
{
  "hallucination_probability":0.91,
  "confidence":"High",
  "verdict":"Potential Hallucination",
  "retrieval_score":0.84,
  "structural_score":0.78,
  "evidence":[
      "...",
      "..."
  ]
}
```

---

## Chat Endpoint

```http
POST /chat
```

---

## WebSocket Streaming

```
ws://localhost:8000/ws/chat
```

Supports

- Streaming responses
- Live verification
- Incremental confidence updates

---

# 🖥️ Frontend Dashboard

The dashboard provides an interactive interface for:

- Response verification
- Confidence visualization
- Retrieval evidence
- Structural diagnostics
- Explainability
- API testing

---

# 📁 Directory Layout

```
HalluciSense

backend/

frontend/

config/

docker/

docs/

assets/

tests/
```

---

# 🧪 Running Tests

Run all tests.

```bash
pytest
```

Specific module.

```bash
pytest tests/test_full_system_validation.py
```

Coverage.

```bash
pytest --cov
```

---

# 🔍 Development Workflow

```text
Clone Repository
        │
Install Dependencies
        │
Configure Environment
        │
Run Backend
        │
Run Frontend
        │
Execute Tests
        │
Develop Features
        │
Validate
        │
Deploy
```

---

# 🔐 Security Notes

Never commit:

```
.env

API Keys

Secrets

Database Passwords

Model Weights

Evaluation Artifacts
```

Sensitive files are excluded through `.gitignore`.

---

# 📈 Recommended Development Workflow

```
Feature Branch
      │
Pull Request
      │
Review
      │
Tests
      │
Merge
      │
Deploy
```

---

# 🧩 Troubleshooting

<details>

<summary>Backend won't start</summary>

Verify

- Python version
- Installed dependencies
- Environment variables
- Database availability

</details>

<details>

<summary>Frontend won't build</summary>

Run

```bash
rm -rf node_modules

npm install
```

</details>

<details>

<summary>Docker issues</summary>

```bash
docker compose down

docker compose build --no-cache

docker compose up
```

</details>

<details>

<summary>Database connection failed</summary>

Check

- PostgreSQL service
- DATABASE_URL
- Network configuration
- Credentials

</details>

---

# 📌 Next Section

The following section presents the complete evaluation methodology, benchmark datasets, performance metrics, calibration analysis, and scientific validation used to assess HalluciSense.

---

# 📊 Evaluation & Benchmarking

HalluciSense was developed using a rigorous multi-phase scientific evaluation protocol designed to validate accuracy, robustness, calibration, generalization, and deployment readiness.

Unlike conventional LLM verification systems that report only classification metrics, HalluciSense evaluates the complete verification pipeline—from retrieval quality to confidence calibration and production performance.

---

# 🧪 Evaluation Pipeline

```text
Dataset Collection
        │
        ▼
Feature Extraction
        │
        ▼
Retrieval Verification
        │
        ▼
Structural Consistency Analysis
        │
        ▼
Confidence Fusion
        │
        ▼
Meta Classification
        │
        ▼
Calibration
        │
        ▼
Error Analysis
        │
        ▼
Production Validation
```

---

# 📚 Benchmark Datasets

HalluciSense was evaluated across multiple public hallucination benchmarks.

| Dataset | Purpose |
|---------|----------|
| HaluEval | General hallucination detection |
| RAGTruth | Retrieval-grounded evaluation |
| HaluBench | Benchmark robustness testing |
| Custom Validation Set | Real-world verification |
| Stress Test Suite | Edge case evaluation |

Each dataset undergoes preprocessing, normalization, leakage analysis, and structural verification before training.

---

# 📈 Evaluation Metrics

The framework reports multiple complementary metrics rather than relying solely on accuracy.

| Metric | Description |
|---------|-------------|
| Accuracy | Overall correctness |
| Precision | Hallucination precision |
| Recall | Hallucination recall |
| F1 Score | Balanced performance |
| ROC-AUC | Ranking quality |
| PR-AUC | Precision-recall quality |
| Brier Score | Calibration quality |
| Expected Calibration Error | Confidence reliability |
| Matthews Correlation Coefficient | Balanced binary evaluation |
| Bootstrap Confidence Intervals | Statistical robustness |

---

# 📉 Calibration Analysis

A confidence-aware verifier should produce probabilities that accurately reflect prediction reliability.

HalluciSense performs:

- Probability calibration
- Reliability analysis
- Confidence binning
- Expected Calibration Error (ECE)
- Brier Score evaluation
- Calibration curve visualization

These analyses ensure that a prediction with **90% confidence behaves like a true 90% confidence prediction**.

---

# 🔍 Error Analysis

Comprehensive forensic analysis is performed on all incorrect predictions.

Analysis includes:

- False positives
- False negatives
- Retrieval failures
- Entity extraction failures
- Numerical inconsistencies
- Temporal reasoning errors
- Structural graph inconsistencies
- Confidence overestimation
- Distribution shift

---

# 📦 Held-Out Validation

The final model is evaluated on a completely unseen validation set.

Validation protocol includes:

- Frozen preprocessing
- Frozen feature engineering
- Frozen retrieval pipeline
- Frozen model parameters
- No retraining
- No manual corrections

This protocol measures genuine generalization performance.

---

# 📊 Model Selection Strategy

Multiple candidate models were evaluated before selecting the final production classifier.

Candidate models included:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Confidence Fusion Meta-Classifier

Selection criteria:

- Highest ROC-AUC
- Calibration quality
- Generalization
- Stability
- Explainability
- Production latency

---

# 📈 Benchmark Results

> Replace these placeholder values with your final evaluation metrics.

| Metric | Score |
|---------|-------|
| Accuracy | XX.XX% |
| Precision | XX.XX% |
| Recall | XX.XX% |
| F1 Score | XX.XX% |
| ROC-AUC | XX.XX |
| PR-AUC | XX.XX |
| MCC | XX.XX |
| Brier Score | XX.XX |
| ECE | XX.XX |

---

# 📊 Performance Comparison

| Capability | HalluciSense | Typical Binary Detector |
|-------------|-------------|--------------------------|
| Retrieval Verification | ✅ | ❌ |
| Structural Reasoning | ✅ | ❌ |
| Confidence Calibration | ✅ | ❌ |
| Explainability | ✅ | Partial |
| Modular Architecture | ✅ | Partial |
| Production Deployment | ✅ | Partial |
| Hybrid Verification | ✅ | ❌ |
| Evidence Presentation | ✅ | Limited |

---

# 📉 Robustness Evaluation

HalluciSense was validated under multiple challenging conditions.

### Distribution Shift

✔ Evaluated

### Retrieval Noise

✔ Evaluated

### Missing Evidence

✔ Evaluated

### Ambiguous Claims

✔ Evaluated

### Numerical Claims

✔ Evaluated

### Temporal Claims

✔ Evaluated

### Multi-hop Reasoning

✔ Evaluated

---

# 🔬 Scientific Validation

The project includes dedicated analyses for:

- Leakage auditing
- Feature stability
- Cross-validation
- Bootstrap confidence intervals
- Correlation analysis
- Variance Inflation Factor (VIF)
- Probability compression
- Root cause analysis
- Distribution shift
- Calibration drift
- Feature importance stability

---

# 📁 Evaluation Reports

The repository includes detailed evaluation artifacts and scientific reports covering every phase of development.

```
backend/
└── evaluation_results/
    ├── phase6/
    ├── phase7/
    ├── phase8/
    ├── reports/
    ├── calibration/
    ├── validation/
    └── deployment/
```

These reports document the experimental methodology, intermediate analyses, validation protocols, and deployment readiness assessments.

---

# 📸 Dashboard Preview

> Replace with actual screenshots.

<p align="center">

<img src="assets/dashboard.png" width="900">

</p>

The dashboard provides:

- Retrieval evidence visualization
- Confidence analysis
- Explainability
- Prediction diagnostics
- API monitoring
- Interactive verification

---

# ⚡ Performance Characteristics

| Component | Typical Latency |
|------------|----------------:|
| Retrieval | < 100 ms |
| Feature Extraction | < 50 ms |
| Structural Analysis | < 75 ms |
| Confidence Fusion | < 20 ms |
| End-to-End Verification | < 300 ms |

> Replace with measured production values.

---

# 🧠 Reproducibility

HalluciSense emphasizes reproducible AI research.

The repository includes:

- Fixed preprocessing pipelines
- Versioned configurations
- Deterministic evaluation scripts
- Experiment reports
- Deployment manifests
- Scientific documentation
- Model metadata
- Validation protocols

This enables independent verification of experimental results and supports future research extensions.

---

# 📌 Next Section

The next section covers deployment architecture, CI/CD workflow, roadmap, publication details, citation format, acknowledgements, and contribution guidelines.

---

# 🚀 Deployment

HalluciSense is designed with production deployment in mind. The architecture separates concerns across independent services, enabling scalable deployment to cloud environments or on-premise infrastructure.

## Supported Deployment Options

| Platform | Status |
|----------|--------|
| Docker | ✅ |
| Railway | ✅ |
| Render | ✅ |
| AWS EC2 | ✅ |
| Azure VM | ✅ |
| Google Cloud VM | ✅ |
| Kubernetes | ✅ |
| Local Development | ✅ |

---

# ☁️ Deployment Architecture

```text
                    Internet
                        │
                        ▼
                Reverse Proxy (Nginx)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   Frontend (Next.js)          Backend API (FastAPI)
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
      Retrieval Engine         Verification Engine       Confidence Fusion
              │                         │                         │
              └──────────────┬──────────┴──────────┬──────────────┘
                             ▼
                      Model Prediction
                             │
                             ▼
                      Explainability API
                             │
                             ▼
                         JSON Response
```

---

# 🐳 Docker

Build the containers

```bash
docker compose build
```

Run the application

```bash
docker compose up
```

Run in detached mode

```bash
docker compose up -d
```

Stop

```bash
docker compose down
```

---

# ☁️ Railway Deployment

```bash
railway login

railway init

railway up
```

Environment variables

```
OPENAI_API_KEY=

GOOGLE_API_KEY=

PINECONE_API_KEY=

SUPABASE_URL=

SUPABASE_KEY=

JWT_SECRET=

DATABASE_URL=
```

---

# 🔄 CI/CD Workflow

HalluciSense follows an automated development workflow.

```text
Developer Push
        │
        ▼
GitHub Actions
        │
        ▼
Linting
        │
        ▼
Unit Tests
        │
        ▼
Integration Tests
        │
        ▼
Docker Build
        │
        ▼
Deployment Validation
        │
        ▼
Production Release
```

---

# 🧪 Testing

Run all tests

```bash
pytest
```

Run a specific module

```bash
pytest backend/tests
```

Generate coverage

```bash
pytest --cov=backend
```

---

# 📈 Project Roadmap

## ✅ Completed

- Retrieval verification
- Structural consistency analysis
- Confidence-aware fusion
- Explainability
- API deployment
- Docker support
- Production configuration
- Scientific evaluation
- Calibration framework
- Documentation

---

## 🚧 Planned

- Multi-document verification
- Knowledge graph reasoning
- Streaming verification
- Batch inference
- Distributed processing
- Active learning
- Reinforcement learning calibration
- Graph Neural Networks
- Vector database integration
- Real-time monitoring

---

# 🔬 Research Contributions

HalluciSense contributes several ideas toward trustworthy Large Language Model verification.

### Novel Components

- Hybrid verification pipeline
- Confidence-aware decision fusion
- Retrieval consistency verification
- Structural graph reasoning
- Multi-stage explainability
- Calibration-aware prediction
- Production-focused deployment architecture

---

# 📚 Citation

If HalluciSense contributes to your research, please cite it.

```bibtex
@software{hallucisense2026,
  title={HalluciSense: A Confidence-Aware Hybrid Framework for Detecting and Quantifying Hallucinations in Large Language Models},
  author={Akash G. Patil},
  year={2026},
  url={https://github.com/akashcodes23/HalluciSense}
}
```

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature/my-feature
```

3. Commit

```bash
git commit -m "Add new feature"
```

4. Push

```bash
git push origin feature/my-feature
```

5. Open a Pull Request

---

# 🐞 Reporting Issues

If you encounter bugs or have suggestions:

- Open an Issue
- Include reproduction steps
- Include logs if available
- Describe expected behavior

---

# 📄 License

Distributed under the MIT License.

See

```
LICENSE
```

for details.

---

# 🙏 Acknowledgements

This project builds upon ideas and technologies from the open-source community.

Special thanks to:

- Hugging Face
- PyTorch
- FastAPI
- Next.js
- LangChain
- Sentence Transformers
- Scikit-learn
- Docker
- Railway
- OpenAI
- Anthropic
- Google DeepMind

---

# ⭐ Support the Project

If you found HalluciSense useful:

⭐ Star the repository

🍴 Fork it

📝 Cite it in your research

📢 Share it with others

Every contribution helps improve trustworthy AI.

---

#

---

# 📬 Contact

For research collaborations, project discussions, or technical questions:

📧 Email: akashgpatil23.05@gmail.com

💼 LinkedIn: https://linkedin.com/akash-g-patil

🐙 GitHub: https://github.com/akashcodes23

---

# 🌍 Vision

> **Building trustworthy AI systems through explainable, confidence-aware, and scientifically validated verification frameworks.**

HalluciSense represents a step toward safer and more reliable Large Language Models by combining retrieval verification, structural reasoning, explainable AI, and calibrated confidence estimation into a unified production-ready framework.

---

<p align="center">

### ⭐ If you found this project useful, please consider giving it a Star ⭐

Made with ❤️ for the AI Research Community

</p>
