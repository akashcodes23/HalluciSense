# HalluciSense

**An availability-aware multi-signal framework for detecting, calibrating, abstaining on, and correcting LLM hallucinations.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js 16](https://img.shields.io/badge/Next.js-16.2+-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scientific Freeze](https://img.shields.io/badge/Scientific%20Benchmark-Locked%20SHA--256-success.svg)](backend/evaluation/results/benchmark_dataset.jsonl)

---

## Table of Contents
1. [Why HalluciSense?](#why-hallucisense)
2. [System Architecture](#system-architecture)
3. [The Three Verification Pillars](#the-three-verification-pillars)
4. [Availability-Aware Adaptive Fusion](#availability-aware-adaptive-fusion)
5. [Scientific Evaluation Results](#scientific-evaluation-results)
6. [Live Production Validation](#live-production-validation)
7. [Research vs. Production Separation](#research-vs-production-separation)
8. [Quick Start & Local Development](#quick-start--local-development)
9. [Project Repository Structure](#project-repository-structure)
10. [Documentation Index](#documentation-index)
11. [License](#license)

---

## Why HalluciSense?

Large Language Models (LLMs) frequently generate plausible-sounding yet factually incorrect assertions—commonly termed *hallucinations*. In high-stakes domains such as medicine, engineering, and scientific literature analysis, undetected hallucinations can propagate critical misinformation and undermine user trust.

Existing mitigation techniques typically rely on a single defensive mechanism: pure external retrieval (which fails when retrieval corpora contain gaps or noise), internal token log-probability introspection (which is unavailable behind black-box commercial APIs like Claude or OpenAI), or multi-sample self-consistency checks (which impose severe inference latency and cost overheads). When any single assumptions breaks, conventional verifiers fail silently or produce miscalibrated certainty scores.

Furthermore, naive heuristic aggregations treat missing signals as "zero risk" or "zero evidence," introducing catastrophic structural bias into risk estimation. In production environments where API access patterns vary dynamically from white-box local checkpoints to closed-source API endpoints, hallucination detection systems must adaptively account for signal availability.

**HalluciSense** resolves this foundational vulnerability by introducing an **availability-aware, multi-signal verification and closed-loop correction architecture**. By dynamically modulating fusion weights based on measured signal presence ($m_i$) and empirical component reliability ($r_i$), applying Platt-scaled calibration, enforcing selective abstention on ambiguous decision boundaries, and validating repairs through an independent re-verification gate, HalluciSense provides reliable, transparent, and auditable hallucination intelligence.

---

## System Architecture

```
                       Input Query (q) + LLM Response (R)
                                       │
                                       ▼
                           Atomic Claim Decomposition
                          {c₁, c₂, ..., cₖ propositions}
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
   │    PILLAR 1     │        │    PILLAR 2     │        │    PILLAR 3     │
   │Evidence Grounding│       │Confidence Est.  │        │Consistency Reas.│
   │      (FE)       │        │      (CG)       │        │      (CF)       │
   │  BM25 + FAISS   │        │ Shannon Entropy │        │ Multi-Candidate │
   │  DeBERTa-v3 NLI │        │ White-Box Gated │        │ SBERT Divergence│
   └────────┬────────┘        └────────┬────────┘        └────────┬────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                     Availability-Aware Adaptive Fusion
                H_adapt = Σ(m_i · r_i · w_i · S_i) / Σ(m_i · r_i · w_i)
                                       │
                                       ▼
                   Platt Scaling Calibration & Selective Abstention
                     P(H=1|S) = σ(a·S + b) ───► Abstain if 0.35 ≤ H ≤ 0.65
                                       │
                                       ▼
                      Closed-Loop Repair & Re-Verification
                     Accept repair only if H_reverify < 0.35
                                       │
                                       ▼
               Auditable Output: H-Score, Risk Tier, Evidence, Trace
```

---

## The Three Verification Pillars

HalluciSense decomposes hallucination detection into three complementary, orthogonal signals:

### Pillar 1: Evidence Grounding (FE — Factual Error)
- **Mechanisms**: Hybrid sparse (BM25) and dense (FAISS with `all-MiniLM-L6-v2`) retrieval over authoritative domain corpora, paired with cross-encoder Natural Language Inference (`cross-encoder/nli-deberta-v3-small`).
- **Signal Score ($S_1$)**: Computed from pairwise premise-hypothesis contradiction and entailment probabilities ($S_1 \in [0, 1]$).
- **Default Weight & Reliability**: Base weight $w_1 = 0.45$, empirical reliability $r_1 = 0.95$. Available in all online environments ($m_1 = 1$).

### Pillar 2: Confidence Estimation (CG — Confidence Gap)
- **Mechanisms**: Token-level generation log-probability analysis, normalized subword sequence probability, and Shannon entropy $H(p) = -\sum p_i \log p_i$.
- **Signal Score ($S_2$)**: High entropy or low token confidence indicates elevated model uncertainty ($S_2 \in [0, 1]$).
- **Default Weight & Reliability**: Base weight $w_2 = 0.30$, empirical reliability $r_2 = 0.85$. Dynamically gated ($m_2 = 1$ if white-box logprobs are accessible; $m_2 = 0$ for black-box APIs).

### Pillar 3: Consistency Reasoning (CF — Consistency Failure)
- **Mechanisms**: Stochastic generation sampling ($k \in [3, 5]$ temperature-perturbed candidate outputs) evaluated via pairwise sentence transformer semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **Signal Score ($S_3$)**: High semantic dispersion across candidate answers indicates stochastic ungroundedness ($S_3 \in [0, 1]$).
- **Default Weight & Reliability**: Base weight $w_3 = 0.25$, empirical reliability $r_3 = 0.80$. Dynamically gated ($m_3 = 1$ when multi-sample generation is enabled; $m_3 = 0$ in single-response inspection).

---

## Availability-Aware Adaptive Fusion

When all three signals are present (white-box generation with multi-sampling), HalluciSense operates in **Mode A (Full Fixed Fusion)**:

$$H = \alpha \cdot \text{FE} + \beta \cdot \text{CG} + \gamma \cdot \text{CF} \quad (\alpha = 0.45, \; \beta = 0.30, \; \gamma = 0.25, \; \alpha+\beta+\gamma = 1.0)$$

In practical deployment, black-box APIs or single-turn constraints frequently make $P_2$ or $P_3$ unavailable. Rather than treating missing signals as zero risk, HalluciSense executes **Mode B (Availability-Aware Adaptive Fusion)**:

$$H_{\text{adaptive}} = \frac{\sum_{i=1}^{3} m_i \cdot r_i \cdot w_i \cdot S_i}{\sum_{i=1}^{3} m_i \cdot r_i \cdot w_i}$$

Where:
- $m_i \in \{0, 1\}$ denotes binary signal availability.
- $r_i \in [0, 1]$ represents component empirical reliability ($r_1=0.95, r_2=0.85, r_3=0.80$).
- $w_i$ represents canonical component weight ($\alpha, \beta, \gamma$).
- $S_i \in [0, 1]$ is the normalized raw signal score.

**Zero-Signal Invariant**: If $\sum_{i=1}^3 m_i = 0$ (all verification components offline or disconnected), HalluciSense explicitly returns `status="FAILED"`, `h_score=None`, and `risk_level=None`. It **never** maps unavailable state to $H = 0.0$.

---

## Scientific Evaluation Results

The scientific performance of HalluciSense was evaluated across held-out domain benchmarks and external evaluation suites under strict data-split isolation (SHA-256: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`).

| Metric / Evaluation Dimension | Value | Scientific Description |
| :--- | :---: | :--- |
| **External Discrimination (AUROC)** | **0.9964** | Area under ROC on held-out multi-domain test distribution |
| **External Precision-Recall (AUPRC)** | **0.9958** | Area under PR curve under class imbalance |
| **Expected Calibration Error (ECE)** | **0.0986** | Empirical calibration error after Platt scaling |
| **Brier Score** | **0.0185** | Mean squared probability error against ground truth |
| **Adaptive Mask $[1, 0, 1]$ AUROC** | **0.9910** | Discrimination with Pillar 2 (logprobs) disabled |
| **Fixed Mask $[1, 0, 1]$ AUROC** | **0.8420** | Naive fixed fusion with Pillar 2 disabled |
| **Adaptive Advantage ($\Delta$AUROC)** | **+0.1490** | Significant improvement under missing signals ($p < 0.001$) |
| **Ablation Effect Size (Cohen's $d$)** | **1.42** | Large effect size validating adaptive renormalization |
| **Correction Success Rate (CSR)** | **88.4%** | Erroneous claims successfully repaired to verified ground truth |
| **Repair Precision Rate (RPR)** | **91.2%** | Precision of generated factual replacements |
| **Corrupted Injection Hallucination Rate (CIHR)** | **2.1%** | Minimal artifact injection rate during closed-loop repair |

*Note: HalluciSense provides statistically calibrated probability estimations and does not claim 100% infallible detection.*

---

## Live Production Validation

The production deployment of HalluciSense is live on Railway infrastructure, running the locked ModelRegistry singleton pipeline with persistent connection pooling and NLI pair caching.

### Live Smoke Test Outcomes (Phase 22 Live Audit)

```
Endpoint: https://hallucisense-production.up.railway.app
```

| Scenario / Prompt | Endpoint | HTTP | Result / Score | Verified Behavior |
| :--- | :--- | :---: | :---: | :--- |
| **System Health & Memory** | `GET /health` | **200** | `status: healthy` (622 MB RSS) | ModelRegistry singletons active |
| **Factual Claim** (*"Capital of Karnataka is Bengaluru"*) | `POST /api/v1/analyze` | **200** | $H = 13.3\%$ (`VERIFIED`) | Correctly verified safe (2.74s) |
| **False Claim** (*"Capital of Karnataka is Mumbai"*) | `POST /api/v1/analyze` | **200** | $H = 99.1\%$ (`LIKELY_HALLUCINATED`) | Entity linking error caught (2.97s) |
| **Closed-Loop Chat** (*"Causes of Type 1 diabetes"*) | `POST /api/v1/chat` | **200** | $H = 1.03\%$ (`VERIFIED`, 5 sources) | End-to-end evidence repair |

---

## Research vs. Production Separation

To maintain scientific integrity and prevent reviewer confusion, HalluciSense maintains strict separation between **Research Benchmark Data** and **Live Production Telemetry**:

| Dimension | Research Evidence (`/scientific`, `/evaluate`) | Production Telemetry (`/overview`, `/traces`) |
| :--- | :--- | :--- |
| **Data Source** | Locked, frozen benchmark dataset (SHA-256 validated) | Live incoming user queries and runtime execution |
| **Key Metrics** | AUROC (0.9964), AUPRC (0.9958), ECE (0.0986), Cohen's $d$ | Total requests, live H-scores, per-stage latency (ms) |
| **Purpose** | Peer-reviewed statistical manuscript support | Operational observability, error tracking, trace audit |
| **Immutability** | Strictly frozen post-Phase 14 validation | Continuously updated via streaming metrics tracker |

---

## Quick Start & Local Development

### Prerequisites
- Python 3.10 or 3.11
- Node.js 18+ and npm
- Git

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/akashcodes23/HalluciSense.git
cd HalluciSense

# Backend Setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Frontend Setup
cd ../frontend
npm install
```

### 2. Run Test Suites
```bash
# Run 72/72 scientific regression and production reliability tests
cd backend
PYTHONPATH=. venv/bin/pytest tests/ -v

# Run frontend build validation
cd ../frontend
npm run build
```

### 3. Launch Development Servers
```bash
# Terminal 1: Backend API (FastAPI)
cd backend
PYTHONPATH=. venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend Dashboard (Next.js)
cd frontend
npm run dev
```

Visit the dashboard at `http://localhost:3000` and the interactive OpenAPI docs at `http://localhost:8000/docs`.

---

## Project Repository Structure

```
HalluciSense/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py             # Configuration & environment settings
│   │   │   ├── rate_limiter.py       # In-memory token bucket rate limiter
│   │   │   └── engine/               # ModelRegistry, Pipeline, Fusion, Calibration
│   │   ├── modules/                  # Chat, Verification, Knowledge, MLOps
│   │   ├── schemas/                  # Pydantic v2 validation contracts
│   │   └── main.py                   # FastAPI application factory & middleware
│   ├── evaluation/                   # Frozen benchmark datasets & evaluation scripts
│   ├── reports/                      # Phase 12–23 audit & release documentation
│   └── tests/                        # 72 pytest regression and reliability suites
├── frontend/
│   ├── src/
│   │   ├── app/(dashboard)/          # Next.js App Router (overview, verify, chat, traces)
│   │   ├── components/               # UI components, layout shell, verdict banners
│   │   ├── services/                 # HalluciSense API client with 60s timeout
│   │   └── store/                    # Zustand state management
│   └── package.json
├── docs/
│   ├── architecture/                 # System architecture diagrams (.svg, .png)
│   ├── demo/                         # 5-minute live demo script & walkthrough
│   ├── presentation/                 # Presentation slides outline & Viva prep Q&A
│   ├── research/                     # Scientific contributions & limitations
│   ├── api/                          # OpenAPI endpoint documentation
│   ├── setup/                        # Local setup & Railway deployment guides
│   └── testing/                      # Testing strategy & coverage reports
├── LICENSE                           # MIT License
└── README.md
```

---

## Documentation Index

- [Live Demo Script (5-Minute Walkthrough)](docs/demo/LIVE_DEMO_SCRIPT.md)
- [Viva Voce & Faculty Evaluation Guide](docs/presentation/VIVA_PREPARATION.md)
- [Presentation Slide Deck Outline](docs/presentation/presentation_outline.md)
- [Research Contributions Statement](docs/research/CONTRIBUTIONS.md)
- [Known Engineering & Research Limitations](docs/research/LIMITATIONS.md)
- [REST API Reference](docs/api/API.md)
- [Local Development Guide](docs/setup/LOCAL_DEVELOPMENT.md)
- [Production Deployment Guide](docs/setup/DEPLOYMENT.md)
- [Testing & Quality Assurance](docs/testing/TESTING.md)

---

## License

This project is open-source software licensed under the [MIT License](LICENSE).
