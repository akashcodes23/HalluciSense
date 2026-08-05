# HalluciSense System Architecture & Technical Design Specification (v2.0)

**Document Version**: 2.0.0-RC1  
**Target Platform**: Railway Cloud / Docker Containers  
**Author**: Principal Staff Backend & MLOps Architect  

---

## 1. High-Level System Architecture

```mermaid
graph TD
    Client[React / Next.js 14 Web Frontend] -->|HTTPS / WSS| Gateway[FastAPI API Gateway / Nginx Reverse Proxy]
    Gateway --> Auth[JWT Authentication & Rate Limiter]
    Auth --> Verification[Verification Orchestrator Engine]

    subgraph "Hybrid Multi-Pillar Inference Pipeline"
        Verification --> Pillar1[Pillar 1: External Evidence Grounding]
        Verification --> Pillar2[Pillar 2: Intra-Model Self-Consistency]

        Pillar1 --> WebSearch[Wikipedia / PubMed / CrossRef Retrieval]
        Pillar1 --> CrossEncoder[CrossEncoder Reranker & NLI Model]

        Pillar2 --> Claims[Atomic Claim Extractor]
        Pillar2 --> SelfNLI[Pairwise Claim NLI Matrix]

        Pillar1 --> FeatureVector[19-Dimensional Hybrid Vector Assembly]
        Pillar2 --> FeatureVector

        FeatureVector --> HybridClassifier[Phase 6M HistGradientBoostingClassifier]
    end

    HybridClassifier --> Decision[Decision Threshold Engine (tau* = 0.54)]
    Decision --> Explainability[SHAP Attribution & Topological Graph Engine]

    Explainability --> Response[JSON API Response + SSE Streaming]

    Response --> DB[(PostgreSQL Database)]
    Response --> RedisCache[(Redis Cache & Session Store)]
```

---

## 2. Component Design Specifications

### A. API Gateway Layer (`backend/app/main.py`)
- Fast, asynchronous Python 3.10 / FastAPI runtime.
- CORS middleware with strict origin whitelist.
- GZip compression for payloads &gt; 1KB.
- Structured `structlog` logging with trace correlation IDs.

### B. Hybrid Multi-Pillar Inference Engine (`backend/app/core/inference/`)
- **Pillar 1**: Retrieves top-$K$ passages from external authoritative corpora (Wikipedia, PubMed, CrossRef) and computes cross-encoder entailment and contradiction distributions.
- **Pillar 2**: Extracts fine-grained atomic claims from LLM responses and computes pairwise self-consistency contradiction margins.
- **Hybrid Meta-Classifier**: Combines 19 continuous features through a robustly scaled `HistGradientBoostingClassifier` trained on $N=58,002$ dev samples.

### C. Storage & Caching Layer
- **PostgreSQL**: Stores user accounts, verification history, claims, and ground truth feedback.
- **Redis**: Caches Wikipedia evidence passages, rate-limit buckets, and active WebSocket connections.
