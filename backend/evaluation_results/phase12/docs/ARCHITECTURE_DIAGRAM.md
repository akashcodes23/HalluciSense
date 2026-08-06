# HalluciSense SaaS Infrastructure Architecture

```
User / Web UI / SDK Client
       ↓ (HTTPS / SSL)
Nginx Reverse Proxy & Helmet Security Middleware
       ↓
FastAPI Application Container (JWT Auth, RBAC, Rate Limiting)
       ↓
├── Pillar 1 Statistical NLI Engine (Frozen LogisticRegression)
├── Pillar 2 Multi-LLM Engine (Claim Extractor, Graph Builder, Multi-LLM Consensus)
├── PostgreSQL 15 Database (Normalized Schema: Users, Orgs, Sessions, Audit)
├── Redis 7 Cache & Task Queue Broker
└── Celery Background Workers (Async Verification Tasks)
```
