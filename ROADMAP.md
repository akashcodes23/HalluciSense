# HalluciSense Open-Source Product Roadmap

## Q3 2026 — v1.0 Release Candidate & Open-Source Launch (Current)
- [x] **v1.0 RC1 Release**: Phase 6M Hybrid Meta-Classifier (`0.9501 AUROC`, `0.0257 ECE`).
- [x] **12 Public Benchmark Adapters**: Integrated HaluEval, TruthfulQA, FEVER, SciFact, PubHealth, FreshQA, FActScore.
- [x] **Enterprise SRE Architecture**: Railway PaaS container deployment, sub-150ms P90 latency, OpenTelemetry tracing.
- [x] **Explainability Suite**: Interactive SHAP feature attributions and topological claim-evidence graphs.

---

## Q4 2026 — v1.1 Multi-Modal & Streaming Vision Verification
- [ ] **Multi-Modal Hallucination Detection**: Support for visual claim verification in VLLMs (GPT-4V, LLaVA, Qwen-VL).
- [ ] **Streaming SSE Verification**: Sub-50ms token-by-token streaming hallucination flagging for active chat UI streams.
- [ ] **Custom Domain Retriever Adapters**: One-line plugin architecture for enterprise vector databases (Pinecone, Qdrant, Milvus).

---

## Q1 2027 — v2.0 Distributed Agentic Fact-Checking Mesh
- [ ] **Multi-Agent Fact-Checking Mesh**: Autonomous Web-search agents verifying multi-hop complex scientific claims.
- [ ] **Edge Wasm Deployment**: Client-side WebAssembly inference running Pillar 2 self-consistency directly in browser runtimes.
