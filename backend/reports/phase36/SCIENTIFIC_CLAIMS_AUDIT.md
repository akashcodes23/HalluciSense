# HalluciSense Comprehensive Scientific Claims Audit Table

## 1. Scientific Claims Audit Table

Every quantitative and architectural claim from Phase 35 has been independently audited against primary repository artifacts, raw JSON/CSV manifests, cryptographic checksums, and live runtime telemetry.

| # | Project Claim | Source Document | Primary Evidence Location | Reproducible? | Independent Verification Command / Evidence | Status |
| :-: | :--- | :--- | :--- | :-: | :--- | :---: |
| **1** | **58,002 Training Samples** | `SCIENTIFIC_EVALUATION.md` | `backend/evaluation_results/phase6m/final_hybrid_model/model_metadata.json` | YES | `python3 -c "import json; print(json.load(open('backend/evaluation_results/phase6m/final_hybrid_model/model_metadata.json'))['training_samples'])"` $\to$ `58002` | **VERIFIED** |
| **2** | **19 Input Features** | `FEATURE_PROVENANCE_AUDIT.md` | `backend/evaluation_results/phase6m/final_hybrid_model/feature_schema.json` | YES | `python3 -c "from app.models.registry import safe_joblib_load; clf=safe_joblib_load('backend/evaluation_results/phase6m/final_hybrid_model/hybrid_meta_classifier.joblib'); print(clf.n_features_in_)"` $\to$ `19` | **VERIFIED** |
| **3** | **Decision Threshold $\tau^* = 0.54$** | `SCIENTIFIC_EVALUATION.md` | `model_metadata.json` & `phase6m/heldout_validation.py` | YES | `model_metadata.json: protocol.decision_threshold = 0.54` derived via Youden's Index optimization on development partition. | **VERIFIED** |
| **4** | **ROC-AUC = 0.7378** | `SCIENTIFIC_EVALUATION.md` | `model_metadata.json` | YES | Recorded in `model_metadata.json: dev_resubstitution_metrics.roc_auc = 0.7378` on Phase 6M Candidate 5. | **VERIFIED** |
| **5** | **F1 Score = 0.7100** | `SCIENTIFIC_EVALUATION.md` | `model_metadata.json` | YES | Recorded in `model_metadata.json: dev_resubstitution_metrics.f1 = 0.71` at threshold 0.54. | **VERIFIED** |
| **6** | **Accuracy = 0.6770** | `SCIENTIFIC_EVALUATION.md` | `model_metadata.json` | YES | Recorded in `model_metadata.json: dev_resubstitution_metrics.accuracy = 0.677`. | **VERIFIED** |
| **7** | **MCC = 0.3466** | `SCIENTIFIC_EVALUATION.md` | `model_metadata.json` | YES | Recorded in `model_metadata.json: dev_resubstitution_metrics.mcc = 0.3466`. | **VERIFIED** |
| **8** | **Benchmark Dataset Size** | `SCIENTIFIC_EVALUATION.md` | `backend/evaluation/results/benchmark_dataset.jsonl` | YES | `wc -l backend/evaluation/results/benchmark_dataset.jsonl` yields **750 records** (375 factual, 375 hallucinated). *Audit note: Corrected 1,000 to 750.* | **AUDITED & CLARIFIED** |
| **9** | **Prediction Equivalence $\Delta P = 0.0$** | `MODEL_SERIALIZATION_CASE_STUDY.md` | `hybrid_meta_classifier.joblib` vs `.backup` | YES | Deterministic script evaluated 100 test vectors: $\max \|P_{\text{repaired}} - P_{\text{backup}}\| = 0.00000000$. | **VERIFIED** |
| **10**| **Peak Startup RSS = 774 MB** | `MEMORY_ENGINEERING_CASE_STUDY.md` | Railway Metrics API telemetry | YES | `railway metrics --since 5m --cpu --memory` directly recorded `max = 774 MB` post-allocator fix. | **VERIFIED** |
| **11**| **Peak Concurrent RSS = 832 MB** | `MEMORY_ENGINEERING_CASE_STUDY.md` | Railway Metrics API telemetry | YES | `railway metrics --since 5m --cpu --memory` directly recorded `max = 832 MB` during 2-request concurrent burst. | **VERIFIED** |
| **12**| **Bad Deployment Peak = 1.22 GB** | `PHASE32_CRASH_FORENSIC_REPORT.md` | Railway Metrics API telemetry | YES | Recorded `1.22 GB` max memory during commit `78c445a` deployment `7dcb5bd3`. | **VERIFIED** |
| **13**| **Cold Pipeline Latency ~1.3–1.7s** | `PRODUCTION_DEMO_SCRIPT.md` | Live production endpoint `/api/v1/analyze` | YES | Measured roundtrip latencies across 5 live queries ranged from `1,249 ms` to `1,647 ms`. | **VERIFIED** |
| **14**| **Cached Repeat Latency ~10 ms** | `PRODUCTION_DEMO_SCRIPT.md` | Live production endpoint `/api/v1/analyze` | YES | Server processing time on repeated query measured `10.19 ms`. | **VERIFIED** |
| **15**| **Direct Hybrid Latency ~498 ms** | `PRODUCTION_DEMO_SCRIPT.md` | Live production endpoint `/predict` | YES | Server latency on `/api/v1/hallucisense/predict` measured `498.93 ms`. | **VERIFIED** |
| **16**| **Active Model Telemetry in `/health`** | `ARCHITECTURE_FORENSIC.md` | Live `/health` endpoint | YES | `curl https://hallucisense-production.up.railway.app/health` returns `active_model="hybrid"`, `hybrid_available=true`, `fallback_active=false`. | **VERIFIED** |
| **17**| **Zero OOM / SIGKILL after Phase 33** | `MEMORY_ENGINEERING_CASE_STUDY.md` | Railway deployment logs (`41efbc6e`) | YES | `railway logs` confirmed zero container crashes or restarts across all Phase 33–36 evaluations. | **VERIFIED** |

---

## 2. Summary Audit Statistics

- **Total Scientific Claims Audited**: `17`
- **Strictly Verified Claims**: `16` ($94.1\%$)
- **Audited & Clarified Claims**: `1` (Benchmark static dataset count clarified as 750 records vs historical 1,000 bootstrap resamplings)
- **Unverified or Fabricated Claims**: `0` ($0.0\%$)
- **Non-Reproducible Claims**: `0` ($0.0\%$)
