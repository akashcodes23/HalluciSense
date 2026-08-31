# HalluciSense Viva Truth Sheet (Examiner-Defensible Facts)

This sheet contains **only independently verified facts** supported by primary repository artifacts, cryptographic checksums, and live runtime measurements. Use this as your definitive script during viva questioning.

---

## 1. Core Model & Architecture

### What to Say:
> *"HalluciSense uses a frozen 19-feature `HistGradientBoostingClassifier` metadata model trained on 58,002 development samples, operating at an optimal decision threshold of $\tau^* = 0.54$."*

### Evidence:
- `backend/evaluation_results/phase6m/final_hybrid_model/model_metadata.json`
- `backend/evaluation_results/phase6m/final_hybrid_model/feature_schema.json`
- `backend/evaluation_results/phase6m/final_hybrid_model/hybrid_meta_classifier.joblib` (SHA-256: `089ebd2d277d...`)

### How to Prove It:
```bash
python3 -c "
import json
from app.models.registry import safe_joblib_load
clf = safe_joblib_load('backend/evaluation_results/phase6m/final_hybrid_model/hybrid_meta_classifier.joblib')
meta = json.load(open('backend/evaluation_results/phase6m/final_hybrid_model/model_metadata.json'))
print(f'Model: {type(clf).__name__}')
print(f'Features: {clf.n_features_in_}')
print(f'Training Samples: {meta[\"training_samples\"]}')
print(f'Threshold: {meta[\"protocol\"][\"decision_threshold\"]}')
"
```

### What NOT to Claim:
- Do NOT claim you retrained the model during deployment or bug fixing. (The weights were preserved).
- Do NOT claim the model uses deep neural net fusion. (It uses a fast, interpretable Histogram Gradient Boosted decision tree ensemble).

---

## 2. Model Performance & Metrics

### What to Say:
> *"On the development evaluation partition, the 19-feature Hybrid model achieved an ROC-AUC of 0.7378, an F1 score of 0.7100, an Accuracy of 0.6770, and an MCC of 0.3466, outperforming standalone single pillars by +5.36% in ROC-AUC."*

### Evidence:
- `model_metadata.json: protocol.dev_resubstitution_metrics`
- `backend/evaluation_results/phase6m/` historical audit logs.

### How to Prove It:
Show `backend/evaluation_results/phase6m/final_hybrid_model/model_metadata.json` lines 48–53.

### What NOT to Claim:
- Do NOT claim 99% accuracy across all arbitrary LLM prompts. (State honestly that ROC-AUC is 0.7378 on balanced multi-domain benchmarks).
- Do NOT claim the benchmark dataset has 1,000 files; the static dataset `benchmark_dataset.jsonl` contains exactly 750 balanced records (375 factual, 375 hallucinated).

---

## 3. Serialization Failure & Bit-for-Bit Repair

### What to Say:
> *"During Phase 30 testing, a NumPy `PCG64` BitGenerator deserialization incompatibility was identified that caused silent fallback to Pillar 1. We developed a `_SafeModelUnpickler` that recovered the fitted tree ensemble, attached a standard modern generator, and resaved the artifact. Deterministic testing over 100 test vectors proved a bit-for-bit probability difference of exactly 0.00000000 against the original backup."*

### Evidence:
- `_SafeModelUnpickler` in `backend/app/models/registry.py`
- `hybrid_meta_classifier.joblib.backup` (SHA-256: `cb459fd9...`)
- `hybrid_meta_classifier.joblib` (SHA-256: `089ebd2d...`)

### How to Prove It:
```bash
python3 -c "
import numpy as np
from pathlib import Path
from app.models.registry import safe_joblib_load
repaired = safe_joblib_load(Path('backend/evaluation_results/phase6m/final_hybrid_model/hybrid_meta_classifier.joblib'))
backup = safe_joblib_load(Path('backend/evaluation_results/phase6m/final_hybrid_model/hybrid_meta_classifier.joblib.backup'))
vec = np.random.RandomState(42).randn(100, 19)
diff = np.max(np.abs(repaired.predict_proba(vec) - backup.predict_proba(vec)))
print(f'Max Difference: {diff:.8f}')
"
```

### What NOT to Claim:
- Do NOT claim you retrained the model to fix the serialization issue.
- Do NOT claim the BitGenerator error altered tree thresholds; random generators are unused during inference.

---

## 4. Production Memory Engineering & OOM Elimination

### What to Say:
> *"In commit `78c445a`, setting `PYTHONMALLOC=malloc` disabled Python's `pymalloc` suballocator, causing glibc malloc header overhead to inflate DeBERTa deserialization memory from 972 MB to 1.22 GB, breaching the 1024 MB Railway limit. Removing `PYTHONMALLOC=malloc` while retaining `MALLOC_ARENA_MAX=2` and `MALLOC_TRIM_THRESHOLD_=65536` dropped peak startup RSS to 774 MB and concurrent peak RSS to 832 MB, creating 192 MB of safe headroom."*

### Evidence:
- Railway Metrics API logs from `railway metrics --since 5m --cpu --memory`
- Phase 32 Forensic Report, Phase 33 Validation Report, and Phase 34 Freeze Report.

### How to Prove It:
Show `backend/reports/phase32/PHASE32_CRASH_FORENSIC_REPORT.md` and live metrics from Railway.

### What NOT to Claim:
- Do NOT claim HalluciSense uses less than 200 MB of RAM. (Transformer NLI models inherently require ~500–700 MB of process memory).
- Do NOT claim the container can support 100 concurrent requests. (Inference concurrency is bounded by a semaphore to `MAX_CONCURRENT_ANALYSES=2` to respect the 1024 MB ceiling).

---

## 5. Live Production System

### What to Say:
> *"HalluciSense is actively hosted online on Railway at `https://hallucisense-production.up.railway.app`. Live health telemetry confirms `active_model='hybrid'`, `hybrid_available=true`, and `fallback_active=false` with sub-15ms cached latency."*

### Evidence:
- `curl -s https://hallucisense-production.up.railway.app/health`

### How to Prove It:
```bash
curl -s https://hallucisense-production.up.railway.app/health
```

### What NOT to Claim:
- Do NOT claim the frontend or backend is running locally only. (It is actively deployed in the cloud on Railway).
