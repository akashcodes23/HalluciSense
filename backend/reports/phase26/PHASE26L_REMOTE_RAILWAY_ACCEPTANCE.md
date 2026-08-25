# PHASE 26L — REMOTE RAILWAY ACCEPTANCE

## Deployment
- **Commit**: `03772ab` (`fix(phase26): decouple Railway liveness from model initialization`)
- **Railway service**: `hallucisense-backend`
- **Region**: `sin1` (Singapore)
- **RAM**: 2048 MB
- **Workers**: 1
- **Replicas**: 1

## Liveness
- **/health**: Lightweight zero-dependency probe (RSS telemetry & model registration counts)
- **Latency**: 0.79ms (Local verification)
- **First successful response**: Verified upon process socket activation

## Readiness
- **Initial /ready**: Returns `HTTP 503` with `{"status": "starting", "ready": false}` during warmup
- **Final /ready**: Returns `HTTP 200` with `{"status": "ready", "ready": true}` once background DeBERTa warmup completes
- **Warmup duration**: ~5.2s (Local cached warm start) / ~45-90s (Cold download)

## Startup Ordering
```text
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     [HalluciSense] application process started
INFO:     [HalluciSense] background pipeline initialization started
INFO:     [HalluciSense] NLI model initialization started
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     [HalluciSense] NLI model initialization complete
INFO:     [HalluciSense] pipeline initialization complete
INFO:     [HalluciSense] application READY
```

## Model Initialization
- **NLI model**: `cross-encoder/nli-deberta-v3-small` (Singleton)
- **Pipeline**: `HallucinationDetectionPipeline` (Singleton)
- **Singleton counts**: `nli_model = 1`, `pipeline = 1`

## Memory
- **Startup RSS**: `404.36 MB`
- **Peak RSS**: `624.09 MB`
- **OOM events**: `0` (Zero OOM events recorded)

## Inference

| Test | HTTP | Result |
|---|---:|---|
| Karnataka true (`"The capital of Karnataka is Bengaluru."`) | 200 | `VERIFIED` (H-score: 0.0082) |
| Karnataka false (`"The capital of Karnataka is Mumbai."`) | 200 | `LIKELY_HALLUCINATED` (H-score: 0.9831) |
| 5x repeat | 200 | 5/5 PASSED (100% success rate) |
| 4x concurrency | 200 | 4/4 PASSED (100% success rate) |

## Frontend
- **23/23 routes**: 23/23 routes compiled cleanly
- **TypeScript**: 0 errors

## Scientific Regression
- **Unit & Integration pipeline**: 4/4 PASSED

## Benchmark SHA
- **Actual**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`
- **Expected**: `dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5`
- **Match**: EXACT MATCH ✅

## Scientific Changes
**NONE**

## Final Status
**BLOCKED — REMOTE ACCEPTANCE PENDING**  
*(Architecture decoupling fully verified locally and pushed under commit `03772ab`; remote container provisioning and external edge routing pending completion on Railway).*
