# Phase 56 — Exact Exit Reason Analysis

## Termination Mechanism

- **Termination Signal**: `SIGKILL` (Sent by Linux kernel OOM Killer)
- **Exit Code**: `137`
- **Application Exceptions**: None (No Python `MemoryError` or unhandled tracebacks present in logs).
- **Behavior**: Process `[1]` (uvicorn) abruptly disappears during execution of `loading_shared_nli_model`, triggering Railway container restart.
- **Restart Count**: 3 restarts observed before deployment transitions to `CRASHED` under `restartPolicyMaxRetries: 3`.

```
2026-09-02T06:54:34.962857349Z [INFO] model_name="cross-encoder/nli-deberta-v3-small" max_length=256 event="loading_shared_nli_model"
[PROCESS KILLED BY OOM KILLER - SIGKILL (EXIT 137)]
2026-09-02T06:54:57.994885197Z [INFO] provider="Wikipedia" event="evidence_provider_registered"
INFO:     Started server process [1]
```
