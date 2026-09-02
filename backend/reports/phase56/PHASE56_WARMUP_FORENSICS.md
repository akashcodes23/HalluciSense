# Phase 56 — Warmup Forensics

## Background Warmup Lifecycle

In `backend/app/main.py`:
```python
async def _background_warmup(app: FastAPI):
    import asyncio
    await asyncio.to_thread(_sync_warmup, app)
```

1. **Immediate Port Binding**: The uvicorn HTTP server binds to `$PORT` (`8080`) immediately so Railway's `/health` probe passes.
2. **Background Execution**: `asyncio.to_thread(_sync_warmup, app)` runs `ModelRegistry.get_pipeline()` in a dedicated thread.
3. **Synchronization**: `threading.RLock()` ensures that if an incoming API request arrives while warmup is in progress, the request waits safely on the singleton lock rather than triggering a duplicate model load.
4. **Readiness State**: Sets `app.state.readiness_status = "READY"` once complete, exposing granular readiness via `/ready`.
