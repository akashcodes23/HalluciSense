# HalluciSense v1.0 Production Operations Runbook

**Document Owner**: Production Infrastructure & Site Reliability Engineering (SRE)  
**System Target**: HalluciSense FastAPI Backend & Next.js 16 Frontend  

---

## 1. Daily Operational Maintenance

### 1.1 Health & Readiness Probe Monitoring
- **Liveness Probe**: `GET /health` (Expected HTTP 200 OK within 5s)
- **Readiness Probe**: `GET /ready` (Expected HTTP 200 OK with `status: "ready"`)
- **Telemetry Check**: `GET /api/v1/metrics` or `GET /api/v1/metrics/prometheus`

### 1.2 Volume Storage Maintenance (`/data`)
Verify Railway volume disk usage:
```bash
df -h /data
# Ensure /data/traces, /data/cache, /data/models are under 80% capacity
```

---

## 2. Deployment & Upgrades Procedure

1. **Staging Validation**: Run `python3 backend/scripts/run_railway_smoke_tests.py http://127.0.0.1:8000`.
2. **Container Image Build**: Docker builds automatically via `railway.toml`.
3. **Zero-Downtime Swap**: Railway performs rolling updates ensuring container healthiness before terminating previous instances.

---

## 3. Log Inspection & Troubleshooting

Logs are formatted in JSON via `structlog`. Filter logs by correlation IDs:
```json
{"event": "request_handled", "request_id": "473dc4fb-aa3c", "trace_id": "TRACE_CF09362933E4", "status_code": 200, "duration_ms": 120.12}
```
