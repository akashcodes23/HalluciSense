# HalluciSense v1.0 Production Incident Response Protocol

**Severity Levels**: SEV-1 (Critical Outage), SEV-2 (Degraded Performance), SEV-3 (Minor Warning)  

---

## 1. Triage Workflow

```
[PagerDuty Alert Triggered]
            │
            ▼
[Verify /health & /ready Probes] ──► (HTTP 503?) ──► Initiate Container Restart
            │
            ▼ (HTTP 200 OK)
[Check /api/v1/metrics/prometheus] ──► Inspect Memory & Latency Gauges
            │
            ▼
[Filter JSON Logs by X-Trace-ID] ──► Locate Root Cause & Remediate
```

---

## 2. Escalation & Remediation Matrix

| Symptom | Probable Cause | Action |
| :--- | :--- | :--- |
| **HTTP 503 Readiness Failure** | Pipeline model component failed loading | Restart container; check `/data` volume permissions |
| **High Latency (> 1s)** | Large payload processing or HuggingFace model download bottleneck | Verify model cache pre-warming in `lifespan` |
| **Memory Out-of-Memory (OOM)** | PyTorch memory leak or large payload batch | Scale Railway instance memory tier to 2GB |
