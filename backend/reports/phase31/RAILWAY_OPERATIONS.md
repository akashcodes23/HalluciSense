# RAILWAY OPERATIONS & RUNBOOK — HALLUCISENSE

**Service Name**: `HalluciSense`  
**Railway Project**: `passionate-contentment` (`2c0fdad7-7765-475c-a41a-7315afb700b7`)  
**Environment**: `production` (`b69f4974-053f-4f1f-bbf8-68991e501f39`)  
**Production URL**: `https://hallucisense-production.up.railway.app`  

---

## 1. Quick Status & Health Checks

### Check Live Deployment Status (CLI):
```bash
railway status
```

### Check Deployment History (Last 10 Deployments):
```bash
railway deployment list --limit 10
```

### Live Health & Memory Telemetry (HTTP):
```bash
curl -i https://hallucisense-production.up.railway.app/health
```
*Expected output: HTTP 200 with `"active_model": "hybrid"`, `"hybrid_available": true`, `"fallback_active": false`.*

### Live Pipeline Readiness Probe (HTTP):
```bash
curl -i https://hallucisense-production.up.railway.app/ready
```
*Expected output: HTTP 200 with `"ready": true`, `"active_model": "hybrid"`.*

---

## 2. Logs & Real-Time Monitoring

### Stream Live Production Runtime Logs:
```bash
railway logs --lines 100
```

### Inspect Build Logs for Specific Deployment:
```bash
railway logs <DEPLOYMENT_ID> --build --lines 200
```

### Grep Startup Lifecyle & Model Loading Events:
```bash
railway logs --lines 200 | grep -E "hybrid_model_loaded_successfully|application READY|shared_nli_model_loaded"
```

---

## 3. Metrics & Resource Utilization

### Inspect CPU and Memory Usage (Last 1 Hour):
```bash
railway metrics --since 1h --cpu --memory
```

### Inspect HTTP Traffic, Latencies, and Status Codes:
```bash
railway metrics --since 1h --http
```

---

## 4. Production Smoke Testing

Execute the automated end-to-end smoke test against the live service:
```bash
python3 backend/tests/test_smoke_production.py
```
Or override target host:
```bash
TARGET_URL="https://hallucisense-production.up.railway.app" python3 backend/tests/test_smoke_production.py
```

---

## 5. Rollback Procedure (Emergency Only)

If a new deployment exhibits regressions:
1. **Identify the Last Known Good Deployment**:
   ```bash
   railway deployment list --limit 5
   ```
   *Verified Gold Baseline: `5b4c5a29-d502-433d-9ec9-99549b585cd7` (Commit `b1aafb3`).*
2. **Rollback via Railway CLI**:
   ```bash
   railway redeploy --deployment <DEPLOYMENT_ID>
   ```
   Or deploy the known-good commit directly:
   ```bash
   git checkout b1aafb3
   railway up --detach
   ```
3. **Verify Post-Rollback Health**:
   ```bash
   curl -s https://hallucisense-production.up.railway.app/health | grep '"active_model":"hybrid"'
   ```

---

## 6. Maintenance & Known Deprecations

- **Railway Config-as-Code Deprecation**: Railway warns `Config as Code (railway.json / railway.toml) is deprecated. Prefer Infrastructure as Code (.railway/railway.ts)`. Existing configuration remains valid through December 1, 2026. Migration to `.railway/railway.ts` should be scheduled as a routine IaC maintenance task.
