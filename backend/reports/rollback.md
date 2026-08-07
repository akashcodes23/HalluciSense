# HalluciSense v1.0 Zero-Downtime Rollback Procedure

---

## 1. Automated Railway Rollback Procedure

1. Open **Railway Dashboard** $\rightarrow$ Select `HalluciSense Backend` service.
2. Go to **Deployments** tab.
3. Locate the last known good deployment commit.
4. Click **Redeploy** on the target deployment.
5. Railway performs a zero-downtime container swap:
   - Starts the target container image.
   - Waits for `/health` to return HTTP 200 OK.
   - Re-routes traffic away from the failing image.

---

## 2. Manual CLI Rollback Command

```bash
# Roll back to target Docker git commit SHA
git checkout <LAST_KNOWN_GOOD_SHA>
docker build -t hallucisense-backend:rollback backend/
docker-compose up -d --build
```
