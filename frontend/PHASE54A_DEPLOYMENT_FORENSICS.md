# PHASE 54A — RAILWAY DEPLOYMENT FORENSICS REPORT

**Date**: 2026-09-02  
**Author**: Antigravity  
**Repository**: akashcodes23/HalluciSense  

---

## 1. Git Repository

| Field | Value |
| :--- | :--- |
| Repository | `akashcodes23/HalluciSense` |
| Remote URL | `https://github.com/akashcodes23/HalluciSense.git` |
| Branch | `main` |
| Current SHA | `565ace7` |
| Phase 54A commit SHA | `2a672c6` |
| Phase 54A deploy fix SHA | `565ace7` |

Phase 54A commit `2a672c6` confirmed on `origin/main` via `git fetch origin`.

---

## 2. Phase 54A Commit Contents (Verified Clean)

Files in commit `2a672c6`:
```
A  frontend/PHASE54A_3D_FRONTEND_REPORT.md
A  frontend/implementation_plan.md
M  frontend/package-lock.json
M  frontend/package.json
M  frontend/src/app/(auth)/login/page.tsx
M  frontend/src/app/(dashboard)/scientific/page.tsx
M  frontend/src/app/page.tsx
A  frontend/src/components/Hero3DCanvas.tsx
M  frontend/src/store/analysis-store.ts
```

All changes strictly within `frontend/`. No backend/ML files modified.

---

## 3. Railway Project Topology

**Project**: `passionate-contentment` (`2c0fdad7-7765-475c-a41a-7315afb700b7`)  
**Environment**: `production` (`b69f4974-053f-4f1f-bbf8-68991e501f39`)

| Service Name | Service ID | Role | Root Dir | URL | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HalluciSense** | `a449c886-d20f-4eb3-b461-81cb5b9944ea` | Backend FastAPI | `/backend` | `https://hallucisense-production.up.railway.app` | ● Online (SUCCESS) |
| **enchanting-wonder** | `33b027b3-7745-40c2-861e-7bb55b237287` | Frontend Next.js | `/frontend` | `https://enchanting-wonder-production-9a0b.up.railway.app` | ● Online (OLD CODE) |

**Deployment topology**: OPTION C — two separate Railway services sharing one GitHub repository (monorepo).

---

## 4. Railway Service Configuration (at time of investigation)

### enchanting-wonder (frontend)

| Field | Value |
| :--- | :--- |
| Connected Repository | `akashcodes23/HalluciSense` |
| Connected Branch | `main` |
| Root Directory | `/frontend` (confirmed via Railway GraphQL API) |
| Railway Config File | `null` (auto-discovered) |
| Build Command | `npm ci && npm run build` (from `frontend/railway.json`) |
| Start Command | `npm start` (from `frontend/railway.json`) |
| Healthcheck Path | `/` |
| Port | `8080` (Railway-injected `$PORT`) |
| Public Domain | `enchanting-wonder-production-9a0b.up.railway.app` |

### HalluciSense (backend)

| Field | Value |
| :--- | :--- |
| Connected Repository | `akashcodes23/HalluciSense` |
| Connected Branch | `main` |
| Root Directory | `/backend` (confirmed via Railway GraphQL API) |
| Build Command | `DOCKERFILE` from `backend/railway.toml` |
| Start Command | `python start.py` |

---

## 5. Deployment History (enchanting-wonder)

| Deployment ID | Status | Timestamp | Cause |
| :--- | :--- | :--- | :--- |
| `ad44888d` | **SUCCESS** | 2026-09-02 11:15:03 | Rollback to old container (pre-Phase54A) |
| `90259ba0` | **FAILED** | 2026-09-02 11:13:42 | Triggered by Phase 54A commit push |
| `fb852857` | **FAILED** | 2026-09-02 11:02:54 | Pre-existing issue |
| `2d51f463` | **FAILED** | 2026-09-02 10:43:28 | Pre-existing issue |
| `96e25ea0` | **FAILED** | 2026-09-02 10:18:13 | Pre-existing issue |
| `ef30d7fa` | **FAILED** | 2026-09-01 22:47:07 | Pre-existing issue |
| ... | **FAILED** | 2026-09-01 13:16+ | Pre-existing issue (all failures) |

**Key Finding**: The `ad44888d` "SUCCESS" at 11:15:03 happened 90 seconds after `90259ba0` FAILED — too fast for a real build. This is Railway retaining the previously running OLD container after the new build failed.

---

## 6. Exact Build Failure (from Railway GraphQL API)

Deployment `45f5539b` (direct upload via `railway deployment up`) revealed the complete build log:

```
fetched snapshot sha256:505186fadc00565a817123010559d4e0621a0a0cf26f78a3a23c64859d31d1c5 (59 MB)
[internal] load build definition from Dockerfile
[internal] load metadata for docker.io/library/python:3.10-slim
[builder 4/7] COPY backend/requirements.txt .
[runner 6/7] COPY backend/ /app/backend/
[runner 7/7] COPY start.py /app/start.py
error [runner 7/7] COPY start.py /app/start.py

Build Failed: failed to solve: failed to compute cache key:
"/start.py": not found
```

---

## 7. Root Cause Classification

**Category**: **H — Monorepo Configuration / Root Config Override**

### Explanation

The repository root contained **`railway.toml`** specifying:

```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"
```

This Dockerfile is the **backend Python Dockerfile** (a 2-stage Python 3.10-slim build with `pip install`, `COPY backend/`, `COPY start.py`).

Even though the `enchanting-wonder` service has `rootDirectory=/frontend` and `frontend/railway.json` specifies `NIXPACKS` builder, Railway's Config-as-Code walked UP the directory tree from `frontend/` and found the root `railway.toml`, which **overrode** the `frontend/railway.json` NIXPACKS configuration.

Result: Railway built the **backend Python Dockerfile** for the **frontend Node.js service**. When building with `rootDirectory=/frontend` as context, `start.py` (a backend file) was not present in the build context → build failed with `/start.py: not found`.

---

## 8. Why the Backend Was Unaffected

The backend service `HalluciSense` has `rootDirectory=/backend`. Its build context is set to `backend/`. The root `railway.toml` was overriding from a higher directory level, but Railway may have been resolving the Dockerfile relative to the backend's root directory correctly. The backend service was consistently `● Online` throughout.

---

## 9. Pre-existing Nature of the Failure

The root `railway.toml` conflict caused failures starting at least **2026-09-01 13:16** (`3e1f41b1` deployment). This predates the Phase 54A commit (`2a672c6` at 11:13 on 2026-09-02) by ~22 hours. Phase 54A did NOT cause the Railway build failure — the failure was pre-existing.

Phase 54A introduced the Three.js packages which locally also encountered TypeScript errors (`login/page.tsx`, `scientific/page.tsx`) that were fixed within the same commit. Those TypeScript fixes ensure `npm run build` passes cleanly.

---

## 10. Fix Applied

**Commit**: `565ace7`  
**Action**: Deleted root `railway.toml`  
**Commit message**: `fix(deploy): remove conflicting root railway.toml that overrode frontend NIXPACKS build`

After deletion:
- `enchanting-wonder` (root=`/frontend`) → finds `frontend/railway.json` → **NIXPACKS builder** → `npm ci && npm run build` → `npm start` ✅
- `HalluciSense` (root=`/backend`) → finds `backend/railway.toml` → **DOCKERFILE builder** → `python start.py` ✅

No other configuration changes were made. No backend ML files were touched.

---

## 11. Local Build Verification

```
Node version:  v22.23.1
npm version:   10.9.8
Next.js:       16.2.11
React:         19.2.4
```

```
npm run build (Phase 54A code, 2a672c6):

✓ Compiled successfully in 2.5s
✓ TypeScript: 0 errors
✓ Generating static pages (23/23)

Routes:
  ○ /          ← new Phase 54A landing page (Hero3DCanvas)
  ○ /verify
  ○ /overview
  ○ /traces
  ... (all 23 routes)
```

---

## 12. Post-Fix Deployment Status

> **To be updated once Railway build for commit `565ace7` completes.**

| Check | Status |
| :--- | :--- |
| Fix committed | ✅ `565ace7` |
| Fix pushed to `origin/main` | ✅ |
| Railway build triggered | ✅ (auto-triggered by GitHub push) |
| Railway build SUCCESS | ✅ Deployment `354eecec` — SUCCESS |
| New deployment healthy | ✅ 1 running replica, 0 crashed |
| Production URL verified | ✅ HTTP 200 |
| New 3D hero visible | ✅ Confirmed by browser subagent |
| Dashboard routes functional | ✅ `/verify`, `/overview` confirmed |
| Console errors | ✅ 0 errors (1 minor Three.js Clock deprecation warning only) |

---

## 13. Backend Runtime Status

- Backend `HalluciSense` service: **● Online** (`SUCCESS`)
- Latest backend deployment: `80057fde` at 2026-09-02 05:43:42 UTC
- Backend URL: `https://hallucisense-production.up.railway.app`
- No OOM evidence: Railway shows 1 running replica, 0 crashed, 0 exited
- Phase 38 forensic confirmed: Exit 137 / SIGKILL Count = **0** for all Phase 33+ deployments

---

## 14. Railway OOM Status

- **FRONTEND BUILD FAILURE**: Confirmed root cause = `railway.toml` builder conflict (NOT OOM)
- **BACKEND RUNTIME OOM**: No OOM signals detected in current deployment
- **FRONTEND RUNTIME OOM**: Not applicable (Next.js static site, negligible memory footprint)

**Conclusion**: The build failures were NOT OOM-related. They were a configuration conflict.

---

## 15. Missing Environment Variables (Non-Blocking)

The `enchanting-wonder` Railway service is missing `NEXT_PUBLIC_API_BASE_URL` as an explicit env var. The `next.config.ts` fallback defaults to `https://hallucisense-production.up.railway.app`, which is correct. This is non-blocking but should be set explicitly for robustness.

---

## 16. NIXPACKS_NO_CACHE Warning

The Railway environment variable `NIXPACKS_NO_CACHE=1` is set on `enchanting-wonder`. This forces full package reinstallation on every build (no dependency caching). While not a direct cause of the current failure, it significantly increases build time and cost. Consider removing this after the fix is validated.

---

## 17. Current Final Status

```
DEPLOYED_SUCCESSFULLY
```

### Production Verification (Browser Subagent — 2026-09-02 11:52)

| Route | HTTP | Observation |
| :--- | :---: | :--- |
| `/` | 200 | Hero headline: **"AI answers. We verify them."** — State pills `DETECT` / `CONFIDENCE` / `VERIFY` visible. Three.js wireframe neural canvas rendering on the right. Split editorial layout confirmed. |
| `/verify` | 200 | Dashboard workbench fully functional. No 3D canvas or layout errors. |
| `/overview` | 200 | Command Center with live telemetry and pipeline health: `pipeline`, `nli model`, `p1 hybrid`, `retriever`, `fusion engine` all `✓ Verified`. |

**Confirmed Railway deployment**: `354eecec` at 2026-09-02 11:44:18 IST
