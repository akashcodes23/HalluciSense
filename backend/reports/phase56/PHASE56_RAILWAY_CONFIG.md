# Phase 56 — Railway Service Configuration

## Service Identification

| Property | Value |
| :--- | :--- |
| **Project Name** | `passionate-contentment` |
| **Project ID** | `2c0fdad7-7765-475c-a41a-7315afb700b7` |
| **Environment** | `production` (`b69f4974-053f-4f1f-bbf8-68991e501f39`) |
| **Service Name** | `HalluciSense` |
| **Service ID** | `a449c886-d20f-4eb3-b461-81cb5b9944ea` |
| **Public Domain** | `https://hallucisense-production.up.railway.app` |
| **Root Directory** | `/backend` |
| **Builder** | `DOCKERFILE` |
| **Start Command** | `python start.py` |
| **Healthcheck Path** | `/health` (Timeout: 300s) |
| **Restart Policy** | `ON_FAILURE` (Max Retries: 3) |
| **Replicas** | 1 (sfo region) |
| **Plan** | Trial / Hobby (1024 MB RAM limit) |
