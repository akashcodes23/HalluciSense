# HalluciSense Phase 10 — Deployment Guide

*Generated: 2026-08-03T05:01:37.151249+00:00*

Pillar 2 runs as an integrated service within the HalluciSense FastAPI application container.
Docker CMD starts Uvicorn workers serving `/api/v1/pillar2/*` endpoints.
