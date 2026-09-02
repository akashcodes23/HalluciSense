# Phase 56 — Deployment History Timeline

## Backend Deployment History (Service: `HalluciSense`)

| Deployment ID | Status | Created At | Commit / Source | Finding |
| :--- | :--- | :--- | :--- | :--- |
| `ebf3c68e-1991-4fe9-816d-c50ac05e5e65` | **CRASHED** | 2026-09-02 12:11:18 IST | `6a47927` | OOM Killer SIGKILL during DeBERTa loading in background warmup |
| `bad0caad-6dd3-4498-9ef8-08cafb8dc9a0` | **REMOVED** | 2026-09-02 11:52:30 IST | `2778598` | Superseded |
| `eef312b5-7bac-405d-b528-6c20bc078fe1` | **REMOVED** | 2026-09-02 11:44:18 IST | `565ace7` | Build succeeded; OOM crash on warmup |
| `80057fde-4b38-4122-afc5-db921dad3f5d` | **REMOVED** | 2026-09-02 11:13:42 IST | `2a672c6` | OOM crash during model load |
| `b64d0555-1d95-4512-8568-ec9bc016e044` | **REMOVED** | 2026-09-02 11:02:54 IST | `8d894f1` | OOM crash during model load |
| `6bbfe0db-b68e-4cfb-b257-d0ee450b9a03` | **REMOVED** | 2026-09-02 10:43:28 IST | `2fada89` | OOM crash during model load |
| `3ef2191a-ff60-4d28-bbec-b9823649bfb6` | **REMOVED** | 2026-09-02 10:18:13 IST | `f591b11` | OOM crash during model load |
| `c6656524-769d-4f6d-a9c9-f659ed7fe7b5` | **REMOVED** | 2026-09-01 22:47:07 IST | `8afe3b7` | OOM crash during model load |
| `baebf1c9-cdff-4a64-b65b-43f44cc6d3f3` | **REMOVED** | 2026-09-01 22:19:27 IST | `5689cc0` | OOM crash during model load |

**Correlation**: Failures began when standard fp32 `AutoModelForSequenceClassification.from_pretrained` was introduced without memory streaming or explicit glibc allocator trimming, causing transient allocation spikes >1024 MB during container initialization.
