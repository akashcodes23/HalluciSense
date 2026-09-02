# Phase 56 — Memory Telemetry & Time Series

## Railway Time Series (Spike Window: 06:35 to 06:45 UTC)

| Timestamp (UTC) | Allocated Memory (GB) | Allocated Memory (MB) | Status |
| :--- | :--- | :--- | :--- |
| `06:37:00` | 0.567 GB | 567.7 MB | Baseline Python + FastAPI |
| `06:38:00` | 0.567 GB | 567.4 MB | Background task starts |
| `06:39:00` | 0.893 GB | 893.8 MB | HuggingFace downloads / reads weights |
| `06:40:00` | 0.894 GB | 894.5 MB | State dict in-memory deserialization |
| `06:41:00` | 0.894 GB | 894.7 MB | Module allocation |
| `06:42:00` | **1.107 GB** | **1,107.5 MB** | **EXCEEDS 1024 MB LIMIT** |
| `06:43:00` | 0.261 GB | 261.2 MB | Container restarted (fresh PID [1]) |
| `06:54:00` | 0.507 GB | 507.0 MB | Post-retry resting state |

### Telemetry Summary
- **Baseline Idle**: 261 MB – 507 MB
- **Peak Observed**: 1,672.82 MB (recorded max aggregate) / 1,107.5 MB (time-series sample)
- **Container Limit**: 1,023.99 MB
- **Spike Factor**: 2.1x over baseline
