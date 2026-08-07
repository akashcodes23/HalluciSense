# HalluciSense v1.0 Service Level Agreement (SLA) & Service Level Objectives (SLO)

---

## 1. Service Level Objectives (SLO) & Indicators (SLI)

| Service Metric | Service Level Objective (SLO) | Measurement Method |
| :--- | :---: | :--- |
| **Availability (Uptime)** | **99.9% Uptime** | `/health` probe HTTP 200 OK |
| **API Latency P90** | $< 250\text{ ms}$ | `hallucisense_request_latency_seconds` |
| **API Latency P95** | $< 350\text{ ms}$ | `hallucisense_request_latency_seconds` |
| **Successful Request Rate** | $> 99.5\%$ | `hallucisense_success_rate_percent` |
| **Trace Generation Reliability** | $100.0\%$ | `/api/v1/debug/{trace_id}` availability |

---

## 2. Service Level Agreement (SLA) Guarantees

1. **Availability**: Monthly uptime commitment of 99.9%.
2. **Support Response Time**: SEV-1 issues acknowledged within 15 minutes.
