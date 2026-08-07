# HalluciSense v1.0 Monitoring Dashboard & Alert Specifications

**Target Scraper**: Prometheus / Datadog / Grafana  
**Scrape Target**: `GET /api/v1/metrics/prometheus` or `GET /metrics`  

---

## 1. Grafana Dashboard Panel Specifications

| Panel Title | Metric Expression | Type | Threshold |
| :--- | :--- | :---: | :---: |
| **Request Throughput** | `rate(hallucisense_requests_total[1m])` | Time Series Graph | $> 10\text{ req/s}$ |
| **Average Latency** | `hallucisense_request_latency_seconds` | Gauge / Graph | $< 0.25\text{ s}$ |
| **Process RAM RSS** | `hallucisense_process_memory_bytes / 1024 / 1024` | Stat Card | $< 1500\text{ MB}$ |
| **Average H-Score** | `hallucisense_average_h_score * 100` | Gauge | $0 - 100\%$ |
| **Success Rate %** | `hallucisense_success_rate_percent` | Stat Card | $> 99.0\%$ |

---

## 2. Production Alert Rules Configuration

```yaml
groups:
  - name: hallucisense_alerts
    rules:
      - alert: HighLatencyAlert
        expr: hallucisense_request_latency_seconds > 0.500
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "P95 request latency exceeded 500ms for 5 consecutive minutes"

      - alert: MemorySpikeAlert
        expr: hallucisense_process_memory_bytes > 1610612736
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Backend process memory RSS exceeded 1.5 GB"

      - alert: ReadinessFailureAlert
        expr: hallucisense_success_rate_percent < 95.0
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "System request success rate dropped below 95%"
```
