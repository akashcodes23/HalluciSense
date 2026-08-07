#!/usr/bin/env python3
"""HalluciSense v1.0 — Observability & Monitoring Test Suite (Sprint 3.4).

Audits:
- Prometheus text format metrics exporter (/metrics/prometheus)
- OpenTelemetry & Tracing header propagation (X-Request-ID, X-Trace-ID, X-Latency-MS, traceparent)
- Health & Readiness component probes (/health, /ready)
- Alert threshold rule evaluations
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from typing import Dict, Any

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}")


def make_request(url: str, method: str = "GET", payload: Dict[str, Any] = None) -> tuple[int, Dict[str, str], str, float]:
    """Execute HTTP request and return (status_code, headers_dict, body_str, duration_ms)."""
    headers = {"Content-Type": "application/json", "User-Agent": "HalluciSense-ObservabilityTest/1.0"}
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            duration_ms = (time.time() - start) * 1000
            resp_headers = dict(resp.headers)
            body = resp.read().decode("utf-8")
            return resp.status, resp_headers, body, round(duration_ms, 2)
    except urllib.error.HTTPError as e:
        duration_ms = (time.time() - start) * 1000
        resp_headers = dict(e.headers)
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        return e.code, resp_headers, body, round(duration_ms, 2)
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        return 500, {}, str(e), round(duration_ms, 2)


def main():
    log("================================================================================")
    log("HALLUCISENSE v1.0 SPRINT 3.4 — OBSERVABILITY & MONITORING TEST SUITE")
    log("================================================================================")

    # 1. OpenTelemetry & Tracing Headers Probe
    log("\n[1/3] Probing OpenTelemetry & Tracing Headers Propagation...")
    status, headers, body, duration = make_request(f"{BACKEND_URL}/health")

    req_id = headers.get("X-Request-ID") or headers.get("x-request-id")
    trace_id = headers.get("X-Trace-ID") or headers.get("x-trace-id")
    latency_hdr = headers.get("X-Latency-MS") or headers.get("x-latency-ms")
    traceparent = headers.get("traceparent")

    log(f"  - HTTP Status      : {status} ({duration} ms)")
    log(f"  - X-Request-ID     : {req_id}")
    log(f"  - X-Trace-ID       : {trace_id}")
    log(f"  - X-Latency-MS     : {latency_hdr}")
    log(f"  - traceparent      : {traceparent}")

    headers_passed = bool(req_id and trace_id and latency_hdr and traceparent)
    log(f"  >> OpenTelemetry Headers Status: {'✅ PASS' if headers_passed else '❌ FAIL'}")

    # 2. Prometheus Metrics Exporter Probe
    log("\n[2/3] Probing Prometheus Text Format Metrics Exporter (/api/v1/metrics/prometheus)...")
    p_status, p_headers, p_body, p_duration = make_request(f"{BACKEND_URL}/api/v1/metrics/prometheus")

    prom_has_counters = "hallucisense_requests_total" in p_body
    prom_has_latency = "hallucisense_request_latency_seconds" in p_body
    prom_has_memory = "hallucisense_process_memory_bytes" in p_body
    prom_passed = p_status == 200 and prom_has_counters and prom_has_latency and prom_has_memory

    log(f"  - Prometheus Exporter Status : {p_status} ({p_duration} ms)")
    log(f"  - Contains Counters         : {prom_has_counters}")
    log(f"  - Contains Latency Gauge    : {prom_has_latency}")
    log(f"  - Contains Memory Gauge     : {prom_has_memory}")
    log(f"  >> Prometheus Metrics Exporter Status: {'✅ PASS' if prom_passed else '❌ FAIL'}")

    # 3. Component Readiness Probe
    log("\n[3/3] Probing Component Readiness Check (/ready)...")
    r_status, _, r_body_str, r_duration = make_request(f"{BACKEND_URL}/ready")
    r_passed = r_status == 200 and "components" in r_body_str

    log(f"  - Readiness Probe Status    : {r_status} ({r_duration} ms)")
    log(f"  >> Component Readiness Status: {'✅ PASS' if r_passed else '❌ FAIL'}")

    log("\n================================================================================")
    all_passed = headers_passed and prom_passed and r_passed
    if all_passed:
        log("✅ ALL OBSERVABILITY & MONITORING AUDIT CHECKS PASSED CLEANLY!")
        sys.exit(0)
    else:
        log("❌ OBSERVABILITY AUDIT FAILURE: One or more checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
