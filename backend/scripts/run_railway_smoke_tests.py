#!/usr/bin/env python3
"""HalluciSense v1.0 — Railway Production Smoke Test Suite (Part 7).

Executes live smoke verification against local or deployed Railway backend instance:
- GET /
- GET /health
- GET /ready
- POST /api/v1/analyze
- POST /api/v1/explain
- GET /api/v1/metrics
- GET /api/v1/debug/latest
- GET /api/v1/debug/{trace_id}

Usage:
  python3 backend/scripts/run_railway_smoke_tests.py [TARGET_URL]
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from typing import Dict, Any

TARGET_URL = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TARGET_URL", "http://127.0.0.1:8000")


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}")


def make_request(url: str, method: str = "GET", payload: Dict[str, Any] = None) -> tuple[int, Dict[str, Any], float]:
    """Execute HTTP request and return (status_code, response_json, duration_ms)."""
    headers = {"Content-Type": "application/json", "User-Agent": "HalluciSense-Railway-SmokeTest/1.0"}
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            duration_ms = (time.time() - start) * 1000
            content = resp.read().decode("utf-8")
            return resp.status, json.loads(content), round(duration_ms, 2)
    except urllib.error.HTTPError as e:
        duration_ms = (time.time() - start) * 1000
        try:
            content = e.read().decode("utf-8")
            body = json.loads(content)
        except Exception:
            body = {"detail": str(e)}
        return e.code, body, round(duration_ms, 2)
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        return 500, {"detail": str(e)}, round(duration_ms, 2)


def main():
    log("================================================================================")
    log(f"HALLUCISENSE v1.0 — RAILWAY PRODUCTION BACKEND SMOKE TEST SUITE")
    log(f"Target URL: {TARGET_URL}")
    log("================================================================================")

    endpoints = [
        ("Root Route /", f"{TARGET_URL}/", "GET", None, 200, ["message", "version", "environment"]),
        ("Health Probe /health", f"{TARGET_URL}/health", "GET", None, 200, ["status"]),
        ("Readiness Probe /ready", f"{TARGET_URL}/ready", "GET", None, 200, ["status", "components"]),
        ("Metrics Telemetry /api/v1/metrics", f"{TARGET_URL}/api/v1/metrics", "GET", None, 200, ["requests", "average_latency_ms", "success_rate"]),
        ("Latest Debug Trace /api/v1/debug/latest", f"{TARGET_URL}/api/v1/debug/latest", "GET", None, 200, ["trace_id", "stages", "summary"]),
        ("Analyze Endpoint /api/v1/analyze", f"{TARGET_URL}/api/v1/analyze", "POST", {
            "query": "Who invented the telephone?",
            "response": "Alexander Graham Bell invented the telephone in 1876.",
            "model_name": "GPT-4"
        }, 200, ["trace_id", "overall_h_score", "risk_level", "pillar_scores"]),
        ("Explain Endpoint /api/v1/explain", f"{TARGET_URL}/api/v1/explain", "POST", {
            "query": "Who invented the telephone?",
            "response": "Alexander Graham Bell invented the telephone in 1876.",
            "model_name": "GPT-4"
        }, 200, ["trace_id", "retrieved_evidence", "reasoning_chain", "adaptive_weights"]),
    ]

    failed_count = 0
    trace_id_to_check = None

    for name, url, method, payload, expected_status, required_keys in endpoints:
        status_code, body, duration_ms = make_request(url, method, payload)
        keys_present = all(k in body for k in required_keys) if isinstance(body, dict) else False
        passed = status_code == expected_status and keys_present

        if passed:
            log(f"  - {name:<42} -> Status: {status_code} ({duration_ms} ms) | ✅ PASS")
        else:
            log(f"  - {name:<42} -> Status: {status_code} ({duration_ms} ms) | ❌ FAIL (Missing keys: {required_keys})")
            failed_count += 1

        if name.startswith("Analyze Endpoint") and isinstance(body, dict):
            trace_id_to_check = body.get("trace_id")

    # Trace ID test
    if trace_id_to_check:
        url = f"{TARGET_URL}/api/v1/debug/{trace_id_to_check}"
        status_code, body, duration_ms = make_request(url, "GET")
        passed = status_code == 200 and "trace_id" in body
        if passed:
            log(f"  - Debug Trace by ID /api/v1/debug/{{id}}     -> Status: {status_code} ({duration_ms} ms) | ✅ PASS")
        else:
            log(f"  - Debug Trace by ID /api/v1/debug/{{id}}     -> Status: {status_code} ({duration_ms} ms) | ❌ FAIL")
            failed_count += 1

    log("================================================================================")
    if failed_count == 0:
        log("✅ ALL RAILWAY PRODUCTION BACKEND SMOKE TESTS PASSED CLEANLY!")
        sys.exit(0)
    else:
        log(f"❌ SMOKE TEST FAILURE: {failed_count} endpoint(s) failed smoke verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
