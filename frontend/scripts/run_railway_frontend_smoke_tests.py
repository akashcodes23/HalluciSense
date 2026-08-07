#!/usr/bin/env python3
"""HalluciSense v1.0 — Railway Frontend Production Smoke Test Suite (Sprint 3.1B).

Executes automated production smoke testing against live Next.js frontend routes
and verifies integration with the production backend API.

Usage:
  python3 frontend/scripts/run_railway_frontend_smoke_tests.py [FRONTEND_URL] [BACKEND_URL]
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from typing import Dict, Any

FRONTEND_URL = sys.argv[1] if len(sys.argv) > 1 else os.getenv("FRONTEND_URL", "http://127.0.0.1:3000")
BACKEND_URL = sys.argv[2] if len(sys.argv) > 2 else os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}")


def make_request(url: str, method: str = "GET", payload: Dict[str, Any] = None) -> tuple[int, str, float]:
    """Execute HTTP request and return (status_code, body_string, duration_ms)."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/html,application/json,application/xhtml+xml,*/*",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            duration_ms = (time.time() - start) * 1000
            content = resp.read().decode("utf-8")
            return resp.status, content, round(duration_ms, 2)
    except urllib.error.HTTPError as e:
        duration_ms = (time.time() - start) * 1000
        try:
            content = e.read().decode("utf-8")
        except Exception:
            content = str(e)
        return e.code, content, round(duration_ms, 2)
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        return 500, str(e), round(duration_ms, 2)


def main():
    log("================================================================================")
    log(f"HALLUCISENSE v1.0 — RAILWAY FRONTEND PRODUCTION SMOKE TEST SUITE")
    log(f"Frontend URL : {FRONTEND_URL}")
    log(f"Backend URL  : {BACKEND_URL}")
    log("================================================================================")

    fe_routes = [
        ("Landing Page /", f"{FRONTEND_URL}/", "HalluciSense"),
        ("Analyzer Workspace /analyze", f"{FRONTEND_URL}/analyze", "Analyzer"),
        ("Pipeline Traces /traces", f"{FRONTEND_URL}/traces", "Traces"),
        ("Metrics Telemetry /metrics", f"{FRONTEND_URL}/metrics", "Metrics"),
        ("Client Settings /settings", f"{FRONTEND_URL}/settings", "Settings"),
    ]

    failed_count = 0

    log("\n[1/2] Auditing Next.js App Router Static & Dynamic Pages...")
    for name, url, expected_text in fe_routes:
        status_code, body, duration_ms = make_request(url, "GET")
        passed = status_code == 200 and expected_text in body
        if passed:
            log(f"  - {name:<38} -> Status: {status_code} ({duration_ms} ms) | ✅ PASS")
        else:
            log(f"  - {name:<38} -> Status: {status_code} ({duration_ms} ms) | ❌ FAIL (Expected text: '{expected_text}')")
            failed_count += 1

    log("\n[2/2] Verifying Live Backend Integration from Frontend Client...")
    status_code, body, duration_ms = make_request(f"{BACKEND_URL}/api/v1/analyze", "POST", {
        "query": "Who invented the telephone?",
        "response": "Alexander Graham Bell invented the telephone in 1876.",
        "model_name": "GPT-4"
    })
    backend_passed = status_code == 200 and "overall_h_score" in body
    if backend_passed:
        log(f"  - Production API Connection /api/v1/analyze -> Status: {status_code} ({duration_ms} ms) | ✅ PASS")
    else:
        log(f"  - Production API Connection /api/v1/analyze -> Status: {status_code} ({duration_ms} ms) | ❌ FAIL")
        failed_count += 1

    log("\n================================================================================")
    if failed_count == 0:
        log("✅ ALL RAILWAY PRODUCTION FRONTEND SMOKE TESTS PASSED CLEANLY!")
        sys.exit(0)
    else:
        log(f"❌ SMOKE TEST FAILURE: {failed_count} route(s)/API test(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
