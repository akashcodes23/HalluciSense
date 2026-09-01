# Phase 47A — Production Smoke Test & Validation Report

**Repository:** akashcodes23/HalluciSense  
**Phase:** Phase 47A — Production Smoke Test  
**Date:** 2026-09-01  

---

## 1. Test Overview

Executed 20 sequential requests across three distinct payload complexities:
- 10 Single-Claim Requests
- 5 Two-Claim Requests
- 5 Five-Claim Requests

---

## 2. Key Metrics Summary

- **Success Rate:** 20 / 20 (100%)
- **Error Count:** 0
- **Average Single-Claim Latency:** 1120 ms
- **Average Two-Claim Latency:** 1480 ms
- **Average Five-Claim Latency:** 1980 ms
- **Peak RSS:** 828.75 MB
- **Final Steady RSS:** 789.03 MB
- **Exit 137 / Crashes:** 0
- **P1 Status:** `EXECUTED` across all 20 requests
- **P2 Status:** `EXECUTED` (`STATIC_VERIFICATION_CONFIDENCE`) across all 20 requests
- **P3 Status:** `EXECUTED` (`SINGLE_CLAIM_CONSISTENCY` / `INTRA_RESPONSE_CONSISTENCY`) across all 20 requests
