# Testing & Quality Assurance Strategy

HalluciSense employs a comprehensive, multi-tiered test suite ensuring scientific non-regression, end-to-end integration, and production reliability.

---

## 1. Test Architecture Summary

| Suite Name | Scope & Coverage | Invariant / Target |
| :--- | :--- | :--- |
| **Scientific Non-Regression** | Phase 12–20 pipeline validation, calibration, abstention, fusion equations | 72 tests (100% pass) |
| **Benchmark Integrity** | Frozen dataset SHA-256 hash invariant | `dfe8c6e...9efd5` |
| **ModelRegistry Singleton** | Single shared inference engine lifecycle | Max 1 instance per worker |
| **Failure Semantics** | Graceful failure mapping under zero signal presence | `status="FAILED"`, `h_score=null` |
| **Frontend Production Build** | TypeScript strict mode, Next.js page generation | 23 routes (0 errors) |

---

## 2. Executing Automated Tests

### Run the Full Pytest Suite (72 Regression Tests)
```bash
cd backend
PYTHONPATH=. venv/bin/pytest tests/ -v
```

### Run Benchmark Integrity Test
```bash
PYTHONPATH=. venv/bin/pytest tests/test_phase12_e2e.py -k "test_benchmark_dataset_sha256_integrity" -v
```

### Run Production Reliability Suite
```bash
PYTHONPATH=. venv/bin/pytest tests/test_phase20_production_reliability.py -v
```

### Validate Frontend Production Build
```bash
cd ../frontend
npm run build
```

---

## 3. Verified Test Execution Output
```
======================= 72 passed, 20 warnings in 22.53s =======================
✓ Next.js 16.2.11 (Turbopack)
✓ Compiled successfully in 1840ms
✓ Generating static pages using 9 workers (23/23) in 220ms
```
