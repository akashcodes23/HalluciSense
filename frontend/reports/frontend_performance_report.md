# HalluciSense v1.0 Frontend Performance Report

**Date**: 2026-08-07  
**Target Thresholds**: LCP < 1.5s, FCP < 0.8s, TTFB < 100ms, Bundle JS < 300KB  

---

## Core Web Vitals & Loading Performance

| Metric | Target Threshold | Empirical Value | Verdict |
| :--- | :---: | :---: | :---: |
| Largest Contentful Paint (LCP) | <= 1500 ms | 740 ms | ✅ PASS |
| First Contentful Paint (FCP) | <= 800 ms | 380 ms | ✅ PASS |
| Time to First Byte (TTFB) | <= 100 ms | 28 ms | ✅ PASS |
| Cumulative Layout Shift (CLS) | <= 0.10 | 0.002 | ✅ PASS |
| Interaction to Next Paint (INP) | <= 200 ms | 45 ms | ✅ PASS |
| First Load JS Bundle Size | <= 300 KB | 184 KB | ✅ PASS |

---

## Production Build Bundle Breakdown

- **Next.js Turbopack Compilation Time**: 2.4s
- **TypeScript Type Checking**: 2.0s
- **Static HTML Prerendering**: 204ms (18/18 static pages)
- **Zero Hydration Errors**: Verified across all 8 application routes.