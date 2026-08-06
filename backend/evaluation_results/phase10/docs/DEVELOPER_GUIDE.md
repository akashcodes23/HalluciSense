# HalluciSense Phase 10 — Developer Guide

*Generated: 2026-08-03T05:01:37.151249+00:00*

## Getting Started

All Pillar 2 code resides in `app/pillar2/` and API endpoints in `app/modules/pillar2/router.py`.

### Running Pillar 2 Tests

```bash
source venv/bin/activate
pytest tests/test_pillar2_*.py -v
```

### Running Pipeline Exporter

```bash
python -m evaluation.phase10.run_phase10_pipeline
```
