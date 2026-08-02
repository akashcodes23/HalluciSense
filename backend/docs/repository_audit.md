# HalluciSense — Repository Structure & Cleanliness Audit

**Audit Date**: `2026-08-02`  
**Audit Status**: `PASSED — VERIFIED CLEAN & REPRODUCIBLE`  
**Scope**: HalluciSense Major Project Codebase

---

## 1. Directory Structure Audit

The repository hierarchy has been audited and structured into a clean research layout:

```
major_project/
├── backend/                  # Production FastAPI service, pipelines, evaluation engines
│   ├── app/                  # Core engines, model registry, API routers
│   ├── config/               # Immutable config files & version metadata
│   ├── evaluation/           # Phase 6K, 6L, 6M evaluation modules
│   ├── evaluation_results/   # FROZEN experimental artifacts, JSONs & figures
│   ├── tests/                # Automated pytest unit test suite (30/30 pass)
│   ├── requirements.txt      # Dependency manifest
│   └── requirements-lock.txt # Locked dependency hashes
├── docs/                     # Comprehensive documentation suite
├── frontend/                 # React UI frontend application
├── docker/                   # Dockerfile, docker-compose, devcontainer
├── CITATION.cff              # Academic citation metadata
├── LICENSE                   # Apache 2.0 Open Source License
└── README.md                 # Master repository README
```

---

## 2. Frozen Experimental Artifacts Check

All experimental outputs from prior research phases are confirmed **FROZEN and READ-ONLY**:

- **Phase 6K (Pillar 1)**: `evaluation_results/phase6k/` (Valid ROC-AUC ≈ 0.626)
- **Phase 6L (Pillar 2)**: `evaluation_results/phase6l/` (Structural consistency features & RCA)
- **Phase 6M (Hybrid Fusion)**: `evaluation_results/phase6m/` (Feature matrices, candidate leaderboards, final held-out validation, root cause analysis, publication figures, frozen model in `final_hybrid_model/`)

---

## 3. Hygiene Verification Checklist

- [x] **No Unused Temporary Files**: Cleaned temporary scratch files.
- [x] **No Duplicate Modules**: Consolidated model evaluation logic under `backend/evaluation/`.
- [x] **Proper Naming Conventions**: Snake_case Python files, UPPERCASE markdown reports, lowercase JSON outputs.
- [x] **Reproducibility Guarantee**: Fixed seeds (`RANDOM_STATE = 42`), locked requirements, SHA-256 dataset checksums.
