# HalluciSense Scientific Reproducibility Manifest

## 1. Environment & Software Lock

| Dependency | Verified Version | Package Manager / Source | Role |
| :--- | :--- | :--- | :--- |
| **Python** | `3.10.12` / `3.11-slim` | Standard CPython | Base Runtime |
| **NumPy** | `1.26.4` | PyPI | Array and Tensor Math |
| **SciPy** | `1.15.3` | PyPI | Scientific / Statistical Functions |
| **scikit-learn** | `1.7.2` | PyPI | HistGradientBoosting & RobustScaler |
| **joblib** | `1.5.2` | PyPI | Model Persistence |
| **PyTorch** | `2.5.1` (CPU) | PyPI (torch) | DeBERTa Tensor Inference |
| **Transformers** | `4.47.1` | PyPI (HuggingFace) | AutoModel & AutoTokenizer |
| **sentence-transformers** | `3.3.1` | PyPI | Dense Semantic Embeddings |
| **FAISS (CPU)** | `1.9.0` | PyPI (faiss-cpu) | Dense Vector Search |
| **Accelerate** | `1.14.0` | PyPI | PyTorch Memory Helper |

---

## 2. Cryptographic Checksum Registry

### A. Production Model Artifacts
```
backend/evaluation_results/phase6m/final_hybrid_model/hybrid_meta_classifier.joblib:
  089ebd2d277d1c21adc0541b71f1bf3e4cb5927d6e74f3ed96b1d00b15337cad (218,104 bytes)

backend/evaluation_results/phase6m/final_hybrid_model/hybrid_meta_classifier.joblib.backup:
  cb459fd99b3da606f78c5777cbf87dee482e59ef60e27168f7656306b4a22fbf (218,344 bytes)

backend/evaluation_results/phase6m/final_hybrid_model/preprocessing.joblib:
  bdbd42e3f386b7b2602e95b1fc32b6ded1ac404779498190442d17aec2f97e90 (799 bytes)

backend/evaluation_results/phase6m/final_hybrid_model/feature_schema.json:
  942df39475c1cabc54b5f472d2ef111cfa511b3ba24050115b9bb57177db0388 (485 bytes)

backend/evaluation_results/phase6m/final_hybrid_model/model_metadata.json:
  69d8c63219de4fa27a62b0a351d78a1fdea1107775b871fc2f0391f353b11f74 (1,356 bytes)
```

### B. Datasets & Requirements
```
backend/requirements.txt:
  72ed66de4f3c99d0642fdf95dd948bb5dfb272b862fe55dcc2ca67143d4d0e9a (1,543 bytes)

backend/evaluation/results/benchmark_dataset.jsonl:
  dfe8c6e48d9b8250667de019047cbc85957eac3a189c7b6ef58de0ef6059efd5 (295,354 bytes, 750 records)
```

---

## 3. Production Deployment Metadata

- **Platform**: Railway Cloud Containers
- **Project ID**: `2c0fdad7-7765-475c-a41a-7315afb700b7`
- **Service ID**: `a449c886-d20f-4eb3-b461-81cb5b9944ea`
- **Active Deployment ID**: `41efbc6e-4124-49eb-be3e-4c702f685a9f`
- **Git Commit**: `c548e96` / `63d4f32`
- **Live Endpoint**: `https://hallucisense-production.up.railway.app`

---

## 4. Reproducibility Classification Table

| Experiment / Capability | Reproducibility Status | Verification Command |
| :--- | :--- | :--- |
| **Model Bit-for-Bit Equivalence** | **FULLY REPRODUCIBLE** | `python3 -c "from app.models.registry import safe_joblib_load; ..."` |
| **Model Metadata & Schema** | **FULLY REPRODUCIBLE** | `cat backend/evaluation_results/phase6m/final_hybrid_model/model_metadata.json` |
| **Benchmark Dataset Checksum & Counts**| **FULLY REPRODUCIBLE** | `wc -l backend/evaluation/results/benchmark_dataset.jsonl` (750 lines) |
| **Unit & Pipeline Regression** | **FULLY REPRODUCIBLE** | `python3 -m pytest backend/tests/test_unit_pipeline.py -v` |
| **Production Live Health & Inference** | **FULLY REPRODUCIBLE** | `python3 backend/tests/test_smoke_production.py` |
| **Full 58k Sample Training Loop** | **OFFLINE REPRODUCIBLE** | Requires full feature extraction matrix (documented in `phase6m/run_phase6m_3.py`) |
