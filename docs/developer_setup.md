# Developer Setup & Contribution Environment

Set up a full development environment for HalluciSense.

---

## 1. Environment Setup
```bash
git clone https://github.com/akashcodes23/HalluciSense.git
cd HalluciSense

# Backend virtualenv
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Frontend npm dependencies
cd ../frontend
npm install
```

---

## 2. Running Tests
```bash
# Run pytest verification suite
cd backend
pytest tests/test_unit_pipeline.py tests/test_integration_api.py tests/test_benchmark_runner.py tests/test_statistical_tests.py -v
```
