# Sub-10 Minute Quickstart Guide

Get **HalluciSense v1.0** up and running in under 10 minutes.

---

## 1. Prerequisites
- **Python 3.10+** (Python 3.10.12 or 3.11 recommended)
- **Node.js 18+** (For frontend web dashboard)
- **Docker & Docker Compose** (Optional for containerized launch)

---

## 2. One-Command Master Reproducibility & Server Launch

```bash
# Clone the repository
git clone https://github.com/akashcodes23/HalluciSense.git
cd HalluciSense/backend

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the master scientific reproducibility pipeline
python run_all_experiments.py

# Start local FastAPI server
python start.py
```

The API server will launch at `http://localhost:8000`. Access interactive API documentation at `http://localhost:8000/docs`.

---

## 3. Verify Server Health

```bash
curl http://localhost:8000/health
```
**Expected Response**:
```json
{
  "status": "healthy",
  "service": "HalluciSense API",
  "version": "1.0.0-rc1"
}
```
