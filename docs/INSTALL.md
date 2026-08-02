# HalluciSense Installation Guide

## Requirements
- Python 3.10+
- Git & Git LFS
- 8GB RAM minimum (16GB recommended for NLI cross-encoder inference)

## Local Installation

```bash
# 1. Clone repository
git clone https://github.com/akashgpatil/hallucisense.git
cd hallucisense/backend

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Install locked dependencies
pip install -r requirements-lock.txt

# 4. Verify installation
python -m pytest tests/ -v
```

## Docker Installation

```bash
cd hallucisense
docker-compose -f docker/docker-compose.yml up --build -d
```
