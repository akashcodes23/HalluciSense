# HalluciSense Artifact Installation Instructions

## System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS 12+, or Windows WSL2
- **Python**: Version 3.10 or higher
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Disk Space**: 5 GB available storage

---

## Installation Steps

### Option A: Local Python Environment
```bash
# 1. Clone repository
git clone https://github.com/akashcodes23/HalluciSense.git
cd HalluciSense

# 2. Execute automated installer
bash scripts/fresh_install.sh
```

### Option B: Docker Container
```bash
# 1. Build & launch via Docker Compose
docker compose up --build
```
