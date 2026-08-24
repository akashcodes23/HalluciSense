# Local Development Setup Guide

This guide details instructions for setting up, running, and debugging HalluciSense on macOS, Linux, and Windows (WSL2).

---

## 1. System Requirements
- **Python**: 3.10.x or 3.11.x
- **Node.js**: v18.x or v20.x
- **npm**: v9.x or v10.x
- **RAM**: Minimum 8 GB (16 GB recommended for local vector indexing)
- **Disk**: 5 GB free disk space for sentence-transformer and NLI models

---

## 2. Backend Setup

### Step 1: Virtual Environment Creation
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a local `.env` file from the provided template:
```bash
cp .env.example .env
```

*Sample Configuration (`backend/.env`)*:
```ini
APP_ENV=development
PORT=8000
HOST=0.0.0.0
CORS_ORIGINS=*

# AI Provider Keys (Optional for basic verification)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Engine Parameters
ALPHA_FACTUAL_ERROR=0.45
BETA_CONFIDENCE_GAP=0.30
GAMMA_CONSISTENCY_FAILURE=0.25
VERIFIED_THRESHOLD=0.35
HALLUCINATED_THRESHOLD=0.65
RATE_LIMIT_PER_MINUTE=100
```

### Step 4: Run Backend Server
```bash
PYTHONPATH=. venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Check health: `curl http://localhost:8000/health`

---

## 3. Frontend Setup

### Step 1: Install Node Dependencies
```bash
cd ../frontend
npm install
```

### Step 2: Configure Environment
Create `.env.local`:
```ini
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### Step 3: Run Development Server
```bash
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## 4. Running Regression Tests
```bash
cd backend
PYTHONPATH=. venv/bin/pytest tests/ -v
```

## 5. Building for Production
```bash
# Verify frontend production bundle
cd frontend
npm run build
```
