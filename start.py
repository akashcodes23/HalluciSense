"""Root Entrypoint for Railway / Production Deployment."""

import os
import sys
from pathlib import Path

# Add backend directory to sys.path
root_dir = Path(__file__).resolve().parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import uvicorn

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
