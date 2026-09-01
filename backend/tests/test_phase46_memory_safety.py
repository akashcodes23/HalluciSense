"""Phase 46 — Memory Safety & Bounded Resource Tests."""

import os
import psutil
import pytest
from app.core.pipeline import HalluciSensePipeline

def test_memory_below_production_ceiling():
    """Ensure fresh process memory remains strictly below 1024 MB Railway limit."""
    import subprocess
    import sys
    code = """
import psutil, os
from app.core.pipeline import HalluciSensePipeline
pipeline = HalluciSensePipeline()
for i in range(5):
    pipeline.predict(response_text=f"Batch iteration {i} asserting that 2 + 2 = 4.")
proc = psutil.Process(os.getpid())
rss_mb = proc.memory_info().rss / (1024 * 1024)
print(f"ISOLATED_RSS:{rss_mb:.2f}")
assert rss_mb < 900.0, f"Memory {rss_mb} exceeded 900MB"
"""
    from pathlib import Path
    backend_dir = Path(__file__).resolve().parent.parent
    res = subprocess.run([sys.executable, "-c", code], cwd=str(backend_dir), capture_output=True, text=True)
    assert res.returncode == 0, f"Process failed: {res.stderr}\n{res.stdout}"
