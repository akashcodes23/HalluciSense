"""Debug single request timing and find slow step."""
import time
import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.engine.model_registry import ModelRegistry

def test_req_11():
    text = "Paris is the capital of France. Berlin is the capital of Germany."
    pipeline = ModelRegistry.get_pipeline()

    print("1. Extract claims...")
    t0 = time.perf_counter()
    claims = pipeline.p1_engine.extract_claims(text)
    print(f"  Claims ({time.perf_counter()-t0:.2f}s): {claims}")

    print("2. Retrieve evidence...")
    t0 = time.perf_counter()
    ev = pipeline._retrieve_evidence(text)
    print(f"  Evidence count ({time.perf_counter()-t0:.2f}s): {len(ev)}")

    print("3. Analyze response...")
    t0 = time.perf_counter()
    rep = pipeline.analyze(text)
    print(f"  Complete analyze ({time.perf_counter()-t0:.2f}s): overall_h={rep.overall_h_score}")

if __name__ == "__main__":
    test_req_11()
