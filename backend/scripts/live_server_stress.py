"""Live Uvicorn Server Memory Forensics & 50-Request Benchmark.

Measures the ACTUAL production memory profile of the uvicorn process receiving live HTTP requests.
"""

import os
import sys
import time
import subprocess
import requests
import statistics
import concurrent.futures
from pathlib import Path
import psutil

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import socket

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_live_server_benchmark():
    print("=" * 80)
    print("PHASE 48 — LIVE UVICORN PRODUCTION SERVER MEMORY BENCHMARK")
    print("=" * 80)

    port = get_free_port()
    print(f"Allocated free dynamic port: {port}")

    # 1. Start live uvicorn server process
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["APP_ENV"] = "production"
    env["MALLOC_ARENA_MAX"] = "2"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(backend_dir)
    env["RATE_LIMIT_PER_MINUTE"] = "1000"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
        cwd=str(backend_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    time.sleep(3.0)  # Wait for startup
    server_ps = psutil.Process(proc.pid)

    # Poll /ready until READY
    ready = False
    for _ in range(60):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/ready", timeout=1.0)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        proc.terminate()
        out, _ = proc.communicate(timeout=5.0)
        print(f"Server startup output:\n{out}")
        assert ready, "Uvicorn server failed to become READY"

    startup_rss = server_ps.memory_info().rss / (1024 * 1024)
    startup_vms = server_ps.memory_info().vms / (1024 * 1024)
    print(f"1. Live Server Startup Memory: RSS={startup_rss:.2f} MB | VMS={startup_vms:.2f} MB | PID={proc.pid}")

    # 2. Warmup request
    t0 = time.perf_counter()
    w_res = requests.post(f"http://127.0.0.1:{port}/api/v1/analyze", json={"response": "The Moon orbits the Earth.", "model_name": "gpt-4o"}, timeout=15.0)
    w_lat = (time.perf_counter() - t0) * 1000.0
    if w_res.status_code != 200:
        print(f"Warmup failed ({w_res.status_code}): {w_res.text}")
    assert w_res.status_code == 200
    warm_rss = server_ps.memory_info().rss / (1024 * 1024)
    print(f"2. Live Server Warm Memory:    RSS={warm_rss:.2f} MB | Latency={w_lat:.1f}ms")

    # 3. 50 Sequential Live HTTP Requests
    print("\n--- Sending 50 Sequential Live HTTP Requests to Uvicorn ---")
    prompts_pool = [
        "The capital of France is Paris.",
        "The capital of France is Berlin.",
        "What is the capital of France?",
        "12 multiplied by 8 equals 96.",
        "12 multiplied by 8 equals 95.",
        "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        "The chemical formula for water is H2O.",
        "The chemical formula for water is CO2.",
        "Jupiter is the largest planet in our solar system.",
        "Albert Einstein developed the theory of general relativity.",
        "Paris is the capital of France. Berlin is the capital of Germany.",
        "Paris is the capital of France. Berlin is the capital of France.",
        "The speed of light in a vacuum is 299792458 meters per second.",
        "The Sun is a yellow dwarf star at the center of our solar system.",
        "Photosynthesis is the process by which plants use sunlight to synthesize nutrients.",
        "Oxygen has atomic number 8.",
        "Helium is the second lightest and second most abundant element in the universe.",
        "Mount Everest is Earth's highest mountain above sea level.",
        "The Amazon River is the largest river by discharge volume of water in the world.",
        "Python is a widely used high-level, general-purpose programming language."
    ]

    rss_series = []
    latencies = []
    session = requests.Session()

    for i in range(1, 51):
        prompt = prompts_pool[(i - 1) % len(prompts_pool)]
        t_req0 = time.perf_counter()
        print(f"  --> Sending Req #{i:02d}: '{prompt[:35]}...'")
        try:
            resp = requests.post(
                f"http://127.0.0.1:{port}/api/v1/analyze",
                json={"response": prompt, "model_name": "gpt-4o"},
                headers={"Connection": "close"},
                timeout=30.0,
            )
            lat = (time.perf_counter() - t_req0) * 1000.0
            assert resp.status_code == 200, f"Req {i} failed: {resp.text}"
            latencies.append(lat)
        except Exception as exc:
            proc.terminate()
            out, _ = proc.communicate(timeout=5.0)
            print(f"\n[SERVER CRASH / HANG LOG on Req #{i}]:\n{out}\n")
            raise exc

        cur_rss = server_ps.memory_info().rss / (1024 * 1024)
        rss_series.append(cur_rss)

        if i in [1, 5, 10, 20, 30, 40, 50] or i % 10 == 0:
            print(f"  Live Req #{i:02d} | Server RSS={cur_rss:6.2f}MB | Lat={lat:6.1f}ms | H={resp.json().get('overall_h_score')}")

    # 4. Concurrency Test on Live Server
    print("\n--- Concurrency Pressure Test on Live Server ---")
    for conc in [2, 4, 8]:
        def send_req(n):
            t_c0 = time.perf_counter()
            r = requests.post(f"http://127.0.0.1:{port}/api/v1/analyze", json={"response": f"Live concurrency probe {n}: Paris is in France.", "model_name": "gpt-4o"}, timeout=30.0)
            return r.status_code, (time.perf_counter() - t_c0) * 1000.0

        t_c_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as executor:
            futures = [executor.submit(send_req, j) for j in range(conc)]
            statuses = [f.result()[0] for f in futures]
            c_lats = [f.result()[1] for f in futures]
        c_dur = (time.perf_counter() - t_c_start) * 1000.0
        c_rss = server_ps.memory_info().rss / (1024 * 1024)
        print(f"  Concurrency Level {conc}: Success={sum(1 for s in statuses if s==200)}/{conc} | Server RSS={c_rss:.2f}MB | AvgLat={statistics.mean(c_lats):.1f}ms | WallTime={c_dur:.1f}ms")

    # Final summary
    min_rss = min(rss_series)
    max_rss = max(rss_series)
    mean_rss = statistics.mean(rss_series)
    median_rss = statistics.median(rss_series)
    first_rss = rss_series[0]
    last_rss = rss_series[-1]
    growth = last_rss - first_rss

    print("\n" + "=" * 80)
    print("LIVE PRODUCTION UVICORN SERVER MEMORY SUMMARY")
    print("=" * 80)
    print(f"Server Startup RSS:      {startup_rss:6.2f} MB")
    print(f"Server Warm RSS:         {warm_rss:6.2f} MB")
    print(f"Min Request RSS:         {min_rss:6.2f} MB")
    print(f"Max Request RSS:         {max_rss:6.2f} MB")
    print(f"Final Request RSS (#50): {last_rss:6.2f} MB")
    print(f"Mean Server RSS:         {mean_rss:6.2f} MB")
    print(f"Median Server RSS:       {median_rss:6.2f} MB")
    print(f"Total RSS Growth:        {growth:6.2f} MB")
    print("=" * 80)

    # Clean shutdown
    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    run_live_server_benchmark()
