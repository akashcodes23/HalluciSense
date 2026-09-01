"""Memory Management and Allocator Trimming Utilities for Production Stability."""

import gc
import ctypes
import os
import psutil
import structlog

logger = structlog.get_logger(__name__)


def trim_process_memory() -> float:
    """Explicitly invoke Python garbage collector and glibc malloc_trim to release memory back to OS.
    
    Returns:
        Current process RSS in MB.
    """
    gc.collect()
    try:
        # Call malloc_trim on Linux (glibc)
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

    try:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0


def get_memory_telemetry() -> dict:
    """Return process memory and thread telemetry."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = round(mem_info.rss / (1024 * 1024), 2)
        vms_mb = round(mem_info.vms / (1024 * 1024), 2)
        threads = process.num_threads()
    except Exception:
        rss_mb = 0.0
        vms_mb = 0.0
        threads = 1

    from app.core.engine.model_registry import ModelRegistry
    init_counts = ModelRegistry.get_init_counts()

    return {
        "pid": os.getpid(),
        "rss_mb": rss_mb,
        "vms_mb": vms_mb,
        "threads": threads,
        "workers": 1,
        "nli_init_count": init_counts.get("nli_model", 0),
        "sentence_transformer_init_count": init_counts.get("sentence_transformer", 0),
        "cross_encoder_reranker_init_count": init_counts.get("cross_encoder_reranker", 0),
        "pipeline_init_count": init_counts.get("pipeline", 0),
    }
