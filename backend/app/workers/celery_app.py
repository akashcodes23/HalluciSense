"""
HalluciSense SaaS — Module 12.6: Celery & Redis Task Queue Configuration
========================================================================
Celery application instance configured for background verification execution.
"""

from __future__ import annotations

import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "hallucisense_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 min max
    task_soft_time_limit=240,
)
