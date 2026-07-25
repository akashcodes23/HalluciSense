"""
Celery Application Factory.
Configures Celery to use Redis as broker and backend for async tasks.
"""
from celery import Celery
from app.core.config import settings

# Initialize Celery app
celery_app = Celery(
    "hallucisense_worker",
    broker=str(settings.REDIS_URL),
    backend=str(settings.REDIS_URL),
    include=["app.workers.tasks.verification_task"]
)

# Optional configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
