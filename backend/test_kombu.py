from celery import Celery
from app.core.config import settings

celery_app = Celery("test", broker=settings.CELERY_BROKER_URL)
try:
    with celery_app.connection_for_write() as conn:
        print("Connected to:", conn)
except Exception as e:
    print("Error:", e)
