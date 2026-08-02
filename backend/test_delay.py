from app.workers.celery_app import celery_app
from app.workers.tasks.verification_task import verify_response_task

try:
    verify_response_task.delay("00000000-0000-0000-0000-000000000000", "test", [0.1])
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Caught:", type(e), str(e))
