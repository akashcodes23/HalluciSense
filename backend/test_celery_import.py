from app.workers.tasks.verification_task import verify_response_task
print("App for task:", verify_response_task.app)
print("Broker URL:", verify_response_task.app.conf.broker_url)
