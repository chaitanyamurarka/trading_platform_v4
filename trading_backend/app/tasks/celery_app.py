# app/tasks/celery_app.py
from celery import Celery
from ..config import settings # Your application settings

# Initialize Celery
# The first argument to Celery is the name of the current module,
# which helps Celery auto-generate names for tasks.
# Or you can use a custom name like "trading_tasks".
celery_application = Celery(
    "trading_platform_tasks", # A name for your Celery application
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[ # Temporily removed the 'app.tasks.optimization_tasks',
             'app.tasks.cache_cleanup_tasks',
             'app.tasks.data_processing_tasks'] # Added the new task module
)

# Optional Celery configuration
celery_application.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Ignore other content
    result_serializer='json',
    timezone='UTC', # Example timezone
    enable_utc=True,
    # You can add more Celery settings here if needed
    # e.g., task_track_started=True,
    # result_expires=3600, # Time in seconds for results to be kept
)

# If you want to auto-discover tasks from all modules named tasks.py in your INSTALLED_APPS
# (like in Django), you'd use app.autodiscover_tasks().
# For FastAPI, explicitly listing modules in `include` is common.

if __name__ == '__main__':
    celery_application.start()