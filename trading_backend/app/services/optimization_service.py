# app/services/optimization_service.py

from celery.result import AsyncResult
from typing import Dict, Any, Optional

from ..tasks.optimization_tasks import optimize_strategy_task # Import your Celery task
from ..tasks.celery_app import celery_application # Import the Celery application instance
from .. import schemas # Your Pydantic schemas

def submit_optimization_job(
    strategy_id: str,
    symbol: str,
    interval: str, # Interval as string
    start_date: str, # Dates as ISO strings
    end_date: str,
    param_grid: Dict[str, Any]
) -> str:
    """
    Submits a strategy optimization task to Celery.
    Returns the task ID.
    """
    print(f"Submitting optimization job: strategy_id={strategy_id}, symbol={symbol}, "
          f"interval={interval}, start_date={start_date}, end_date={end_date}, param_grid={param_grid}")

    # Use .delay() or .apply_async() to send the task to Celery
    # .delay() is a shortcut for .apply_async()
    task_result = optimize_strategy_task.delay(
        strategy_id=strategy_id,
        symbol=symbol,
        interval_str=interval,
        start_date_iso=start_date, # Pass as string
        end_date_iso=end_date,     # Pass as string
        param_grid=param_grid
    )
    print(f"Optimization job submitted. Task ID: {task_result.id}")
    return task_result.id

def get_job_status_and_result(job_id: str) -> schemas.JobStatusResponse:
    """
    Checks the status of a Celery task by its ID and retrieves the result if ready.
    """
    print(f"Checking status for job ID: {job_id}")
    # Create an AsyncResult instance for the given task ID
    # Ensure you pass the 'app' argument if your Celery app instance is not globally default
    # or if you have multiple Celery apps.
    task_result_obj = AsyncResult(job_id, app=celery_application)

    status_str = task_result_obj.status
    task_info: Optional[schemas.OptimizationTaskResult] = None
    
    # Map Celery status strings to our JobStatus enum
    # This is a simple mapping; you might want to refine it.
    job_status_enum_val: schemas.JobStatus
    if status_str == "PENDING":
        job_status_enum_val = schemas.JobStatus.PENDING
        # For PENDING, Celery's info is often None or the state itself if custom.
        # If task.update_state was used with meta, task_result_obj.info might have it.
    elif status_str == "STARTED":
        job_status_enum_val = schemas.JobStatus.STARTED
    elif status_str == "SUCCESS":
        job_status_enum_val = schemas.JobStatus.SUCCESS
        # If the task succeeded, result.result should contain the return value of the task
        result_data = task_result_obj.result
        if isinstance(result_data, dict): # Our task returns a dict
            task_info = schemas.OptimizationTaskResult(**result_data)
        else:
            # Handle unexpected result type
            print(f"Warning: Task {job_id} succeeded but result format is unexpected: {result_data}")
            # task_info remains None, or you can set a default error state in it.
    elif status_str == "FAILURE":
        job_status_enum_val = schemas.JobStatus.FAILURE
        # result.result might contain the exception/traceback.
        # Be careful about exposing raw tracebacks to clients.
    elif status_str == "RETRY":
        job_status_enum_val = schemas.JobStatus.RETRY
    elif status_str == "REVOKED": # Or if you implement cancellation
        job_status_enum_val = schemas.JobStatus.REVOKED
    elif status_str == "PROGRESS": # If using custom state 'PROGRESS'
        job_status_enum_val = schemas.JobStatus.STARTED # Treat PROGRESS as STARTED
        # You might want to extract progress meta if available in task_result_obj.info
        # progress_meta = task_result_obj.info if isinstance(task_result_obj.info, dict) else {}
        # return schemas.JobStatusResponse(job_id=job_id, status=job_status_enum_val, result=None, progress_meta=progress_meta)
    else: # Unknown status
        print(f"Warning: Unknown Celery task status '{status_str}' for job ID {job_id}")
        job_status_enum_val = schemas.JobStatus.PENDING # Default to PENDING or a new UNKNOWN status


    print(f"Job ID: {job_id}, Celery Status: {status_str}, Mapped Status: {job_status_enum_val}, Result (if any): {task_info}")

    return schemas.JobStatusResponse(
        job_id=job_id,
        status=job_status_enum_val,
        result=task_info
    )