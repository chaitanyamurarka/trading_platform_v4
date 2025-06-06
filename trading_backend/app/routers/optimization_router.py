# app/routers/optimization_router.py

from fastapi import APIRouter, HTTPException, status, Path
from datetime import datetime

from .. import schemas # Pydantic models
from ..services import optimization_service # The service we just created

router = APIRouter(
    prefix="/optimize",
    tags=["Strategy Optimization"]
)

@router.post("/",
             response_model=schemas.JobSubmissionResponse,
             status_code=status.HTTP_202_ACCEPTED) # 202 Accepted is good for async tasks
async def start_new_optimization_job(
    request: schemas.OptimizationRequest # FastAPI will parse and validate the request body
):
    """
    Submit a new strategy optimization task.

    - **strategy_id**: Identifier of the trading strategy to optimize.
    - **symbol**: Asset symbol to run the strategy on.
    - **interval**: Time interval for the candles.
    - **start_date**: Start datetime of the historical data range for backtesting.
    - **end_date**: End datetime of the historical data range.
    - **param_grid**: Dictionary of strategy parameters to vary.

    Enqueues a background job to perform the optimization.
    Returns a job ID to track the task.
    """
    if request.start_date >= request.end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date must be earlier than end_date"
        )

    try:
        # Convert datetime objects to ISO format strings for Celery task
        # and Interval enum to its string value
        start_date_iso = request.start_date.isoformat()
        end_date_iso = request.end_date.isoformat()
        interval_str = request.interval.value

        job_id = optimization_service.submit_optimization_job(
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            interval=interval_str,
            start_date=start_date_iso,
            end_date=end_date_iso,
            param_grid=request.param_grid
        )
        return schemas.JobSubmissionResponse(
            job_id=job_id,
            status=schemas.JobStatus.RECEIVED, # Initial status after submission
            message="Optimization job successfully submitted."
        )
    except Exception as e:
        # Log the exception e
        print(f"Error submitting optimization job: {e}") # Basic logging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit optimization job."
        )


@router.get("/{job_id}", response_model=schemas.JobStatusResponse)
async def get_optimization_job_status(
    job_id: str = Path(..., description="The ID of the optimization job to check.")
):
    """
    Get the status and result (if available) of an optimization task.

    - **job_id**: The ID of the job (returned when the task was submitted).
    """
    try:
        status_response = optimization_service.get_job_status_and_result(job_id=job_id)
        # Potentially, if Celery returns PENDING for a non-existent ID after some time,
        # or if the result expired, you might want to map that to a 404 Not Found.
        # For now, we return what the service gives.
        if status_response.status == schemas.JobStatus.PENDING:
             # A task ID that Celery doesn't know about (e.g., wrong ID, or result expired and cleaned up)
             # will also typically show as PENDING. You might add more nuanced handling if needed.
             pass # Or raise HTTPException for truly unknown IDs if service indicates it

        return status_response
    except Exception as e:
        # Log the exception e
        print(f"Error retrieving status for job {job_id}: {e}") # Basic logging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve status for job {job_id}."
        )