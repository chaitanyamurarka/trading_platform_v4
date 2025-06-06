# chaitanyamurarka/trading_platform_v3.1/trading_platform_v3.1-fd71c9072644cabd20e39b57bf2d47b25107e752/trading_backend/app/routers/historical_data_router.py
from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional

from .. import schemas
from ..services import historical_data_service

router = APIRouter(
    prefix="/historical",
    tags=["Historical Data"]
)

@router.get("/", response_model=schemas.HistoricalDataResponse)
async def fetch_initial_historical_data(
    background_tasks: BackgroundTasks,
    session_token: str = Query(..., description="The user's session token."),
    exchange: str = Query(..., description="Exchange name or code (e.g., 'NASDAQ')"),
    token: str = Query(..., description="Asset symbol or token (e.g., 'AAPL')"),
    interval: schemas.Interval = Query(..., description="Data interval (e.g., '1m', '5m', '1d')"),
    start_time: datetime = Query(..., description="Start datetime for the data range (ISO format, e.g., '2023-01-01T00:00:00')"),
    end_time: datetime = Query(..., description="End datetime for the data range (ISO format, e.g., '2023-01-01T12:00:00')"),
):
    """
    Retrieve the initial chunk of historical OHLC data.
    The server processes the entire range, caches it, and returns the most recent data.
    A 'request_id' is returned for fetching older chunks.
    """
    if start_time >= end_time:
        raise HTTPException(status_code=400, detail="start_time must be earlier than end_time")

    response = historical_data_service.get_initial_historical_data(
        background_tasks=background_tasks,
        session_token=session_token,
        exchange=exchange,
        token=token,
        interval_val=interval.value,
        start_time=start_time,
        end_time=end_time
    )
    return response

@router.get("/chunk", response_model=schemas.HistoricalDataChunkResponse)
async def fetch_historical_data_chunk(
    request_id: str = Query(..., description="The unique ID of the data request session."),
    offset: int = Query(..., ge=0, description="The starting index of the data to fetch."),
    limit: int = Query(5000, ge=1, le=10000, description="The number of candles to fetch.")
):
    """
    Retrieve a subsequent chunk of historical OHLC data that has already been processed.
    """
    response = historical_data_service.get_historical_data_chunk(
        request_id=request_id,
        offset=offset,
        limit=limit
    )
    return response