"""
schemas.py

This module defines the Pydantic models that are used for data validation,
serialization, and defining the shapes of API requests and responses. These models
act as a clear and enforceable contract for the data moving through the application.
"""
from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

class Interval(str, Enum):
    """Enumeration of allowed timeframe intervals for OHLC data."""
    SEC_1 = "1s"
    SEC_5 = "5s"
    SEC_10 = "10s"
    SEC_15 = "15s"
    SEC_30 = "30s"
    SEC_45 = "45s"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_10 = "10m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    MIN_45 = "45m"
    HOUR_1 = "1h"
    DAY_1 = "1d"

class CandleBase(BaseModel):
    """
    Base schema for a single OHLC (Open, High, Low, Close) data point.
    It includes a validator to automatically compute the UNIX timestamp.
    """
    timestamp: datetime = Field(..., description="The timestamp of the candle (timezone-aware).")
    open: float = Field(..., description="The opening price for the candle period.")
    high: float = Field(..., description="The highest price for the candle period.")
    low: float = Field(..., description="The lowest price for the candle period.")
    close: float = Field(..., description="The closing price for the candle period.")
    volume: Optional[float] = Field(None, description="The trading volume for the candle period.")
    unix_timestamp: Optional[float] = Field(None, description="The timestamp represented as a UNIX epoch float. Automatically calculated.")

    @model_validator(mode='after')
    def calculate_unix_timestamp(self) -> 'CandleBase':
        """Calculates and sets the UNIX timestamp from the datetime timestamp."""
        if self.timestamp:
            self.unix_timestamp = self.timestamp.timestamp()
        return self

class Candle(CandleBase):
    """
    Represents a single OHLC candle, configured for ORM (Object-Relational Mapping) mode.
    This allows it to be created directly from a SQLAlchemy database object.
    """
    class Config:
        from_attributes = True  # Pydantic v2 setting for ORM mode.

class HistoricalDataResponse(BaseModel):
    """
    Defines the structured response for an initial historical data request.
    It includes the candle data plus metadata for pagination (lazy loading).
    """
    request_id: Optional[str] = Field(None, description="A unique ID for this data session, used for fetching subsequent chunks.")
    candles: List[Candle] = Field(description="The list of OHLC candle data.")
    offset: Optional[int] = Field(None, description="The starting offset of this chunk within the full dataset.")
    total_available: int = Field(description="The total number of candles available on the server for the requested range.")
    is_partial: bool = Field(description="True if the returned 'candles' are a subset of the total available.")
    message: str = Field(description="A descriptive message about the result of the data load.")

class HistoricalDataChunkResponse(BaseModel):
    """Defines the response for a subsequent chunk of historical data."""
    candles: List[Candle]
    offset: int
    limit: int
    total_available: int

class SessionInfo(BaseModel):
    """Schema for returning a new session token to the client."""
    session_token: str

class OptimizationRequest(BaseModel):
    """Defines the request body for submitting a strategy optimization task."""
    strategy_id: str
    symbol: str
    interval: Interval
    start_date: datetime
    end_date: datetime
    param_grid: Dict[str, List[Any]] = Field(description='Example: {"param1": [10, 20], "param2": [0.5, 0.6]}')

class OptimizationTaskResult(BaseModel):
    """Schema for the final result of a completed optimization task."""
    best_params: Dict[str, Any]
    best_score: float

class JobStatus(str, Enum):
    """Enumeration of possible statuses for a background job."""
    PENDING = "PENDING"
    RECEIVED = "RECEIVED"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"

class JobStatusResponse(BaseModel):
    """Schema for the response when querying the status of a background job."""
    job_id: str
    status: JobStatus
    result: Optional[OptimizationTaskResult] = None

class JobSubmissionResponse(BaseModel):
    """Schema for the response after successfully submitting a new job."""
    job_id: str
    status: JobStatus = JobStatus.RECEIVED
    message: Optional[str] = "Job submitted successfully."