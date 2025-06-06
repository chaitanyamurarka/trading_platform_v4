"""
models.py

This module defines the SQLAlchemy ORM (Object-Relational Mapping) models.
These Python classes are mapped to database tables and define the schema
for the data stored in the application's database.
"""

from sqlalchemy import (
    Column, String, DateTime, Float, Integer, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base

# Base class for all ORM models.
Base = declarative_base()

class OHLC(Base):
    """
    Represents the `ohlc_data` table in the database.
    This table stores historical Open, High, Low, Close, and Volume data
    for various financial instruments at different time intervals.
    """
    __tablename__ = "ohlc_data"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    exchange = Column(String(50), nullable=False, doc="The exchange where the instrument is traded (e.g., 'NSE', 'NASDAQ').")
    token = Column(String(50), nullable=False, doc="The trading symbol or token of the instrument (e.g., 'RELIANCE', 'AAPL').")
    interval = Column(String(10), nullable=False, doc="The time interval of the OHLC candle (e.g., '1s', '5m', '1d').")
    timestamp = Column(DateTime, nullable=False, doc="The start time of the OHLC candle period.")

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

    __table_args__ = (
        # A composite unique constraint is essential for preventing duplicate data entries.
        # It ensures that for a given instrument, interval, and timestamp, only one record can exist.
        # This is also critical for database dialects that support "INSERT ... ON DUPLICATE KEY UPDATE".
        UniqueConstraint("exchange", "token", "interval", "timestamp", name="uq_ohlc_data_unique_candle"),

        # A composite index to accelerate queries that filter by these common criteria.
        # This significantly improves performance for fetching historical data.
        Index("idx_ohlc_data_query", "exchange", "token", "interval", "timestamp")
    )

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the OHLC object."""
        return (f"<OHLC(exchange='{self.exchange}', token='{self.token}', interval='{self.interval}', "
                f"timestamp='{self.timestamp}', open={self.open}, high={self.high}, "
                f"low={self.low}, close={self.close}, volume={self.volume})>")

# Note: The `StrategyParameter`, `StrategyInfo`, etc., models were moved to `schemas.py`
# as they are used for data validation and API contracts (Pydantic models), not for
# database persistence (SQLAlchemy models). This separation clarifies their purpose.