"""
data_processing_tasks.py

This module contains Celery tasks related to background data processing.
The primary task defined here is for pre-aggregating and caching historical data
at various time intervals.
"""

import logging
import numpy as np
from datetime import datetime, timezone

from .celery_app import celery_application
from ..core.cache import get_cached_ohlc_data, set_cached_ohlc_data,redis_client
from ..core.numba_resampling_kernels import launch_resample_ohlc
from .. import schemas

# Map of standard intervals to their duration in seconds.
INTERVAL_SECONDS_MAP = {
    "1s": 1, "5s": 5, "10s": 10, "15s": 15, "30s": 30, "45s": 45,
    "1m": 60, "5m": 300, "10m": 600, "15m": 900,
    "30m": 1800, "45m": 2700, "1h": 3600
}

@celery_application.task(name="tasks.resample_and_cache_all_intervals")
def resample_and_cache_all_intervals_task(
    base_1s_data_key: str,
    request_id_prefix: str,
    user_requested_interval: str
):
    """
    A Celery background task that resamples 1-second data to all other standard intervals.

    This task is triggered after a user's initial data request. It takes the base
    1-second data, resamples it to every other interval (e.g., 5s, 1m, 5m, 1h),
    and caches each result. This pre-aggregation makes subsequent requests for the
    same time range but different intervals nearly instantaneous.

    Args:
        base_1s_data_key: The Redis cache key where the temporary 1s base data is stored.
        request_id_prefix: The unique cache key prefix for the user's data request range.
        user_requested_interval: The interval the user originally requested, which is skipped
                                 as it was already processed on-demand.
    """
    logging.info(f"Starting background resampling for data prefix: {request_id_prefix}")
    
    base_1s_candles = get_cached_ohlc_data(base_1s_data_key)
    if not base_1s_candles:
        logging.warning(f"No 1s base data found at key {base_1s_data_key} for background resampling.")
        return
        
    # Prepare NumPy arrays from the 1s data for the Numba kernel.
    timestamps_1s_np = np.array([c.timestamp.timestamp() for c in base_1s_candles], dtype=np.float64)
    open_1s_np = np.array([c.open for c in base_1s_candles], dtype=np.float64)
    high_1s_np = np.array([c.high for c in base_1s_candles], dtype=np.float64)
    low_1s_np = np.array([c.low for c in base_1s_candles], dtype=np.float64)
    close_1s_np = np.array([c.close for c in base_1s_candles], dtype=np.float64)
    volume_1s_np = np.array([c.volume or 0.0 for c in base_1s_candles], dtype=np.float64)

    # Iterate through all standard intervals to resample and cache.
    for interval, agg_seconds in INTERVAL_SECONDS_MAP.items():
        # Skip the interval the user already requested (it's already cached) and the base 1s data.
        if interval == user_requested_interval or interval == "1s":
            continue

        logging.debug(f"Background resampling to {interval} for {request_id_prefix}")
        
        # Resample using the high-performance Numba kernel.
        (ts_agg, o_agg, h_agg, l_agg, c_agg, v_agg, num_agg_bars) = launch_resample_ohlc(
            timestamps_1s_np, open_1s_np, high_1s_np, low_1s_np, close_1s_np, volume_1s_np,
            agg_seconds
        )
        
        if num_agg_bars > 0:
            # Convert the resampled NumPy arrays back to Pydantic models.
            resampled_candles = [
                schemas.Candle(
                    timestamp=datetime.fromtimestamp(ts_agg[i], tz=timezone.utc),
                    open=o_agg[i], high=h_agg[i], low=l_agg[i], close=c_agg[i], volume=v_agg[i]
                ) for i in range(num_agg_bars)
            ]
            
            # Cache the result with a specific key for this interval.
            target_cache_key = f"{request_id_prefix}:{interval}"
            set_cached_ohlc_data(target_cache_key, resampled_candles, expiration=3600)
    
    # Clean up the temporary 1s data key after processing.
    redis_client.delete(base_1s_data_key)
    logging.info(f"Finished and cleaned up background resampling for {request_id_prefix}")