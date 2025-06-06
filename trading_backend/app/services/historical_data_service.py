"""
historical_data_service.py

This service handles all logic related to fetching, processing, and caching
historical OHLC (Open, High, Low, Close) data.
"""

import logging
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, HTTPException
from pydantic import TypeAdapter

from .. import pyiqfeed as iq
from .. import schemas
from ..core.cache import get_cached_ohlc_data, set_cached_ohlc_data, redis_client, CACHE_EXPIRATION_SECONDS
from ..core.numba_resampling_kernels import launch_resample_ohlc
from ..dtn_iq_client import get_iqfeed_history_conn
from ..tasks.data_processing_tasks import resample_and_cache_all_intervals_task

# --- Constants ---
INTERVAL_SECONDS_MAP = {
    "1s": 1, "5s": 5, "10s": 10, "15s": 15, "30s": 30, "45s": 45,
    "1m": 60, "5m": 300, "10m": 600, "15m": 900,
    "30m": 1800, "45m": 2700, "1h": 3600
}
INITIAL_FETCH_LIMIT = 5000

# --- Helper Functions ---

def _build_request_id(session_token: str, exchange: str, token: str, start_time: datetime, end_time: datetime) -> str:
    """Builds a consistent, unique prefix for all cache keys related to a specific data request range."""
    return f"chart_data:{session_token}:{exchange}:{token}:{start_time.isoformat()}:{end_time.isoformat()}"

def _parse_and_filter_dtn_data(
    api_response_data: np.ndarray,
    start_time: datetime,
    end_time: datetime,
    trading_symbol: str
) -> List[schemas.CandleBase]:
    """
    Parses raw NumPy structured array from DTN IQFeed into Pydantic models
    and filters it to the precise requested time range.
    """
    if api_response_data.size == 0:
        return []

    if 'time' in api_response_data.dtype.names:
        timestamps_dt64 = api_response_data['date'] + api_response_data['time']
    else:
        timestamps_dt64 = api_response_data['date']

    start_time_np = np.datetime64(start_time.replace(tzinfo=None))
    end_time_np = np.datetime64(end_time.replace(tzinfo=None))

    mask = (timestamps_dt64 >= start_time_np) & (timestamps_dt64 <= end_time_np)
    filtered_data = api_response_data[mask]

    if filtered_data.size == 0:
        logging.info(f"No data remains for {trading_symbol} after time filtering.")
        return []

    python_timestamps = pd.to_datetime(timestamps_dt64[mask], utc=True).to_pydatetime()
    
    # --- BUG FIX START ---
    # The original code used `rec.get()`, which is a dictionary method, on a NumPy record (`rec`).
    # This caused the `AttributeError`. The corrected code below iterates through the records
    # and safely checks for the existence of volume fields (`prd_vlm` or `tot_vlm`)
    # in the NumPy dtype before accessing them.

    candle_dicts = []
    dtype_names = filtered_data.dtype.names
    for ts, rec in zip(python_timestamps, filtered_data):
        volume = 0.0
        if 'prd_vlm' in dtype_names:
            volume = float(rec['prd_vlm'])
        elif 'tot_vlm' in dtype_names:
            volume = float(rec['tot_vlm'])
        
        candle_dicts.append({
            "timestamp": ts,
            "open": float(rec['open_p']),
            "high": float(rec['high_p']),
            "low": float(rec['low_p']),
            "close": float(rec['close_p']),
            "volume": volume
        })
    # --- BUG FIX END ---

    if not candle_dicts:
        return []

    candle_adapter = TypeAdapter(List[schemas.CandleBase])
    return candle_adapter.validate_python(candle_dicts)


def _fetch_from_dtn_iq_api(
    trading_symbol: str,
    interval_val: str,
    start_time: datetime,
    end_time: datetime,
) -> List[schemas.CandleBase]:
    """
    Fetches historical data directly from the DTN IQFeed API.
    """
    logging.info(f"Fetching from DTN IQFeed for {trading_symbol}, Interval: {interval_val}, Range: {start_time} to {end_time}")
    hist_conn = get_iqfeed_history_conn()
    if not hist_conn:
        logging.error("DTN IQFeed History Connection not available.")
        return []

    try:
        with iq.ConnConnector([hist_conn]):
            api_response_data = hist_conn.request_bars_in_period(
                ticker=trading_symbol,
                interval_len=1,
                interval_type='s',
                bgn_prd=start_time,
                end_prd=end_time,
                ascend=True,
            )
            
            if api_response_data is None or not isinstance(api_response_data, np.ndarray):
                logging.warning(f"Unexpected data type from IQFeed: {type(api_response_data)} for {trading_symbol}")
                return []
            
            logging.info(f"Received {len(api_response_data)} base records from IQFeed for {trading_symbol}.")
            return _parse_and_filter_dtn_data(api_response_data, start_time, end_time, trading_symbol)

    except iq.NoDataError:
        logging.info(f"IQFeed (NoDataError): No data for {trading_symbol} in the specified range.")
    except Exception as e:
        logging.error(f"Exception during IQFeed API call for {trading_symbol}: {e}", exc_info=True)
    
    return []


def _get_and_prepare_1s_data_for_range(
    session_token: str,
    exchange: str,
    token: str,
    start_time: datetime,
    end_time: datetime
) -> List[schemas.Candle]:
    """
    Retrieves 1-second data for a given date range, utilizing cache first and
    fetching from the DTN API as a fallback.
    """
    all_1s_candles = []
    date_range = pd.date_range(start_time.date(), end_time.date())
    daily_cache_keys = [f"1s_data:{exchange}:{token}:{day.strftime('%Y-%m-%d')}" for day in date_range]
    
    cached_results = redis_client.mget(daily_cache_keys) if daily_cache_keys else []
    
    missing_dates = []
    for i, result in enumerate(cached_results):
        day = date_range[i].date()
        if result:
            try:
                deserialized = json.loads(result)
                all_1s_candles.extend([schemas.Candle(**item) for item in deserialized])
            except (json.JSONDecodeError, TypeError):
                missing_dates.append(day)
        else:
            missing_dates.append(day)

    if missing_dates:
        fetch_start = datetime.combine(min(missing_dates), datetime.min.time())
        fetch_end = datetime.combine(max(missing_dates), datetime.max.time())
        
        newly_fetched_data = _fetch_from_dtn_iq_api(token, "1s", fetch_start, fetch_end)
        
        if newly_fetched_data:
            all_1s_candles.extend(newly_fetched_data)
            new_data_df = pd.DataFrame([c.model_dump() for c in newly_fetched_data])
            new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp'])
            new_data_df['date_key'] = new_data_df['timestamp'].dt.strftime('%Y-%m-%d')
            
            pipe = redis_client.pipeline()
            for day in missing_dates:
                date_str = day.strftime('%Y-%m-%d')
                day_cache_key = f"1s_data:{exchange}:{token}:{date_str}"
                day_df = new_data_df[new_data_df['date_key'] == date_str]
                records_to_cache = day_df.to_dict(orient='records')
                for record in records_to_cache:
                    record['timestamp'] = record['timestamp'].isoformat()
                pipe.set(day_cache_key, json.dumps(records_to_cache), ex=CACHE_EXPIRATION_SECONDS)
            pipe.execute()

    all_1s_candles.sort(key=lambda c: c.timestamp)
    start_ts = start_time.replace(tzinfo=timezone.utc)
    end_ts = end_time.replace(tzinfo=timezone.utc)
    return [c for c in all_1s_candles if start_ts <= c.timestamp <= end_ts]

def get_initial_historical_data(
    background_tasks: BackgroundTasks,
    session_token: str,
    exchange: str,
    token: str,
    interval_val: str,
    start_time: datetime,
    end_time: datetime,
) -> schemas.HistoricalDataResponse:
    """
    Main entry point for fetching historical data.
    """
    request_id = _build_request_id(session_token, exchange, token, start_time, end_time)
    target_data_key = f"{request_id}:{interval_val}"

    full_data = get_cached_ohlc_data(target_data_key)
    
    if not full_data:
        logging.info(f"Cache MISS for {target_data_key}. Processing from base 1s data.")
        base_1s_candles = _get_and_prepare_1s_data_for_range(session_token, exchange, token, start_time, end_time)

        if not base_1s_candles:
            return schemas.HistoricalDataResponse(candles=[], total_available=0, is_partial=False, message="No data available.", request_id=None, offset=None)

        if interval_val == "1s":
            full_data = base_1s_candles
        else:
            aggregation_seconds = INTERVAL_SECONDS_MAP.get(interval_val)
            if not aggregation_seconds:
                raise HTTPException(status_code=400, detail=f"Unsupported interval for resampling: {interval_val}")

            timestamps_1s_np = np.array([c.timestamp.timestamp() for c in base_1s_candles], dtype=np.float64)
            open_1s_np = np.array([c.open for c in base_1s_candles], dtype=np.float64)
            high_1s_np = np.array([c.high for c in base_1s_candles], dtype=np.float64)
            low_1s_np = np.array([c.low for c in base_1s_candles], dtype=np.float64)
            close_1s_np = np.array([c.close for c in base_1s_candles], dtype=np.float64)
            volume_1s_np = np.array([c.volume or 0.0 for c in base_1s_candles], dtype=np.float64)
            
            (ts_agg, o_agg, h_agg, l_agg, c_agg, v_agg, num_bars) = launch_resample_ohlc(
                timestamps_1s_np, open_1s_np, high_1s_np, low_1s_np, close_1s_np, volume_1s_np, aggregation_seconds
            )
            
            resampled_candles = [
                schemas.Candle(
                    timestamp=datetime.fromtimestamp(ts_agg[i], tz=timezone.utc),
                    open=o_agg[i], high=h_agg[i], low=l_agg[i], close=c_agg[i], volume=v_agg[i]
                ) for i in range(num_bars)
            ]
            full_data = resampled_candles

        if full_data:
            set_cached_ohlc_data(target_data_key, full_data, expiration=3600)
            
            task_triggered_key = f"{request_id}:task_triggered"
            if not redis_client.get(task_triggered_key):
                base_data_key_for_task = f"temp_1s_data:{uuid.uuid4()}"
                set_cached_ohlc_data(base_data_key_for_task, base_1s_candles, expiration=600)
                
                background_tasks.add_task(
                    resample_and_cache_all_intervals_task,
                    base_1s_data_key=base_data_key_for_task,
                    request_id_prefix=request_id,
                    user_requested_interval=interval_val
                )
                redis_client.set(task_triggered_key, "true", ex=3600)

    total_available = len(full_data)
    initial_offset = max(0, total_available - INITIAL_FETCH_LIMIT)
    candles_to_send = full_data[initial_offset:]
    
    return schemas.HistoricalDataResponse(
        request_id=target_data_key,
        candles=candles_to_send,
        offset=initial_offset,
        total_available=total_available,
        is_partial=(total_available > len(candles_to_send)),
        message=f"Initial data loaded. Displaying last {len(candles_to_send)} of {total_available} candles."
    )

def get_historical_data_chunk(
    request_id: str,
    offset: int,
    limit: int = 5000
) -> schemas.HistoricalDataChunkResponse:
    """
    Retrieves a subsequent chunk of historical data from the cache using the request_id.
    """
    full_data = get_cached_ohlc_data(request_id)
    if full_data is None:
        raise HTTPException(status_code=404, detail="Data for this request not found or has expired.")

    total_available = len(full_data)
    if offset < 0 or offset >= total_available:
        return schemas.HistoricalDataChunkResponse(candles=[], offset=offset, limit=limit, total_available=total_available)
        
    chunk = full_data[offset: offset + limit]
    
    return schemas.HistoricalDataChunkResponse(
        candles=chunk,
        offset=offset,
        limit=limit,
        total_available=total_available
    )