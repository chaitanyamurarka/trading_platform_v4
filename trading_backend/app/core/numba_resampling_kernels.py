"""
numba_resampling_kernels.py

This module provides high-performance resampling functions accelerated by Numba.
Numba's Just-In-Time (JIT) compiler translates Python functions into optimized
machine code, making it ideal for CPU-bound, numerically-intensive tasks like
resampling large arrays of OHLC data.
"""

import numpy as np
import numba

@numba.jit(nopython=True)
def resample_ohlc_cpu_jit(
    timestamps_1s: np.ndarray,
    open_1s: np.ndarray,
    high_1s: np.ndarray,
    low_1s: np.ndarray,
    close_1s: np.ndarray,
    volume_1s: np.ndarray,
    out_timestamps: np.ndarray,
    out_open: np.ndarray,
    out_high: np.ndarray,
    out_low: np.ndarray,
    out_close: np.ndarray,
    out_volume: np.ndarray,
    aggregation_seconds: int
) -> int:
    """
    Performs OHLC resampling on sorted 1-second data using a Numba JIT-compiled function.

    This function iterates through the 1-second data and aggregates it into larger
    time intervals (e.g., 60 seconds for 1-minute bars).

    Args:
        timestamps_1s: NumPy array of UNIX timestamps for the 1s data.
        open_1s, high_1s, low_1s, close_1s, volume_1s: NumPy arrays for the 1s OHLCV data.
        out_timestamps, out_open, ...: Pre-allocated NumPy arrays to store the resampled results.
        aggregation_seconds: The target interval in seconds (e.g., 60 for 1m, 300 for 5m).

    Returns:
        The total number of aggregated bars created.
    """
    if len(timestamps_1s) == 0:
        return 0

    # Initialize variables for the first aggregated bar.
    current_bar_idx = 0
    # Determine the start of the very first time bucket by rounding down the first timestamp.
    bucket_start_ts = np.floor(timestamps_1s[0] / aggregation_seconds) * aggregation_seconds
    
    # Initialize the first output bar with the values from the first input bar.
    out_timestamps[current_bar_idx] = bucket_start_ts
    out_open[current_bar_idx] = open_1s[0]
    out_high[current_bar_idx] = high_1s[0]
    out_low[current_bar_idx] = low_1s[0]
    out_close[current_bar_idx] = close_1s[0]
    out_volume[current_bar_idx] = volume_1s[0]

    # Loop through the rest of the 1-second data.
    for i in range(1, len(timestamps_1s)):
        ts = timestamps_1s[i]
        
        # Check if this 1s bar belongs to a new time bucket.
        if ts >= bucket_start_ts + aggregation_seconds:
            # The current aggregated bar is complete, so we start a new one.
            current_bar_idx += 1
            bucket_start_ts = np.floor(ts / aggregation_seconds) * aggregation_seconds
            
            # Initialize the new bar.
            out_timestamps[current_bar_idx] = bucket_start_ts
            out_open[current_bar_idx] = open_1s[i]
            out_high[current_bar_idx] = high_1s[i]
            out_low[current_bar_idx] = low_1s[i]
            out_volume[current_bar_idx] = 0.0  # Reset volume for summation.
        
        # Update high, low, close, and volume for the current aggregated bar.
        if high_1s[i] > out_high[current_bar_idx]:
            out_high[current_bar_idx] = high_1s[i]
        
        if low_1s[i] < out_low[current_bar_idx]:
            out_low[current_bar_idx] = low_1s[i]
            
        out_close[current_bar_idx] = close_1s[i]  # Always update with the latest close.
        out_volume[current_bar_idx] += volume_1s[i]

    # The number of bars is the final index + 1.
    return current_bar_idx + 1


def launch_resample_ohlc(
    timestamps_1s: np.ndarray, open_1s: np.ndarray, high_1s: np.ndarray, 
    low_1s: np.ndarray, close_1s: np.ndarray, volume_1s: np.ndarray,
    aggregation_seconds: int
) -> tuple:
    """
    A wrapper function to launch the Numba-JIT compiled resampling kernel.

    This function handles the memory allocation for the output arrays and
    calls the high-performance JIT function. It then trims the output arrays
    to their actual size.

    Args:
        (See resample_ohlc_cpu_jit for descriptions of the input arrays)
        aggregation_seconds: The target interval in seconds.

    Returns:
        A tuple containing the trimmed output arrays (timestamps, open, high,
        low, close, volume) and the number of aggregated bars.
    """
    num_1s_records = len(timestamps_1s)
    if num_1s_records == 0:
        return (np.array([]),) * 6 + (0,)  # Return 6 empty arrays and a count of 0.

    # Pre-allocate output arrays with a generous size. The actual size will be smaller.
    max_out_bars = num_1s_records
    out_timestamps_host = np.zeros(max_out_bars, dtype=np.float64)
    out_open_host = np.zeros(max_out_bars, dtype=np.float64)
    out_high_host = np.zeros(max_out_bars, dtype=np.float64)
    out_low_host = np.zeros(max_out_bars, dtype=np.float64)
    out_close_host = np.zeros(max_out_bars, dtype=np.float64)
    out_volume_host = np.zeros(max_out_bars, dtype=np.float64)

    # Call the Numba JIT-compiled function.
    actual_bars = resample_ohlc_cpu_jit(
        timestamps_1s, open_1s, high_1s, low_1s, close_1s, volume_1s,
        out_timestamps_host, out_open_host, out_high_host, out_low_host, out_close_host, out_volume_host,
        aggregation_seconds
    )
    
    # Trim the output arrays to the actual number of bars produced for memory efficiency.
    return (
        out_timestamps_host[:actual_bars], 
        out_open_host[:actual_bars], 
        out_high_host[:actual_bars], 
        out_low_host[:actual_bars], 
        out_close_host[:actual_bars], 
        out_volume_host[:actual_bars],
        actual_bars
    )