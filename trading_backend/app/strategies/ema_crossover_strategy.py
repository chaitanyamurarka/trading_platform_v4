# app/strategies/ema_crossover_strategy.py
import pandas as pd
import numpy as np
import numba
from numba import cuda # If using CUDA, otherwise just numba for CPU
from typing import Dict, Any, List, Optional

# Assuming your models.py is in the parent directory of 'strategies'
# Adjust the import path if your structure is different
# from .. import models  # If models.py is in 'app/' and this is 'app/strategies/'
# For demonstration, let's assume models are accessible.
# You'll need to define these Pydantic models or import them correctly.
# Example stubs for models used:
class StrategyParameter(dict): pass # Placeholder
class StrategyInfo(dict): pass    # Placeholder
class IndicatorSeries(dict): pass # Placeholder
class Trade(dict): pass           # Placeholder

from .base_strategy import BaseStrategy, PortfolioState # Assuming base_strategy.py is in the same directory
# from ..config import logger # Assuming logger is set up in app/config.py

# --- Numba Kernels and Launcher (from your numba_kernels.py) ---
# Define constants for Numba loop status
POSITION_NONE = 0
POSITION_LONG = 1
POSITION_SHORT = -1
MAX_TRADES_FOR_DETAILED_OUTPUT = 2000 # Max trades to pre-allocate

# If you are NOT using CUDA, you would use @numba.jit(nopython=True) for CPU kernels
# and remove cuda-specific syntax like cuda.grid(1), cuda.to_device, etc.
# The kernel below is written for CUDA.
@cuda.jit
def ema_crossover_kernel(
    # Data arrays (1D) - device arrays
    open_prices_global: np.ndarray,
    high_prices_global: np.ndarray,
    low_prices_global: np.ndarray,
    close_prices_global: np.ndarray,

    # Parameter arrays (1D, one entry per combination) - device arrays
    fast_ema_periods_global: np.ndarray, # Not directly used if k_fast/slow are precomputed
    slow_ema_periods_global: np.ndarray, # Not directly used if k_fast/slow are precomputed
    stop_loss_pcts_global: np.ndarray,
    take_profit_pcts_global: np.ndarray,
    execution_price_types_global: np.ndarray, # 1 for open, 0 for close (example)

    initial_capital: float,
    n_candles: int,
    detailed_output_requested: bool, # True if k=0 detailed output is needed

    # Output arrays - device arrays
    cash_arr_global: np.ndarray,
    position_arr_global: np.ndarray,
    entry_price_arr_global: np.ndarray,
    sl_price_arr_global: np.ndarray,
    tp_price_arr_global: np.ndarray,

    final_pnl_arr_global: np.ndarray,
    total_trades_arr_global: np.ndarray,
    winning_trades_arr_global: np.ndarray,
    losing_trades_arr_global: np.ndarray,
    
    equity_arr_global: np.ndarray,
    peak_equity_arr_global: np.ndarray,
    max_drawdown_arr_global: np.ndarray,

    # Pre-calculated EMA smoothing factors
    k_fast_arr_global: np.ndarray,
    k_slow_arr_global: np.ndarray,

    # Detailed output arrays for k=0 (if requested) - device arrays
    equity_curve_values_k0_global: np.ndarray,
    fast_ema_series_k0_global: np.ndarray,
    slow_ema_series_k0_global: np.ndarray,
    trade_entry_bar_indices_k0_global: np.ndarray,
    trade_exit_bar_indices_k0_global: np.ndarray,
    trade_entry_prices_k0_global: np.ndarray,
    trade_exit_prices_k0_global: np.ndarray,
    trade_types_k0_global: np.ndarray,
    trade_pnls_k0_global: np.ndarray,
    trade_count_k0_val_arr_global: np.ndarray # 1-element array for trade_count_k0
):
    k = cuda.grid(1) # Get the unique ID for this thread, corresponding to param combination 'k'

    if k >= k_fast_arr_global.shape[0]: # Check against a parameter array size
        return

    # Parameters for this specific combination 'k'
    stop_loss_pct = stop_loss_pcts_global[k]
    take_profit_pct = take_profit_pcts_global[k]
    execution_price_type_val = execution_price_types_global[k] # 1 for open, 0 for close
    kf = k_fast_arr_global[k]
    ks = k_slow_arr_global[k]

    # Local state variables for this thread (combination k)
    cash_k = initial_capital
    position_k = POSITION_NONE
    entry_price_k = 0.0
    sl_price_k = 0.0
    tp_price_k = 0.0

    current_fast_ema_k = 0.0
    prev_fast_ema_k = 0.0
    current_slow_ema_k = 0.0
    prev_slow_ema_k = 0.0

    total_trades_k = 0
    winning_trades_k = 0
    losing_trades_k = 0

    equity_k = initial_capital
    peak_equity_k = initial_capital
    max_drawdown_k = 0.0
    
    local_trade_count_k0 = 0 # For detailed output when k=0

    # Main Loop over candles
    for i in range(n_candles):
        current_open = open_prices_global[i]
        current_high = high_prices_global[i]
        current_low = low_prices_global[i]
        current_close = close_prices_global[i]

        # EMA Calculation
        if i == 0:
            current_fast_ema_k = current_close
            current_slow_ema_k = current_close
        else:
            current_fast_ema_k = (current_close * kf) + (prev_fast_ema_k * (1.0 - kf))
            current_slow_ema_k = (current_close * ks) + (prev_slow_ema_k * (1.0 - ks))

        if detailed_output_requested and k == 0 and i < fast_ema_series_k0_global.shape[0]:
            fast_ema_series_k0_global[i] = current_fast_ema_k
            slow_ema_series_k0_global[i] = current_slow_ema_k
        
        # Warm-up: Skip trading logic for the very first candle (i=0) to let EMAs initialize
        if i < 1: # Or a longer period if slow_ema_period is large and you want it more stable
            # Update equity for the first bar (no trades yet)
            equity_k = cash_k # No position, equity is cash
            if equity_k > peak_equity_k:
                peak_equity_k = equity_k
            # Max drawdown calculation is not very meaningful on first bar without trades
            if detailed_output_requested and k == 0 and i < equity_curve_values_k0_global.shape[0]:
                 equity_curve_values_k0_global[i] = equity_k
            prev_fast_ema_k = current_fast_ema_k
            prev_slow_ema_k = current_slow_ema_k
            continue

        action_taken_this_bar = False
        exec_price = current_open if execution_price_type_val == 1 else current_close

        # SL/TP Checks
        if position_k != POSITION_NONE:
            pnl_val = 0.0
            exit_price_sl_tp = 0.0
            closed_by_sl_tp = False
            current_trade_idx_for_log = local_trade_count_k0 -1 

            if position_k == POSITION_LONG:
                if sl_price_k > 0.0 and current_low <= sl_price_k:
                    exit_price_sl_tp = sl_price_k; closed_by_sl_tp = True
                elif tp_price_k > 0.0 and current_high >= tp_price_k:
                    exit_price_sl_tp = tp_price_k; closed_by_sl_tp = True
                if closed_by_sl_tp: pnl_val = exit_price_sl_tp - entry_price_k
            
            elif position_k == POSITION_SHORT:
                if sl_price_k > 0.0 and current_high >= sl_price_k:
                    exit_price_sl_tp = sl_price_k; closed_by_sl_tp = True
                elif tp_price_k > 0.0 and current_low <= tp_price_k:
                    exit_price_sl_tp = tp_price_k; closed_by_sl_tp = True
                if closed_by_sl_tp: pnl_val = entry_price_k - exit_price_sl_tp
            
            if closed_by_sl_tp:
                cash_k += pnl_val # For LONG. For SHORT, cash was already credited at entry (simplified model)
                                 # A more accurate short model: cash_k += (entry_price_k - exit_price_sl_tp) + entry_price_k
                                 # Let's assume PnL is added to cash for simplicity here.
                if pnl_val > 0.0: winning_trades_k += 1
                elif pnl_val < 0.0: losing_trades_k += 1
                
                if detailed_output_requested and k == 0 and \
                   current_trade_idx_for_log >= 0 and \
                   current_trade_idx_for_log < trade_entry_bar_indices_k0_global.shape[0]:
                    trade_exit_bar_indices_k0_global[current_trade_idx_for_log] = i
                    trade_exit_prices_k0_global[current_trade_idx_for_log] = exit_price_sl_tp
                    trade_pnls_k0_global[current_trade_idx_for_log] = pnl_val
                
                position_k = POSITION_NONE; entry_price_k = 0.0
                sl_price_k = 0.0; tp_price_k = 0.0
                action_taken_this_bar = True

        # Crossover Signal Logic (only if not closed by SL/TP)
        if not action_taken_this_bar:
            is_bullish_crossover = prev_fast_ema_k <= prev_slow_ema_k and current_fast_ema_k > current_slow_ema_k
            is_bearish_crossover = prev_fast_ema_k >= prev_slow_ema_k and current_fast_ema_k < current_slow_ema_k
            current_trade_idx_for_log_on_signal_close = local_trade_count_k0 - 1

            if is_bullish_crossover:
                if position_k == POSITION_SHORT: # Close short
                    pnl_val = entry_price_k - exec_price
                    cash_k += pnl_val # Simplified
                    if pnl_val > 0.0: winning_trades_k += 1
                    elif pnl_val < 0.0: losing_trades_k += 1
                    if detailed_output_requested and k == 0 and \
                       current_trade_idx_for_log_on_signal_close >= 0 and \
                       current_trade_idx_for_log_on_signal_close < trade_entry_bar_indices_k0_global.shape[0]:
                        trade_exit_bar_indices_k0_global[current_trade_idx_for_log_on_signal_close] = i
                        trade_exit_prices_k0_global[current_trade_idx_for_log_on_signal_close] = exec_price
                        trade_pnls_k0_global[current_trade_idx_for_log_on_signal_close] = pnl_val
                    position_k = POSITION_NONE; entry_price_k = 0.0; sl_price_k = 0.0; tp_price_k = 0.0
                
                if position_k == POSITION_NONE: # Open long
                    position_k = POSITION_LONG; entry_price_k = exec_price
                    total_trades_k += 1
                    if stop_loss_pct > 0.0: sl_price_k = exec_price * (1.0 - stop_loss_pct)
                    if take_profit_pct > 0.0: tp_price_k = exec_price * (1.0 + take_profit_pct)
                    if detailed_output_requested and k == 0 and local_trade_count_k0 < trade_entry_bar_indices_k0_global.shape[0]:
                        trade_entry_bar_indices_k0_global[local_trade_count_k0] = i
                        trade_entry_prices_k0_global[local_trade_count_k0] = exec_price
                        trade_types_k0_global[local_trade_count_k0] = POSITION_LONG
                        trade_exit_bar_indices_k0_global[local_trade_count_k0] = -1 # Mark as open
                        local_trade_count_k0 += 1
            
            elif is_bearish_crossover:
                if position_k == POSITION_LONG: # Close long
                    pnl_val = exec_price - entry_price_k
                    cash_k += pnl_val
                    if pnl_val > 0.0: winning_trades_k += 1
                    elif pnl_val < 0.0: losing_trades_k += 1
                    if detailed_output_requested and k == 0 and \
                       current_trade_idx_for_log_on_signal_close >= 0 and \
                       current_trade_idx_for_log_on_signal_close < trade_entry_bar_indices_k0_global.shape[0]:
                        trade_exit_bar_indices_k0_global[current_trade_idx_for_log_on_signal_close] = i
                        trade_exit_prices_k0_global[current_trade_idx_for_log_on_signal_close] = exec_price
                        trade_pnls_k0_global[current_trade_idx_for_log_on_signal_close] = pnl_val
                    position_k = POSITION_NONE; entry_price_k = 0.0; sl_price_k = 0.0; tp_price_k = 0.0

                if position_k == POSITION_NONE: # Open short
                    position_k = POSITION_SHORT; entry_price_k = exec_price
                    total_trades_k += 1
                    if stop_loss_pct > 0.0: sl_price_k = exec_price * (1.0 + stop_loss_pct)
                    if take_profit_pct > 0.0: tp_price_k = exec_price * (1.0 - take_profit_pct)
                    if detailed_output_requested and k == 0 and local_trade_count_k0 < trade_entry_bar_indices_k0_global.shape[0]:
                        trade_entry_bar_indices_k0_global[local_trade_count_k0] = i
                        trade_entry_prices_k0_global[local_trade_count_k0] = exec_price
                        trade_types_k0_global[local_trade_count_k0] = POSITION_SHORT
                        trade_exit_bar_indices_k0_global[local_trade_count_k0] = -1 # Mark as open
                        local_trade_count_k0 += 1
        
        # Update Equity Curve
        current_unrealized_pnl_val = 0.0
        if position_k == POSITION_LONG: current_unrealized_pnl_val = current_close - entry_price_k
        elif position_k == POSITION_SHORT: current_unrealized_pnl_val = entry_price_k - current_close
        
        equity_k = cash_k + current_unrealized_pnl_val # For long. For short, it's more complex if cash isn't adjusted at entry.
                                                    # Assuming a simplified model where PnL reflects in equity.

        if equity_k > peak_equity_k: peak_equity_k = equity_k
        
        current_dd_val = 0.0
        if peak_equity_k > 0.000001: # Avoid division by zero
             current_dd_val = (peak_equity_k - equity_k) / peak_equity_k
        if current_dd_val > max_drawdown_k: max_drawdown_k = current_dd_val

        if detailed_output_requested and k == 0 and i < equity_curve_values_k0_global.shape[0]:
            equity_curve_values_k0_global[i] = equity_k
        
        prev_fast_ema_k = current_fast_ema_k
        prev_slow_ema_k = current_slow_ema_k
    # --- End of candles loop ---

    # Final PNL Calculation
    last_close_price = close_prices_global[n_candles - 1] if n_candles > 0 else 0.0
    realized_pnl_sum_k = cash_k - initial_capital # This is PnL from closed trades
    unrealized_pnl_at_close_k = 0.0
    if position_k == POSITION_LONG: unrealized_pnl_at_close_k = last_close_price - entry_price_k
    elif position_k == POSITION_SHORT: unrealized_pnl_at_close_k = entry_price_k - last_close_price
    
    final_pnl_arr_global[k] = realized_pnl_sum_k + unrealized_pnl_at_close_k

    total_trades_arr_global[k] = total_trades_k
    winning_trades_arr_global[k] = winning_trades_k
    losing_trades_arr_global[k] = losing_trades_k
    max_drawdown_arr_global[k] = max_drawdown_k

    # Store final state (optional, for debugging or other metrics)
    cash_arr_global[k] = cash_k 
    position_arr_global[k] = position_k
    entry_price_arr_global[k] = entry_price_k
    sl_price_arr_global[k] = sl_price_k
    tp_price_arr_global[k] = tp_price_k
    equity_arr_global[k] = equity_k # This is final equity (cash + unrealized PnL of open position)
    peak_equity_arr_global[k] = peak_equity_k

    if detailed_output_requested and k == 0:
        trade_count_k0_val_arr_global[0] = local_trade_count_k0
        if position_k != POSITION_NONE and local_trade_count_k0 > 0:
            last_trade_idx_for_log = local_trade_count_k0 - 1
            if last_trade_idx_for_log < trade_entry_bar_indices_k0_global.shape[0] and \
               trade_exit_bar_indices_k0_global[last_trade_idx_for_log] == -1: # If still open
                trade_exit_bar_indices_k0_global[last_trade_idx_for_log] = n_candles - 1
                trade_exit_prices_k0_global[last_trade_idx_for_log] = last_close_price
                trade_pnls_k0_global[last_trade_idx_for_log] = unrealized_pnl_at_close_k

def run_ema_crossover_optimization_numba(
    open_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, close_prices: np.ndarray,
    fast_ema_periods: np.ndarray, slow_ema_periods: np.ndarray,
    stop_loss_pcts: np.ndarray, take_profit_pcts: np.ndarray,
    execution_price_types: np.ndarray, # Should be int array (e.g., 1 for open, 0 for close)
    initial_capital: float,
    detailed_output_requested: bool = False # Default to False
) -> tuple:
    n_combinations = fast_ema_periods.shape[0]
    n_candles = close_prices.shape[0]

    if n_candles == 0: # Handle empty price data
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return (np.zeros(n_combinations, dtype=np.float64), np.zeros(n_combinations, dtype=np.int64),
                np.zeros(n_combinations, dtype=np.int64), np.zeros(n_combinations, dtype=np.int64),
                np.zeros(n_combinations, dtype=np.float64),
                empty_float, empty_float, empty_float, # equity_curve, fast_ema, slow_ema
                empty_int, empty_int, empty_float, empty_float, empty_int, empty_float, # trades
                np.array([0], dtype=np.int64))


    # Host-side Output Arrays
    final_pnl_arr_host = np.zeros(n_combinations, dtype=np.float64)
    total_trades_arr_host = np.zeros(n_combinations, dtype=np.int64)
    winning_trades_arr_host = np.zeros(n_combinations, dtype=np.int64)
    losing_trades_arr_host = np.zeros(n_combinations, dtype=np.int64)
    max_drawdown_arr_host = np.zeros(n_combinations, dtype=np.float64)
    
    cash_arr_host = np.full(n_combinations, initial_capital, dtype=np.float64)
    position_arr_host = np.full(n_combinations, POSITION_NONE, dtype=np.int64)
    entry_price_arr_host = np.zeros(n_combinations, dtype=np.float64)
    sl_price_arr_host = np.zeros(n_combinations, dtype=np.float64)
    tp_price_arr_host = np.zeros(n_combinations, dtype=np.float64)
    equity_arr_host = np.full(n_combinations, initial_capital, dtype=np.float64)
    peak_equity_arr_host = np.full(n_combinations, initial_capital, dtype=np.float64)

    k_fast_arr_host = 2.0 / (fast_ema_periods.astype(np.float64) + 1.0)
    k_slow_arr_host = 2.0 / (slow_ema_periods.astype(np.float64) + 1.0)

    # Detailed Output Arrays (Host side)
    # Only allocate if detailed output is requested AND for a single combination scenario (n_combinations == 1)
    # Or, if you always want it for k=0 regardless of n_combinations, adjust this logic.
    # For simplicity, let's assume detailed output is primarily for single runs (n_combinations=1)
    # or the first combination (k=0) in a multi-combination run.
    # The kernel is written to only populate k=0 detailed arrays.
    
    equity_curve_size = n_candles if detailed_output_requested else 0
    equity_curve_values_k0_host = np.empty(equity_curve_size, dtype=np.float64)
    fast_ema_series_k0_host = np.empty(equity_curve_size, dtype=np.float64)
    slow_ema_series_k0_host = np.empty(equity_curve_size, dtype=np.float64)

    trade_array_size = MAX_TRADES_FOR_DETAILED_OUTPUT if detailed_output_requested else 0
    trade_entry_bar_indices_k0_host = np.full(trade_array_size, -1, dtype=np.int64) # Init with -1
    trade_exit_bar_indices_k0_host = np.full(trade_array_size, -1, dtype=np.int64)
    trade_entry_prices_k0_host = np.full(trade_array_size, np.nan, dtype=np.float64)
    trade_exit_prices_k0_host = np.full(trade_array_size, np.nan, dtype=np.float64)
    trade_types_k0_host = np.full(trade_array_size, POSITION_NONE, dtype=np.int64)
    trade_pnls_k0_host = np.full(trade_array_size, np.nan, dtype=np.float64)
    trade_count_k0_val_arr_host = np.array([0], dtype=np.int64)


    # Transfer data to GPU
    d_open_prices = cuda.to_device(open_prices.astype(np.float64)) # Ensure float64 for prices
    d_high_prices = cuda.to_device(high_prices.astype(np.float64))
    d_low_prices = cuda.to_device(low_prices.astype(np.float64))
    d_close_prices = cuda.to_device(close_prices.astype(np.float64))

    d_fast_ema_periods = cuda.to_device(fast_ema_periods.astype(np.int32)) # Periods are int
    d_slow_ema_periods = cuda.to_device(slow_ema_periods.astype(np.int32))
    d_stop_loss_pcts = cuda.to_device(stop_loss_pcts.astype(np.float64))
    d_take_profit_pcts = cuda.to_device(take_profit_pcts.astype(np.float64))
    d_execution_price_types = cuda.to_device(execution_price_types.astype(np.int32)) # Int for type
    d_k_fast_arr = cuda.to_device(k_fast_arr_host)
    d_k_slow_arr = cuda.to_device(k_slow_arr_host)

    d_cash_arr = cuda.to_device(cash_arr_host)
    d_position_arr = cuda.to_device(position_arr_host)
    d_entry_price_arr = cuda.to_device(entry_price_arr_host)
    d_sl_price_arr = cuda.to_device(sl_price_arr_host)
    d_tp_price_arr = cuda.to_device(tp_price_arr_host)
    
    d_final_pnl_arr = cuda.to_device(final_pnl_arr_host)
    d_total_trades_arr = cuda.to_device(total_trades_arr_host)
    d_winning_trades_arr = cuda.to_device(winning_trades_arr_host)
    d_losing_trades_arr = cuda.to_device(losing_trades_arr_host)
    d_equity_arr = cuda.to_device(equity_arr_host)
    d_peak_equity_arr = cuda.to_device(peak_equity_arr_host)
    d_max_drawdown_arr = cuda.to_device(max_drawdown_arr_host)

    d_equity_curve_values_k0 = cuda.to_device(equity_curve_values_k0_host)
    d_fast_ema_series_k0 = cuda.to_device(fast_ema_series_k0_host)
    d_slow_ema_series_k0 = cuda.to_device(slow_ema_series_k0_host)
    d_trade_entry_bar_indices_k0 = cuda.to_device(trade_entry_bar_indices_k0_host)
    d_trade_exit_bar_indices_k0 = cuda.to_device(trade_exit_bar_indices_k0_host)
    d_trade_entry_prices_k0 = cuda.to_device(trade_entry_prices_k0_host)
    d_trade_exit_prices_k0 = cuda.to_device(trade_exit_prices_k0_host)
    d_trade_types_k0 = cuda.to_device(trade_types_k0_host)
    d_trade_pnls_k0 = cuda.to_device(trade_pnls_k0_host)
    d_trade_count_k0_val_arr = cuda.to_device(trade_count_k0_val_arr_host)

    threads_per_block = 256 
    blocks_per_grid = (n_combinations + (threads_per_block - 1)) // threads_per_block

    ema_crossover_kernel[blocks_per_grid, threads_per_block](
        d_open_prices, d_high_prices, d_low_prices, d_close_prices,
        d_fast_ema_periods, d_slow_ema_periods,
        d_stop_loss_pcts, d_take_profit_pcts, d_execution_price_types,
        initial_capital, n_candles, detailed_output_requested,
        d_cash_arr, d_position_arr, d_entry_price_arr,
        d_sl_price_arr, d_tp_price_arr,
        d_final_pnl_arr, d_total_trades_arr,
        d_winning_trades_arr, d_losing_trades_arr,
        d_equity_arr, d_peak_equity_arr, d_max_drawdown_arr,
        d_k_fast_arr, d_k_slow_arr,
        d_equity_curve_values_k0,
        d_fast_ema_series_k0, d_slow_ema_series_k0,
        d_trade_entry_bar_indices_k0, d_trade_exit_bar_indices_k0,
        d_trade_entry_prices_k0, d_trade_exit_prices_k0,
        d_trade_types_k0, d_trade_pnls_k0,
        d_trade_count_k0_val_arr
    )
    cuda.synchronize()

    # Copy results back
    final_pnl_arr = d_final_pnl_arr.copy_to_host()
    total_trades_arr = d_total_trades_arr.copy_to_host()
    winning_trades_arr = d_winning_trades_arr.copy_to_host()
    losing_trades_arr = d_losing_trades_arr.copy_to_host()
    max_drawdown_arr = d_max_drawdown_arr.copy_to_host()

    if detailed_output_requested : # Always copy back k=0 detail if requested
        equity_curve_values_k0 = d_equity_curve_values_k0.copy_to_host()
        fast_ema_series_k0 = d_fast_ema_series_k0.copy_to_host()
        slow_ema_series_k0 = d_slow_ema_series_k0.copy_to_host()
        
        trade_count_k0_val_arr_host = d_trade_count_k0_val_arr.copy_to_host()
        trade_count_k0 = trade_count_k0_val_arr_host[0]

        actual_trades_entry_bar_indices = d_trade_entry_bar_indices_k0.copy_to_host()[:trade_count_k0]
        actual_trades_exit_bar_indices = d_trade_exit_bar_indices_k0.copy_to_host()[:trade_count_k0]
        actual_trades_entry_prices = d_trade_entry_prices_k0.copy_to_host()[:trade_count_k0]
        actual_trades_exit_prices = d_trade_exit_prices_k0.copy_to_host()[:trade_count_k0]
        actual_trades_types = d_trade_types_k0.copy_to_host()[:trade_count_k0]
        actual_trades_pnls = d_trade_pnls_k0.copy_to_host()[:trade_count_k0]
        trade_count_k0_arr_ret = np.array([trade_count_k0], dtype=np.int64)
    else: # No detailed output requested, return empty arrays for detailed parts
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        equity_curve_values_k0 = empty_float
        fast_ema_series_k0 = empty_float
        slow_ema_series_k0 = empty_float
        actual_trades_entry_bar_indices = empty_int
        actual_trades_exit_bar_indices = empty_int
        actual_trades_entry_prices = empty_float
        actual_trades_exit_prices = empty_float
        actual_trades_types = empty_int
        actual_trades_pnls = empty_float
        trade_count_k0_arr_ret = np.array([0], dtype=np.int64)

    return (
        final_pnl_arr, total_trades_arr, winning_trades_arr, losing_trades_arr, max_drawdown_arr,
        equity_curve_values_k0, fast_ema_series_k0, slow_ema_series_k0,
        actual_trades_entry_bar_indices, actual_trades_exit_bar_indices,
        actual_trades_entry_prices, actual_trades_exit_prices,
        actual_trades_types, actual_trades_pnls,
        trade_count_k0_arr_ret # This is an array containing the count
    )

# --- EMA Crossover Strategy Class ---
class EMACrossoverStrategy(BaseStrategy):
    strategy_id = "ema_crossover_numba_cuda" # Changed ID to reflect Numba CUDA
    strategy_name = "EMA Crossover (Numba CUDA)"
    strategy_description = "A simple EMA crossover strategy accelerated with Numba CUDA for optimization."

    # Numba-specific attribute
    has_numba_optimization = True # Flag to indicate this strategy uses the Numba optimizer

    def __init__(self, shared_ohlc_data: pd.DataFrame, params: Dict[str, Any], portfolio: PortfolioState):
        self.fast_ema_series = pd.Series(dtype=float)
        self.slow_ema_series = pd.Series(dtype=float)
        super().__init__(shared_ohlc_data, params, portfolio)
        # _initialize_strategy_state is called by super().__init__

    def _initialize_strategy_state(self):
        """Initializes EMAs. Called by BaseStrategy constructor."""
        close_prices = self.shared_ohlc_data['close']
        fast_period = self.params.get('fast_ema_period', 10)
        slow_period = self.params.get('slow_ema_period', 20)

        if not close_prices.empty:
            self.fast_ema_series = close_prices.ewm(span=fast_period, adjust=False).mean()
            self.slow_ema_series = close_prices.ewm(span=slow_period, adjust=False).mean()
        else:
            self.fast_ema_series = pd.Series(dtype=float, index=close_prices.index)
            self.slow_ema_series = pd.Series(dtype=float, index=close_prices.index)


    def update_indicators_and_generate_signals(self, bar_index: int, current_ohlc_bar: pd.Series) -> Optional[str]:
        """
        Generates trading signals for a single bar.
        This method is used for single backtests (non-Numba).
        The Numba kernel has its own embedded EMA calculation and signal logic.
        """
        if bar_index < 1 or bar_index >= len(self.fast_ema_series): # Need at least one previous bar
            return None

        # Get EMAs for current and previous bar
        # current_fast_ema = self.fast_ema_series.iloc[bar_index] # Already calculated in _initialize_strategy_state
        # prev_fast_ema = self.fast_ema_series.iloc[bar_index - 1]
        # current_slow_ema = self.slow_ema_series.iloc[bar_index]
        # prev_slow_ema = self.slow_ema_series.iloc[bar_index - 1]
        
        # For on-the-fly calculation matching Numba kernel's iterative EMA approach:
        # This part is tricky if _initialize_strategy_state already batch-calculated EMAs.
        # The BaseStrategy process_bar expects this method to calculate for the *current* bar.
        # For simplicity in this example, we'll use the pre-calculated series.
        # A more advanced BaseStrategy might pass EMA state between bars.

        if bar_index == 0: # Should not happen due to check above, but defensive
             self.current_fast_ema = current_ohlc_bar['close']
             self.current_slow_ema = current_ohlc_bar['close']
        else:
             # Iterative EMA calculation (if not using precomputed series directly)
             # This requires storing prev_ema from the previous call to this function.
             # For this example, we'll assume self.fast_ema_series and self.slow_ema_series
             # are correctly populated for bar_index and bar_index-1 by _initialize_strategy_state
             pass


        # Use pre-calculated series for signal generation
        # Ensure series are aligned with bar_index
        if bar_index < len(self.fast_ema_series) and bar_index -1 >= 0:
            current_fast_ema = self.fast_ema_series.iloc[bar_index]
            prev_fast_ema = self.fast_ema_series.iloc[bar_index - 1]
            current_slow_ema = self.slow_ema_series.iloc[bar_index]
            prev_slow_ema = self.slow_ema_series.iloc[bar_index - 1]
        else: # Not enough data for EMAs
            return None


        signal = None
        # Bullish crossover: fast EMA crosses above slow EMA
        if prev_fast_ema <= prev_slow_ema and current_fast_ema > current_slow_ema:
            signal = "BUY"
            # logger.info(f"{current_ohlc_bar.name}: BUY signal - FastEMA({current_fast_ema:.2f}) crossed above SlowEMA({current_slow_ema:.2f})")
        # Bearish crossover: fast EMA crosses below slow EMA
        elif prev_fast_ema >= prev_slow_ema and current_fast_ema < current_slow_ema:
            signal = "SELL"
            # logger.info(f"{current_ohlc_bar.name}: SELL signal - FastEMA({current_fast_ema:.2f}) crossed below SlowEMA({current_slow_ema:.2f})")
        
        return signal

    def get_indicator_series(self, ohlc_timestamps: List[pd.Timestamp]) -> List[IndicatorSeries]:
        """Returns EMA series for charting."""
        indicators = []
        # Ensure series are reindexed to the passed timestamps if necessary, or assume they align
        # For simplicity, assume self.fast_ema_series and self.slow_ema_series are already calculated
        # on self.shared_ohlc_data which should have these timestamps as index.

        if not self.fast_ema_series.empty:
            fast_ema_points = [
                # models.IndicatorDataPoint(time=ts.to_pydatetime(), value=round(val, 2))
                {"time": ts.to_pydatetime(), "value": round(val, 2)} # Simpler dict for now
                for ts, val in self.fast_ema_series.reindex(ohlc_timestamps).dropna().items()
            ]
            indicators.append(
                # models.IndicatorSeries(
                #     name=f"FastEMA({self.params.get('fast_ema_period', 'N/A')})",
                #     type="line", data=fast_ema_points,
                #     config=models.IndicatorConfig(color="blue", lineWidth=1)
                # )
                 {"name": f"FastEMA({self.params.get('fast_ema_period', 'N/A')})", "type": "line", "data": fast_ema_points, "config": {"color": "blue"}}
            )

        if not self.slow_ema_series.empty:
            slow_ema_points = [
                # models.IndicatorDataPoint(time=ts.to_pydatetime(), value=round(val, 2))
                {"time": ts.to_pydatetime(), "value": round(val, 2)}
                for ts, val in self.slow_ema_series.reindex(ohlc_timestamps).dropna().items()
            ]
            indicators.append(
                # models.IndicatorSeries(
                #     name=f"SlowEMA({self.params.get('slow_ema_period', 'N/A')})",
                #     type="line", data=slow_ema_points,
                #     config=models.IndicatorConfig(color="red", lineWidth=1)
                # )
                {"name": f"SlowEMA({self.params.get('slow_ema_period', 'N/A')})", "type": "line", "data": slow_ema_points, "config": {"color": "red"}}
            )
        return indicators

    @classmethod
    def get_info(cls) -> StrategyInfo:
        """Provides metadata about the strategy and its parameters."""
        return StrategyInfo(
            id=cls.strategy_id,
            name=cls.strategy_name,
            description=cls.strategy_description,
            parameters=[
                StrategyParameter(name="fast_ema_period", label="Fast EMA Period", type="int", default=10, min_value=2, max_value=100, step=1, description="Period for the fast Exponential Moving Average."),
                StrategyParameter(name="slow_ema_period", label="Slow EMA Period", type="int", default=20, min_value=10, max_value=500, step=10, description="Period for the slow Exponential Moving Average."),
                StrategyParameter(name="stop_loss_pct", label="Stop Loss %", type="float", default=0.0, min_value=0.0, max_value=100.0, step=0.5, description="Stop loss percentage from entry price. 0 to disable."),
                StrategyParameter(name="take_profit_pct", label="Take Profit %", type="float", default=0.0, min_value=0.0, max_value=100.0, step=0.5, description="Take profit percentage from entry price. 0 to disable."),
                StrategyParameter(name="execution_price_type", label="Execution Price", type="choice", default="close", options=["open", "close"], description="Price to use for execution (open or close of the signal bar).")            ]
        )

    # This method would be called by the OptimizerEngine if has_numba_optimization is True
    def run_numba_optimization(self,
                               open_p: np.ndarray, high_p: np.ndarray, low_p: np.ndarray, close_p: np.ndarray,
                               param_combinations: List[Dict[str, Any]], # List of dicts, each dict is one param set
                               initial_capital: float,
                               detailed_output_for_first_run: bool = False
                              ) -> pd.DataFrame: # Returns a DataFrame of results
        """
        Prepares data and calls the Numba CUDA kernel launcher.
        """
        num_combinations = len(param_combinations)
        if num_combinations == 0:
            return pd.DataFrame()

        # Convert list of param dicts to numpy arrays for Numba
        # Ensure all param_combinations have the same keys and order
        # This assumes get_info().parameters defines the parameters and their order
        
        # Initialize arrays with defaults or from the first combination
        # This needs to be robust to missing keys if param_combinations are not uniform
        # For simplicity, assume all combinations have all keys defined by get_info()
        
        # Helper to safely get param or its default if not in a specific combo
        # This is more for the Python-based backtester. Numba expects full arrays.
        # For Numba, ensure param_combinations are expanded to full arrays correctly by the optimizer_engine.

        fast_ema_periods_arr = np.array([c['fast_ema_period'] for c in param_combinations], dtype=np.int32)
        slow_ema_periods_arr = np.array([c['slow_ema_period'] for c in param_combinations], dtype=np.int32)
        
        # Handle optional SL/TP: if not present or 0, pass 0.0 to Numba kernel (kernel logic should handle 0 as disabled)
        stop_loss_pcts_arr = np.array([c.get('stop_loss_pct', 0.0) / 100.0 for c in param_combinations], dtype=np.float64) # Convert to fraction
        take_profit_pcts_arr = np.array([c.get('take_profit_pct', 0.0) / 100.0 for c in param_combinations], dtype=np.float64) # Convert to fraction
        
        # Map execution_price_type string to int for Numba (e.g., "open" -> 1, "close" -> 0)
        exec_price_map = {"open": 1, "close": 0}
        execution_price_types_arr = np.array([exec_price_map.get(c.get('execution_price_type', "close").lower(), 0) for c in param_combinations], dtype=np.int32)

        # Call the Numba launcher function
        (final_pnl, total_trades, winning_trades, losing_trades, max_drawdown,
         equity_curve, fast_ema_s, slow_ema_s,
         trade_entry_bars, trade_exit_bars, trade_entry_prices, trade_exit_prices,
         trade_types, trade_pnls, trade_count_arr) = run_ema_crossover_optimization_numba(
            open_prices=open_p, high_prices=high_p, low_prices=low_p, close_prices=close_p,
            fast_ema_periods=fast_ema_periods_arr,
            slow_ema_periods=slow_ema_periods_arr,
            stop_loss_pcts=stop_loss_pcts_arr,
            take_profit_pcts=take_profit_pcts_arr,
            execution_price_types=execution_price_types_arr,
            initial_capital=initial_capital,
            detailed_output_requested=detailed_output_for_first_run # Pass this flag
        )

        results_list = []
        for i in range(num_combinations):
            params_used = param_combinations[i]
            results_list.append({
                "params": params_used,
                "final_pnl": final_pnl[i],
                "total_trades": total_trades[i],
                "winning_trades": winning_trades[i],
                "losing_trades": losing_trades[i],
                "max_drawdown": max_drawdown[i] * 100 if max_drawdown[i] is not None else None, # Convert to percentage
                # Add other metrics as needed
            })
        
        results_df = pd.DataFrame(results_list)

        # If detailed output was requested (typically for the first/single run),
        # you can attach it to the DataFrame or handle it separately.
        # For now, the Numba function returns them, and they could be passed along
        # by the optimizer engine if needed.
        # The detailed output (equity_curve, EMAs, trades) corresponds to k=0 (first combination).
        
        # Store detailed output if requested and available
        # This part is for the optimizer engine to handle and potentially return.
        # For now, the strategy's role is just to execute the Numba part.
        self.detailed_run_data = None
        if detailed_output_for_first_run and num_combinations > 0 : # and len(equity_curve) > 0:
            self.detailed_run_data = {
                "equity_curve": equity_curve.tolist() if isinstance(equity_curve, np.ndarray) else equity_curve,
                "fast_ema_series": fast_ema_s.tolist() if isinstance(fast_ema_s, np.ndarray) else fast_ema_s,
                "slow_ema_series": slow_ema_s.tolist() if isinstance(slow_ema_s, np.ndarray) else slow_ema_s,
                "trades": {
                    "entry_bar": trade_entry_bars.tolist() if isinstance(trade_entry_bars, np.ndarray) else trade_entry_bars,
                    "exit_bar": trade_exit_bars.tolist() if isinstance(trade_exit_bars, np.ndarray) else trade_exit_bars,
                    "entry_price": trade_entry_prices.tolist() if isinstance(trade_entry_prices, np.ndarray) else trade_entry_prices,
                    "exit_price": trade_exit_prices.tolist() if isinstance(trade_exit_prices, np.ndarray) else trade_exit_prices,
                    "type": trade_types.tolist() if isinstance(trade_types, np.ndarray) else trade_types, # 1 for LONG, -1 for SHORT
                    "pnl": trade_pnls.tolist() if isinstance(trade_pnls, np.ndarray) else trade_pnls,
                    "count": trade_count_arr[0] if isinstance(trade_count_arr, np.ndarray) and len(trade_count_arr)>0 else 0
                }
            }
            # print(f"Detailed run data for first combo: {self.detailed_run_data['trades']['count']} trades")


        return results_df

# To make this file discoverable as a strategy, ensure EMACrossoverStrategy is defined.
# The strategy loader will look for classes inheriting from BaseStrategy.
