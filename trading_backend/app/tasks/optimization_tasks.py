# app/tasks/optimization_tasks.py
from celery.exceptions import Ignore # For handling known non-retryable errors
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

from .celery_app import celery_application
from .. import schemas # Pydantic models for data structures
# For DB access from Celery task:
from ..services import historical_data_service # To fetch data
from ..core import strategy_loader # To load the strategy class

import logging
from typing import List,Dict,Any

# Helper to manage DB session within a task
def get_db_session_for_task():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper function to make data JSON serializable
def make_json_serializable(data):
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(i) for i in data]
    elif isinstance(data, (np.integer, np.int64)): # Handle numpy integers
        return int(data)
    elif isinstance(data, (np.floating, np.float32, np.float64)): # Handle numpy floats
        return float(data)
    elif isinstance(data, np.ndarray): # Handle numpy arrays
        return make_json_serializable(data.tolist())
    elif isinstance(data, pd.Timestamp): # Handle pandas Timestamps
        return data.isoformat()
    elif pd.isna(data): # Handle pandas NaT or NaN (must be checked before number types)
        return None
    # Add other specific type conversions if needed (e.g., np.bool_ for booleans)
    elif isinstance(data, np.bool_):
        return bool(data)
    return data

@celery_application.task(name="tasks.optimize_strategy", bind=True, acks_late=True, reject_on_worker_lost=True)
def optimize_strategy_task(self, # Task instance for state updates
                           strategy_id: str,
                           symbol: str, # This is the trading symbol string, e.g., "RELIANCE"
                           interval_str: str, # Interval as string, e.g., "1m"
                           start_date_iso: str, # Dates as ISO strings
                           end_date_iso: str,
                           param_grid: dict,
                           initial_capital: float = 100000.0, # Default initial capital
                           detailed_output_for_first_run: bool = True # For Numba strategy details
                           ) -> dict:
    """
    Celery task to perform strategy optimization using dynamically loaded strategies.
    """
    self.update_state(state='PENDING', meta={'status': 'Initializing optimization...'})
    
    db_session_gen = get_db_session_for_task()
    db = next(db_session_gen)

    try:
        start_time = datetime.fromisoformat(start_date_iso)
        end_time = datetime.fromisoformat(end_date_iso)

        self.update_state(state='PROGRESS', meta={'status': f'Fetching historical data for {symbol} {interval_str} from {start_date_iso} to {end_date_iso}...'})
        
        # 1. Fetch Historical Data using the service
        # Note: historical_data_service.get_historical_data_with_fetch expects 'token' to be the trading symbol string
        ohlc_candles_schemas: List[schemas.Candle] = historical_data_service.get_historical_data_with_fetch(
            db=db,
            exchange="NSE", # Assuming NSE for now, or pass as param
            token=symbol, # 'token' here is the trading symbol string
            interval_val=interval_str,
            start_time=start_time,
            end_time=end_time
        )

        if not ohlc_candles_schemas:
            error_msg = f"No historical data found for {symbol} from {start_date_iso} to {end_date_iso} for interval {interval_str}."
            self.update_state(state='FAILURE', meta={'status': error_msg, 'exc_type': 'DataNotFoundError', 'exc_message': error_msg})
            # raise Ignore() # Or return an error structure
            return {"error": error_msg, "best_params": None, "best_score": float("-inf")}


        self.update_state(state='PROGRESS', meta={'status': f'Preparing data and loading strategy {strategy_id}...'})
        
        # Convert to DataFrame and then to NumPy arrays
        ohlc_df = pd.DataFrame([candle.model_dump() for candle in ohlc_candles_schemas])
        if ohlc_df.empty:
            error_msg = f"Historical data was empty after conversion for {symbol}."
            self.update_state(state='FAILURE', meta={'status': error_msg, 'exc_type': 'DataEmptyError', 'exc_message': error_msg})
            return {"error": error_msg, "best_params": None, "best_score": float("-inf")}

        ohlc_df['timestamp'] = pd.to_datetime(ohlc_df['timestamp'])
        ohlc_df = ohlc_df.set_index('timestamp').sort_index()

        open_p = ohlc_df['open'].to_numpy(dtype=np.float64)
        high_p = ohlc_df['high'].to_numpy(dtype=np.float64)
        low_p = ohlc_df['low'].to_numpy(dtype=np.float64)
        close_p = ohlc_df['close'].to_numpy(dtype=np.float64)
        # volume_p = ohlc_df['volume'].to_numpy(dtype=np.float64) # If needed by strategy

        # 2. Load Strategy
        StrategyClass = strategy_loader.get_strategy_class(strategy_id)
        if not StrategyClass:
            error_msg = f"Strategy with ID '{strategy_id}' not found."
            self.update_state(state='FAILURE', meta={'status': error_msg, 'exc_type': 'StrategyNotFoundError', 'exc_message': error_msg})
            return {"error": error_msg, "best_params": None, "best_score": float("-inf")}

        strategy_instance_for_info = StrategyClass(shared_ohlc_data=ohlc_df, params={}, portfolio=None) # Temp for info if needed

        # 3. Prepare parameter combinations
        param_names = list(param_grid.keys())
        param_value_lists = [param_grid[name] for name in param_names]
        raw_parameter_combinations = list(itertools.product(*param_value_lists))
        
        parameter_combinations_dicts: List[Dict[str, Any]] = []
        for combo_tuple in raw_parameter_combinations:
            parameter_combinations_dicts.append(dict(zip(param_names, combo_tuple)))

        total_combinations = len(parameter_combinations_dicts)
        if total_combinations == 0:
            error_msg = "No parameter combinations to process."
            self.update_state(state='FAILURE', meta={'status': error_msg, 'exc_type': 'NoParametersError', 'exc_message': error_msg})
            return {"error": error_msg, "best_params": None, "best_score": float("-inf")}

        self.update_state(state='PROGRESS', meta={'status': f'Starting optimization for {total_combinations} combinations...'})

        best_score = float("-inf")
        best_params_dict = None
        all_run_results_df = pd.DataFrame() # To store results from Numba or Python backtests

        # 4. Execute Strategy
        # Check if the strategy has a Numba-accelerated optimization method
        if hasattr(StrategyClass, 'has_numba_optimization') and StrategyClass.has_numba_optimization and \
           hasattr(strategy_instance_for_info, 'run_numba_optimization'):
            
            self.update_state(state='PROGRESS', meta={'status': f'Running Numba-accelerated optimization for {strategy_id}...'})
            try:
                # The run_numba_optimization method in the strategy should handle
                # converting param_combinations_dicts to the required numpy arrays.
                all_run_results_df = strategy_instance_for_info.run_numba_optimization(
                    open_p=open_p, high_p=high_p, low_p=low_p, close_p=close_p,
                    param_combinations=parameter_combinations_dicts, # Pass list of dicts
                    initial_capital=initial_capital,
                    detailed_output_for_first_run=detailed_output_for_first_run and (total_combinations > 0)
                )
                # Assuming run_numba_optimization returns a DataFrame with columns like
                # 'params' (dict), 'final_pnl', 'total_trades', etc.
                # And the 'score' to optimize on is 'final_pnl' for this example.
                if not all_run_results_df.empty:
                    # Find best score (e.g., max PnL)
                    # Ensure 'final_pnl' and 'params' columns exist
                    if 'final_pnl' in all_run_results_df.columns and 'params' in all_run_results_df.columns:
                        best_run_idx = all_run_results_df['final_pnl'].idxmax()
                        best_score = all_run_results_df.loc[best_run_idx, 'final_pnl']
                        best_params_dict = all_run_results_df.loc[best_run_idx, 'params']
                    else:
                        logging.error("Numba optimization result DataFrame missing 'final_pnl' or 'params' column.")
                        # Fallback or error
                else:
                    logging.warning("Numba optimization returned empty results.")

            except Exception as e_numba:
                error_msg = f"Error during Numba optimization for {strategy_id}: {e_numba}"
                logging.error(error_msg, exc_info=True)
                self.update_state(state='FAILURE', meta={'status': error_msg, 'exc_type': type(e_numba).__name__, 'exc_message': str(e_numba)})
                return {"error": error_msg, "best_params": None, "best_score": float("-inf")}
        
        else: # Fallback to iterative Python-based backtesting (bar-by-bar)
            self.update_state(state='PROGRESS', meta={'status': f'Running Python-based optimization for {strategy_id}...'})
            all_python_results = []
            for i, params_dict in enumerate(parameter_combinations_dicts):
                self.update_state(state='PROGRESS',
                                  meta={'current': i + 1, 'total': total_combinations,
                                        'status': f'Python Backtest {i+1}/{total_combinations}: {params_dict}'})
                
                # For each param combination, run a full backtest
                # This requires PortfolioState and Strategy instantiation per run
                from app.strategies.base_strategy import PortfolioState # Ensure import
                portfolio = PortfolioState(initial_capital=initial_capital)
                strategy_instance = StrategyClass(shared_ohlc_data=ohlc_df.copy(), params=params_dict, portfolio=portfolio)
                
                for bar_idx in range(len(ohlc_df)):
                    strategy_instance.process_bar(bar_idx)
                
                # After all bars, calculate final PnL or other metric
                # This logic should be part of PortfolioState or BaseStrategy ideally
                final_equity = portfolio.equity_curve[-1]['equity'] if portfolio.equity_curve else initial_capital
                current_score = final_equity - initial_capital # Example: PnL
                
                all_python_results.append({"params": params_dict, "final_pnl": current_score}) # Store PnL as score

                if current_score > best_score:
                    best_score = current_score
                    best_params_dict = params_dict
            
            all_run_results_df = pd.DataFrame(all_python_results)


        # Final result structure
        result = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "interval": interval_str,
            "start_date": start_date_iso,
            "end_date": end_date_iso,
            "best_params": best_params_dict,
            "best_score": best_score, # Define what score means (e.g., PnL, Sharpe)
            "total_combinations_processed": total_combinations,
            # "all_results_summary": all_run_results_df.to_dict(orient='records') # Optional: summary of all runs
        }
        if hasattr(strategy_instance_for_info, 'detailed_run_data') and strategy_instance_for_info.detailed_run_data:
            result["detailed_output_first_run"] = strategy_instance_for_info.detailed_run_data

         # **** Convert the entire result dictionary to be JSON serializable ****
        serializable_result = make_json_serializable(result)    
        self.update_state(state='SUCCESS', meta=serializable_result) # Store full result in meta for SUCCESS
        return result

    except Exception as e:
        error_msg = f"Unhandled exception in optimization task: {e}"
        logging.error(error_msg, exc_info=True)
        error_result = {
            "error": error_msg,
            "best_params": None,
            "best_score": float("-inf"), # Ensure this is a standard float
            "exc_type": type(e).__name__,
            "exc_message": str(e)
        }
        self.update_state(state='FAILURE', meta=make_json_serializable(error_result))
        # Do not re-raise the exception if you want Celery to mark it as failed based on state
        # Re-raising might cause retries depending on Celery config.
        # raise Ignore() # To ensure it's marked as failed and not retried by Celery
        return {"error": error_msg, "best_params": None, "best_score": float("-inf")} # Return error structure
    finally:
        # Ensure DB session is closed for the task
        try:
            next(db_session_gen) # This will trigger the finally block in get_db_session_for_task
        except StopIteration:
            pass # Expected
        except Exception as e_db_close:
            logging.error(f"Error closing DB session in Celery task: {e_db_close}", exc_info=True)

