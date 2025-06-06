# app/strategies/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List, Optional

from .. import models

class PortfolioState:
    # ... (other parts of PortfolioState class remain the same)
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.current_position_qty = 0
        self.current_position_avg_price = 0.0
        self.current_position_type = None # "LONG" or "SHORT"
        self.trades: List[models.Trade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.open_trade: Optional[models.Trade] = None
        self.stop_loss_price: Optional[float] = None
        self.take_profit_price: Optional[float] = None

    def record_equity(self, timestamp: pd.Timestamp, current_market_price: float):
        current_value = self.current_cash
        position_value_change = 0
        if self.current_position_qty > 0:
            if self.current_position_type == "LONG":
                position_value_change = (current_market_price - self.current_position_avg_price) * self.current_position_qty
            elif self.current_position_type == "SHORT":
                position_value_change = (self.current_position_avg_price - current_market_price) * self.current_position_qty
            # This equity calculation assumes cash does not include proceeds from short sell directly until closure.
            current_value = self.current_cash + position_value_change 
        self.equity_curve.append({"time": timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp,
                                  "equity": round(current_value, 2)})

    def _reset_sl_tp(self):
        self.stop_loss_price = None
        self.take_profit_price = None

    def _update_sl_tp(self, entry_price: float, position_type: str,
                      stop_loss_pct: Optional[float], take_profit_pct: Optional[float]):
        self._reset_sl_tp() # Reset SL/TP prices first
        if position_type == "LONG":
            # Only set stop_loss_price if stop_loss_pct is provided and greater than 0
            if stop_loss_pct is not None and stop_loss_pct > 0:
                self.stop_loss_price = round(entry_price * (1 - stop_loss_pct / 100.0), 2)
            # Only set take_profit_price if take_profit_pct is provided and greater than 0
            if take_profit_pct is not None and take_profit_pct > 0:
                self.take_profit_price = round(entry_price * (1 + take_profit_pct / 100.0), 2)
        elif position_type == "SHORT":
            # Only set stop_loss_price if stop_loss_pct is provided and greater than 0
            if stop_loss_pct is not None and stop_loss_pct > 0:
                self.stop_loss_price = round(entry_price * (1 + stop_loss_pct / 100.0), 2)
            # Only set take_profit_price if take_profit_pct is provided and greater than 0
            if take_profit_pct is not None and take_profit_pct > 0:
                self.take_profit_price = round(entry_price * (1 - take_profit_pct / 100.0), 2)

    def buy(self, timestamp: pd.Timestamp, price: float, qty: int = 1,
            stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None):
        action_time = timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp
        if self.current_position_type == "SHORT":
            self.close_position(timestamp, price)
        cost = price * qty
        if self.current_position_type == "LONG":
            new_total_qty = self.current_position_qty + qty
            new_avg_price = ((self.current_position_avg_price * self.current_position_qty) + (price * qty)) / new_total_qty
            self.current_position_avg_price = round(new_avg_price, 2)
            self.current_position_qty = new_total_qty
            self._update_sl_tp(self.current_position_avg_price, "LONG", stop_loss_pct, take_profit_pct)
        else:
            self.current_position_avg_price = price
            self.current_position_qty = qty
            self.current_position_type = "LONG"
            self.open_trade = models.Trade(entry_time=action_time, entry_price=price, trade_type="LONG", qty=qty, status="OPEN")
            self._update_sl_tp(price, "LONG", stop_loss_pct, take_profit_pct)
        self.current_cash -= cost

    def sell(self, timestamp: pd.Timestamp, price: float, qty: int = 1,
             stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None):
        action_time = timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp
        if self.current_position_type == "LONG":
            self.close_position(timestamp, price)
        if self.current_position_type == "SHORT":
            new_total_qty = self.current_position_qty + qty
            new_avg_price = ((self.current_position_avg_price * self.current_position_qty) + (price * qty)) / new_total_qty
            self.current_position_avg_price = round(new_avg_price, 2)
            self.current_position_qty = new_total_qty
            self._update_sl_tp(self.current_position_avg_price, "SHORT", stop_loss_pct, take_profit_pct)
        else:
            self.current_position_avg_price = price
            self.current_position_qty = qty
            self.current_position_type = "SHORT"
            self.open_trade = models.Trade(entry_time=action_time, entry_price=price, trade_type="SHORT", qty=qty, status="OPEN")
            self._update_sl_tp(price, "SHORT", stop_loss_pct, take_profit_pct)

    def close_position(self, timestamp: pd.Timestamp, price: float):
        action_time = timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp
        if self.current_position_qty == 0 or not self.open_trade: return

        pnl = 0.0
        entry_price_for_pnl = self.open_trade.entry_price 
        qty_closed = self.current_position_qty

        if self.current_position_type == "LONG":
            self.current_cash += qty_closed * price
            pnl = (price - entry_price_for_pnl) * qty_closed
        elif self.current_position_type == "SHORT":
            pnl = (entry_price_for_pnl - price) * qty_closed
            self.current_cash += pnl # In short selling, cash is affected by PnL directly upon closing.
        
        self.open_trade.exit_time = action_time
        self.open_trade.exit_price = price
        self.open_trade.pnl = round(pnl, 2)
        self.open_trade.status = "CLOSED"
        self.trades.append(self.open_trade.model_copy(deep=True))
        
        self.current_position_qty = 0
        self.current_position_avg_price = 0.0
        self.current_position_type = None
        self.open_trade = None
        self._reset_sl_tp()

class BaseStrategy(ABC): # Ensure (ABC)
    strategy_id: str = "base_strategy"
    strategy_name: str = "Base Strategy"
    strategy_description: str = "Base class for strategies with on-the-fly indicator calculation."

    def __init__(self, shared_ohlc_data: pd.DataFrame, params: Dict[str, Any], portfolio: PortfolioState):
        self.shared_ohlc_data = shared_ohlc_data
        self.params = params
        self.portfolio = portfolio
        self._initialize_strategy_state()

    @abstractmethod
    def _initialize_strategy_state(self):
        pass

    @abstractmethod
    def update_indicators_and_generate_signals(self, bar_index: int, current_ohlc_bar: pd.Series) -> Optional[str]:
        pass

    def process_bar(self, bar_index: int):
        if bar_index >= len(self.shared_ohlc_data): return

        current_ohlc_bar = self.shared_ohlc_data.iloc[bar_index]
        timestamp = current_ohlc_bar.name # This is pd.Timestamp
        
        # Check and process SL/TP before generating new signals for the bar
        if self.portfolio.current_position_qty > 0 and self.portfolio.open_trade:
            exit_price_sl_tp = None
            if self.portfolio.current_position_type == "LONG":
                if self.portfolio.stop_loss_price and current_ohlc_bar['low'] <= self.portfolio.stop_loss_price:
                    exit_price_sl_tp = self.portfolio.stop_loss_price
                    logger.info(f"{timestamp}: LONG SL hit at {exit_price_sl_tp} (Low: {current_ohlc_bar['low']})")
                elif self.portfolio.take_profit_price and current_ohlc_bar['high'] >= self.portfolio.take_profit_price:
                    exit_price_sl_tp = self.portfolio.take_profit_price
                    logger.info(f"{timestamp}: LONG TP hit at {exit_price_sl_tp} (High: {current_ohlc_bar['high']})")
            elif self.portfolio.current_position_type == "SHORT":
                if self.portfolio.stop_loss_price and current_ohlc_bar['high'] >= self.portfolio.stop_loss_price:
                    exit_price_sl_tp = self.portfolio.stop_loss_price
                    logger.info(f"{timestamp}: SHORT SL hit at {exit_price_sl_tp} (High: {current_ohlc_bar['high']})")
                elif self.portfolio.take_profit_price and current_ohlc_bar['low'] <= self.portfolio.take_profit_price:
                    exit_price_sl_tp = self.portfolio.take_profit_price
                    logger.info(f"{timestamp}: SHORT TP hit at {exit_price_sl_tp} (Low: {current_ohlc_bar['low']})")
            
            if exit_price_sl_tp is not None:
                self.portfolio.close_position(timestamp, exit_price_sl_tp)
                # After closing due to SL/TP, record equity and return to avoid further actions on this bar
                self.portfolio.record_equity(timestamp, exit_price_sl_tp) # Record equity at exit price
                return # Important to return after SL/TP closure

        signal = self.update_indicators_and_generate_signals(bar_index, current_ohlc_bar)
        
        execution_price_type = self.params.get("execution_price_type", "close")
        action_price = current_ohlc_bar['open'] if execution_price_type == "open" else current_ohlc_bar['close']
        
        # Get SL/TP percentages from strategy parameters
        sl_pct = self.params.get("stop_loss_pct")
        tp_pct = self.params.get("take_profit_pct")

        if signal == "BUY":
            if self.portfolio.current_position_type != "LONG": # Avoid re-entry if already long
                 if self.portfolio.current_position_type == "SHORT": # Close short if flipping
                     self.portfolio.close_position(timestamp, action_price)
                 self.portfolio.buy(timestamp, action_price, stop_loss_pct=sl_pct, take_profit_pct=tp_pct)
        elif signal == "SELL":
            if self.portfolio.current_position_type != "SHORT": # Avoid re-entry if already short
                if self.portfolio.current_position_type == "LONG": # Close long if flipping
                    self.portfolio.close_position(timestamp, action_price)
                self.portfolio.sell(timestamp, action_price, stop_loss_pct=sl_pct, take_profit_pct=tp_pct)
        elif signal == "CLOSE_LONG" and self.portfolio.current_position_type == "LONG":
            self.portfolio.close_position(timestamp, action_price)
        elif signal == "CLOSE_SHORT" and self.portfolio.current_position_type == "SHORT":
            self.portfolio.close_position(timestamp, action_price)
        
        # Record equity at the end of processing the bar, using the bar's close price
        # This is now handled by the main backtesting loop after process_bar returns.

    @abstractmethod
    def get_indicator_series(self, ohlc_timestamps: List[pd.Timestamp]) -> List[models.IndicatorSeries]:
        """
        Returns all calculated indicator series for charting.
        This should be callable after all bars are processed or on demand for charting.
        ohlc_timestamps: A list of pd.Timestamp objects corresponding to the OHLC data index.
        """
        pass
    
    @classmethod
    def get_info(cls) -> models.StrategyInfo:
        return models.StrategyInfo(
            id=cls.strategy_id, name=cls.strategy_name,
            description=cls.strategy_description, parameters=[]
        )