# app/dtn_live_listener.py
from . import pyiqfeed as iq
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PlatformLiveBarListener(iq.SilentBarListener):
    def __init__(self, name="PlatformLiveBarListener", symbol: str = "UNKNOWN"):
        super().__init__(name)
        self._name = f"{name}-{symbol}"
        self.symbol = symbol

    def process_live_bar(self, bar_data: np.ndarray) -> None:
        # This is where you'd process the completed live bar.
        # For example, push to a WebSocket, save to a real-time DB, or trigger an event.
        logger.debug(f"[{self._name}] Live Bar Completed for {self.symbol}: {bar_data}")
        # Example: Convert to schemas.CandleBase and process
        # candle = self._parse_bar_data(bar_data)
        # if candle:
        #     # Your processing logic here, e.g.:
        #     # await websocket_manager.broadcast_candle(self.symbol, candle)
        #     # await live_data_cache.update_latest_candle(self.symbol, candle)
        pass

    def process_latest_bar_update(self, bar_data: np.ndarray) -> None:
        # This is called for updates to the currently forming bar.
        # Can be very frequent; use judiciously.
        # logger.debug(f"[{self._name}] Live Bar Update for {self.symbol}: {bar_data}")
        pass

    def process_history_bar(self, bar_data: np.ndarray) -> None:
        logger.debug(f"[{self._name}] History Bar (lookback for live) for {self.symbol}: {bar_data}")
        # Process lookback data if needed
        pass

    def process_error(self, fields):
        logger.error(f"[{self._name}] Listener Error for {self.symbol}: {fields}")

    def feed_is_stale(self) -> None:
        logger.warning(f"[{self._name}] Feed is stale for {self.symbol}.")

    def feed_is_fresh(self) -> None:
        logger.info(f"[{self._name}] Feed is fresh for {self.symbol}.")

    # You might want a helper to parse bar_data to your CandleBase schema
    # def _parse_bar_data(self, bar_data_item: np.void) -> Optional[schemas.CandleBase]:
    #     # Similar to parse_iqfeed_bar_data in historical_data_service.py
    #     # but adapted for the live bar data structure if it differs, or reusable
    #     try:
    #         # Assuming bar_data_item is a single element from the np.ndarray
    #         # The structure of bar_data from BarConn is:
    #         # ('symbol', 'S64'), ('date', '<M8[D]'), ('time', '<u8'),
    #         # ('open_p', '<f8'), ('high_p', '<f8'), ('low_p', '<f8'), ('close_p', '<f8'),
    #         # ('tot_vlm', '<u8'), ('prd_vlm', '<u8'), ('num_trds', '<u8'), ('req_id', 'S64')
    #         # The 'time' field is epoch microseconds from midnight.
            
    #         item = bar_data_item[0] # If bar_data is an array with one element
    #         dt_date = item['date'].astype(datetime)
    #         ts_value = datetime.combine(dt_date, datetime.min.time()) + timedelta(microseconds=int(item['time']))

    #         return schemas.CandleBase(
    #             timestamp=ts_value,
    #             open=float(item['open_p']),
    #             high=float(item['high_p']),
    #             low=float(item['low_p']),
    #             close=float(item['close_p']),
    #             volume=float(item['prd_vlm']) # prd_vlm is period volume
    #         )
    #     except Exception as e:
    #         logger.error(f"Error parsing live bar data: {e}, data: {bar_data_item}")
    #         return None