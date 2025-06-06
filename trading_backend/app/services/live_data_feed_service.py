# app/services/live_data_feed_service.py
from .. import pyiqfeed as iq
import logging
import time
# Only import get_iqfeed_history_conn; we will not rely on the global flag directly here.
from app.dtn_iq_client import get_iqfeed_history_conn
from app.dtn_live_listener import PlatformLiveBarListener

logger = logging.getLogger(__name__)

class LiveDataFeedService:
    def __init__(self):
        self.bar_conn = None
        self.listeners = {}
        self.subscribed_symbols = set()

    def _ensure_connection(self):
        logger.debug("LiveDataFeedService: Ensuring IQFeed service is responsive.")
        
        # Call get_iqfeed_history_conn(). This function attempts to launch IQFeed
        # if not running and checks admin port connectivity.
        # If it returns None, IQFeed is not in a usable state.
        hist_conn_check_obj = get_iqfeed_history_conn()

        if hist_conn_check_obj is None:
            logger.error("get_iqfeed_history_conn() indicated IQFeed service is not available. Cannot establish BarConn for live data.")
            # At this point, get_iqfeed_history_conn() would have logged details if launch failed.
            raise ConnectionError("IQFeed service appears unresponsive; cannot initialize live data BarConn.")
        
        # If we got here, get_iqfeed_history_conn() succeeded, meaning IQFeed's admin port is responsive.
        # This implies IQConnect.exe is running. We can now proceed with BarConn.
        # We don't need to keep hist_conn_check_obj, it was just for the check/launch side-effect.
        logger.info("IQFeed service responded to initial check (via HistoryConn attempt). Proceeding with BarConn setup.")

        if self.bar_conn is None:
            logger.info("Creating and connecting BarConn for live data.")
            self.bar_conn = iq.BarConn(name="PlatformLiveBarFeed")
            try:
                self.bar_conn.connect() # Connect to the derivatives port (default 9400)
                time.sleep(1) # Allow a moment for the connection to establish
                
                # pyiqfeed's BarConn.connected() checks if IQFeed client is connected to DTN servers.
                if not self.bar_conn.connected():
                     logger.warning("BarConn socket to IQConnect.exe established, but IQConnect.exe not yet reported as connected to DTN servers. Live data might be delayed or fail if login issues persist.")
                logger.info("BarConn socket connection initiated.")
            except Exception as e:
                logger.error(f"Failed to connect BarConn: {e}", exc_info=True)
                self.bar_conn = None 
                raise ConnectionError(f"Failed to connect BarConn for live data: {e}")
        # Note: Further checks for an existing but disconnected self.bar_conn could be added if needed,
        # but for initial setup, this covers the main path.

    # --- subscribe_to_symbol, unsubscribe_from_symbol, disconnect methods ---
    # (These should remain largely the same as the last version that correctly used self.bar_conn directly)

    def subscribe_to_symbol(self, symbol: str, interval_len: int = 60, interval_type: str = 's', lookback_bars: int = 5, update_interval_secs: int = 0):
        self._ensure_connection() 
        
        if symbol in self.subscribed_symbols:
            logger.info(f"Already subscribed to live bars for {symbol}")
            return

        listener = PlatformLiveBarListener(symbol=symbol)
        if not self.bar_conn: # Should be caught by _ensure_connection if it failed
            logger.error("BarConn is not initialized. Cannot subscribe.")
            raise ConnectionError("BarConn not initialized.")

        self.bar_conn.add_listener(listener)
        self.listeners[symbol] = listener

        logger.info(f"Subscribing to live {interval_len}{interval_type} bars for {symbol} with {lookback_bars} lookback, update: {update_interval_secs}s.")
        try:
            self.bar_conn.watch(
                symbol=symbol,
                interval_len=interval_len,
                interval_type=interval_type,
                update=update_interval_secs,
                lookback_bars=lookback_bars
            )
            self.subscribed_symbols.add(symbol)
            logger.info(f"Successfully initiated watch for {symbol}")
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}", exc_info=True)
            if symbol in self.listeners: 
                if self.bar_conn:
                    self.bar_conn.remove_listener(self.listeners[symbol])
                del self.listeners[symbol]
            # Do not remove from subscribed_symbols here if add failed
            raise

    def unsubscribe_from_symbol(self, symbol: str):
        if self.bar_conn and symbol in self.subscribed_symbols:
            logger.info(f"Unsubscribing from live bars for {symbol}")
            listener_to_remove = self.listeners.get(symbol)
            try:
                self.bar_conn.unwatch(symbol)
            except Exception as e:
                logger.error(f"Error sending unwatch command for {symbol}: {e}", exc_info=True)
            # Always try to clean up listener and tracking set
            if listener_to_remove and self.bar_conn:
                try:
                    self.bar_conn.remove_listener(listener_to_remove)
                except Exception as e_rem:
                    logger.error(f"Error removing listener for {symbol}: {e_rem}", exc_info=True)
            if symbol in self.listeners:
                del self.listeners[symbol]
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
            logger.info(f"Successfully processed unwatch for {symbol}")
        else:
            logger.warning(f"Not subscribed to {symbol} or BarConn not active.")

    def disconnect(self):
        if self.bar_conn:
            logger.info("Disconnecting BarConn and cleaning up subscriptions.")
            for symbol_to_unwatch in list(self.subscribed_symbols): # Iterate over a copy
                self.unsubscribe_from_symbol(symbol_to_unwatch)
            
            try:
                self.bar_conn.disconnect()
                logger.info("BarConn disconnected successfully.")
            except Exception as e:
                logger.error(f"Error during BarConn disconnect: {e}", exc_info=True)
        
        self.bar_conn = None # Crucial to reset
        self.listeners.clear()
        self.subscribed_symbols.clear()

live_feed_service = LiveDataFeedService()