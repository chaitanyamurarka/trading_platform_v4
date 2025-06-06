# app/core/cache.py
import redis
import json
from typing import Optional, List, Any
from ..config import settings # Your application settings
from .. import schemas # Your Pydantic schemas

# Redis connection (URL from environment or defaults)
REDIS_URL = settings.REDIS_URL
redis_client = redis.Redis.from_url(REDIS_URL)

# Define a cache expiration time for user-specific data (e.g., 35 minutes)
CACHE_EXPIRATION_SECONDS = 60 * 35

def get_cached_ohlc_data(cache_key: str) -> Optional[List[schemas.Candle]]:
    """Attempts to retrieve and deserialize OHLC data from Redis cache."""
    cached_data = redis_client.get(cache_key)
    if cached_data:
        try:
            # Assuming data is stored as a JSON string of a list of candle dicts
            deserialized_data = json.loads(cached_data)
            # Convert list of dicts back to list of Pydantic models
            return [schemas.Candle(**item) for item in deserialized_data]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error deserializing cached data for key {cache_key}: {e}")
            return None
    return None

def set_cached_ohlc_data(cache_key: str, data: List[schemas.Candle], expiration: int = CACHE_EXPIRATION_SECONDS):
    """Serializes and stores OHLC data in Redis cache with an expiration."""
    try:
        # Convert list of Pydantic models to list of dicts for JSON serialization
        serializable_data = [item.model_dump(mode='json') for item in data] # Pydantic v2
        # serializable_data = [item.dict() for item in data] # Pydantic v1
        redis_client.set(cache_key, json.dumps(serializable_data), ex=expiration)
    except TypeError as e:
        print(f"Error serializing data for cache key {cache_key}: {e}")

def build_ohlc_cache_key(
    exchange: str,
    token: str,
    interval: str,
    date_str: str, # Changed from start_time_iso, end_time_iso
    session_token: Optional[str] = None
) -> str:
    """
    Builds a consistent cache key for OHLC data queries.
    For 1s data, the key is now based on the date string (YYYY-MM-DD).
    For other intervals, it uses the full date string which represents the query range.
    """
    if session_token and interval == "1s":
        # User-specific cache key for 1s data, now per-day.
        return f"user:{session_token}:ohlc:{exchange}:{token}:1s:{date_str}"
    else:
        # Generic, shared cache key for aggregated data (uses start_end string).
        return f"ohlc:{exchange}:{token}:{interval}:{date_str}"