"""
cache_cleanup_tasks.py

This module defines a periodic Celery task responsible for cleaning up expired
user sessions and their associated data from the Redis cache.
"""
import time
import logging
from .celery_app import celery_application
from ..core.cache import redis_client

# Define the session timeout duration (e.g., 30 minutes).
# A session is considered expired if no heartbeat has been received within this period.
SESSION_TIMEOUT_SECONDS = 30 * 60

@celery_application.task(name="tasks.cleanup_expired_sessions")
def cleanup_expired_sessions_task():
    """
    A periodic Celery task that scans for and removes expired session data from Redis.

    This task should be scheduled to run periodically (e.g., every hour) using
    Celery Beat. It performs the following steps:
    1. Scans for all keys matching the "session:*" pattern.
    2. Checks the timestamp of each session key to see if it has expired.
    3. If a session is expired, it deletes the session key itself.
    4. It then finds and deletes all other data associated with that session's
       token (e.g., cached chart data matching "user:<session_token>:*").

    This proactive cleanup prevents the Redis database from growing indefinitely
    with stale data from inactive users.
    """
    logging.info("Starting expired session cleanup task...")
    try:
        # Scan for all session keys in Redis. `scan_iter` is memory-efficient for large databases.
        session_keys = redis_client.scan_iter("session:*")
        current_time = int(time.time())
        expired_sessions_count = 0
        deleted_data_keys_count = 0

        for session_key_bytes in session_keys:
            session_key = session_key_bytes.decode('utf-8')
            last_seen_timestamp_bytes = redis_client.get(session_key)
            
            if last_seen_timestamp_bytes:
                last_seen_timestamp = int(last_seen_timestamp_bytes)
                
                # Check if the session has timed out.
                if current_time - last_seen_timestamp > SESSION_TIMEOUT_SECONDS:
                    expired_sessions_count += 1
                    session_token = session_key.split(":", 1)[1]
                    logging.info(f"Session {session_token[:8]}... has expired. Deleting associated data.")

                    # Use a pipeline for efficient bulk deletion of keys.
                    pipe = redis_client.pipeline()
                    # Delete the main session key.
                    pipe.delete(session_key)

                    # Find and delete all associated user data keys.
                    user_data_pattern = f"user:{session_token}:*"
                    keys_to_delete = [key for key in redis_client.scan_iter(user_data_pattern)]
                    
                    if keys_to_delete:
                        for key in keys_to_delete:
                            pipe.delete(key)
                        deleted_data_keys_count += len(keys_to_delete)
                        logging.debug(f"Queued {len(keys_to_delete)} data keys for deletion for expired session.")
                    
                    # Execute all deletion commands in a single round-trip to Redis.
                    pipe.execute()

        logging.info(f"Cleanup task finished. Cleaned up {expired_sessions_count} expired sessions "
                     f"and deleted {deleted_data_keys_count} associated data keys.")
        return {"status": "success", "sessions_cleaned": expired_sessions_count, "data_keys_deleted": deleted_data_keys_count}

    except Exception as e:
        logging.error(f"An error occurred during the cache cleanup task: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}