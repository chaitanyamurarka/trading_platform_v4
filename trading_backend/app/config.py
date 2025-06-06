"""
config.py

This module centralizes application configuration management.

It uses pydantic-settings to load settings from environment variables or a .env file,
providing a single, type-hinted source of truth for configuration values like
database URLs, API keys, and other service credentials.
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional

# Load environment variables from a .env file if it exists.
# This is useful for local development.
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    """
    Defines the application's configuration settings.
    Pydantic automatically reads these from environment variables (case-insensitive)
    or from a .env file.
    """
    # URL for the Redis instance, used for caching and as a Celery message broker/result backend.
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Celery configuration, defaulting to the same Redis instance.
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    # DTN IQFeed Credentials for market data.
    # These are optional as they might be configured directly in the IQConnect client.
    DTN_PRODUCT_ID: Optional[str] = os.getenv("DTN_PRODUCT_ID")
    DTN_LOGIN: Optional[str] = os.getenv("DTN_LOGIN")
    DTN_PASSWORD: Optional[str] = os.getenv("DTN_PASSWORD")

    class Config:
        # Specifies the name of the environment file to load.
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single, globally-accessible instance of the settings.
settings = Settings()