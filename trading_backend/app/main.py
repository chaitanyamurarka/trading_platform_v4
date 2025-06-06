"""
main.py

This module serves as the main entry point for the Trading Platform's FastAPI backend.

It initializes the FastAPI application, configures middleware (CORS, GZip),
mounts the static frontend files, sets up application startup and shutdown events,
and includes the API routers for different functionalities.
"""

import logging
import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import application components
from .dtn_iq_client import launch_iqfeed_service_if_needed
# Temporily Commenting this
# from .core import strategy_loader
from .services.live_data_feed_service import live_feed_service
from .routers import historical_data_router, utility_router

# --- Basic Logging Configuration ---
# Configures logging to output to standard output with a detailed format.
# This ensures that logs are easily visible in containerized environments or when running locally.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="Trading Platform API",
    description="Backend API for historical data, live data feeds, and strategy execution.",
    version="1.0.0"
)

# --- Static Frontend File Serving ---
# Determine the correct path to the frontend directory to serve the single-page application.
# The path is calculated relative to this file's location.
script_dir = os.path.dirname(__file__)
backend_root_dir = os.path.dirname(script_dir)
project_root_dir = os.path.dirname(backend_root_dir)
frontend_dir = os.path.join(project_root_dir, "frontend")
static_dir = os.path.join(frontend_dir, "static")

# Mount the 'static' directory to serve CSS, JS, and other assets.
# A request to '/static/css/style.css' will serve the file from 'frontend/static/css/style.css'.
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logging.info(f"Mounted static directory: {static_dir}")
else:
    logging.error(f"Static directory not found at: {static_dir}. Static files will not be served.")


# --- Middleware Configuration ---

# Add GZip middleware to compress responses for better network performance.
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add CORS (Cross-Origin Resource Sharing) middleware to allow requests from any origin.
# This is crucial for the frontend (served, e.g., on localhost:xxxx) to communicate with the backend API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- Application Lifecycle Events ---

@app.on_event("startup")
async def startup_event():
    """
    Actions to perform when the application starts up.
    - Initializes the connection to the DTN IQFeed service.
    - Pre-loads available trading strategies.
    """
    logging.info("Application starting up...")
    # Attempt to launch the IQFeed connection client if it's not already running.
    # This is a critical prerequisite for fetching market data.
    launch_iqfeed_service_if_needed()
    # Pre-load all available strategies from the 'strategies' directory.
    # Temporily Commenting
    # strategy_loader.load_strategies()
    logging.info("Application startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform when the application is shutting down.
    - Gracefully disconnects the live data feed service.
    """
    logging.info("Application shutting down...")
    live_feed_service.disconnect()
    logging.info("Live feed service disconnected.")


# --- API Routers ---

# Include routers from other modules to organize API endpoints.
app.include_router(historical_data_router.router)
app.include_router(utility_router.router)
# To add the optimization router, uncomment the following lines:
# from .routers import optimization_router
# app.include_router(optimization_router.router)


# --- Root Endpoint ---

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serves the main `index.html` file of the frontend application.
    This allows the backend to act as the web server for the SPA.
    """
    index_html_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_html_path):
        with open(index_html_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    else:
        logging.error(f"index.html not found at: {index_html_path}")
        raise HTTPException(status_code=404, detail="index.html not found")