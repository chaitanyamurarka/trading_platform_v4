@echo off
REM Batch file to start Celery, IQFeed Keep Alive, and Uvicorn server.

ECHO Starting services...
ECHO Each service will open in a new PowerShell window.
ECHO To stop a service, go to its window, press CTRL+C, and then you can close the window.
ECHO Closing this main batch file window will NOT stop the services.

REM --- Configuration (modify if your paths or commands differ) ---
SET VENV_ACTIVATION_SCRIPT=.\.venv\Scripts\Activate.ps1
SET TRADING_BACKEND_DIR=.\trading_backend\

REM --- Check if venv activation script exists ---
IF NOT EXIST "%VENV_ACTIVATION_SCRIPT%" (
    echo ERROR: Virtual environment activation script not found at %VENV_ACTIVATION_SCRIPT%
    echo Please ensure the path is correct and the virtual environment exists.
    pause
    exit /b
)

REM --- Check if trading_backend directory exists ---
IF NOT EXIST "%TRADING_BACKEND_DIR%" (
    echo ERROR: Trading backend directory not found at %TRADING_BACKEND_DIR%
    echo Please ensure the path is correct.
    pause
    exit /b
)

REM --- 1. Start Celery Worker ---
ECHO Starting Celery worker...
START "Celery Worker" powershell -ExecutionPolicy Bypass -NoExit -Command "& %VENV_ACTIVATION_SCRIPT%; Write-Host 'Virtual environment activated for Celery.'; cd %TRADING_BACKEND_DIR%; Write-Host ('Current directory: ' + (Get-Location).Path); celery -A app.tasks.celery_app.celery_application worker -l info -P eventlet"

REM Add a small delay to allow the first window to initialize if needed, though usually not necessary
timeout /t 2 /nobreak >nul

REM --- 2. Start IQFeed Keep Alive script ---
ECHO Starting IQFeed Keep Alive script...
START "IQFeed Keep Alive" powershell -ExecutionPolicy Bypass -NoExit -Command "& %VENV_ACTIVATION_SCRIPT%; Write-Host 'Virtual environment activated for IQFeed.'; cd %TRADING_BACKEND_DIR%; Write-Host ('Current directory: ' + (Get-Location).Path); python .\iqfeed_keep_alive.py"

timeout /t 2 /nobreak >nul

REM --- 3. Start Uvicorn Server ---
ECHO Starting Uvicorn server...
START "Uvicorn Server" powershell -ExecutionPolicy Bypass -NoExit -Command "& %VENV_ACTIVATION_SCRIPT%; Write-Host 'Virtual environment activated for Uvicorn.'; cd %TRADING_BACKEND_DIR%; Write-Host ('Current directory: ' + (Get-Location).Path); uvicorn app.main:app --host 0.0.0.0 --reload --log-level debug"

ECHO.
ECHO All services have been launched in separate windows.
ECHO Remember to stop them by pressing CTRL+C in each respective window.
ECHO This batch file will now close. (Or use 'pause' below if you want it to wait)