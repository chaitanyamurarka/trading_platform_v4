@echo off
REM Batch file to drop and create a MySQL database

REM --- Configuration ---
SET MYSQL_USER=root
SET MYSQL_PASSWORD=root123
SET DATABASE_NAME=app_data
REM Update this path if your mysql.exe is in a different location
SET MYSQL_PATH="C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"
REM For XAMPP users, it might be something like:
REM SET MYSQL_PATH="C:\xampp\mysql\bin\mysql.exe"

REM --- Check if MYSQL_PATH is set correctly ---
IF NOT EXIST %MYSQL_PATH% (
    echo ERROR: MySQL executable not found at %MYSQL_PATH%
    echo Please update the MYSQL_PATH variable in this script.
    pause
    exit /b
)

echo Connecting to MySQL to drop and create database '%DATABASE_NAME%'...

REM --- SQL Commands ---
REM 1. Drop the database if it exists
REM 2. Create the database
(
    echo DROP DATABASE IF EXISTS %DATABASE_NAME%;
    echo CREATE DATABASE %DATABASE_NAME%;
    echo FLUSH PRIVILEGES;
) > temp_mysql_commands.sql

REM --- Execute SQL Commands ---
%MYSQL_PATH% -u %MYSQL_USER% -p%MYSQL_PASSWORD% < temp_mysql_commands.sql

REM --- Check for errors (basic check) ---
IF ERRORLEVEL 1 (
    echo ERROR: MySQL command execution failed.
    echo Please check your MySQL credentials, server status, and the commands.
) ELSE (
    echo Database '%DATABASE_NAME%' dropped and recreated successfully.
)

REM --- Cleanup ---
IF EXIST temp_mysql_commands.sql (
    del temp_mysql_commands.sql
)

echo.
echo Batch file execution finished.
pause
