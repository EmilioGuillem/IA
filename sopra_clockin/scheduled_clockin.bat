@echo off
REM ===================================================================
REM SopraGP4U Clock In/Out Automation - Windows Batch Script
REM ===================================================================
REM This script is designed to be executed by Windows Task Scheduler
REM
REM Features:
REM - Logs output to file
REM - Handles Python environment activation
REM - Error handling and exit codes
REM
REM Usage: scheduled_clockin.bat
REM ===================================================================

setlocal enabledelayedexpansion

REM Set the project directory
set PROJECT_DIR=%~dp0

REM Set Python executable (update this path if your Python is installed elsewhere)
set PYTHON_EXE=python

REM Set log file for scheduling
set SCHEDULE_LOG=%PROJECT_DIR%logs\schedule_run.log

REM Append timestamp to log
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set date=%%c-%%a-%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set time=%%a%%b)
echo. >> "%SCHEDULE_LOG%"
echo ===================================================================== >> "%SCHEDULE_LOG%"
echo Scheduled run at %date% %time% >> "%SCHEDULE_LOG%"
echo ===================================================================== >> "%SCHEDULE_LOG%"

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Set browser preference (options: chrome, edge)
REM Uncomment and modify the line below to use Edge instead of Chrome:
REM set SOPRA_BROWSER=edge
REM Default is Chrome if not set
if not defined SOPRA_BROWSER (
    set SOPRA_BROWSER=chrome
)

REM Check if Python is installed
%PYTHON_EXE% --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Python is not installed or not in PATH >> "%SCHEDULE_LOG%"
    exit /b 1
)

REM Run the main script
echo Starting automation script with browser: %SOPRA_BROWSER% >> "%SCHEDULE_LOG%"
%PYTHON_EXE% src\sopra_clockin.py >> "%SCHEDULE_LOG%" 2>&1

if !errorlevel! equ 0 (
    echo Automation completed successfully >> "%SCHEDULE_LOG%"
    exit /b 0
) else (
    echo Automation failed with error code !errorlevel! >> "%SCHEDULE_LOG%"
    exit /b 1
)

endlocal
