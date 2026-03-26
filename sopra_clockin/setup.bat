@echo off
REM ===================================================================
REM Setup Script for SopraGP4U Clock In/Out Automation
REM ===================================================================
REM This script handles initial setup and dependency installation
REM ===================================================================

setlocal enabledelayedexpansion

echo.
echo ===================================================================
echo SopraGP4U Automation - Setup Script
echo ===================================================================
echo.

REM Check Python version
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [OK] Python is installed
python --version

echo.
echo Installing required dependencies...
echo.

REM Install requirements
pip install -r requirements.txt

if !errorlevel! equ 0 (
    echo.
    echo [OK] Dependencies installed successfully
    echo.
    echo Next steps:
    echo 1. Configure credentials by setting environment variables:
    echo    - SOPRA_USERNAME: Your SopraGP4U username
    echo    - SOPRA_PASSWORD: Your SopraGP4U password
    echo.
    echo 2. Test the script:
    echo    python src/sopra_clockin.py
    echo.
    echo 3. Configure Windows Task Scheduler to run scheduled_clockin.bat
    echo    See README.md for detailed instructions
    echo.
) else (
    echo [ERROR] Failed to install dependencies
    echo Check your internet connection and try again
    exit /b 1
)

pause
endlocal
