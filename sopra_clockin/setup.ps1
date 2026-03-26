# ===================================================================
# Setup Helper Script for SopraGP4U Clock In/Out Automation
# ===================================================================
# Run this script as Administrator in PowerShell for initial setup
#
# Usage: .\setup.ps1
# ===================================================================

param(
    [switch]$SetCredentials = $false,
    [switch]$InstallDependencies = $false,
    [switch]$CreateScheduledTasks = $false,
    [switch]$All = $false
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogsDir = Join-Path $ProjectDir "logs"

Write-Host "=========================================" -ForegroundColor Green
Write-Host "SopraGP4U Automation - Setup Helper" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

# Check if running as administrator
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "WARNING: This script should be run as Administrator for full functionality" -ForegroundColor Yellow
    Write-Host ""
}

# Install dependencies
if ($InstallDependencies -or $All) {
    Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
    
    try {
        python -m pip install -r (Join-Path $ProjectDir "requirements.txt")
        Write-Host "[OK] Dependencies installed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Failed to install dependencies: $_" -ForegroundColor Red
        exit 1
    }
}

# Set credentials
if ($SetCredentials -or $All) {
    Write-Host ""
    Write-Host "Setting up environment variables..." -ForegroundColor Cyan
    
    Write-Host "Enter your SopraGP4U username:" -NoNewline -ForegroundColor Yellow
    $username = Read-Host " "
    
    Write-Host "Enter your SopraGP4U password:" -NoNewline -ForegroundColor Yellow
    $password = Read-Host -AsSecureString
    $plainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToCoTaskMemUnicode($password))
    
    try {
        [Environment]::SetEnvironmentVariable("SOPRA_USERNAME", $username, "User")
        [Environment]::SetEnvironmentVariable("SOPRA_PASSWORD", $plainPassword, "User")
        Write-Host "[OK] Credentials set as environment variables" -ForegroundColor Green
        Write-Host "Note: Variables will be available in new PowerShell/CMD sessions" -ForegroundColor Gray
    }
    catch {
        Write-Host "[ERROR] Failed to set environment variables: $_" -ForegroundColor Red
    }
}

# Create scheduled tasks
if ($CreateScheduledTasks -or $All) {
    Write-Host ""
    Write-Host "Creating scheduled tasks..." -ForegroundColor Cyan
    
    $action = New-ScheduledTaskAction -Execute (Join-Path $ProjectDir "scheduled_clockin.bat")
    
    # CLOCK IN task (8:00 AM)
    Write-Host "Creating 'SopraGP4U Clock In' task at 08:00..." -ForegroundColor Gray
    $trigger = New-ScheduledTaskTrigger -Daily -At "08:00"
    
    try {
        Register-ScheduledTask -TaskName "SopraGP4U Clock In" `
            -Action $action `
            -Trigger $trigger `
            -RunLevel Highest `
            -Force | Out-Null
        Write-Host "[OK] Clock In task created" -ForegroundColor Green
    }
    catch {
        Write-Host "[WARNING] Clock In task creation failed (may already exist): $_" -ForegroundColor Yellow
    }
    
    # CLOCK OUT task (5:15 PM)
    Write-Host "Creating 'SopraGP4U Clock Out' task at 17:15..." -ForegroundColor Gray
    $trigger = New-ScheduledTaskTrigger -Daily -At "17:15"
    
    try {
        Register-ScheduledTask -TaskName "SopraGP4U Clock Out" `
            -Action $action `
            -Trigger $trigger `
            -RunLevel Highest `
            -Force | Out-Null
        Write-Host "[OK] Clock Out task created" -ForegroundColor Green
    }
    catch {
        Write-Host "[WARNING] Clock Out task creation failed (may already exist): $_" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Test the script: python src\sopra_clockin.py" -ForegroundColor Gray
Write-Host "2. Check logs in: $LogsDir" -ForegroundColor Gray
Write-Host "3. View Task Scheduler tasks" -ForegroundColor Gray
Write-Host ""
