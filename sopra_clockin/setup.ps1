# This file has been created (totally or partially) with the assistance of
# artificial intelligence tools. All content has been generated under the
# direct supervision of a named individual and the AI.Backbone Orchestrator
# Compliance framework.

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
    Write-Host "Creating hourly logon task..." -ForegroundColor Cyan

    $taskName = "SopraGP4U Hourly Check"
    $runner = Join-Path $ProjectDir "hourly_clockin.ps1"
    $powershell = (Get-Command powershell.exe).Source
    $action = New-ScheduledTaskAction `
        -Execute $powershell `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`""
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $settings = New-ScheduledTaskSettingsSet `
        -MultipleInstances IgnoreNew `
        -StartWhenAvailable

    try {
        Register-ScheduledTask -TaskName $taskName `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Description "Comprueba SopraGP4U al iniciar sesión y cada hora." `
            -Force | Out-Null

        Unregister-ScheduledTask -TaskName "SopraGP4U Clock In" -Confirm:$false -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName "SopraGP4U Clock Out" -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "[OK] Hourly logon task created: $taskName" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Hourly task creation failed: $_" -ForegroundColor Red
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
