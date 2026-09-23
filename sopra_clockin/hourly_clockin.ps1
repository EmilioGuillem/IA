# This file has been created (totally or partially) with the assistance of
# artificial intelligence tools. All content has been generated under the
# direct supervision of a named individual and the AI.Backbone Orchestrator
# Compliance framework.

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $ProjectDir "logs\hourly_runner.log"

function Write-RunnerLog([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

try {
    $Python = (Get-Command python.exe -ErrorAction Stop).Source
}
catch {
    Write-RunnerLog "FATAL: python.exe not found in PATH. Install Python or fix PATH, then re-run."
    Read-Host "Press Enter to close"
    exit 1
}

Set-Location $ProjectDir
$env:SOPRA_BROWSER = "edge"
$env:SOPRA_DRY_RUN = "false"
Write-RunnerLog "Hourly runner started for user $env:USERNAME (python: $Python)"

while ($true) {
    try {
        Write-RunnerLog "Starting hourly check"
        & $Python (Join-Path $ProjectDir "src\sopra_clockin.py") *>> $LogFile
        Write-RunnerLog "Hourly check finished with exit code $LASTEXITCODE"
    }
    catch {
        Write-RunnerLog "Hourly check failed: $($_.Exception.Message)"
    }

    Write-RunnerLog "Sleeping for 1 hour until next check"
    Start-Sleep -Seconds 3600
}