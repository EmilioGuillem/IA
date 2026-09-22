# This file has been created (totally or partially) with the assistance of
# artificial intelligence tools. All content has been generated under the
# direct supervision of a named individual and the AI.Backbone Orchestrator
# Compliance framework.

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $ProjectDir "logs\hourly_runner.log"
$Python = (Get-Command python.exe -ErrorAction Stop).Source

function Write-RunnerLog([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

Set-Location $ProjectDir
$env:SOPRA_DRY_RUN = "false"
Write-RunnerLog "Hourly runner started for user $env:USERNAME"

while ($true) {
    try {
        Write-RunnerLog "Starting hourly check"
        & $Python (Join-Path $ProjectDir "src\sopra_clockin.py") *>> $LogFile
        Write-RunnerLog "Hourly check finished with exit code $LASTEXITCODE"
    }
    catch {
        Write-RunnerLog "Hourly check failed: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds 3600
}