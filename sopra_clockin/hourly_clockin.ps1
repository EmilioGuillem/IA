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
    Write-RunnerLog "[KO] python.exe not found in PATH"
    exit 1
}

Set-Location $ProjectDir
$env:SOPRA_BROWSER = "edge"
$env:SOPRA_DRY_RUN = "false"
$env:SOPRA_QUIET_LOGS = "true"

while ($true) {
    try {
        Write-RunnerLog "[CHECK] Starting hourly verification"
        & $Python (Join-Path $ProjectDir "src\sopra_clockin.py")
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            Write-RunnerLog "[OK] Hourly verification completed"
        }
        else {
            Write-RunnerLog "[KO] Hourly verification failed (exit code $exitCode)"
        }
    }
    catch {
        Write-RunnerLog "[KO] Hourly verification failed: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds 3600
}