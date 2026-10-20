# This file has been created (totally or partially) with the assistance of artificial intelligence tools.
# All content has been generated under the direct supervision of a named individual,
# and under the AI.Backbone Orchestrator Compliance framework
$ErrorActionPreference = 'Stop'

$packageDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDirectory = Join-Path (Split-Path $packageDirectory -Parent) 'MyBoda'

if (-not (Test-Path (Join-Path $sourceDirectory 'package.json'))) {
  throw "No se ha encontrado la aplicación fuente en $sourceDirectory"
}

$excluded = @('node_modules', 'data', 'dist', 'dist-server', '.env', '*.sqlite', '*.sqlite-shm', '*.sqlite-wal')
Get-ChildItem -LiteralPath $sourceDirectory -Force | Where-Object {
  $name = $_.Name
  -not ($excluded -contains $name -or $name -like '*.sqlite*')
} | ForEach-Object {
  Copy-Item -LiteralPath $_.FullName -Destination $packageDirectory -Recurse -Force
}

Write-Host 'MyBoda-Online preparado. Ejecuta npm.cmd install y npm.cmd run build para validar el paquete.'