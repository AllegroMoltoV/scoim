[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "activity-selection-replay-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

Push-Location $workspaceRoot
try {
    & $venvPython -m llm_musical_composer.activity_selection *> $logPath
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($exitCode -ne 0) {
    throw "activity selection replay failed with exit code $exitCode"
}
