[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "closure-judge-protocol-revision-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

Push-Location $workspaceRoot
try {
    & $venvPython -m llm_musical_composer.closure_judge --protocol revision *> $logPath
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($exitCode -ne 0) {
    throw "closure judge protocol revision failed with exit code $exitCode. See $logPath"
}
