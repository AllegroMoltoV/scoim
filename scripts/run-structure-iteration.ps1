[CmdletBinding()]
param(
    [string]$RunId
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "structure-iteration-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

Push-Location $workspaceRoot
try {
    if ($RunId) {
        & $venvPython -m llm_musical_composer.structure_iteration --run-id $RunId *> $logPath
    } else {
        & $venvPython -m llm_musical_composer.structure_iteration *> $logPath
    }
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($exitCode -ne 0) {
    throw "structure iteration failed with exit code $exitCode"
}
