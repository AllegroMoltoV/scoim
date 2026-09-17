[CmdletBinding()]
param(
    [ValidateRange(2, 232)]
    [int]$Count = 16,

    [int]$Seed = 20260811
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "closure-calibration-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$sourceRoot = Join-Path $workspaceRoot '.appendix\source-smf'
$auditRoot = Join-Path $workspaceRoot '.appendix\corpus-audit'
$outputRoot = Join-Path $workspaceRoot '.appendix\closure-calibration'

if ($Count % 2 -ne 0) {
    throw 'Count must be even so that few-shot and holdout sets are balanced'
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

& $venvPython -m llm_musical_composer.closure_calibration `
    $sourceRoot `
    $auditRoot `
    $outputRoot `
    --count $Count `
    --seed $Seed *> $logPath
$buildExitCode = $LASTEXITCODE
Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($buildExitCode -ne 0) {
    throw "Closure calibration build failed with exit code $buildExitCode"
}
