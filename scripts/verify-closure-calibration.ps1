[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "verify-closure-calibration-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$outputRoot = Join-Path $workspaceRoot '.appendix\closure-calibration'
$auditRoot = Join-Path $workspaceRoot '.appendix\corpus-audit'
$verificationScript = Join-Path $PSScriptRoot 'verify_closure_calibration.py'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

& $venvPython $verificationScript $outputRoot $auditRoot *> $logPath
$verificationExitCode = $LASTEXITCODE
Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($verificationExitCode -ne 0) {
    throw "Closure calibration verification failed with exit code $verificationExitCode"
}
