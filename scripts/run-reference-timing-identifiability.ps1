param(
    [string]$OutputRoot = '.appendix\reference-timing-identifiability-v1'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$python = '.\.venv\Scripts\python.exe'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDirectory = New-Item -ItemType Directory -Path '.logs' -Force
$logPath = Join-Path $logDirectory.FullName "reference-timing-identifiability-$timestamp.log"

& $python -m llm_musical_composer.reference_timing_run --output-dir $OutputRoot *> $logPath
$exitCode = $LASTEXITCODE
Get-Content -LiteralPath $logPath
exit $exitCode
