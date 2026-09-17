param(
    [string]$SplitRoot = '.appendix\score-timing-split-v1',
    [string]$OutputRoot = '.appendix\score-timing-development-v1'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$python = '.\.venv\Scripts\python.exe'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDirectory = New-Item -ItemType Directory -Path '.logs' -Force
$logPath = Join-Path $logDirectory.FullName "score-timing-development-v1-$timestamp.log"
$arguments = @(
    '-m', 'llm_musical_composer.score_timing_development_run',
    '--staging-dir', (Join-Path $SplitRoot 'development-smf'),
    '--staging-manifest', (Join-Path $SplitRoot 'manifest.json'),
    '--output-dir', $OutputRoot
)

& $python @arguments *> $logPath
$exitCode = $LASTEXITCODE
Get-Content -LiteralPath $logPath
exit $exitCode
