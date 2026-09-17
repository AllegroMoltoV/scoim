param(
    [string]$SourceRoot = '.appendix\reference-corpus-v1\source',
    [string]$OutputRoot = '.appendix\reference-decomposition-v1'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$python = '.\.venv\Scripts\python.exe'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDirectory = New-Item -ItemType Directory -Path '.logs' -Force
$logPath = Join-Path $logDirectory.FullName "reference-decomposition-v1-$timestamp.log"
$arguments = @(
    '-m', 'llm_musical_composer.reference_decomposition_run',
    '--source-dir', $SourceRoot,
    '--output-dir', $OutputRoot,
    '--reference-manifest', '.appendix\reference-profile-v1\manifest.json',
    '--reference-files', '.appendix\reference-profile-v1\files.jsonl',
    '--reference-summary', '.appendix\reference-profile-v1\summary.json',
    '--structure-manifest', '.appendix\reference-structure-analysis\manifest.json',
    '--structure-files', '.appendix\reference-structure-analysis\files.jsonl',
    '--structure-summary', '.appendix\reference-structure-analysis\summary.json',
    '--structure-controls', '.appendix\reference-structure-analysis\controls.json',
    '--audit-path', '.appendix\corpus-audit\files.jsonl',
    '--audit-path', '.appendix\corpus-audit\summary.json',
    '--audit-path', '.appendix\corpus-audit\duplicate-candidates.json',
    '--audit-path', '.appendix\corpus-audit\evaluation-groups.json',
    '--known-source-run', '.appendix\reference-variance-smoke-v7\runs\reference-a-candidate-2'
)

& $python @arguments *> $logPath
$exitCode = $LASTEXITCODE
Get-Content -LiteralPath $logPath
exit $exitCode
