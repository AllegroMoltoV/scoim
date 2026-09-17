[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "run-corpus-audit-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$inputRoot = Join-Path $workspaceRoot '.appendix\source-smf'
$outputRoot = Join-Path $workspaceRoot '.appendix\corpus-audit'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

& $venvPython -m llm_musical_composer.corpus_audit $inputRoot $outputRoot *> $logPath
$auditExitCode = $LASTEXITCODE
Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($auditExitCode -ne 0) {
    throw "Corpus audit failed with exit code $auditExitCode"
}
