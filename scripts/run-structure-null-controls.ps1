[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "structure-null-controls-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$recordsPath = Join-Path $workspaceRoot '.appendix\reference-structure-analysis\files.jsonl'
$outputRoot = Join-Path $workspaceRoot '.appendix\structure-null-controls'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

& $venvPython -m llm_musical_composer.structure_null_controls `
    --records $recordsPath `
    --output-dir $outputRoot *> $logPath
$analysisExitCode = $LASTEXITCODE
Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($analysisExitCode -ne 0) {
    throw "Structure null controls failed with exit code $analysisExitCode"
}
