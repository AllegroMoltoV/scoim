[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "analyze-reference-structure-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$inputRoot = Join-Path $workspaceRoot '.appendix\source-smf'
$outputRoot = Join-Path $workspaceRoot '.appendix\reference-structure-analysis'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

& $venvPython -m llm_musical_composer.reference_structure $inputRoot $outputRoot *> $logPath
$analysisExitCode = $LASTEXITCODE
Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($analysisExitCode -ne 0) {
    throw "Reference structure analysis failed with exit code $analysisExitCode"
}
