[CmdletBinding()]
param(
    [string]$SourceDir = ".appendix\source-smf",
    [string]$OutputDir = ".appendix\reference-profile-v1"
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

& $pythonPath -m llm_musical_composer.reference_controls `
    --source-dir (Join-Path $workspaceRoot $SourceDir) `
    --output-dir (Join-Path $workspaceRoot $OutputDir) `
    --neighbor-count 7

if ($LASTEXITCODE -ne 0) {
    throw "reference profile build failed with exit code $LASTEXITCODE"
}
