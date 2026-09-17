[CmdletBinding()]
param(
    [string]$SourceDir = ".appendix\source-smf",
    [string]$ReferenceDir = ".appendix\reference-profile-v1",
    [string]$OutputDir = ".appendix\control-axis-anchors-v1",
    [string]$Registry = "configs\evaluator-registry-v1.json",
    [string[]]$BackupAxes = @()
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

$arguments = @(
    '-m', 'llm_musical_composer.control_axis_anchors',
    '--source-dir', (Join-Path $workspaceRoot $SourceDir),
    '--reference-dir', (Join-Path $workspaceRoot $ReferenceDir),
    '--output-dir', (Join-Path $workspaceRoot $OutputDir),
    '--registry', (Join-Path $workspaceRoot $Registry),
    '--project-root', $workspaceRoot
)
if ($BackupAxes.Count -gt 0) {
    $arguments += '--backup-axes'
    $arguments += $BackupAxes
}

& $pythonPath @arguments

if ($LASTEXITCODE -ne 0) {
    throw "control-axis anchor build stopped with exit code $LASTEXITCODE"
}
