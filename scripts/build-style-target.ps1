[CmdletBinding()]
param(
    [string]$Anchor,
    [ValidateRange(0.0, 1.0)][Nullable[double]]$Density,
    [ValidateRange(0.0, 1.0)][Nullable[double]]$Polyphony,
    [ValidateRange(0.0, 1.0)][Nullable[double]]$Velocity,
    [ValidateRange(0.0, 1.0)][Nullable[double]]$Register,
    [ValidateRange(3, 232)][int]$NeighborCount = 7
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "build-style-target-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$recordsPath = Join-Path $workspaceRoot '.appendix\reference-structure-analysis\files.jsonl'
$outputRoot = Join-Path $workspaceRoot '.appendix\style-target'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

$arguments = @(
    '-m',
    'llm_musical_composer.style_target',
    '--records', $recordsPath,
    '--output-dir', $outputRoot,
    '--neighbor-count', $NeighborCount
)
if ($Anchor) {
    $arguments += @('--anchor', $Anchor)
}
if ($null -ne $Density) {
    $arguments += @('--density', $Density)
}
if ($null -ne $Polyphony) {
    $arguments += @('--polyphony', $Polyphony)
}
if ($null -ne $Velocity) {
    $arguments += @('--velocity', $Velocity)
}
if ($null -ne $Register) {
    $arguments += @('--register', $Register)
}

& $venvPython @arguments *> $logPath
$targetExitCode = $LASTEXITCODE
Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($targetExitCode -ne 0) {
    throw "Style target build failed with exit code $targetExitCode"
}
