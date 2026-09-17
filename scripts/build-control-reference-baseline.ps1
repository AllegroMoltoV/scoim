[CmdletBinding()]
param(
    [string]$SourceDir,
    [string]$ReferenceManifest,
    [string]$OutputDir,
    [string[]]$Observation
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $repositoryRoot '.venv\Scripts\python.exe'

if (-not $SourceDir) {
    $SourceDir = Join-Path $repositoryRoot '.appendix\source-smf'
}
if (-not $ReferenceManifest) {
    $ReferenceManifest = Join-Path $repositoryRoot '.appendix\reference-profile-v1\manifest.json'
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $repositoryRoot '.appendix\control-reference-baseline-v3'
}
if (-not $Observation) {
    $Observation = @(
        (Join-Path $repositoryRoot '.appendix\multiscale-calibration-run-v8\outputs\final.mid')
    )
}

$implementationPaths = @(
    (Join-Path $repositoryRoot 'src\llm_musical_composer\control_reference_baseline.py'),
    (Join-Path $repositoryRoot 'src\llm_musical_composer\reference_profile.py'),
    (Join-Path $repositoryRoot 'src\llm_musical_composer\smf_notes.py'),
    (Join-Path $repositoryRoot 'src\llm_musical_composer\control_axis_anchors.py'),
    $PSCommandPath
)

$commandArguments = @(
    '-m',
    'llm_musical_composer.control_reference_baseline',
    '--source-dir',
    $SourceDir,
    '--reference-manifest',
    $ReferenceManifest,
    '--output-dir',
    $OutputDir
)
foreach ($path in $Observation) {
    $commandArguments += @('--observation', $path)
}
foreach ($path in $implementationPaths) {
    $commandArguments += @('--implementation', $path)
}

Push-Location $repositoryRoot
try {
    & $pythonExecutable @commandArguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
