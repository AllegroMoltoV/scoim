[CmdletBinding()]
param(
    [switch]$Coverage
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "test-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$systemTemp = [System.IO.Path]::GetTempPath()
$pytestBaseTemp = Join-Path $systemTemp "llm-musical-composer-pytest-$timestamp-$PID"
$scoimSource = Join-Path $workspaceRoot 'src\scoim'
$allTestPaths = @(Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'tests') -Filter 'test_*.py' -File)
$scoimTestPaths = @($allTestPaths | Where-Object Name -Like 'test_scoim*.py' | Select-Object -ExpandProperty FullName)
$legacyRuntimeTestNames = @(
    'test_activity_selection.py'
    'test_generic_pipeline_quality.py'
    'test_material_development.py'
    'test_music_dsl.py'
    'test_performance_pipeline.py'
    'test_piano_texture_pilot.py'
    'test_piano_texture_register_placement.py'
    'test_pilot_features.py'
    'test_pilot_loop.py'
    'test_pipeline_dsl.py'
    'test_recurrence_quality.py'
    'test_run_state.py'
    'test_section_contrast.py'
    'test_smf_notes.py'
    'test_smf_render.py'
    'test_structure_features.py'
    'test_sustain_profile.py'
)
$legacyRuntimeTestPaths = @(
    $allTestPaths |
        Where-Object Name -In $legacyRuntimeTestNames |
        Select-Object -ExpandProperty FullName
)
$productTestPaths = @($scoimTestPaths + $legacyRuntimeTestPaths)
$pytestArguments = @('-m', 'pytest') + $productTestPaths + @('--basetemp', $pytestBaseTemp)
if ($Coverage) {
    $pytestArguments += @('--cov=scoim', '--cov-branch', '--cov-report=term-missing')
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

if ($scoimTestPaths.Count -eq 0) {
    throw 'SCoIM tests were not found'
}

if ($legacyRuntimeTestPaths.Count -ne $legacyRuntimeTestNames.Count) {
    throw 'One or more current runtime dependency tests were not found'
}

& $venvPython @pytestArguments *> $logPath
$pytestExitCode = $LASTEXITCODE
if ($pytestExitCode -ne 0) {
    Write-Output (Get-Content -LiteralPath $logPath -Raw)
    throw "pytest failed with exit code $pytestExitCode"
}

& $venvPython -m ruff check . *>> $logPath
$ruffExitCode = $LASTEXITCODE
if ($ruffExitCode -ne 0) {
    Write-Output (Get-Content -LiteralPath $logPath -Raw)
    throw "ruff failed with exit code $ruffExitCode"
}

& $venvPython (Join-Path $PSScriptRoot 'check-markdown-links.py') *>> $logPath
$markdownExitCode = $LASTEXITCODE
if ($markdownExitCode -ne 0) {
    Write-Output (Get-Content -LiteralPath $logPath -Raw)
    throw "Markdown link check failed with exit code $markdownExitCode"
}

& $venvPython -m ruff format --check $scoimSource @scoimTestPaths *>> $logPath
$formatExitCode = $LASTEXITCODE
Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($formatExitCode -ne 0) {
    throw "ruff format check failed with exit code $formatExitCode"
}
