param(
    [switch]$PrepareOnly,
    [string]$ArtifactRoot = ".appendix/texture-rhythm-content-two-stage-pilot-v1"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv/Scripts/python.exe"
$action = if ($PrepareOnly) { "prepare" } else { "run" }

& $python -m llm_musical_composer.texture_rhythm_content_pilot_run $action `
    --project-root $projectRoot `
    --artifact-root (Join-Path $projectRoot $ArtifactRoot)
exit $LASTEXITCODE
