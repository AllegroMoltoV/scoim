param(
    [string]$InputJsonl = ".appendix/reference-profile-v1/files.jsonl",
    [string]$ReferenceSummary = ".appendix/reference-profile-v1/summary.json",
    [string]$OutputDir = ".appendix/intensity-control-v1"
)

$ErrorActionPreference = "Stop"

& ".\.venv\Scripts\python.exe" -m llm_musical_composer.intensity_control `
    --input-jsonl $InputJsonl `
    --reference-summary $ReferenceSummary `
    --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
