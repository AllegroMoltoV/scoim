param(
    [switch]$ApplyListeningResponse
)

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDirectory = Join-Path $PSScriptRoot "../.logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory "brightness-control-$timestamp.log"
$python = Join-Path $PSScriptRoot "../.venv/Scripts/python.exe"
$arguments = @("-m", "llm_musical_composer.brightness_control_run")
if ($ApplyListeningResponse) {
    $arguments += "--apply-listening-response"
}

& $python @arguments *> $logPath
$result = $LASTEXITCODE
Get-Content -LiteralPath $logPath -Encoding utf8
exit $result
