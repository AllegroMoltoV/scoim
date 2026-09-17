$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDirectory = Join-Path $PSScriptRoot "..\.logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory "long-form-repair-$timestamp.log"

& (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe") -m llm_musical_composer.long_form_repair @args *> $logPath
$result = $LASTEXITCODE
Get-Content -LiteralPath $logPath -Encoding utf8
exit $result
