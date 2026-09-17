[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "setup-$timestamp.log"
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    py -3.13 -m venv (Join-Path $workspaceRoot '.venv') *> $logPath
    $venvExitCode = $LASTEXITCODE
    if ($venvExitCode -ne 0) {
        Write-Output (Get-Content -LiteralPath $logPath -Raw)
        throw "venv creation failed with exit code $venvExitCode"
    }
}

& $venvPython -m pip install --disable-pip-version-check --editable "${workspaceRoot}[dev]" *>> $logPath
$installExitCode = $LASTEXITCODE
if ($installExitCode -ne 0) {
    Write-Output (Get-Content -LiteralPath $logPath -Raw)
    throw "pip install failed with exit code $installExitCode"
}

& $venvPython -m pip check *>> $logPath
$checkExitCode = $LASTEXITCODE
Write-Output (Get-Content -LiteralPath $logPath -Raw)
if ($checkExitCode -ne 0) {
    throw "pip check failed with exit code $checkExitCode"
}
