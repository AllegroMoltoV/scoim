[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logRoot = Join-Path $workspaceRoot '.logs'
$logPath = Join-Path $logRoot "package-test-$timestamp.log"
$workRoot = Join-Path ([System.IO.Path]::GetTempPath()) "scoim-package-test-$timestamp"
$sourceDistRoot = Join-Path $workRoot 'source-dist'
$extractedRoot = Join-Path $workRoot 'extracted'
$wheelRoot = Join-Path $workRoot 'wheel'
$wheelhouse = Join-Path $workRoot 'wheelhouse'
$versionPath = Join-Path $workRoot 'distribution-version.txt'
$cleanVenv = Join-Path $workRoot 'venv'
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$cleanPython = Join-Path $cleanVenv 'Scripts\python.exe'
$cleanScoim = Join-Path $cleanVenv 'Scripts\scoim.exe'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType Directory -Path $sourceDistRoot -Force | Out-Null
New-Item -ItemType Directory -Path $extractedRoot -Force | Out-Null
New-Item -ItemType Directory -Path $wheelRoot -Force | Out-Null
New-Item -ItemType Directory -Path $wheelhouse -Force | Out-Null
New-Item -ItemType File -Path $logPath -Force | Out-Null

function Invoke-Logged {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Executable @Arguments *>> $logPath
    if ($LASTEXITCODE -ne 0) {
        Write-Output (Get-Content -LiteralPath $logPath -Raw)
        throw "$Executable failed with exit code $LASTEXITCODE"
    }
}

Invoke-Logged $venvPython -m build --sdist --outdir $sourceDistRoot $workspaceRoot
$sdist = @(Get-ChildItem -LiteralPath $sourceDistRoot -Filter '*.tar.gz' -File -ErrorAction Stop)
if ($sdist.Count -ne 1) {
    throw "Expected one sdist, found $($sdist.Count)"
}
$sdistPath = $sdist[0].FullName
Invoke-Logged tar.exe --force-local -xzf $sdistPath -C $extractedRoot
$sourceRoot = @(Get-ChildItem -LiteralPath $extractedRoot -Directory -ErrorAction Stop)
if ($sourceRoot.Count -ne 1) {
    throw "Expected one extracted source directory, found $($sourceRoot.Count)"
}
$sourcePath = $sourceRoot[0].FullName
Invoke-Logged $venvPython -m build --wheel --outdir $wheelRoot $sourcePath
$wheel = @(Get-ChildItem -LiteralPath $wheelRoot -Filter '*.whl' -File -ErrorAction Stop)
if ($wheel.Count -ne 1) {
    throw "Expected one wheel, found $($wheel.Count)"
}
$wheelPath = $wheel[0].FullName
Invoke-Logged $venvPython (Join-Path $PSScriptRoot 'check-distributions.py') --sdist $sdistPath --wheel $wheelPath --version-file $versionPath
$distributionVersion = (Get-Content -LiteralPath $versionPath -Raw).Trim()
if (-not $distributionVersion) {
    throw 'The distribution metadata has no package version'
}

# Network access is isolated to dependency acquisition. Everything below installs
# and runs from the acquired wheelhouse only.
Invoke-Logged $venvPython -m pip download --disable-pip-version-check --dest $wheelhouse $wheelPath
Invoke-Logged py -3.13 -m venv $cleanVenv
Invoke-Logged $cleanPython -m pip install --disable-pip-version-check --no-index --find-links $wheelhouse "scoim==$distributionVersion"
Invoke-Logged $cleanPython -m pip check
Invoke-Logged $cleanScoim --help

$fixedFlowExample = Join-Path $sourcePath 'examples\fixed-flow'
$fixedV2Example = Join-Path $sourcePath 'examples\fixed-v2'
$legacyExample = Join-Path $sourcePath 'examples\fixed-aba'
Invoke-Logged $cleanPython (Join-Path $sourcePath 'scripts\smoke-installed-public.py') --flow (Join-Path $fixedFlowExample 'approved-flow.json') --responses (Join-Path $fixedFlowExample 'responses.json') --v2-flow (Join-Path $fixedV2Example 'approved-flow.json') --v2-responses (Join-Path $fixedV2Example 'responses.json') --legacy-script (Join-Path $legacyExample 'approved-script.json') --legacy-response (Join-Path $legacyExample 'frozen-response.json') --output (Join-Path $workRoot 'public-smoke') --forbid-root (Join-Path $workspaceRoot 'src')

Write-Output (Get-Content -LiteralPath $logPath -Raw)
Write-Output "Package test artifacts: $workRoot"
