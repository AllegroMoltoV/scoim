[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SourceRoot
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$destinationRoot = Join-Path $workspaceRoot '.appendix\source-smf'
$manifestPath = Join-Path $workspaceRoot '.appendix\source-smf-copy-manifest.json'
$logRoot = Join-Path $workspaceRoot '.logs'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logPath = Join-Path $logRoot "copy-source-smf-$timestamp.log"

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
Start-Transcript -LiteralPath $logPath | Out-Null

try {
    if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
        throw "Source directory is not readable: $sourceRoot"
    }

    $sourceEntries = @(Get-ChildItem -LiteralPath $sourceRoot -Force)
    $sourceDirectories = @($sourceEntries | Where-Object { $_.PSIsContainer })
    if ($sourceDirectories.Count -ne 0) {
        throw "Source contains directories. Expected none, found $($sourceDirectories.Count)."
    }

    $sourceFiles = @($sourceEntries | Where-Object { -not $_.PSIsContainer })
    $unsupported = @(
        $sourceFiles | Where-Object {
            $_.Extension.ToLowerInvariant() -notin @('.mid', '.musicloop')
        }
    )
    if ($unsupported.Count -ne 0) {
        $names = ($unsupported.Name | Sort-Object) -join ', '
        throw "Source contains unsupported files: $names"
    }

    New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null

    $records = [System.Collections.Generic.List[object]]::new()
    foreach ($sourceFile in ($sourceFiles | Sort-Object Name)) {
        $destinationPath = Join-Path $destinationRoot $sourceFile.Name
        $sourceHash = (Get-FileHash -LiteralPath $sourceFile.FullName -Algorithm SHA256).Hash
        $copyStatus = 'copied'

        if (Test-Path -LiteralPath $destinationPath -PathType Leaf) {
            $destinationInfo = Get-Item -LiteralPath $destinationPath -Force
            $destinationHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash
            if (
                $destinationInfo.Length -ne $sourceFile.Length -or
                $destinationHash -ne $sourceHash
            ) {
                throw "Existing destination differs from source: $($sourceFile.Name)"
            }
            $copyStatus = 'already_matched'
        }
        else {
            Copy-Item -LiteralPath $sourceFile.FullName -Destination $destinationPath
        }

        $copiedInfo = Get-Item -LiteralPath $destinationPath -Force
        $copiedHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash
        if ($copiedInfo.Length -ne $sourceFile.Length -or $copiedHash -ne $sourceHash) {
            throw "Copied file verification failed: $($sourceFile.Name)"
        }

        $records.Add([pscustomobject]@{
            name = $sourceFile.Name
            extension = $sourceFile.Extension.ToLowerInvariant()
            bytes = [int64]$sourceFile.Length
            sha256 = $sourceHash
            copy_status = $copyStatus
        })
    }

    $destinationFiles = @(Get-ChildItem -LiteralPath $destinationRoot -File -Force)
    $sourceNames = @($sourceFiles.Name | Sort-Object)
    $destinationNames = @($destinationFiles.Name | Sort-Object)
    if (Compare-Object -ReferenceObject $sourceNames -DifferenceObject $destinationNames) {
        throw 'Source and destination file-name sets differ.'
    }

    $summary = [ordered]@{
        schema_version = 1
        copied_at = (Get-Date).ToString('o')
        source = $sourceRoot
        destination = $destinationRoot
        file_count = $records.Count
        total_bytes = [int64](($records | Measure-Object -Property bytes -Sum).Sum)
        extension_counts = [ordered]@{
            mid = @($records | Where-Object { $_.extension -eq '.mid' }).Count
            musicloop = @($records | Where-Object { $_.extension -eq '.musicloop' }).Count
        }
        files = $records
    }

    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
    Write-Output ($summary | ConvertTo-Json -Depth 3)
}
finally {
    Stop-Transcript | Out-Null
}
