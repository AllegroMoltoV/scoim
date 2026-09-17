param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [string]$ManifestPath = '.appendix\reference-structure-analysis\manifest.json',
    [string]$OutputRoot = '.appendix\reference-corpus-v1\source'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 1) {
    throw "Unsupported manifest schema_version: $($manifest.schema_version)"
}
if ($manifest.source_files.Count -ne 232) {
    throw "Expected 232 source files, got $($manifest.source_files.Count)"
}

$destination = New-Item -ItemType Directory -Path $OutputRoot -Force
$copied = 0
$reused = 0
foreach ($record in $manifest.source_files) {
    $name = [string]$record.name
    if ([System.IO.Path]::GetFileName($name) -ne $name) {
        throw "Manifest source name is not a file name: $name"
    }
    $source = [System.IO.Path]::Combine($SourceRoot, $name)
    $target = [System.IO.Path]::Combine($destination.FullName, $name)
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Source SMF is missing: $name"
    }
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    $expectedHash = ([string]$record.sha256).ToLowerInvariant()
    if ($sourceHash -ne $expectedHash) {
        throw "Source SMF SHA-256 mismatch: $name"
    }
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($targetHash -ne $expectedHash) {
            throw "Existing local SMF SHA-256 mismatch: $name"
        }
        $reused += 1
        continue
    }
    Copy-Item -LiteralPath $source -Destination $target
    $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($targetHash -ne $expectedHash) {
        throw "Copied SMF SHA-256 mismatch: $name"
    }
    $copied += 1
}

[pscustomobject]@{
    status = 'pass'
    source_count = $manifest.source_files.Count
    copied = $copied
    reused = $reused
    output_root = $destination.FullName
} | ConvertTo-Json -Depth 3
