# The one Windows release packer: local builds, unsigned releases and the signing
# job all archive through here. Compress-Archive silently skips OS-hidden
# entries; ZipFile walks the whole payload, including Playwright's
# .local-browsers. The archive is then extracted once more and must hold the
# same tree (every directory, every file byte for byte), or nothing ships.
param(
    [Parameter(Mandatory = $true)][string]$PayloadRoot,
    [Parameter(Mandatory = $true)][string]$Archive
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem  # Windows PowerShell 5.1

function Get-TreeDigest([string]$Root, [string]$Prefix) {
    $Tree = @{}
    $Base = (Get-Item -LiteralPath $Root -Force).FullName.TrimEnd('\', '/')
    foreach ($Item in Get-ChildItem -LiteralPath $Root -Recurse -Force) {
        $Relative = $Prefix + $Item.FullName.Substring($Base.Length).TrimStart('\', '/').Replace('\', '/')
        $Tree[$Relative] = if ($Item.PSIsContainer) { 'directory' } else {
            (Get-FileHash -LiteralPath $Item.FullName -Algorithm SHA256).Hash
        }
    }
    $Tree
}

$Payload = (Resolve-Path -LiteralPath $PayloadRoot).Path.TrimEnd('\', '/')
$Archive = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Archive)
if (Test-Path -LiteralPath $Archive) { throw "Windows archive already exists: $Archive" }
# Short scratch root: the payload path guard leaves room for a user's
# extraction folder, not for a long temporary prefix.
$TempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$Audit = Join-Path $TempRoot ('ob-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
try {
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $Payload, $Archive, [System.IO.Compression.CompressionLevel]::Optimal, $true)
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Archive, $Audit)
    $Leaf = Split-Path -Leaf $Payload
    $Expected = Get-TreeDigest $Payload "$Leaf/"
    $Expected[$Leaf] = 'directory'
    $Actual = Get-TreeDigest $Audit ''
    $Differs = @(@($Expected.Keys) + @($Actual.Keys) | Sort-Object -Unique |
        Where-Object { $Expected[$_] -ne $Actual[$_] })
    if ($Differs.Count) {
        Remove-Item -LiteralPath $Archive -Force -ErrorAction SilentlyContinue
        throw "Windows archive differs from its payload at $($Differs.Count) path(s), first: $($Differs[0])"
    }
    Write-Host "Packed $($Expected.Count) payload entries into $Archive"
} finally {
    Remove-Item -LiteralPath $Audit -Recurse -Force -ErrorAction SilentlyContinue
}
