# Verify an Inno Setup compiler path without executing it: official Inno Setup
# 7.1.1-x64 leaves 0.0.0.0 in PE VersionInfo for both ISCC.exe and ISCmplr.dll,
# so identity is checked by exact SHA-256 of the CLI and its sibling compiler
# DLL against packaging\windows\inno_setup_pin.json. Returns the resolved path
# to ISCC.exe on success; throws an actionable diagnostic on any mismatch.
param([Parameter(Mandatory = $true)][string]$Compiler)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
$Pin = Get-Content -Raw -LiteralPath (Join-Path $Repo 'packaging\windows\inno_setup_pin.json') | ConvertFrom-Json
$ExpectedIscc = if ($Pin.compilerSha256) { "$($Pin.compilerSha256.'ISCC.exe')".Trim().ToLowerInvariant() } else { '' }
$ExpectedDll = if ($Pin.compilerSha256) { "$($Pin.compilerSha256.'ISCmplr.dll')".Trim().ToLowerInvariant() } else { '' }
if ($ExpectedIscc -notmatch '^[0-9a-f]{64}\z' -or $ExpectedDll -notmatch '^[0-9a-f]{64}\z') {
    throw 'packaging\windows\inno_setup_pin.json is missing valid SHA-256 compilerSha256 digests for ISCC.exe and ISCmplr.dll'
}
if (-not (Test-Path -LiteralPath $Compiler -PathType Leaf)) {
    throw "Inno Setup compiler not found: $Compiler"
}
$Iscc = (Resolve-Path -LiteralPath $Compiler).ProviderPath
$ActualIscc = (Get-FileHash -LiteralPath $Iscc -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualIscc -ne $ExpectedIscc) {
    throw "$Iscc has SHA-256 $ActualIscc, not the pinned Inno Setup $($Pin.version) ISCC.exe ($ExpectedIscc)"
}
$Dll = Join-Path (Split-Path -Parent $Iscc) 'ISCmplr.dll'
if (-not (Test-Path -LiteralPath $Dll -PathType Leaf)) {
    throw "Pinned Inno Setup $($Pin.version) companion ISCmplr.dll is missing next to $Iscc ($Dll)"
}
$ActualDll = (Get-FileHash -LiteralPath $Dll -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualDll -ne $ExpectedDll) {
    throw "$Dll has SHA-256 $ActualDll, not the pinned Inno Setup $($Pin.version) ISCmplr.dll ($ExpectedDll)"
}
return $Iscc
