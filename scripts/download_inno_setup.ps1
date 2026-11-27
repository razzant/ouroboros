# Stage the pinned Inno Setup compiler (packaging\windows\inno_setup_pin.json) for
# scripts\pack_windows_setup.ps1: the official installer is accepted only with
# exactly the pinned bytes, then installs in its own portable mode for the current
# user into -Destination (no administrator rights, Start menu entries, file
# associations or uninstall entry). Prints the path of ISCC.exe.
param([Parameter(Mandatory = $true)][string]$Destination)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false  # the installer's exit code is judged below
$Pin = Get-Content -Raw -LiteralPath (Join-Path (Split-Path -Parent $PSScriptRoot) 'packaging\windows\inno_setup_pin.json') |
    ConvertFrom-Json
$Target = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination).TrimEnd('\', '/')
$Iscc = Join-Path $Target 'ISCC.exe'
$Verify = Join-Path $PSScriptRoot 'verify_inno_setup.ps1'

if (Test-Path -LiteralPath $Iscc -PathType Leaf) {
    try {
        $null = & $Verify -Compiler $Iscc
    } catch {
        throw "$Target already holds an unverified Inno Setup compiler, not the pinned $($Pin.version): $($_.Exception.Message)"
    }
    Write-Host "Inno Setup $($Pin.version) is already staged in $Target"
    return $Iscc
}
if ((Test-Path -LiteralPath $Target) -and @(Get-ChildItem -LiteralPath $Target -Force).Count) {
    throw "Destination must be a new or empty folder, or hold the pinned compiler: $Target"
}
$TempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
$Work = Join-Path $TempRoot ('is-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
try {
    New-Item -ItemType Directory -Path $Work | Out-Null
    $Download = Join-Path $Work $Pin.installer
    Invoke-WebRequest -Uri $Pin.url -OutFile $Download
    $Size = (Get-Item -LiteralPath $Download).Length
    $Digest = (Get-FileHash -LiteralPath $Download -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Size -ne $Pin.sizeBytes -or $Digest -ne $Pin.sha256) {
        throw "$($Pin.installer) is $Size bytes with SHA-256 $Digest, not the pinned $($Pin.sizeBytes) bytes with $($Pin.sha256)"
    }
    $Log = Join-Path $Work 'install.log'
    $Process = Start-Process -FilePath $Download -Wait -PassThru -ArgumentList @(
        '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-', '/CURRENTUSER', '/PORTABLE=1',
        "/DIR=`"$Target`"", "/LOG=`"$Log`"")
    if ($Process.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $Log) { Get-Content -LiteralPath $Log | Write-Host }
        throw "The Inno Setup $($Pin.version) installer exited $($Process.ExitCode)"
    }
    try {
        $null = & $Verify -Compiler $Iscc
    } catch {
        if (Test-Path -LiteralPath $Log) { Get-Content -LiteralPath $Log | Write-Host }
        throw "The Inno Setup $($Pin.version) installer did not stage the pinned compiler in $($Target): $($_.Exception.Message)"
    }
} finally {
    Remove-Item -LiteralPath $Work -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Host "Staged Inno Setup $($Pin.version) in $Target"
return $Iscc
