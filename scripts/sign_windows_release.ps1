# Tag-only step of the windows-sign job, the only job bound to the
# windows-release-signing Environment and its eSigner secrets. It extracts the
# build's unsigned ZIP, signs Ouroboros.exe with SSL.com CodeSignTool 1.3.2
# (pinned by exact archive digest), verifies signer and timestamp and repacks
# the release ZIP. Credentials never enter checked-in files or the log.
param([Parameter(Mandatory = $true)][string]$UnsignedArchive)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false  # exit codes are judged below
$Required = @('ESIGNER_USERNAME', 'ESIGNER_PASSWORD', 'ESIGNER_CREDENTIAL_ID', 'ESIGNER_TOTP_SECRET', 'ESIGNER_CERT_SHA1')
$Missing = @($Required | Where-Object { -not [Environment]::GetEnvironmentVariable($_) })
if ($Missing.Count) { throw "Missing Windows signing configuration: $($Missing -join ', ')" }
# A set but malformed thumbprint is a broken configuration, never "unsigned".
if ($env:ESIGNER_CERT_SHA1 -notmatch '^[a-fA-F0-9]{40}\z') { throw 'ESIGNER_CERT_SHA1 is not a 40-hex SHA-1 certificate thumbprint' }
if ($env:GITHUB_EVENT_NAME -ne 'push') { throw 'Windows release signing requires a pushed release tag' }
$Version = (Get-Content VERSION).Trim()
if ($env:GITHUB_REF -ne "refs/tags/v$Version") { throw 'Windows release tag must match VERSION exactly' }
if (-not $env:RUNNER_TEMP) { throw 'Windows release signing requires a temporary runner directory' }
# CodeSignTool's bundled JDK 11.0.2 (2019) does not trust the root behind
# cs.ssl.com's current TLS chain; the runner's maintained Java 11 does.
$Java = if ($env:JAVA_HOME_11_X64) { Join-Path $env:JAVA_HOME_11_X64 'bin\java.exe' }
if (-not $Java -or -not (Test-Path -LiteralPath $Java -PathType Leaf)) {
    throw 'The runner has no maintained Java 11 (JAVA_HOME_11_X64) to run CodeSignTool'
}
$Archive = "dist\Ouroboros-$Version-windows-x64.zip"
if (Test-Path $Archive) { throw 'Windows release archive unexpectedly exists before signing' }
$Work = Join-Path $env:RUNNER_TEMP ('es-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$Download = Join-Path $Work 'CodeSignTool-v1.3.2-windows.zip'
$Tool = Join-Path $Work 'tool'
$Payload = Join-Path $Work 'p'
$Signed = Join-Path $Work 'signed'
try {
    New-Item -ItemType Directory -Force -Path $Tool, $Signed, 'dist' | Out-Null
    Expand-Archive -LiteralPath $UnsignedArchive -DestinationPath $Payload
    $Executable = Join-Path $Payload 'Ouroboros\Ouroboros.exe'
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw 'Unsigned Windows archive has no Ouroboros\Ouroboros.exe' }
    # SSL.com's own versioned GitHub release asset, rather than the download
    # page's "current version" link. The digest below pins the accepted bytes.
    Invoke-WebRequest -Uri 'https://github.com/SSLcom/CodeSignTool/releases/download/v1.3.2/CodeSignTool-v1.3.2-windows.zip' -OutFile $Download
    $ExpectedDigest = '4afc32e8b7f79bbe1de7e4e7049aaad4e0f754357613b9bbec0e3052f06fd36b'.ToUpperInvariant()
    $Digest = (Get-FileHash -LiteralPath $Download -Algorithm SHA256).Hash
    if ($Digest -ne $ExpectedDigest) { throw "CodeSignTool archive digest $Digest is not the pinned $ExpectedDigest" }
    Expand-Archive -LiteralPath $Download -DestinationPath $Tool
    # Fixed layout (jar\, conf\, the unused bundled JDK and AppleDouble __MACOSX\
    # junk); locating the JAR spares this script the folder names.
    $Jar = Get-ChildItem -LiteralPath $Tool -Recurse -File -Filter 'code_sign_tool-*.jar' |
        Where-Object { $_.FullName -notmatch '__MACOSX' } | Select-Object -First 1
    if (-not $Jar) { throw 'Pinned CodeSignTool archive has no code_sign_tool JAR' }
    # The vendor BAT expands %* through cmd.exe without quoting, so Java runs
    # the JAR directly. The vendor CLI takes credentials on argv and may echo
    # request details: its output is discarded, never printed or uploaded, and
    # the outcome is judged by exit code, signed file and signature alone.
    $env:CODE_SIGN_TOOL_PATH = $Jar.Directory.Parent.FullName
    Push-Location $env:CODE_SIGN_TOOL_PATH  # match the vendor launcher's configuration root
    try {
        & $Java -jar $Jar.FullName sign "-username=$env:ESIGNER_USERNAME" "-password=$env:ESIGNER_PASSWORD" `
            "-credential_id=$env:ESIGNER_CREDENTIAL_ID" "-totp_secret=$env:ESIGNER_TOTP_SECRET" `
            "-input_file_path=$Executable" "-output_dir_path=$Signed" *> $null
        $SignExit = $LASTEXITCODE
    } catch {
        # Never render this exception: it may carry vendor output.
        throw 'CodeSignTool could not be started with the runner Java 11'
    } finally { Pop-Location }
    $SignedExecutable = Join-Path $Signed 'Ouroboros.exe'
    if ($SignExit -ne 0) {
        throw ("CodeSignTool sign exited $SignExit. Its output is withheld, so the cause is not visible here: " +
            'check the eSigner credentials and TOTP seed, the signing balance and SSL.com service state')
    }
    if (-not (Test-Path -LiteralPath $SignedExecutable -PathType Leaf)) {
        throw 'CodeSignTool exited 0 but wrote no signed Ouroboros.exe; its output is withheld and the cause is unknown'
    }
    & "$PSScriptRoot\verify_windows_signature.ps1" -Executable $SignedExecutable -ExpectedThumbprint $env:ESIGNER_CERT_SHA1 | Out-Null
    Copy-Item -LiteralPath $SignedExecutable -Destination $Executable -Force
    & "$PSScriptRoot\verify_windows_signature.ps1" -Executable $Executable -ExpectedThumbprint $env:ESIGNER_CERT_SHA1 | Out-Null
    & "$PSScriptRoot\pack_windows_archive.ps1" -PayloadRoot (Join-Path $Payload 'Ouroboros') -Archive $Archive
} finally {
    Remove-Item -LiteralPath $Work -Recurse -Force -ErrorAction SilentlyContinue
}
