# Tag-only step of the windows-sign job, the only job bound to the
# windows-release-signing Environment and its eSigner secrets. It extracts the
# build's unsigned ZIP, signs exactly the executables windows_signing_catalog.json
# names with SSL.com CodeSignTool 1.3.2 (pinned by exact archive digest),
# verifies each signer and timestamp, repacks the release ZIP and reports every
# file's hash before and after signing as the step output signed_files.
# Credentials never enter checked-in files or the log.
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
if (-not $env:GITHUB_OUTPUT) { throw 'Windows release signing requires a step output file for its signing record' }
# CodeSignTool's bundled JDK 11.0.2 (2019) does not trust the root behind
# cs.ssl.com's current TLS chain; the runner's maintained Java 11 does.
$Java = if ($env:JAVA_HOME_11_X64) { Join-Path $env:JAVA_HOME_11_X64 'bin\java.exe' }
if (-not $Java -or -not (Test-Path -LiteralPath $Java -PathType Leaf)) {
    throw 'The runner has no maintained Java 11 (JAVA_HOME_11_X64) to run CodeSignTool'
}
$Catalog = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'windows_signing_catalog.json') | ConvertFrom-Json
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
    # Every eSigner request is billed, retried or not. Each catalogued file must
    # therefore be present, unsigned and, where the catalog pins upstream
    # bytes, exactly those bytes before anything is downloaded or signed. A
    # file that already carries a signature, a vendor's included, is refused
    # rather than replaced.
    $Targets = @(foreach ($Entry in $Catalog.files) {
        $File = Join-Path $Payload ($Entry.path -replace '/', [IO.Path]::DirectorySeparatorChar)
        if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Unsigned Windows archive has no $($Entry.path)" }
        $Status = (Get-AuthenticodeSignature -LiteralPath $File).Status
        if ("$Status" -ne 'NotSigned') {
            throw "$($Entry.path) already has an Authenticode signature state ($Status); release signing never replaces one"
        }
        $Hash = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Entry.upstreamSha256 -and $Hash -ne $Entry.upstreamSha256) {
            throw "$($Entry.path) has SHA-256 $Hash, not the catalogued upstream $($Entry.upstreamSha256)"
        }
        [pscustomobject]@{ Path = $Entry.path; File = $File; UnsignedSha256 = $Hash }
    })
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
    $env:CODE_SIGN_TOOL_PATH = $Jar.Directory.Parent.FullName
    $Records = @()
    foreach ($Target in $Targets) {
        # One request per file, never retried: a failure names how many
        # requests this run has made, because earlier ones stay billed.
        $Request = "signing request $($Records.Count + 1) of $($Targets.Count) ($($Target.Path))"
        $Output = Join-Path $Signed "$($Records.Count)"
        New-Item -ItemType Directory -Path $Output | Out-Null
        # The vendor BAT expands %* through cmd.exe without quoting, so Java runs
        # the JAR directly. The vendor CLI takes credentials on argv and may echo
        # request details: its output is discarded, never printed or uploaded, and
        # the outcome is judged by exit code, signed file and signature alone.
        Push-Location $env:CODE_SIGN_TOOL_PATH  # match the vendor launcher's configuration root
        try {
            & $Java -jar $Jar.FullName sign "-username=$env:ESIGNER_USERNAME" "-password=$env:ESIGNER_PASSWORD" `
                "-credential_id=$env:ESIGNER_CREDENTIAL_ID" "-totp_secret=$env:ESIGNER_TOTP_SECRET" `
                "-input_file_path=$($Target.File)" "-output_dir_path=$Output" *> $null
            $SignExit = $LASTEXITCODE
        } catch {
            # Never render this exception: it may carry vendor output.
            throw "CodeSignTool could not be started with the runner Java 11 at $Request"
        } finally { Pop-Location }
        $SignedFile = Join-Path $Output (Split-Path -Leaf $Target.File)
        if ($SignExit -ne 0) {
            throw ("CodeSignTool sign exited $SignExit at $Request. Its output is withheld, so the cause is not visible here: " +
                'check the eSigner credentials and TOTP seed, the signing balance and SSL.com service state')
        }
        if (-not (Test-Path -LiteralPath $SignedFile -PathType Leaf)) {
            throw "CodeSignTool exited 0 but wrote no signed file at $Request; its output is withheld and the cause is unknown"
        }
        $Returned = & "$PSScriptRoot\verify_windows_signature.ps1" -Executable $SignedFile -ExpectedThumbprint $env:ESIGNER_CERT_SHA1
        Copy-Item -LiteralPath $SignedFile -Destination $Target.File -Force
        $Final = & "$PSScriptRoot\verify_windows_signature.ps1" -Executable $Target.File -ExpectedThumbprint $env:ESIGNER_CERT_SHA1
        if ($Final.Sha256 -ne $Returned.Sha256) { throw "Copied $($Target.Path) differs from the verified signed file" }
        $Records += [pscustomobject][ordered]@{
            path = $Target.Path; unsignedSha256 = $Target.UnsignedSha256; sha256 = $Final.Sha256.ToLowerInvariant()
        }
    }
    & "$PSScriptRoot\pack_windows_archive.ps1" -PayloadRoot (Join-Path $Payload 'Ouroboros') -Archive $Archive
    # windows-proof checks these hashes against the final ZIP it re-verifies.
    Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value ('signed_files=' + (ConvertTo-Json -InputObject $Records -Compress))
} finally {
    Remove-Item -LiteralPath $Work -Recurse -Force -ErrorAction SilentlyContinue
}
