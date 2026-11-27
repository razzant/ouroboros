# Tag-only step of the windows-sign job, the only job bound to the
# windows-release-signing Environment and its eSigner secrets. It extracts the
# build's unsigned ZIP, signs exactly the executables windows_signing_catalog.json
# names with SSL.com CodeSignTool 1.3.2 (pinned by exact archive digest),
# verifies each signer and timestamp, builds BOTH the release ZIP and the
# per-user Setup installer from that exact same signed payload tree, signs and
# verifies the outer Setup .exe, and reports every file's hash before and after
# signing as the step outputs signed_files and signed_installer.
# Credentials never enter checked-in files or the log.
param(
    [Parameter(Mandatory = $true)][string]$UnsignedArchive,
    # Optional path to a pre-staged pinned ISCC.exe; by default staged via download_inno_setup.ps1 before signing.
    [string]$InnoCompiler = ''
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false  # exit codes are judged below
$Required = @('ESIGNER_USERNAME', 'ESIGNER_PASSWORD', 'ESIGNER_CREDENTIAL_ID', 'ESIGNER_TOTP_SECRET', 'ESIGNER_CERT_SHA1')
$Missing = @($Required | Where-Object { -not [Environment]::GetEnvironmentVariable($_) })
if ($Missing.Count) { throw "Missing Windows signing configuration: $($Missing -join ', ')" }
$SignerUsername = $env:ESIGNER_USERNAME
$SignerPassword = $env:ESIGNER_PASSWORD
$SignerCredentialId = $env:ESIGNER_CREDENTIAL_ID
$SignerTotpSecret = $env:ESIGNER_TOTP_SECRET
$SignerCertSha1 = $env:ESIGNER_CERT_SHA1
# Keep credentials in script-local variables for CodeSignTool alone so no child
# process (Inno installer, ISCC.exe, packers, signtool, Java env) inherits ESIGNER_*.
foreach ($SecretName in @(Get-ChildItem Env:ESIGNER_* -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })) {
    Remove-Item -LiteralPath ('Env:' + $SecretName) -ErrorAction Stop
}
# A set but malformed thumbprint is a broken configuration, never "unsigned".
if ($SignerCertSha1 -notmatch '^[a-fA-F0-9]{40}\z') { throw 'ESIGNER_CERT_SHA1 is not a 40-hex SHA-1 certificate thumbprint' }
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
$ArchiveName = "Ouroboros-$Version-windows-x64.zip"
$InstallerName = "Ouroboros-Setup-$Version-windows-x64.exe"
$Archive = Join-Path 'dist' $ArchiveName
$Installer = Join-Path 'dist' $InstallerName
if (Test-Path -LiteralPath $Archive) { throw 'Windows release archive unexpectedly exists before signing' }
if (Test-Path -LiteralPath $Installer) { throw 'Windows release installer unexpectedly exists before signing' }
if ($InnoCompiler -and -not (Test-Path -LiteralPath $InnoCompiler -PathType Leaf)) {
    throw "Configured Inno Setup compiler not found: $InnoCompiler"
}
$Work = Join-Path $env:RUNNER_TEMP ('es-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
$Download = Join-Path $Work 'CodeSignTool-v1.3.2-windows.zip'
$Tool = Join-Path $Work 'tool'
$InnoDir = Join-Path $Work 'inno'
$Payload = Join-Path $Work 'p'
$Signed = Join-Path $Work 'signed'
$SetupStage = Join-Path $Work 'setup-unsigned'
$StagedArchive = Join-Path $Work $ArchiveName
$StagedInstaller = Join-Path $Work $InstallerName
try {
    New-Item -ItemType Directory -Force -Path $Tool, $Signed | Out-Null
    Expand-Archive -LiteralPath $UnsignedArchive -DestinationPath $Payload
    # Every eSigner request is billed, retried or not. Each catalogued file and
    # the installer payload shape must therefore be valid, unsigned and, where
    # the catalog pins upstream bytes, exactly those bytes before anything is
    # downloaded or signed. A file that already carries a signature, a vendor's
    # included, is refused rather than replaced.
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
    $PayloadApp = Join-Path $Payload 'Ouroboros'
    $ExpectedRootEntries = @('Ouroboros.exe', '_internal', 'bin')
    $ActualRootEntries = @(Get-ChildItem -LiteralPath $PayloadApp -Force | ForEach-Object { $_.Name })
    if (@($ActualRootEntries | Where-Object { $ExpectedRootEntries -notcontains $_ }).Count -or
        -not (Test-Path -LiteralPath (Join-Path $PayloadApp 'Ouroboros.exe') -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $PayloadApp '_internal') -PathType Container) -or
        -not (Test-Path -LiteralPath (Join-Path $PayloadApp 'bin') -PathType Container)) {
        throw "Unsigned Windows payload must hold exactly Ouroboros.exe, _internal\ and bin\ (found: $($ActualRootEntries -join ', '))"
    }
    $PayloadVersionFile = Join-Path $PayloadApp '_internal\VERSION'
    if (-not (Test-Path -LiteralPath $PayloadVersionFile -PathType Leaf)) {
        throw 'Unsigned Windows payload has no _internal\VERSION'
    }
    $PayloadVersion = (Get-Content -Raw -LiteralPath $PayloadVersionFile).Trim()
    if ($PayloadVersion -ne $Version) {
        throw "Unsigned Windows payload VERSION '$PayloadVersion' does not match checkout VERSION '$Version'"
    }
    # Stage every external tool (pinned CodeSignTool 1.3.2 and pinned Inno Setup
    # 7.1.1) before the first billed eSigner request, so a tool fetch or digest
    # failure spends zero signings.
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
    if (-not $InnoCompiler) {
        $InnoCompiler = & (Join-Path $PSScriptRoot 'download_inno_setup.ps1') -Destination $InnoDir
    }
    $TotalRequests = $Targets.Count + 1
    function Invoke-CodeSignRequest {
        param(
            [Parameter(Mandatory = $true)][string]$SourceFile,
            [Parameter(Mandatory = $true)][string]$DestinationFile,
            [Parameter(Mandatory = $true)][string]$Label,
            [Parameter(Mandatory = $true)][int]$Index
        )
        # One request per file, never retried: a failure names how many
        # requests this run has made, because earlier ones stay billed.
        $Request = "signing request $Index of $TotalRequests ($Label)"
        $Output = Join-Path $Signed "$Index"
        New-Item -ItemType Directory -Path $Output | Out-Null
        # The vendor BAT expands %* through cmd.exe without quoting, so Java runs
        # the JAR directly. The vendor CLI takes credentials on argv and may echo
        # request details: its output is discarded, never printed or uploaded, and
        # the outcome is judged by exit code, signed file and signature alone.
        Push-Location $env:CODE_SIGN_TOOL_PATH  # match the vendor launcher's configuration root
        try {
            & $Java -jar $Jar.FullName sign "-username=$SignerUsername" "-password=$SignerPassword" `
                "-credential_id=$SignerCredentialId" "-totp_secret=$SignerTotpSecret" `
                "-input_file_path=$SourceFile" "-output_dir_path=$Output" *> $null
            $SignExit = $LASTEXITCODE
        } catch {
            # Never render this exception: it may carry vendor output.
            throw "CodeSignTool could not be started with the runner Java 11 at $Request"
        } finally { Pop-Location }
        $SignedFile = Join-Path $Output (Split-Path -Leaf $SourceFile)
        if ($SignExit -ne 0) {
            throw ("CodeSignTool sign exited $SignExit at $Request. Its output is withheld, so the cause is not visible here: " +
                'check the eSigner credentials and TOTP seed, the signing balance and SSL.com service state')
        }
        if (-not (Test-Path -LiteralPath $SignedFile -PathType Leaf)) {
            throw "CodeSignTool exited 0 but wrote no signed file at $Request; its output is withheld and the cause is unknown"
        }
        $Returned = & "$PSScriptRoot\verify_windows_signature.ps1" -Executable $SignedFile -ExpectedThumbprint $SignerCertSha1
        Copy-Item -LiteralPath $SignedFile -Destination $DestinationFile -Force
        $Final = & "$PSScriptRoot\verify_windows_signature.ps1" -Executable $DestinationFile -ExpectedThumbprint $SignerCertSha1
        if ($Final.Sha256 -ne $Returned.Sha256) { throw "Copied $Label differs from the verified signed file" }
        $Final
    }
    $Records = @()
    foreach ($Target in $Targets) {
        $Final = Invoke-CodeSignRequest -SourceFile $Target.File -DestinationFile $Target.File `
            -Label $Target.Path -Index ($Records.Count + 1)
        $Records += [pscustomobject][ordered]@{
            path = $Target.Path; unsignedSha256 = $Target.UnsignedSha256; sha256 = $Final.Sha256.ToLowerInvariant()
        }
    }
    # Pack BOTH the portable ZIP and the per-user Setup installer from the exact
    # same signed Ouroboros\ payload tree before moving either artifact into dist\.
    & "$PSScriptRoot\pack_windows_archive.ps1" -PayloadRoot $PayloadApp -Archive $StagedArchive
    & (Join-Path $PSScriptRoot 'pack_windows_setup.ps1') -PayloadRoot $PayloadApp -OutputDir $SetupStage -Compiler $InnoCompiler
    $UnsignedSetup = Join-Path $SetupStage $InstallerName
    if (-not (Test-Path -LiteralPath $UnsignedSetup -PathType Leaf)) {
        throw "Windows installer packer did not produce $InstallerName"
    }
    $SetupStatus = (Get-AuthenticodeSignature -LiteralPath $UnsignedSetup).Status
    if ("$SetupStatus" -ne 'NotSigned') {
        throw "$InstallerName already has an Authenticode signature state ($SetupStatus); release signing never replaces one"
    }
    $UnsignedSetupSha256 = (Get-FileHash -LiteralPath $UnsignedSetup -Algorithm SHA256).Hash.ToLowerInvariant()
    $FinalSetup = Invoke-CodeSignRequest -SourceFile $UnsignedSetup -DestinationFile $StagedInstaller `
        -Label $InstallerName -Index $TotalRequests
    $SignedSetupSha256 = $FinalSetup.Sha256.ToLowerInvariant()
    if ($SignedSetupSha256 -eq $UnsignedSetupSha256) {
        throw "Signed $InstallerName hash unexpectedly equals its unsigned hash"
    }
    New-Item -ItemType Directory -Force -Path 'dist' | Out-Null
    Copy-Item -LiteralPath $StagedArchive -Destination $Archive -Force
    Copy-Item -LiteralPath $StagedInstaller -Destination $Installer -Force
    $PublishedSetup = & "$PSScriptRoot\verify_windows_signature.ps1" -Executable $Installer -ExpectedThumbprint $SignerCertSha1
    if ($PublishedSetup.Sha256.ToLowerInvariant() -ne $SignedSetupSha256) {
        throw "Published $InstallerName differs from the verified signed installer"
    }
    $InstallerRecord = [pscustomobject][ordered]@{
        name = $InstallerName; unsignedSha256 = $UnsignedSetupSha256; sha256 = $SignedSetupSha256
    }
    # windows-proof checks these hashes against the final ZIP and Setup installer it re-verifies.
    Add-Content -LiteralPath $env:GITHUB_OUTPUT -Value (
        ('signed_files=' + (ConvertTo-Json -InputObject $Records -Compress)),
        ('signed_installer=' + (ConvertTo-Json -InputObject $InstallerRecord -Compress))
    )
} catch {
    Remove-Item -LiteralPath $Archive, $Installer -Force -ErrorAction SilentlyContinue
    throw
} finally {
    $SignerUsername = $SignerPassword = $SignerCredentialId = $SignerTotpSecret = $null
    Remove-Item -LiteralPath $Work -Recurse -Force -ErrorAction SilentlyContinue
}
