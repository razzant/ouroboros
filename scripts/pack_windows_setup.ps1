# The one Windows installer packer: it compiles packaging\windows\ouroboros.iss
# with the pinned Inno Setup over an existing collected payload, the Ouroboros
# folder the portable ZIP holds, so local builds, QA and the release jobs all
# install exactly the ZIP's files; a tree with Hidden entries, which Setup would
# leave out, is refused. Release signing, when configured, signs the result
# afterwards (scripts\sign_windows_release.ps1).
#
#   pwsh -NoProfile -File scripts/pack_windows_setup.ps1 -PayloadRoot <extracted ZIP>\Ouroboros -OutputDir <empty folder>
#
# writes <OutputDir>\Ouroboros-Setup-<version>-windows-x64.exe, where the version
# is the payload's own _internal\VERSION: the version of the files it installs.
# Only the compiler is a build dependency; nobody needs anything to run the result.
param(
    [Parameter(Mandatory = $true)][string]$PayloadRoot,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    # ISCC.exe of the pinned Inno Setup; by default PATH, then its usual install folders.
    [string]$Compiler = ''
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false  # the compiler's exit code is judged below
$Repo = Split-Path -Parent $PSScriptRoot
$Pin = Get-Content -Raw -LiteralPath (Join-Path $Repo 'packaging\windows\inno_setup_pin.json') | ConvertFrom-Json

$Payload = (Resolve-Path -LiteralPath $PayloadRoot).Path.TrimEnd('\', '/')
# The collected payload must hold exactly Ouroboros.exe, _internal\ and bin\, matching
# the portable ZIP layout and preventing stray build files from leaking into Setup.
$Expected = @('Ouroboros.exe', '_internal', 'bin')
$Names = @(Get-ChildItem -LiteralPath $Payload -Force | ForEach-Object { $_.Name })
if (@($Names | Where-Object { $Expected -notcontains $_ }).Count -or
    -not (Test-Path -LiteralPath (Join-Path $Payload 'Ouroboros.exe') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $Payload '_internal') -PathType Container) -or
    -not (Test-Path -LiteralPath (Join-Path $Payload 'bin') -PathType Container)) {
    throw "$Payload is not a collected Ouroboros payload of exactly Ouroboros.exe, _internal\ and bin\ (found: $($Names -join ', '))"
}
# ouroboros.iss's wildcard Source skips Hidden entries, which the ZIP keeps (pack_windows_archive.ps1): refuse
# such a tree instead of building a Setup that installs less than its ZIP. An extracted ZIP carries no attributes.
$Hidden = @(Get-ChildItem -LiteralPath $Payload -Recurse -Force |
    Where-Object { $_.Attributes -band [IO.FileAttributes]::Hidden } | Select-Object -First 3 | ForEach-Object { $_.FullName })
if ($Hidden.Count) {
    throw ("Setup would leave out Hidden entries that the ZIP includes, such as $($Hidden -join ', '). " +
           "Pass the Ouroboros folder of an extracted ZIP as -PayloadRoot, as the release does.")
}
$VersionFile = Join-Path $Payload '_internal\VERSION'
if (-not (Test-Path -LiteralPath $VersionFile -PathType Leaf)) { throw "The payload has no _internal\VERSION" }
$Version = (Get-Content -Raw -LiteralPath $VersionFile).Trim()
if (-not ($Version -match '^(\d+)\.(\d+)\.(\d+)(?:-?(?:rc|alpha|beta|a|b)\.?\d+)?$')) {
    throw "The payload's VERSION '$Version' is not a release version"
}
$NumericVersion = "$($Matches[1]).$($Matches[2]).$($Matches[3]).0"
$Checkout = (Get-Content -Raw -LiteralPath (Join-Path $Repo 'VERSION')).Trim()
if ($Checkout -ne $Version) { Write-Host "Packaging the payload's version $Version; this checkout's VERSION is $Checkout" }
$Name = "Ouroboros-Setup-$Version-windows-x64"

$Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir).TrimEnd('\', '/')
if ([String]::Equals($Output, $Payload, [StringComparison]::OrdinalIgnoreCase) -or
    $Output.StartsWith($Payload + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "OutputDir must lie outside the payload: $Output"
}
if (Test-Path -LiteralPath $Output) {
    if (-not (Test-Path -LiteralPath $Output -PathType Container) -or @(Get-ChildItem -LiteralPath $Output -Force).Count) {
        throw "OutputDir must be a new or empty folder: $Output"
    }
} else {
    New-Item -ItemType Directory -Path $Output | Out-Null
}

$Verify = Join-Path $PSScriptRoot 'verify_inno_setup.ps1'
if ($Compiler) {
    $Iscc = & $Verify -Compiler $Compiler
} else {
    $Candidates = @(Get-Command ISCC.exe -All -ErrorAction SilentlyContinue | ForEach-Object { $_.Source }) + @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 7\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe")
    $Seen = @{}
    $Rejected = @()
    $Iscc = $null
    foreach ($Candidate in $Candidates) {
        if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { continue }
        $Resolved = (Resolve-Path -LiteralPath $Candidate).ProviderPath
        if ($Seen.ContainsKey($Resolved)) { continue }
        $Seen[$Resolved] = $true
        try {
            $Iscc = & $Verify -Compiler $Resolved
            break
        } catch {
            $Rejected += $_.Exception.Message
        }
    }
    if (-not $Iscc) {
        $Detail = if ($Rejected.Count) { " (rejected: $($Rejected -join '; '))" } else { '' }
        throw ("Inno Setup $($Pin.version) (ISCC.exe) was not found$Detail. Stage the pinned compiler with " +
               "scripts\download_inno_setup.ps1 -Destination <folder>, then pass -Compiler <folder>\ISCC.exe")
    }
}

Write-Host "Compiling $Name.exe from $Payload with Inno Setup $($Pin.version) ($Iscc)"
$global:LASTEXITCODE = 0
& $Iscc /Qp "/DAppVersion=$Version" "/DAppNumericVersion=$NumericVersion" "/DPayloadRoot=$Payload" `
    "/O$Output" "/F$Name" (Join-Path $Repo 'packaging\windows\ouroboros.iss')
if ($LASTEXITCODE -ne 0) { throw "The Inno Setup compiler exited $LASTEXITCODE" }
$Setup = Join-Path $Output "$Name.exe"
$Produced = @(Get-ChildItem -LiteralPath $Output -Force)
if ($Produced.Count -ne 1 -or -not (Test-Path -LiteralPath $Setup -PathType Leaf)) {
    throw "The compiler did not write exactly $Name.exe into $Output (found: $(($Produced | ForEach-Object { $_.Name }) -join ', '))"
}
$Digest = (Get-FileHash -LiteralPath $Setup -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Installer: $Setup ($((Get-Item -LiteralPath $Setup).Length) bytes, SHA-256 $Digest)"
