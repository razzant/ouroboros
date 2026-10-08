# Verify a signed Ouroboros.exe: a valid Authenticode signature from the expected
# certificate, timestamped. Holds no credentials, so CI and anyone checking a
# downloaded release run the same script with the thumbprint the release
# evidence publishes. Returns the verified signer for the caller's proof.
param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [Parameter(Mandatory = $true)][string]$ExpectedThumbprint
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false  # exit codes are judged below
if ($ExpectedThumbprint -notmatch '^[a-fA-F0-9]{40}\z') { throw 'Expected SHA-1 certificate thumbprint is missing or malformed' }
$Expected = $ExpectedThumbprint.ToUpperInvariant()
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Windows executable missing: $Executable" }
$Signature = Get-AuthenticodeSignature -LiteralPath $Executable
if ($Signature.Status -ne 'Valid' -or -not $Signature.SignerCertificate) {
    throw "Authenticode signature is not valid: $($Signature.Status)"
}
$Actual = $Signature.SignerCertificate.Thumbprint
if ($Actual -ne $Expected) { throw "Authenticode signer $Actual is not the configured certificate $Expected" }
# RFC 3161 timestamps need not populate TimeStamperCertificate in this API.
# SignTool's documented exit codes decide instead of its localized text:
# 0 verified; 1 failed; 2 warnings, which /tw raises for a missing timestamp.
$SignTool = Get-Command signtool.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
if (-not $SignTool) {
    $Kits = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    if (Test-Path $Kits) {
        $SignTool = Get-ChildItem $Kits -Directory | Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName 'x64\signtool.exe' } |
            Where-Object { Test-Path $_ } | Select-Object -First 1
    }
}
if (-not $SignTool) { throw 'Windows SDK signtool.exe is required for timestamp verification' }
& $SignTool verify /pa /tw /v $Executable *> $null
switch ($LASTEXITCODE) {
    0 { }
    2 { throw 'SignTool verified the signature only with warnings (exit 2); /tw warns when it carries no timestamp' }
    default { throw "SignTool rejected the signature (exit $LASTEXITCODE)" }
}
$Signer = [pscustomobject]@{
    Thumbprint = $Actual
    Publisher  = $Signature.SignerCertificate.GetNameInfo(
        [System.Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false)
    Sha256     = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
}
Write-Host "Verified Authenticode signer $($Signer.Publisher) ($Actual), timestamped; executable SHA-256 $($Signer.Sha256)"
$Signer
