[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$RunRoot)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'Native Windows required; nothing measured.' }
if (-not [IO.Path]::IsPathFullyQualified($RunRoot)) { throw 'Use a new absolute disposable RunRoot.' }
if (Test-Path -LiteralPath $RunRoot) { throw 'RunRoot already exists; it will not be reused or deleted.' }
[IO.Directory]::CreateDirectory($RunRoot) | Out-Null
$evidence = Join-Path $RunRoot 'evidence'
[IO.Directory]::CreateDirectory($evidence) | Out-Null
$pins = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'pins.json') -Raw | ConvertFrom-Json
$probeExit = 2
Start-Transcript -LiteralPath (Join-Path $evidence 'bootstrap.txt') | Out-Null
try {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'pins.json') -Destination $evidence
    $archive = Join-Path $RunRoot 'python.zip'
    Invoke-WebRequest -Uri $pins.python_url -OutFile $archive
    if ((Get-Item -LiteralPath $archive).Length -ne $pins.python_size) { throw 'Python size mismatch.' }
    $expected512 = [Convert]::ToHexString([Convert]::FromBase64String($pins.python_sha512_base64))
    $actual512 = (Get-FileHash -LiteralPath $archive -Algorithm SHA512).Hash
    $actual256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    if ($actual512 -ne $expected512 -or $actual256 -ne $pins.python_sha256) { throw 'Python hash mismatch.' }
    @{ url = $pins.python_url; sha512 = $actual512; sha256 = $actual256 } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $evidence 'download.json') -Encoding utf8
    $runtime = Join-Path $RunRoot 'runtime'
    Expand-Archive -LiteralPath $archive -DestinationPath $runtime
    $probePython = Join-Path $runtime 'tools/python.exe'
    # No setup.ps1, installer, registry change, PATH change, or NuGet client.
    & $probePython -B -s -E -m unittest discover -s $PSScriptRoot -p 'test_portable.py' -v 2>&1 |
        Tee-Object -FilePath (Join-Path $evidence 'portable-tests.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Portable tests failed on Windows.' }
    & $probePython -B -s -E (Join-Path $PSScriptRoot 'probe.py') --root (Join-Path $RunRoot 'measurement') 2>&1 |
        Tee-Object -FilePath (Join-Path $evidence 'driver.txt')
    $probeExit = $LASTEXITCODE
    Write-Host "Probe exit code: $probeExit (this is not an activation verdict)."
}
finally {
    # Copy only direct raw evidence. Never traverse or delete a generation/junction.
    $measurement = Join-Path $RunRoot 'measurement'
    $events = Join-Path $measurement 'events.jsonl'
    if (Test-Path -LiteralPath $events) { Copy-Item -LiteralPath $events -Destination $evidence }
    if (Test-Path -LiteralPath $measurement) {
        foreach ($caseDir in Get-ChildItem -LiteralPath $measurement -Directory) {
            if ($caseDir.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
            foreach ($log in Get-ChildItem -LiteralPath $caseDir.FullName -File) {
                if ($log.Extension -in '.stdout', '.stderr') {
                    Copy-Item -LiteralPath $log.FullName -Destination (Join-Path $evidence ($caseDir.Name + '--' + $log.Name))
                }
            }
        }
    }
    Write-Host "All run files retained at $RunRoot. No cleanup attempted."
    Stop-Transcript | Out-Null
    Get-ChildItem -LiteralPath $evidence -File | Sort-Object Name | ForEach-Object {
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $($_.Name)"
    } | Set-Content -LiteralPath (Join-Path $evidence 'SHA256SUMS.txt') -Encoding utf8
}
exit $probeExit
