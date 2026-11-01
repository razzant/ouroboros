# Windows Update qualification: provision, check the equipment on this host, run, always export evidence.
# Equipment, not product. Online only while provisioning (pinned downloads, the wheelhouse); the product
# phases get PIP_NO_INDEX + this wheelhouse and an official Git URL mapped to a local mirror (qualify.py).
# Two identities stay apart: the candidate (inputs.json's sha and tree, checked out on its own) and the equipment
# commit this job runs on, whose one parent must be that candidate (qualify.py --bind-only, before the downloads).
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunRoot,          # new, absolute, short (D:\wq-<hex>): Windows path budget
    [Parameter(Mandatory = $true)][string]$EquipmentRepo,    # checkout of the equipment commit, full history
    [Parameter(Mandatory = $true)][string]$EquipmentCommit,  # its sha (GITHUB_SHA)
    [string]$Cases = ''                                      # empty: the full required selection
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:OS -ne 'Windows_NT') { throw 'Native Windows required; nothing measured.' }
if (-not [IO.Path]::IsPathFullyQualified($RunRoot)) { throw 'Use a new absolute RunRoot.' }
if (Test-Path -LiteralPath $RunRoot) { throw 'RunRoot already exists; it is never reused or deleted.' }
if ($EquipmentCommit -notmatch '^[0-9a-f]{40}$') { throw 'EquipmentCommit must be 40-hex.' }
$inputsPath = Join-Path $PSScriptRoot 'inputs.json'
$inputs = Get-Content -LiteralPath $inputsPath -Raw | ConvertFrom-Json
$CandidateSha, $CandidateTree = [string]$inputs.candidate.sha, [string]$inputs.candidate.tree
if ($CandidateSha -notmatch '^[0-9a-f]{40}$' -or $CandidateTree -notmatch '^[0-9a-f]{40}$') {
    throw 'inputs.json does not bind the candidate sha and tree (40-hex); nothing was downloaded or run.'
}
$evidence = Join-Path $RunRoot 'evidence'
$downloads = Join-Path $RunRoot 'downloads'
$wheelhouse = Join-Path $RunRoot 'wheelhouse'
foreach ($dir in @($RunRoot, $evidence, $downloads, $wheelhouse)) { [IO.Directory]::CreateDirectory($dir) | Out-Null }
$driverExit = 2
Start-Transcript -LiteralPath (Join-Path $evidence 'bootstrap.txt') | Out-Null
try {
    $effective = Join-Path $evidence 'inputs.effective.json'
    Copy-Item -LiteralPath $inputsPath -Destination $effective

    # The candidate on its own, at exactly its commit and tree (never the equipment commit's tree).
    $Candidate = Join-Path $RunRoot 'candidate'
    git clone --quiet --no-checkout --no-hardlinks $EquipmentRepo $Candidate
    if ($LASTEXITCODE -ne 0) { throw 'candidate clone failed' }
    git -C $Candidate -c advice.detachedHead=false checkout --quiet --detach $CandidateSha
    if ($LASTEXITCODE -ne 0) { throw 'the candidate commit is not in the equipment commit''s history' }
    $tree = git -C $Candidate rev-parse 'HEAD^{tree}'
    if ($tree -ne $CandidateTree) { throw "candidate tree is $tree, inputs bind $CandidateTree" }

    function Get-Verified([string]$Url, [string]$Path, [long]$Size, [string]$Sha256, [string]$Sha512b64) {
        Invoke-WebRequest -Uri $Url -OutFile $Path
        if ((Get-Item -LiteralPath $Path).Length -ne $Size) { throw "Size mismatch: $Url" }
        if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) { throw "SHA-256 mismatch: $Url" }
        if ($Sha512b64) {
            $expected = [Convert]::ToHexString([Convert]::FromBase64String($Sha512b64))
            if ((Get-FileHash -LiteralPath $Path -Algorithm SHA512).Hash -ne $expected) { throw "SHA-512 mismatch: $Url" }
        }
    }

    # The PSF NuGet CPython pinned by the NTFS probe: one copy runs the driver, another is mode F's bare subject.
    $nupkg = Join-Path $downloads 'python.zip'
    Get-Verified $inputs.python.url $nupkg $inputs.python.size $inputs.python.sha256 $inputs.python.sha512_base64
    $driverRoot = Join-Path $RunRoot 'driver-python'
    $bareRoot = Join-Path $RunRoot 'bare py'   # a space: quoting of the interpreter path the product hands on
    Expand-Archive -LiteralPath $nupkg -DestinationPath $driverRoot
    Expand-Archive -LiteralPath $nupkg -DestinationPath $bareRoot
    $driverPython = Join-Path $driverRoot 'tools\python.exe'
    $barePython = Join-Path $bareRoot 'tools\python.exe'

    # Both identities, apart, before the 772 MB download: evidence\bindings.json, or a refusal with its reason.
    $identity = @('--inputs', $effective, '--run-root', $RunRoot, '--candidate', $Candidate,
                  '--equipment-repo', $EquipmentRepo, '--equipment-commit', $EquipmentCommit)
    & $driverPython -E -s -B (Join-Path $PSScriptRoot 'qualify.py') --bind-only @identity 2>&1 |
        Tee-Object -FilePath (Join-Path $evidence 'bind.txt')
    if ($LASTEXITCODE -ne 0) { throw 'The equipment commit is not exactly the bound candidate plus this equipment; nothing was run.' }

    # The unchanged released app, verified against the parent's pin, extracted under a path with a space.
    $zip = Join-Path $downloads 'release.zip'
    Get-Verified $inputs.release.url $zip $inputs.release.size $inputs.release.sha256 ''
    $releaseDir = Join-Path $RunRoot 'released app'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($zip, $releaseDir)
    $exe = @(Get-ChildItem -LiteralPath $releaseDir -Recurse -File -Filter 'Ouroboros.exe')
    if ($exe.Count -ne 1) { throw "Expected one Ouroboros.exe, found $($exe.Count)" }
    $appPython = Join-Path $exe[0].DirectoryName '_internal\python-standalone\python.exe'

    # Host facts that can explain a failure: OS build, file system, Git and its system config, long paths.
    $facts = [ordered]@{
        os = (Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber)
        filesystem = (Get-Volume -DriveLetter $RunRoot.Substring(0, 1)).FileSystemType
        git = (& git --version)
        git_system_config = (& git config --system --list --show-origin 2>&1 | Out-String)
        long_paths = (Get-ItemPropertyValue 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction SilentlyContinue)
        candidate = (& git -C $Candidate rev-parse HEAD 'HEAD^{tree}')
        equipment = (& git -C $EquipmentRepo rev-parse HEAD 'HEAD^{tree}')
        release_exe_sha256 = (Get-FileHash -LiteralPath $exe[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $facts | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $evidence 'host.json') -Encoding utf8

    # The complete wheelhouse of the candidate lock, for the app's interpreter and for the bare one (cp310).
    $env:PIP_CONFIG_FILE = 'NUL'; $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:PIP_CACHE_DIR = Join-Path $RunRoot 'pip-cache'
    $lock = Join-Path $Candidate 'requirements-runtime.lock'
    foreach ($python in @($appPython, $barePython)) {
        & $python -E -s -B -m pip download --only-binary=:all: --dest $wheelhouse -r $lock 2>&1 |
            Tee-Object -FilePath (Join-Path $evidence 'wheelhouse.txt') -Append
        if ($LASTEXITCODE -ne 0) { throw "Wheelhouse download failed for $python" }
    }
    Remove-Item Env:PIP_CONFIG_FILE, Env:PIP_DISABLE_PIP_VERSION_CHECK, Env:PYTHONDONTWRITEBYTECODE, Env:PIP_CACHE_DIR
    Get-ChildItem -LiteralPath $wheelhouse -File | Sort-Object Name | ForEach-Object {
        "$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $($_.Name)"
    } | Set-Content -LiteralPath (Join-Path $evidence 'wheelhouse.sha256') -Encoding utf8

    # The equipment's own falsifying tests on this host, then the pipeline canary: a refuted verdict must exit non-zero.
    & $driverPython -E -s -B -m unittest discover -s $PSScriptRoot -p 'test_portable.py' -v 2>&1 |
        Tee-Object -FilePath (Join-Path $evidence 'portable-tests-windows.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Portable tests failed on Windows; nothing was qualified.' }
    & $driverPython -E -s -B (Join-Path $PSScriptRoot 'qualify.py') --demonstrate-false-verdict 2>&1 |
        Tee-Object -FilePath (Join-Path $evidence 'canary.txt')
    if ($LASTEXITCODE -eq 0) { throw 'A deliberately false verdict exited 0: the exit-code path is broken.' }

    $arguments = @('-E', '-s', '-B', (Join-Path $PSScriptRoot 'qualify.py'), '--release-dir', $releaseDir,
                   '--bare-python', $barePython, '--wheelhouse', $wheelhouse) + $identity
    if ($Cases) { $arguments += @('--cases', $Cases) }
    & $driverPython @arguments 2>&1 | Tee-Object -FilePath (Join-Path $evidence 'driver.txt')
    $driverExit = $LASTEXITCODE
    Write-Host "Qualification driver exit code: $driverExit (0 only: full selection, every case PASS, custody confirmed)."
}
finally {
    Write-Host "All run files retained at $RunRoot. Nothing deleted."
    Stop-Transcript | Out-Null
    Get-ChildItem -LiteralPath $evidence -Recurse -File | Where-Object Name -ne 'SHA256SUMS.txt' | Sort-Object FullName |
        ForEach-Object {
            "$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $($_.FullName.Substring($evidence.Length + 1))"
        } | Set-Content -LiteralPath (Join-Path $evidence 'SHA256SUMS.txt') -Encoding utf8
}
exit $driverExit
