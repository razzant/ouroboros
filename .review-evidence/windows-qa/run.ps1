# Evidence-only #208 qualification on native Windows. It signs nothing, holds no
# credentials and publishes nothing. Phases: tests (the four existing test files in
# a dependency-only venv under safe_test), native (real Authenticode/signtool and
# bundled-runtime observations of the published v7.8.0 ZIP), package (manifest+ZIP).
param([Parameter(Mandatory = $true)][ValidateSet('tests', 'native', 'package')][string]$Phase)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false  # exit codes are judged below
if (-not $IsWindows) { throw 'Actual Windows is mandatory' }
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Base = Join-Path $env:RUNNER_TEMP 'issue208-qa'
$Out = Join-Path $Base 'out'
$W = Join-Path $env:RUNNER_TEMP 'q'  # short: ZIP members reach 196 characters
$AssetUrl = 'https://github.com/razzant/ouroboros/releases/download/v7.8.0/Ouroboros-7.8.0-windows-x64.zip'
$AssetSize = 772126896
$AssetSha = '44782370dacfed116363d94c6ce92103a3184f0a42476efa56c7000d6249e2a8'
$MainSha = '8580d1b4ec6deb22c946e692f5a4e7a4cfb902a687374d579921ef1e123fb30d'
$Signer = 'A848890E0454AB8B951B5FF6869E5B80D62FFE3A'
$Tests = 'tests/test_windows_signing_powershell.py', 'tests/test_windows_signing_release.py',
    'tests/test_release_proof.py', 'tests/test_build_scripts.py'
$script:Checks = @()

function Check([string]$Name, [bool]$Pass, $Observed) {
    $script:Checks += [pscustomobject][ordered]@{ name = $Name; pass = $Pass; observed = $Observed }
    Write-Host "$(if ($Pass) { 'PASS' } else { 'FAIL' }) $Name"
}
function Q([string]$Text) { "'" + $Text.Replace("'", "''") + "'" }
function Sha([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Assert-Candidate {
    if ((git -C $Repo rev-parse HEAD).Trim() -ne $env:GITHUB_SHA) { throw 'checkout is not the pushed commit' }
    if (git -C $Repo status --porcelain --untracked-files=no) { throw 'tracked source is dirty' }
}
# Lines of the staged ci.yml "Smoke final Windows archive" step, verbatim, from the
# line equal to $First up to (excluding) the first later line starting with $Before.
function Get-Slice([string]$First, [string]$Before) {
    $Lines = Get-Content -LiteralPath (Join-Path $Repo '.github/workflows/ci.yml')
    $Step = @(0..($Lines.Count - 1) | Where-Object { $Lines[$_].Trim() -eq '- name: Smoke final Windows archive' })
    if ($Step.Count -ne 1) { throw 'ci.yml smoke step not found exactly once' }
    $S = @(($Step[0] + 1)..($Lines.Count - 1) | Where-Object { $Lines[$_].Trim() -eq $First })[0]
    $E = @(($S + 1)..($Lines.Count - 1) | Where-Object { $Lines[$_].Trim().StartsWith($Before) })[0]
    if (-not $S -or -not $E -or @($Lines[$Step[0]..$E] | Where-Object { $_.Trim().StartsWith('- name:') }).Count -ne 1) {
        throw "ci.yml slice '$First' is not inside the smoke step"
    }
    $Indent = $Lines[$S].Length - $Lines[$S].TrimStart().Length
    ($Lines[$S..($E - 1)] | ForEach-Object { if ($_.Length -ge $Indent) { $_.Substring($Indent) } else { $_.TrimStart() } }) -join "`n"
}
# A child pwsh in the checkout with a bounded wait; streams and exit code kept.
function Invoke-Child([string]$Name, [string]$Body, [hashtable]$Vars = @{}, [int]$Seconds = 120) {
    $Dir = Join-Path $Out "native/$Name"
    [void][IO.Directory]::CreateDirectory($Dir)
    $Script = Join-Path $Dir 'script.ps1'
    Set-Content -LiteralPath $Script -Value ("`$ErrorActionPreference = 'Stop'`ntry {`n$Body`n} catch { [Console]::Error.WriteLine(`$_.Exception.Message); exit 1 }")
    $Info = [Diagnostics.ProcessStartInfo]::new((Get-Process -Id $PID).Path)
    foreach ($Arg in '-NoLogo', '-NoProfile', '-NonInteractive', '-File', $Script) { $Info.ArgumentList.Add($Arg) }
    $Info.WorkingDirectory = $Repo; $Info.UseShellExecute = $false
    $Info.RedirectStandardOutput = $true; $Info.RedirectStandardError = $true
    foreach ($Key in $Vars.Keys) { $Info.Environment[$Key] = $Vars[$Key] }
    $Process = [Diagnostics.Process]::Start($Info)
    $StdOut = $Process.StandardOutput.ReadToEndAsync(); $StdErr = $Process.StandardError.ReadToEndAsync()
    $Code = if ($Process.WaitForExit($Seconds * 1000)) { $Process.WaitForExit(); $Process.ExitCode } else { $Process.Kill($true); 'timeout' }
    $Result = [pscustomobject][ordered]@{ exit = $Code; stdout = $StdOut.Result; stderr = $StdErr.Result }
    $Result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Dir 'result.json')
    $Result
}
function Observe([string]$Label, [string]$Path) {
    $Sig = Get-AuthenticodeSignature -LiteralPath $Path
    & $script:SignTool verify /pa /all /tw /v $Path *> (Join-Path $Out "native/signtool-$Label.txt")
    $Exit = $LASTEXITCODE
    [pscustomobject][ordered]@{ path = $Path; sha256 = (Sha $Path); status = "$($Sig.Status)"
        signatureType = "$($Sig.SignatureType)"; thumbprint = $Sig.SignerCertificate.Thumbprint
        subject = $Sig.SignerCertificate.Subject; timeStamper = $Sig.TimeStamperCertificate.Subject; signtoolExit = $Exit }
}

function Invoke-Tests {
    $D = Join-Path $Out 'tests'
    Assert-Candidate
    git -C $Repo ls-files -s | Set-Content -LiteralPath (Join-Path $Out 'candidate-ls-files.txt')
    $Venv = Join-Path $env:RUNNER_TEMP 'qa-venv'
    $env:UV_PROJECT_ENVIRONMENT = $Venv; $env:UV_CACHE_DIR = Join-Path $env:RUNNER_TEMP 'uv-cache'; $env:UV_PYTHON_DOWNLOADS = 'never'
    Push-Location $Repo
    try {
        & uv sync --locked --extra browser --group dev --no-install-project --python $env:QA_BASE_PYTHON *> (Join-Path $D 'uv-sync.log')
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed: $LASTEXITCODE" }
        $Py = Join-Path $Venv 'Scripts\python.exe'
        & uv pip list --python $Py --format freeze > (Join-Path $D 'installed.txt')
        & $Py -I -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('ouroboros') is not None)"
        Check 'venv.dependency_only' ($LASTEXITCODE -eq 0) 'ouroboros not importable from the venv itself'
        # The PATH's bash for the bash-executing cases; System32's bash is the WSL launcher.
        $env:PATH = (Join-Path $env:ProgramFiles 'Git\bin') + ';' + $env:PATH
        & $Py -I -c "import json, shutil; print(json.dumps({n: shutil.which(n) for n in ('pwsh', 'bash', 'git')}))" > (Join-Path $D 'tools.json')
        $St = Join-Path $env:RUNNER_TEMP 'st'; [void][IO.Directory]::CreateDirectory($St)
        $Junit = Join-Path $D 'junit.xml'
        $Argv = @('-I', '-S', 'scripts/safe_test.py', '--temp-parent', $St, '--evidence-dir', (Join-Path $D 'safe-test'),
            '--timeout', '420', '--', $Py, '-m', 'pytest') + $Tests + @('-p', 'no:cacheprovider', '--timeout=300', '-ra', "--junitxml=$Junit")
        & $Py @Argv > (Join-Path $D 'safe-test.stdout.txt') 2> (Join-Path $D 'safe-test.stderr.txt')
        Check 'pytest.exit' ($LASTEXITCODE -eq 0) $LASTEXITCODE
    } finally { Pop-Location }
    [xml]$X = Get-Content -Raw -LiteralPath $Junit
    $Cases = @($X.SelectNodes('//testcase'))
    foreach ($T in $Tests) {
        $Mod = $T.Replace('/', '.').Replace('.py', '')
        $Mine = @($Cases | Where-Object { $_.classname -eq $Mod -or $_.classname.StartsWith("$Mod.") })
        $Failed = @($Mine | Where-Object { $_.SelectSingleNode('failure|error') }).Count
        $Skipped = @($Mine | Where-Object { $_.SelectSingleNode('skipped') }).Count
        Check "junit.$Mod" ($Mine.Count -gt 0 -and $Failed -eq 0 -and $Skipped -eq 0) @{ cases = $Mine.Count; failed = $Failed; skipped = $Skipped }
    }
    Check 'junit.total' ($Cases.Count -gt 0) $Cases.Count
}

function Invoke-Native {
    Assert-Candidate
    foreach ($Dir in 'a', 'x', 't', 'h', 'rt') { [void][IO.Directory]::CreateDirectory((Join-Path $W $Dir)) }
    $Asset = Join-Path $W 'a\Ouroboros-7.8.0-windows-x64.zip'
    & curl.exe --fail --location --silent --show-error --retry 3 --connect-timeout 30 --max-time 240 --output $Asset $AssetUrl
    $Size = if (Test-Path -LiteralPath $Asset) { (Get-Item -LiteralPath $Asset).Length } else { 0 }
    $Hash = if ($Size) { Sha $Asset } else { '' }
    Check 'asset.pinned' ($Size -eq $AssetSize -and $Hash -eq $AssetSha) @{ url = $AssetUrl; size = $Size; sha256 = $Hash }
    if ($Hash -ne $AssetSha) { throw 'published ZIP does not match its public digest; nothing else is observed' }
    # Only the launcher and the bundled interpreter tree, exactly as the ZIP holds them.
    $Root = Join-Path $W 'x'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Zip = [IO.Compression.ZipFile]::OpenRead($Asset)
    try {
        foreach ($Entry in $Zip.Entries) {
            $Name = $Entry.FullName
            if ($Name -ne 'Ouroboros/Ouroboros.exe' -and -not $Name.StartsWith('Ouroboros/_internal/python-standalone/')) { continue }
            if ($Name -match '(^|/)\.\.(/|$)|\\|:') { throw "unsafe ZIP member $Name" }
            $Target = Join-Path $Root $Name.Replace('/', '\')
            [void][IO.Directory]::CreateDirectory((Split-Path -Parent $Target))
            if (-not $Name.EndsWith('/')) { [IO.Compression.ZipFileExtensions]::ExtractToFile($Entry, $Target) }
        }
    } finally { $Zip.Dispose() }
    $Catalog = Get-Content -Raw -LiteralPath (Join-Path $Repo 'scripts/windows_signing_catalog.json') | ConvertFrom-Json
    $Main, $PyExe, $PywExe = @($Catalog.files | ForEach-Object { Join-Path $Root $_.path.Replace('/', '\') })
    $Pins = @($Catalog.files[1].upstreamSha256, $Catalog.files[2].upstreamSha256)
    Check 'catalog.pins_match_release' ((Sha $Main) -eq $MainSha -and (Sha $PyExe) -eq $Pins[0] -and (Sha $PywExe) -eq $Pins[1]) $Pins
    $script:SignTool = Get-Command signtool.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
    if (-not $script:SignTool) {
        $script:SignTool = Get-ChildItem (Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin') -Directory | Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName 'x64\signtool.exe' } | Where-Object { Test-Path $_ } | Select-Object -First 1
    }
    $M = Observe 'main' $Main
    Check 'main.valid_timestamped' ($M.status -eq 'Valid' -and $M.thumbprint -eq $Signer -and $M.signtoolExit -eq 0) $M
    foreach ($Pair in @(@('python', $PyExe), @('pythonw', $PywExe))) {
        $O = Observe $Pair[0] $Pair[1]
        Check "$($Pair[0]).unsigned_expected" ($O.status -eq 'NotSigned' -and $O.signtoolExit -ne 0) $O
    }
    $Verify = '& .\scripts\verify_windows_signature.ps1 -Executable {0} -ExpectedThumbprint {1}'
    $R = Invoke-Child 'verify-main' ($Verify -f (Q $Main), (Q $Signer))
    Check 'verifier.main' ($R.exit -eq 0 -and $R.stdout -match 'timestamped') $R.exit
    # The staged windows-proof signature block, verbatim, on the real ZIP: refused at python.exe.
    $Block = Get-Slice 'if ($env:WINDOWS_SIGNER_THUMBPRINT) {' '# PyInstaller 6 onedir'
    $GitHubOutput = Join-Path $Out 'native/catalog-current.github-output'
    $R = Invoke-Child 'catalog-current' ("`$SmokeRoot = $(Q $Root)`n$Block") @{ WINDOWS_SIGNER_THUMBPRINT = $Signer; GITHUB_OUTPUT = $GitHubOutput }
    $Verified = @([regex]::Matches($R.stdout, 'Verified Authenticode signer')).Count  # Ouroboros.exe only
    Check 'catalog.current_zip_refused_expected' ($R.exit -eq 1 -and $R.stderr -match 'not valid: NotSigned' -and $Verified -eq 1 -and
        -not (Test-Path $GitHubOutput)) @{ exit = $R.exit; verified = $Verified }
    # Negative controls: one flipped byte inside the hashed image; a valid foreign signer; a wrong expected thumbprint.
    $Tampered = Join-Path $W 't\Ouroboros.exe'
    $Bytes = [IO.File]::ReadAllBytes($Main)
    $Pe = [BitConverter]::ToInt32($Bytes, 0x3C)
    $DirBase = $Pe + 24 + $(if ([BitConverter]::ToUInt16($Bytes, $Pe + 24) -eq 0x20B) { 112 } else { 96 })
    $Offset = [int]([BitConverter]::ToUInt32($Bytes, $DirBase + 32) / 2)
    $Bytes[$Offset] = $Bytes[$Offset] -bxor 0xFF
    [IO.File]::WriteAllBytes($Tampered, $Bytes)
    $T = Observe 'tampered' $Tampered
    $R = Invoke-Child 'verify-tampered' ($Verify -f (Q $Tampered), (Q $Signer))
    Check 'tampered.refused' ($T.status -eq 'HashMismatch' -and $T.signtoolExit -ne 0 -and $R.exit -eq 1 -and $R.stderr -match 'HashMismatch') @{ offset = $Offset; observed = $T }
    $Foreign = Join-Path $PSHOME 'pwsh.exe'
    $F = Get-AuthenticodeSignature -LiteralPath $Foreign
    $R = Invoke-Child 'verify-foreign' ($Verify -f (Q $Foreign), (Q $Signer))
    Check 'wrong_signer.foreign_refused' ($F.Status -eq 'Valid' -and $R.exit -eq 1 -and $R.stderr -match 'not the configured certificate') $F.SignerCertificate.Subject
    $R = Invoke-Child 'verify-wrong-expected' ($Verify -f (Q $Main), (Q ($Signer.Substring(0, 39) + 'B')))
    Check 'wrong_signer.expected_refused' ($R.exit -eq 1 -and $R.stderr -match 'not the configured certificate') $R.exit
    Check 'main.unchanged' ((Sha $Main) -eq $MainSha) (Sha $Main)
    # The staged python.exe/pythonw.exe smoke, verbatim, on the bundled runtime from this ZIP.
    $Smoke = Get-Slice '$env:HOME = $HomeDir' '& "$SmokeRoot\Ouroboros\bin\ouroboros.cmd" --help'
    $Before = @((Sha $PyExe), (Sha $PywExe))
    $R = Invoke-Child 'smoke' ("`$SmokeRoot = $(Q $Root)`n`$HomeDir = $(Q (Join-Path $W 'h'))`n$Smoke") @{ RUNNER_TEMP = (Join-Path $W 'rt') } 180
    $Probe = Get-Content -Raw -LiteralPath (Join-Path $W "rt\pythonw-probe-$env:GITHUB_RUN_ATTEMPT.txt") -ErrorAction SilentlyContinue
    Check 'runtime.smoke' ($R.exit -eq 0 -and $Probe -eq 'ok' -and "$((Sha $PyExe), (Sha $PywExe))" -eq "$Before") @{ exit = $R.exit; probe = $Probe }
    $Version = & $PyExe -I -B -c 'import sys; print(sys.version.split()[0])'
    Check 'runtime.version_is_catalog_python' ("$Version".Trim() -eq $Catalog.pythonUpstream.pythonVersion) "$Version"
}

function Invoke-Package {
    $Os = Get-CimInstance Win32_OperatingSystem
    git -C $Repo status --porcelain=v1 --untracked-files=all | Set-Content -LiteralPath (Join-Path $Out 'candidate-status-after.txt')
    $Results = @{}
    foreach ($Name in 'tests', 'native') {
        $File = Join-Path $Out "$Name/result.json"
        $Results[$Name] = if (Test-Path $File) { Get-Content -Raw -LiteralPath $File | ConvertFrom-Json } else { 'not run' }
    }
    [ordered]@{ githubSha = $env:GITHUB_SHA; head = (git -C $Repo rev-parse HEAD); tree = (git -C $Repo rev-parse 'HEAD^{tree}')
        runId = $env:GITHUB_RUN_ID; runAttempt = $env:GITHUB_RUN_ATTEMPT; workflowRef = $env:GITHUB_WORKFLOW_REF
        os = "$($Os.Caption) $($Os.BuildNumber)"; image = "$env:ImageOS $env:ImageVersion"; pwsh = "$($PSVersionTable.PSVersion)"
        results = $Results
        limits = 'Mocked tests prove control flow only. No file was signed; no signed python.exe/pythonw.exe exists yet. ' +
            'No SmartScreen, antivirus or release qualification.'
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Out 'summary.json')
    $Lines = Get-ChildItem -LiteralPath $Out -Recurse -File -Force | Sort-Object FullName |
        ForEach-Object { '{0}  {1}' -f (Sha $_.FullName), $_.FullName.Substring($Out.Length + 1).Replace('\', '/') }
    Set-Content -LiteralPath (Join-Path $Out 'SHA256SUMS.txt') -Value $Lines
    $Pack = Join-Path $Base 'pack'; [void][IO.Directory]::CreateDirectory($Pack)
    $Name = "issue208-windows-qa-$($env:GITHUB_SHA.Substring(0, 12))-$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT.zip"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory($Out, (Join-Path $Pack $Name), [IO.Compression.CompressionLevel]::Optimal, $true)
    Set-Content -LiteralPath (Join-Path $Pack "$Name.sha256") -Value ('{0}  {1}' -f (Sha (Join-Path $Pack $Name)), $Name)
    Copy-Item -LiteralPath (Join-Path $Out 'SHA256SUMS.txt'), (Join-Path $Out 'summary.json') -Destination $Pack
}

[void][IO.Directory]::CreateDirectory((Join-Path $Out $Phase))
if ($Phase -eq 'package') { Invoke-Package; exit 0 }
$Failure = ''
try { if ($Phase -eq 'tests') { Invoke-Tests } else { Invoke-Native } } catch { $Failure = "$($_.Exception.Message)" }
$Pass = -not $Failure -and @($script:Checks | Where-Object { -not $_.pass }).Count -eq 0
[ordered]@{ phase = $Phase; pass = $Pass; error = $Failure; checks = $script:Checks } | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $Out "$Phase/result.json")
if (-not $Pass) { Write-Host "Phase $Phase failed: $Failure"; exit 1 }
