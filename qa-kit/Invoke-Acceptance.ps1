# One invokable path, ONLY on a disposable GitHub-hosted Windows runner with a fictitious profile.
#   -Mode visible    leg 1: a CLEAN profile without system Git. The real installer wizard is opened as a person opens it and driven only through UIA
#                    InvokePattern (Install, Finish) inside its own process tree; frames of the wizard and of the app's first-run wizard.
#   -Mode migration  leg 2, ITS OWN runner and profile: the published old release ZIP (-OldZip) creates a profile and a chat via its own loopback APIs -> the new package
#                    through the same ordinary wizard, whose Finish opens the app -> Update + handoff -> cold start by the Start .lnk ->
#                    uninstall -> new ZIP reopen.
# Both need the candidate's ONE built dist\ ZIP: the kit extracts its payload and the author's packer packs Setup from exactly
# the ZIP's files (pack_windows_setup.ps1 refuses a raw PyInstaller dist). No physical input is sent. OUROBOROS_QA_DISPOSABLE=1 is set by whoever
# owns the workflow; this script never sets it. Only allowlisted JSON facts and bounded PNG frames are selected into <WorkRoot>\publish.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('visible', 'migration')][string]$Mode,
    [Parameter(Mandatory)][string]$WorkRoot,         # must not exist: venv, evidence and the publishable selection go here
    [Parameter(Mandatory)][string]$CandidateRoot,    # the candidate checkout: its UIA skill, its tool directories, its dist\ build
    [string]$Installer = '',                         # a pre-built installer; otherwise the author's packer is run on dist\Ouroboros
    [string]$Compiler = '',                          # ISCC.exe of the pinned Inno Setup (scripts\download_inno_setup.ps1)
    [string]$FactsDir = '',                          # the build facts (projection.json, build-binding.json, package.json) to carry into publish
    [string]$OldZip = '',                            # migration: the published old release ZIP, unchanged
    [switch]$UninstallRunnerGit,                     # visible: opt in to #4220's verified uninstall of the runner's own Git
    [string]$LaunchRoute = 'shellexecute',
    [string]$Python = 'python'
)
Set-StrictMode -Version 3
$ErrorActionPreference = 'Stop'
if ($env:OUROBOROS_QA_DISPOSABLE -ne '1' -or $env:GITHUB_ACTIONS -ne 'true' -or $env:RUNNER_ENVIRONMENT -ne 'github-hosted') {
    throw 'Refused: only a disposable GitHub-hosted Windows runner with OUROBOROS_QA_DISPOSABLE=1.'
}
if (Test-Path -LiteralPath $WorkRoot) { throw 'WorkRoot must not exist.' }
$kit = $PSScriptRoot
New-Item -ItemType Directory -Path $WorkRoot | Out-Null
$evidence = Join-Path $WorkRoot 'evidence'
$publish = Join-Path $WorkRoot 'publish'

# Only allowlisted synthetic facts and bounded frames leave the host.
function Select-Publishable([switch]$NoFrames) {
    New-Item -ItemType Directory -Path (Join-Path $publish 'frames') -Force | Out-Null
    foreach ($name in 'result.json', 'host-git.json', 'git-traces.json', 'fixture-model.json') {
        if (Test-Path -LiteralPath (Join-Path $evidence $name)) { Copy-Item (Join-Path $evidence $name) (Join-Path $publish $name) }
    }
    if ($FactsDir) {
        foreach ($name in 'projection.json', 'build-binding.json', 'package.json') {
            if (Test-Path -LiteralPath (Join-Path $FactsDir $name)) { Copy-Item (Join-Path $FactsDir $name) (Join-Path $publish $name) }
        }
    }
    if (-not $NoFrames -and (Test-Path -LiteralPath (Join-Path $evidence 'frames'))) {
        Get-ChildItem -LiteralPath (Join-Path $evidence 'frames') -Filter '*.png' |
            Where-Object { $_.Name -match '^[a-z0-9_]+\.png$' -and $_.Length -le 3MB } | Select-Object -First 80 |
            ForEach-Object { Copy-Item $_.FullName (Join-Path (Join-Path $publish 'frames') $_.Name) }
    }
}

# A stop that left no result: a structured one with fixed words only (preflight writes its own, richer one first). A run that
# started and died may have installed, launched or photographed something: that is "unknown", and its frames stay on the host.
function Stop-Early([string]$phase) {
    $result = Join-Path $evidence 'result.json'
    $ran = $phase -eq 'acceptance_run'
    if (-not (Test-Path -LiteralPath $result)) {  # never overwrite what preflight or the run itself recorded
        New-Item -ItemType Directory -Path $evidence -Force | Out-Null
        $reason = [ordered]@{ verdict = 'unconfirmed'; phase = $phase; code = ($phase + '_failed') }
        $facts = [ordered]@{ acceptance_run_started = $ran }
        $collected = $false
        if ($ran) { $collected = 'unknown' } else { $facts.nothing_installed_or_launched = $true }
        $doc = [ordered]@{
            schema = 5; verdict = 'unconfirmed'; phase = $phase; code = ($phase + '_failed'); visual = 'n/a'; reasons = @($reason)
            checks = @{}; facts = $facts; declared = @{ screenshots_collected = $collected }
        }
        [IO.File]::WriteAllText($result, ($doc | ConvertTo-Json -Depth 5))
    }
    Select-Publishable -NoFrames:$ran
    exit 2
}

$venv = Join-Path $WorkRoot 'qa-venv'
& $Python -m venv $venv
if ($LASTEXITCODE -ne 0) { Stop-Early 'venv' }
$py = Join-Path $venv 'Scripts\python.exe'
# The UIA skill's #4220 pin; migration also uses the old release's own locked WebSocket client version for Python 3.12.
$deps = @('comtypes==1.4.17')
if ($Mode -eq 'migration') { $deps += 'websockets==17.0.1' }
& $py -m pip install --disable-pip-version-check --no-cache-dir @deps
if ($LASTEXITCODE -ne 0) { Stop-Early 'qa_dependencies_install' }
& $py -I -B (Join-Path $kit 'run.py') preflight --out (Join-Path $WorkRoot 'preflight.json') --evidence $evidence --uia --candidate-root $CandidateRoot
if ($LASTEXITCODE -ne 0) { Stop-Early 'preflight' }

$run = @((Join-Path $kit 'run.py'), $Mode, '--evidence', $evidence, '--launch-route', $LaunchRoute, '--candidate-root', $CandidateRoot)
if ($Installer) { $run += @('--installer', $Installer) }
if ($Compiler) { $run += @('--compiler', $Compiler) }
if ($FactsDir -and (Test-Path -LiteralPath (Join-Path $FactsDir 'projection.json'))) { $run += @('--projection-file', (Join-Path $FactsDir 'projection.json')) }
if ($Mode -eq 'migration' -and $OldZip) { $run += @('--old-zip', $OldZip) }
if ($Mode -eq 'visible' -and $UninstallRunnerGit) { $run += '--uninstall-runner-git' }
try { Set-DisplayResolution -Width 1366 -Height 768 -Force } catch { Write-Host "display size kept: $($_.Exception.GetType().Name)" }
& $py -I -B @run
$code = $LASTEXITCODE

if (-not (Test-Path -LiteralPath (Join-Path $evidence 'result.json'))) { Stop-Early 'acceptance_run' }
Select-Publishable
exit $code
