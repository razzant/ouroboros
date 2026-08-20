# Build script for Ouroboros on Windows
# Run from repo root: powershell -ExecutionPolicy Bypass -File build_windows.ps1

$ErrorActionPreference = "Stop"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    & $Command
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        throw "$Description failed with exit code $ExitCode"
    }
}

$Version = (Get-Content VERSION).Trim()
$ArchiveName = "Ouroboros-${Version}-windows-x64.zip"
$ManagedSourceBranch = if ($env:OUROBOROS_MANAGED_SOURCE_BRANCH) { $env:OUROBOROS_MANAGED_SOURCE_BRANCH } else { "ouroboros" }
$env:PYTHONDONTWRITEBYTECODE = "1"
if (-not $env:PYTHONPYCACHEPREFIX) {
    $env:PYTHONPYCACHEPREFIX = Join-Path ([System.IO.Path]::GetTempPath()) "OuroborosBuildPycache"
}
New-Item -ItemType Directory -Force -Path $env:PYTHONPYCACHEPREFIX | Out-Null

Write-Host "=== Building Ouroboros for Windows (v${Version}) ==="

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv 0.12.1 is required. Install it with: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.12.1/install.ps1 | iex"'
}

if (-not (Test-Path "python-standalone\python.exe")) {
    Write-Host "ERROR: python-standalone\ not found."
    Write-Host "Run first: powershell -ExecutionPolicy Bypass -File scripts/download_python_standalone.ps1"
    exit 1
}

# Bundle the official Node.js runtime so node-runtime skills work out of the box.
if (-not (Test-Path "node-standalone\node.exe")) {
    Write-Host "--- Downloading bundled Node.js runtime ---"
    Invoke-NativeChecked "Node.js runtime download" {
        powershell -ExecutionPolicy Bypass -File scripts/download_node_standalone.ps1
    }
}

Write-Host "--- Installing launcher dependencies ---"
$BuildRequirements = Join-Path ([IO.Path]::GetTempPath()) "ouroboros-build-requirements-$PID.txt"
Invoke-NativeChecked "Build dependency export" {
    uv export --locked --no-dev --extra browser --extra desktop --extra build `
        --no-emit-project --no-hashes --no-annotate --output-file $BuildRequirements
}
try {
    Invoke-NativeChecked "Launcher dependency installation" {
        uv pip install --python python -q -r $BuildRequirements
    }
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $BuildRequirements
}

if (-not (Test-Path "ripgrep-standalone\rg.exe")) {
    Write-Host "--- Downloading bundled ripgrep runtime ---"
    Invoke-NativeChecked "ripgrep runtime download" {
        powershell -ExecutionPolicy Bypass -File "scripts\download_ripgrep_standalone.ps1"
    }
}

Write-Host "--- Verifying bundled Betterleaks runtime ---"
Invoke-NativeChecked "Betterleaks runtime installation" {
    & "python-standalone\python.exe" -m ouroboros.betterleaks_runtime install `
        --build-output betterleaks-standalone
}

Write-Host "--- Installing agent dependencies into python-standalone ---"
Invoke-NativeChecked "Agent dependency installation" {
    uv pip install --python "python-standalone\python.exe" -q -r requirements-runtime.lock
}

Write-Host "--- Fetching exact Claudexor runtime seed ---"
Invoke-NativeChecked "Claudexor runtime seed fetch" {
    & "python-standalone\python.exe" scripts/fetch_claudexor_runtime.py --output-dir claudexor-runtime
}

if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }

$env:PYINSTALLER_CONFIG_DIR = Join-Path (Get-Location) ".pyinstaller-cache"
New-Item -ItemType Directory -Force -Path $env:PYINSTALLER_CONFIG_DIR | Out-Null

Write-Host "--- Installing Chromium for browser tools (bundled into python-standalone) ---"
$env:PLAYWRIGHT_BROWSERS_PATH = "0"
Invoke-NativeChecked "Chromium installation" {
    & "python-standalone\python.exe" -m playwright install --only-shell chromium
}

Write-Host "--- Installing WebKit for mobile-grade browser tools (bundled into python-standalone) ---"
Invoke-NativeChecked "WebKit installation" {
    & "python-standalone\python.exe" -m playwright install webkit
}

Write-Host "--- Pruning optional Chromium resources with long Windows paths ---"
$LocalBrowsers = "python-standalone\Lib\site-packages\playwright\driver\package\.local-browsers"
if (Test-Path $LocalBrowsers) {
    Get-ChildItem -Path $LocalBrowsers -Directory -Filter "chromium_headless_shell-*" | ForEach-Object {
        $ShellRoot = $_.FullName
        $OptionalPaths = @(
            "chrome-headless-shell-win64\PrivacySandboxAttestationsPreloaded",
            "chrome-headless-shell-win64\resources\accessibility\reading_mode_gdocs_helper",
            "chrome-headless-shell-win64\resources\accessibility\reading_mode_gdocs_helper_manifest.json"
        )
        foreach ($Rel in $OptionalPaths) {
            $Target = Join-Path $ShellRoot $Rel
            if (Test-Path $Target) {
                Remove-Item -Recurse -Force $Target
            }
        }
    }
}

Write-Host "--- Building embedded managed repo bundle ---"
Invoke-NativeChecked "Managed repo bundle build" {
    python scripts/build_repo_bundle.py --source-branch $ManagedSourceBranch
}

Write-Host "--- Running PyInstaller ---"
Invoke-NativeChecked "PyInstaller build" {
    python -m PyInstaller Ouroboros.spec --clean --noconfirm
}

Write-Host "--- Installing packaged CLI wrappers ---"
New-Item -ItemType Directory -Force -Path "dist\Ouroboros\bin" | Out-Null
Copy-Item "packaging\cli\ouroboros.cmd" "dist\Ouroboros\bin\ouroboros.cmd" -Force
Copy-Item "packaging\cli\install-ouroboros-cli.cmd" "dist\Ouroboros\bin\install-ouroboros-cli.cmd" -Force

# WA6 parity: precompile bytecode instead of deleting it. Windows has no codesign
# seal, so this is purely for start-speed + consistency with the macOS build (where
# precompiled+sealed .pyc keep the signature valid). --invalidation-mode
# unchecked-hash means a read-only payload never rewrites the .pyc at import. Runs
# before the path-length guard so the final archived payload is validated.
Write-Host "--- Precompiling Python bytecode in archive payload (start-speed parity) ---"
$AppEmbeddedPy = Get-ChildItem -Path "dist\Ouroboros" -Recurse -Force -File -Filter "python.exe" `
    | Where-Object { $_.FullName -match "python-standalone" } | Select-Object -First 1
if ($AppEmbeddedPy) {
    $EmbeddedPyPath = $AppEmbeddedPy.FullName
} else {
    $EmbeddedPyPath = (Resolve-Path "python-standalone\python.exe").Path
}
Write-Host "Using embedded interpreter for compileall: $EmbeddedPyPath"
$CompileTargets = @()
$StdlibTarget = Get-ChildItem -Path "dist\Ouroboros" -Recurse -Force -Directory -Filter "python-standalone" | Select-Object -First 1
if ($StdlibTarget) { $CompileTargets += $StdlibTarget.FullName }
$OuroborosTarget = Get-ChildItem -Path "dist\Ouroboros" -Recurse -Force -Directory -Filter "ouroboros" | Select-Object -First 1
if ($OuroborosTarget) { $CompileTargets += $OuroborosTarget.FullName }
if ($CompileTargets.Count -gt 0) {
    # Neutralize the build-time PYTHONDONTWRITEBYTECODE/PYTHONPYCACHEPREFIX for this
    # command only, else compileall writes no in-tree .pyc (start-speed parity).
    $SavedDWB = $env:PYTHONDONTWRITEBYTECODE; $SavedPCP = $env:PYTHONPYCACHEPREFIX
    Remove-Item Env:PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPYCACHEPREFIX -ErrorAction SilentlyContinue
    # compileall returns a non-zero exit if ANY bundled file fails to compile (the
    # python-standalone ships a known tab/space-broken Tcl/Tix WmDefault.py that
    # Ouroboros never imports). That single-file failure must NOT fail the build —
    # the rest of the tree is still sealed (start-speed parity with the POSIX
    # `|| true`). Neutralize Stop-on-native-error (pwsh 7.4+ ties native exit codes
    # to $ErrorActionPreference) for THIS call only, then reset $LASTEXITCODE.
    $PrevEAP = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try {
        & "$EmbeddedPyPath" -m compileall -q -f --invalidation-mode unchecked-hash @CompileTargets
    } catch {
        Write-Host "compileall reported non-fatal per-file failures (ignored): $_"
    }
    $ErrorActionPreference = $PrevEAP
    $global:LASTEXITCODE = 0
    if ($SavedDWB) { $env:PYTHONDONTWRITEBYTECODE = $SavedDWB }
    if ($SavedPCP) { $env:PYTHONPYCACHEPREFIX = $SavedPCP }
} else {
    Write-Host "WARNING: no compileall targets found in dist\Ouroboros (python-standalone / ouroboros)."
}

Write-Host "--- Checking Windows archive path lengths ---"
$TooLong = Get-ChildItem -Path "dist\Ouroboros" -Recurse -Force | Where-Object {
    $_.FullName.Substring((Resolve-Path "dist\Ouroboros").Path.Length).TrimStart('\').Length -gt 200
}
if ($TooLong) {
    $Sample = ($TooLong | Select-Object -First 10 | ForEach-Object { $_.FullName }) -join "`n"
    throw "Windows build contains paths longer than 200 chars under dist\Ouroboros:`n$Sample"
}

Write-Host ""
Write-Host "=== Creating archive ==="
Compress-Archive -Path "dist\Ouroboros" -DestinationPath "dist\$ArchiveName" -Force

Write-Host ""
Write-Host "=== Done ==="
Write-Host "Archive: dist\$ArchiveName"
Write-Host ""
Write-Host "To run: extract and execute Ouroboros\Ouroboros.exe"
Write-Host "To install CLI: Ouroboros\bin\install-ouroboros-cli.cmd"
