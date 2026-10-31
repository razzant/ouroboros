# Downloads python-build-standalone for Windows (x86_64)
# Run from repo root: powershell -ExecutionPolicy Bypass -File scripts/download_python_standalone.ps1
# CI test jobs take the same pinned interpreter without the packaged dependencies:
# scripts/download_python_standalone.ps1 -Dest <dir> -InterpreterOnly
param(
    [string]$Dest = "python-standalone",
    [switch]$InterpreterOnly
)

$ErrorActionPreference = "Stop"

$Release = "20261009"
$PyVersion = "3.10.22"
$Platform = "x86_64-pc-windows-msvc"

# Pinned SHA256 (from the release's SHA256SUMS): a swapped/truncated archive
# fails here instead of becoming the packaged runtime. Update the pin when
# bumping $Release/$PyVersion, exactly as in download_python_standalone.sh, and
# the release signing catalog's python.exe/pythonw.exe hashes with it
# (windows_signing_catalog.json).
$Sha256 = "f20288c45ff11e7fe026f40ddcfb2acfe695cec18860746639e289142a589745"

$Filename = "cpython-${PyVersion}+${Release}-${Platform}-install_only_stripped.tar.gz"
$Url = "https://github.com/astral-sh/python-build-standalone/releases/download/${Release}/${Filename}"

Write-Host "=== Downloading Python ${PyVersion} for ${Platform} ==="
Write-Host "URL: ${Url}"

if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
if (Test-Path "_python_tmp") { Remove-Item -Recurse -Force "_python_tmp" }
New-Item -ItemType Directory -Path "_python_tmp" | Out-Null

$ArchivePath = "_python_tmp\python.tar.gz"
Write-Host "Downloading..."
for ($Attempt = 1; ; $Attempt++) {
    try {
        Invoke-WebRequest -Uri $Url -OutFile $ArchivePath -UseBasicParsing
        break
    } catch {
        if ($Attempt -ge 5) { throw }
        Write-Warning "Download attempt ${Attempt} failed: $($_.Exception.Message); retrying in 5 s"
        Start-Sleep -Seconds 5
    }
}

Write-Host "Verifying SHA256..."
$Actual = (Get-FileHash -Algorithm SHA256 $ArchivePath).Hash.ToLower()
if ($Actual -ne $Sha256.ToLower()) {
    throw "SHA256 mismatch for ${Filename}: expected ${Sha256}, got ${Actual} - refusing to install."
}
Write-Host "Checksum OK: ${Actual}"

Write-Host "Extracting..."
tar -xzf $ArchivePath -C "_python_tmp"

Move-Item "_python_tmp\python" $Dest
Remove-Item -Recurse -Force "_python_tmp"

if ($InterpreterOnly) {
    Write-Host "Python: ${Dest}\python.exe (interpreter only)"
    & "${Dest}\python.exe" --version
    exit 0
}

echo ""
Write-Host "=== Installing agent dependencies ==="
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required for locked dependency installation"
}
uv pip install --python "${Dest}\python.exe" --quiet -r requirements-runtime.lock
if ($LASTEXITCODE -ne 0) {
    throw "Agent dependency installation failed with exit code $LASTEXITCODE"
}

echo ""
Write-Host "=== Installing optional: local model support ==="
try {
    uv pip install --python "${Dest}\python.exe" --quiet "llama-cpp-python[server]" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "llama-cpp-python installation failed with exit code $LASTEXITCODE"
    }
    Write-Host "llama-cpp-python installed successfully"
} catch {
    Write-Warning "llama-cpp-python install failed - local model support will not be available"
}

echo ""
Write-Host "=== Done ==="
Write-Host "Python: ${Dest}\python.exe"
& "${Dest}\python.exe" --version
