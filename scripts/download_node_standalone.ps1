# Stages the official Node.js runtime that ouroboros/claudexor_runtime_pin.json names, as
# node-standalone\node.exe, after checking the archive's exact size and SHA-256
# (scripts/download_node_standalone.py, shared with macOS/Linux and Docker).
# Run from repo root: powershell -ExecutionPolicy Bypass -File scripts/download_node_standalone.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $Root "python-standalone\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
& $Python (Join-Path $PSScriptRoot "download_node_standalone.py") @args
if ($LASTEXITCODE -ne 0) { throw "Node runtime staging failed ($LASTEXITCODE)" }
