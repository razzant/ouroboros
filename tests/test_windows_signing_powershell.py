"""Execute the release scripts with PowerShell; crypto/SDK calls are mocks.

Runs on any host with pwsh. These tests prove packing and script control flow,
not Windows trust, the vendor service, native SDK behavior or a live signature.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pytest
import yaml
from tests._cases import chosen_rows

ROOT = Path(__file__).resolve().parents[1]
PWSH = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(not PWSH, reason="PowerShell is required for script execution")
SIGNER = "AB" * 20
CATALOG = json.loads((ROOT / "scripts" / "windows_signing_catalog.json").read_text(encoding="utf-8"))
CATALOG_PATHS = [entry["path"] for entry in CATALOG["files"]]


def run_ps(tmp_path, body, **overrides):
    wrapper = tmp_path / "test.ps1"
    wrapper.write_text('$ErrorActionPreference = "Stop"\ntry {\n' + body +
                       '\n} catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }\n',
                       encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("ESIGNER_") and key not in ("RUNNER_TEMP", "GITHUB_OUTPUT")}
    env.update(TEST_REPO=str(ROOT), POWERSHELL_TELEMETRY_OPTOUT="1", POWERSHELL_UPDATECHECK="Off")
    env.update(overrides)
    return subprocess.run([PWSH, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(wrapper)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)


def test_powershell_scripts_and_workflow_parse(tmp_path):
    scripts = [ROOT / "build_windows.ps1", *[ROOT / "scripts" / name for name in (
        "download_inno_setup.ps1", "pack_windows_archive.ps1", "pack_windows_setup.ps1",
        "sign_windows_release.ps1", "smoke_windows_setup.ps1",
        "verify_inno_setup.ps1", "verify_windows_signature.ps1")]]
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    for job_id in ("windows-sign", "windows-proof"):
        for index, step in enumerate(jobs[job_id]["steps"]):
            if step.get("shell") == "pwsh":
                script = tmp_path / f"{job_id}-{index}.ps1"
                # Actions interpolates expressions before PowerShell parses.
                script.write_text(re.sub(r"\$\{\{.*?\}\}", "fixture-value", step["run"]))
                scripts.append(script)
    result = run_ps(tmp_path, r'''
foreach ($File in ($env:TEST_FILES | ConvertFrom-Json)) {
    $Tokens = $null; $Errors = $null
    $Ast = [System.Management.Automation.Language.Parser]::ParseFile($File, [ref]$Tokens, [ref]$Errors)
    if ($Errors.Count) { throw ($Errors | Out-String) }
    # Inspect commands, so mentioning Compress-Archive in an explanation is fine.
    $Bad = $Ast.FindAll({ param($Node)
        $Node -is [System.Management.Automation.Language.CommandAst] -and
        $Node.GetCommandName() -eq 'Compress-Archive'
    }, $true)
    if ($Bad.Count) { throw "Compress-Archive invocation in $File" }
}
''', TEST_FILES=json.dumps([str(script) for script in scripts]))
    assert result.returncode == 0, result.stdout + result.stderr


def test_packer_keeps_hidden_files_empty_directories_and_exact_bytes(tmp_path):
    payload = tmp_path / "payload with spaces" / "Ouroboros"
    browser = payload / "_internal" / ".local-browsers"
    browser.mkdir(parents=True)
    (browser / "empty").mkdir()
    files = {"Ouroboros.exe": b"mock executable\x00\xff", ".hidden": b"hidden",
             "_internal/.local-browsers/browser.dll": b"browser\x00",
             "_internal/spaces and unicode-é.txt": b"payload"}
    for name, content in files.items():
        (payload / name).write_bytes(content)
    archive = tmp_path / "output with spaces.zip"
    result = run_ps(tmp_path, r'''
if ($IsWindows) {
    (Get-Item -Force -LiteralPath (Join-Path $env:TEST_PAYLOAD '.hidden')).Attributes = 'Hidden'
    (Get-Item -Force -LiteralPath (Join-Path $env:TEST_PAYLOAD '_internal/.local-browsers')).Attributes = 'Hidden,Directory'
}
& "$env:TEST_REPO/scripts/pack_windows_archive.ps1" -PayloadRoot $env:TEST_PAYLOAD -Archive $env:TEST_ARCHIVE
''', TEST_PAYLOAD=str(payload), TEST_ARCHIVE=str(archive))
    assert result.returncode == 0, result.stdout + result.stderr
    with zipfile.ZipFile(archive) as packed:
        assert {name: packed.read("Ouroboros/" + name) for name in files} == files
        assert "Ouroboros/_internal/.local-browsers/empty/" in packed.namelist()
    original = archive.read_bytes()
    repeated = run_ps(tmp_path, r'''
& "$env:TEST_REPO/scripts/pack_windows_archive.ps1" -PayloadRoot $env:TEST_PAYLOAD -Archive $env:TEST_ARCHIVE
''', TEST_PAYLOAD=str(payload), TEST_ARCHIVE=str(archive))
    assert repeated.returncode != 0 and "already exists" in repeated.stderr
    assert archive.read_bytes() == original


MOCK_VERIFIER = r'''
function Get-AuthenticodeSignature {
    param($LiteralPath)
    if ($env:TEST_VERIFICATION_LOG) { Add-Content -LiteralPath $env:TEST_VERIFICATION_LOG -Value $LiteralPath }
    $Cert = [pscustomobject]@{Thumbprint = $env:TEST_ACTUAL}
    $Cert | Add-Member ScriptMethod GetNameInfo { param($Type, $Issuer); 'Example Publisher' }
    if ($env:TEST_STATUS -eq 'NoCertificate') { $Cert = $null }
    [pscustomobject]@{Status = $env:TEST_STATUS; SignerCertificate = $Cert; TimeStamperCertificate = $null}
}
function Get-Command {
    param($Name, $ErrorAction)
    if ($env:TEST_TOOL_EXIT -ne 'missing') { [pscustomobject]@{Source = 'Invoke-MockSignTool'} }
}
function Invoke-MockSignTool {
    $Leaked = @(Get-ChildItem Env:ESIGNER_* -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
    if ($Leaked.Count) { throw "signtool.exe inherited ESIGNER environment variables: $($Leaked -join ', ')" }
    @($args) | ConvertTo-Json | Set-Content -LiteralPath tool-arguments.json
    # Deliberately localized and misleading; only exit status decides.
    Write-Output 'Successfully verified. Проверка. Warnungen.'
    $global:LASTEXITCODE = [int]$env:TEST_TOOL_EXIT
}
${env:ProgramFiles(x86)} = (Get-Location).Path
'''


@pytest.mark.parametrize(("status", "actual", "tool_exit", "error"), [
    ("Valid", SIGNER, "0", None),
    ("HashMismatch", SIGNER, "0", "not valid"),
    ("NoCertificate", SIGNER, "0", "not valid"),
])
def test_verifier_uses_crypto_identity_and_native_exit_status(tmp_path, status, actual, tool_exit, error):
    executable = tmp_path / "Ouroboros.exe"
    executable.write_bytes(b"mock PE")
    result = run_ps(tmp_path, MOCK_VERIFIER + r'''
$Signer = & "$env:TEST_REPO/scripts/verify_windows_signature.ps1" -Executable './Ouroboros.exe' -ExpectedThumbprint $env:TEST_EXPECTED
$Signer | ConvertTo-Json | Set-Content -LiteralPath signer.json
''', TEST_ACTUAL=actual, TEST_EXPECTED=SIGNER.lower(), TEST_STATUS=status, TEST_TOOL_EXIT=tool_exit)
    if error:
        assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
        assert not (tmp_path / "signer.json").exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads((tmp_path / "signer.json").read_text())["Thumbprint"].upper() == SIGNER
    if (tmp_path / "tool-arguments.json").exists():
        assert json.loads((tmp_path / "tool-arguments.json").read_text()) == [
            "verify", "/pa", "/all", "/tw", "/v", "./Ouroboros.exe"]
    else:
        assert error


# Signature state follows the bytes: "signed:" files carry this run's
# signature (TEST_STATUS), "vendor-signed:" files a vendor's, others none. The
# two upstream-Python markers hash to the catalog pins, so a pinned file is
# accepted only with exactly that content and any other content is refused.
SIGNER_MOCKS = MOCK_VERIFIER + r"""
function Get-AuthenticodeSignature {
    param($LiteralPath)
    if ($env:TEST_VERIFICATION_LOG) { Add-Content -LiteralPath $env:TEST_VERIFICATION_LOG -Value $LiteralPath }
    $Text = [IO.File]::ReadAllText($LiteralPath)
    $Status = if ($Text.StartsWith('signed:')) { $env:TEST_STATUS } elseif ($Text.StartsWith('vendor-signed:')) { 'Valid' } else { 'NotSigned' }
    $Cert = [pscustomobject]@{Thumbprint = $env:TEST_ACTUAL}
    $Cert | Add-Member ScriptMethod GetNameInfo { param($Type, $Issuer); 'Example Publisher' }
    [pscustomobject]@{Status = $Status; SignerCertificate = $Cert; TimeStamperCertificate = $null}
}
function Get-FileHash {
    param($LiteralPath, $Algorithm)
    if ($LiteralPath.EndsWith('CodeSignTool-v1.3.2-windows.zip')) {
        $Hash = if ($env:TEST_MODE -eq 'digest') { '0' * 64 } else {
            '4afc32e8b7f79bbe1de7e4e7049aaad4e0f754357613b9bbec0e3052f06fd36b'
        }
        return [pscustomobject]@{Hash=$Hash.ToUpperInvariant()}
    }
    $Pins = $env:TEST_PINS | ConvertFrom-Json
    $Text = [IO.File]::ReadAllText($LiteralPath)
    if ($Pins.PSObject.Properties.Name -contains $Text) { return [pscustomobject]@{Hash=$Pins.$Text.ToUpperInvariant()} }
    Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $LiteralPath -Algorithm $Algorithm
}
"""
UPSTREAM = {path: f"upstream {path.rsplit('/', 1)[1]}".encode() for path in CATALOG_PATHS[1:]}
UNSIGNED_PAYLOAD = {
    CATALOG_PATHS[0]: b"unsigned mock PE",
    **UPSTREAM,
    "Ouroboros/_internal/VERSION": b"7.6.1-rc.1\n",
    "Ouroboros/bin/ouroboros.cmd": b"@echo off\r\n",
    "Ouroboros/_internal/.local-browsers/hidden": b"browser",
    # Look-alikes outside the catalog: venv launcher templates and a DLL its
    # vendor signed stay byte for byte as built.
    "Ouroboros/_internal/python-standalone/Lib/venv/scripts/nt/python.exe": b"venv launcher template",
    "Ouroboros/_internal/python-standalone/Lib/venv/scripts/nt/pythonw.exe": b"venv windowed template",
    "Ouroboros/_internal/vcruntime140.dll": b"vendor-signed: runtime DLL",
}
ARCHIVE = Path("dist") / "Ouroboros-7.6.1-rc.1-windows-x64.zip"
INSTALLER = Path("dist") / "Ouroboros-Setup-7.6.1-rc.1-windows-x64.exe"
MOCK_SERVICE = r"""
$TestRoot = (Get-Location).Path
function Invoke-WebRequest {
    param($Uri, $OutFile)
    Add-Content -LiteralPath (Join-Path $TestRoot 'network-calls') -Value $Uri
    Copy-Item -LiteralPath (Join-Path $TestRoot 'vendor.zip') -Destination $OutFile
}
function Assert-NoEsignerEnv([string]$Caller) {
    $Leaked = @(Get-ChildItem Env:ESIGNER_* -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
    if ($Leaked.Count) { throw "$Caller inherited ESIGNER environment variables: $($Leaked -join ', ')" }
}
function Invoke-MockDownloadInno {
    param([string]$Destination)
    Assert-NoEsignerEnv 'download_inno_setup.ps1'
    Add-Content -LiteralPath (Join-Path $TestRoot 'inno-download-calls') -Value $Destination
    if ($env:TEST_MODE -eq 'inno-download-failure') { throw 'Pinned Inno Setup digest mismatch' }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $Iscc = Join-Path $Destination 'ISCC.exe'
    Set-Content -LiteralPath $Iscc -Value 'mock-iscc' -NoNewline
    return $Iscc
}
function Invoke-MockPackSetup {
    param([string]$PayloadRoot, [string]$OutputDir, [string]$Compiler)
    Assert-NoEsignerEnv 'pack_windows_setup.ps1'
    Add-Content -LiteralPath (Join-Path $TestRoot 'setup-pack-calls') -Value "$PayloadRoot|$Compiler"
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $Version = (Get-Content -Raw -LiteralPath (Join-Path $PayloadRoot '_internal/VERSION')).Trim()
    $Built = Join-Path $OutputDir "Ouroboros-Setup-$Version-windows-x64.exe"
    $Launcher = [IO.File]::ReadAllText((Join-Path $PayloadRoot 'Ouroboros.exe'))
    $Prefix = if ($env:TEST_MODE -eq 'setup-presigned') { 'vendor-signed:' } else { 'unsigned-setup:' }
    [IO.File]::WriteAllText($Built, $Prefix + $Launcher)
}
function Invoke-MockJava {
    Assert-NoEsignerEnv 'java.exe'
    $Source = @($args | Where-Object { $_ -like '-input_file_path=*' })[0].Substring(17)
    Add-Content -LiteralPath (Join-Path $TestRoot 'java-calls') -Value $Source
    $Call = @(Get-Content -LiteralPath (Join-Path $TestRoot 'java-calls')).Count
    Write-Output 'SYNTHETIC_VENDOR_SECRET_MUST_NOT_REACH_LOG'
    $global:LASTEXITCODE = 0
    $Signature = if ($env:TEST_MODE -eq 'invalid-signature' -and $Call -eq 3) { 'forged:' } else { 'signed:' }
    if ($Call -eq [int]$env:TEST_FAIL_AT) {
        if ($env:TEST_MODE -eq 'start-failure') { throw 'SYNTHETIC_VENDOR_SECRET_MUST_NOT_REACH_LOG' }
        if ($env:TEST_MODE -eq 'vendor-failure') { $global:LASTEXITCODE = 1; return }
        if ($env:TEST_MODE -eq 'no-output') { return }
    }
    $Output = @($args | Where-Object { $_ -like '-output_dir_path=*' })[0].Substring(17)
    [IO.File]::WriteAllText((Join-Path $Output (Split-Path -Leaf $Source)), $Signature + [IO.File]::ReadAllText($Source))
}
Set-Alias -Name (Join-Path $env:JAVA_HOME_11_X64 'bin/java.exe') -Value Invoke-MockJava
Set-Alias -Name (Join-Path $env:TEST_REPO 'scripts/download_inno_setup.ps1') -Value Invoke-MockDownloadInno
Set-Alias -Name (Join-Path $env:TEST_REPO 'scripts/pack_windows_setup.ps1') -Value Invoke-MockPackSetup
& "$env:TEST_REPO/scripts/sign_windows_release.ps1" -UnsignedArchive (Join-Path $TestRoot 'unsigned.zip')
"""


def _run_signer(tmp_path, payload=UNSIGNED_PAYLOAD, **overrides):
    (tmp_path / "VERSION").write_text("7.6.1-rc.1\n")
    java = tmp_path / "java11" / "bin" / "java.exe"
    java.parent.mkdir(parents=True)
    java.touch()
    with zipfile.ZipFile(tmp_path / "unsigned.zip", "w") as archive:
        for name, content in payload.items():
            archive.writestr(name, content)
    with zipfile.ZipFile(tmp_path / "vendor.zip", "w") as archive:
        archive.writestr("vendor/jar/code_sign_tool-test.jar", b"mock JAR, never executed")
    env = {name: "synthetic-test-value" for name in (
        "ESIGNER_USERNAME", "ESIGNER_PASSWORD", "ESIGNER_CREDENTIAL_ID", "ESIGNER_TOTP_SECRET")}
    env.update(ESIGNER_CERT_SHA1=SIGNER, GITHUB_EVENT_NAME="push", GITHUB_REF="refs/tags/v7.6.1-rc.1",
               RUNNER_TEMP=str(tmp_path), JAVA_HOME_11_X64=str(java.parent.parent),
               GITHUB_OUTPUT=str(tmp_path / "github-output"), TEST_MODE="success", TEST_FAIL_AT="0",
               TEST_ACTUAL=SIGNER, TEST_TOOL_EXIT="0", TEST_STATUS="Valid",
               TEST_VERIFICATION_LOG=str(tmp_path / "signature-checks"),
               TEST_PINS=json.dumps({UPSTREAM[entry["path"]].decode(): entry["upstreamSha256"]
                                     for entry in CATALOG["files"][1:]}))
    env.update(overrides)
    return run_ps(tmp_path, SIGNER_MOCKS + MOCK_SERVICE, **env)


def _lines(path):
    return path.read_text().splitlines() if path.exists() else []


@pytest.mark.parametrize(("overrides", "error"), [
    ({"ESIGNER_CERT_SHA1": ""}, "Missing Windows signing configuration"),
    ({"ESIGNER_CERT_SHA1": " "}, "not a 40-hex"),
    ({"GITHUB_EVENT_NAME": "workflow_dispatch"}, "requires a pushed release tag"),
])
def test_signer_refuses_broken_configuration_before_network(tmp_path, overrides, error):
    (tmp_path / "VERSION").write_text("7.6.1-rc.1\n")
    env = {name: "synthetic-test-value" for name in (
        "ESIGNER_USERNAME", "ESIGNER_PASSWORD", "ESIGNER_CREDENTIAL_ID", "ESIGNER_TOTP_SECRET")}
    env.update(ESIGNER_CERT_SHA1=SIGNER, GITHUB_EVENT_NAME="push", GITHUB_REF="refs/tags/v7.6.1-rc.1",
               RUNNER_TEMP=str(tmp_path), JAVA_HOME_11_X64="", GITHUB_OUTPUT=str(tmp_path / "out"))
    env.update(overrides)
    result = run_ps(tmp_path, r'''
function Invoke-WebRequest { throw 'UNEXPECTED_NETWORK_CALL' }
& "$env:TEST_REPO/scripts/sign_windows_release.ps1" -UnsignedArchive './missing.zip'
''', **env)
    assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
    assert "UNEXPECTED_NETWORK_CALL" not in result.stdout + result.stderr
    assert not (tmp_path / "dist").exists()


def test_signer_signs_exactly_the_catalog_and_installer_and_reports_both_hashes(tmp_path):
    result = _run_signer(tmp_path)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "SYNTHETIC_VENDOR_SECRET_MUST_NOT_REACH_LOG" not in output
    # Four billed requests: the 3 catalogued payload files, then the compiled outer Setup .exe.
    java_calls = [line.replace("\\", "/") for line in _lines(tmp_path / "java-calls")]
    assert [line.split("/p/", 1)[1] for line in java_calls[:3]] == CATALOG_PATHS
    assert java_calls[3].endswith("/setup-unsigned/" + INSTALLER.name)
    assert len(java_calls) == 4
    with zipfile.ZipFile(tmp_path / ARCHIVE) as packed:
        files = {info.filename: packed.read(info) for info in packed.infolist() if not info.is_dir()}
    assert set(files) == set(UNSIGNED_PAYLOAD)
    for name, content in UNSIGNED_PAYLOAD.items():
        assert files[name] == (b"signed:" + content if name in CATALOG_PATHS else content), name
    # The compiled Setup .exe was built from the already-signed Ouroboros.exe, then outer-signed.
    expected_unsigned_setup = b"unsigned-setup:signed:" + UNSIGNED_PAYLOAD[CATALOG_PATHS[0]]
    expected_signed_setup = b"signed:" + expected_unsigned_setup
    assert (tmp_path / INSTALLER).read_bytes() == expected_signed_setup
    # Preflight + returned + copied for 3 payload files (9), plus preflight + returned + copied + published for Setup (4) = 13.
    assert len(_lines(tmp_path / "signature-checks")) == 3 * len(CATALOG_PATHS) + 4
    assert json.loads((tmp_path / "tool-arguments.json").read_text())[:3] == ["verify", "/pa", "/all"]
    outputs = dict(line.split("=", 1) for line in _lines(tmp_path / "github-output"))
    record = outputs["signed_files"]
    installer_record = outputs["signed_installer"]
    pins = {entry["path"]: entry.get("upstreamSha256") for entry in CATALOG["files"]}
    assert json.loads(record) == [{
        "path": path,
        "unsignedSha256": pins[path] or hashlib.sha256(UNSIGNED_PAYLOAD[path]).hexdigest(),
        "sha256": hashlib.sha256(b"signed:" + UNSIGNED_PAYLOAD[path]).hexdigest(),
    } for path in CATALOG_PATHS]
    assert json.loads(installer_record) == {
        "name": INSTALLER.name,
        "unsignedSha256": hashlib.sha256(expected_unsigned_setup).hexdigest(),
        "sha256": hashlib.sha256(expected_signed_setup).hexdigest(),
    }
    # The secret-free proof's receipt writer accepts both records for their respective final artifacts.
    receipt = tmp_path / "receipt.json"
    smoke = subprocess.run(
        [sys.executable, "-S", str(ROOT / "scripts" / "release_proof.py"), "record-smoke",
         "--proof-id", "windows-x64", "--artifact", str(tmp_path / ARCHIVE), "--output", str(receipt),
         "--commit", "a" * 40, "--tag", "v7.6.1-rc.1", "--check", "packaged_cli_help",
         "--authenticode-thumbprint", SIGNER, "--authenticode-publisher", "Example Publisher",
         "--authenticode-files", record],
        capture_output=True, text=True, check=False)
    assert smoke.returncode == 0, smoke.stderr
    assert json.loads(receipt.read_text())["authenticode"]["files"] == json.loads(record)
    setup_receipt = tmp_path / "setup-receipt.json"
    setup_smoke = subprocess.run(
        [sys.executable, "-S", str(ROOT / "scripts" / "release_proof.py"), "record-smoke",
         "--proof-id", "windows-setup-x64", "--artifact", str(tmp_path / INSTALLER), "--output", str(setup_receipt),
         "--commit", "a" * 40, "--tag", "v7.6.1-rc.1", "--check", "packaged_cli_help",
         "--authenticode-thumbprint", SIGNER, "--authenticode-publisher", "Example Publisher",
         "--authenticode-installer", installer_record],
        capture_output=True, text=True, check=False)
    assert setup_smoke.returncode == 0, setup_smoke.stderr
    assert json.loads(setup_receipt.read_text())["authenticode"]["installer"] == json.loads(installer_record)
    assert not list(tmp_path.glob("es-*"))


@pytest.mark.parametrize(("change", "error"), chosen_rows([
    *[({path: None}, f"has no {path}") for path in CATALOG_PATHS],
    *[({path: b"rebuilt " + UPSTREAM[path]}, "not the catalogued upstream") for path in CATALOG_PATHS[1:]],
    ({CATALOG_PATHS[0]: b"vendor-signed: launcher"}, "already has an Authenticode signature state (Valid)"),
    ({CATALOG_PATHS[2]: b"vendor-signed: pythonw"}, "already has an Authenticode signature state (Valid)"),
    ({"Ouroboros/_internal/VERSION": b"0.0.0\n"}, "does not match checkout VERSION"),
    ({"Ouroboros/stray.txt": b"stray"}, "must hold exactly"),
], 0, 3, 6, 7, 8))
def test_signer_refuses_wrong_catalog_input_before_any_request(tmp_path, change, error):
    payload = {name: content for name, content in {**UNSIGNED_PAYLOAD, **change}.items() if content is not None}
    result = _run_signer(tmp_path, payload)
    assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
    assert not (tmp_path / "network-calls").exists() and not (tmp_path / "java-calls").exists()
    assert not (tmp_path / ARCHIVE).exists() and not (tmp_path / INSTALLER).exists()
    assert not (tmp_path / "github-output").exists()


@pytest.mark.parametrize(("overrides", "error", "requests"), chosen_rows([
    ({"TEST_MODE": "digest"}, "not the pinned", 0),
    ({"TEST_MODE": "inno-download-failure"}, "Pinned Inno Setup digest mismatch", 0),
    *[({"TEST_MODE": "vendor-failure", "TEST_FAIL_AT": str(n)},
       f"sign exited 1 at signing request {n} of 4 ({CATALOG_PATHS[n - 1]})", n) for n in (1, 2, 3)],
    ({"TEST_MODE": "vendor-failure", "TEST_FAIL_AT": "4"},
     f"sign exited 1 at signing request 4 of 4 ({INSTALLER.name})", 4),
    ({"TEST_MODE": "start-failure", "TEST_FAIL_AT": "2"},
     "could not be started with the runner Java 11 at signing request 2", 2),
    ({"TEST_MODE": "no-output", "TEST_FAIL_AT": "4"}, "wrote no signed file at signing request 4", 4),
    ({"TEST_MODE": "setup-presigned"}, "already has an Authenticode signature state (Valid)", 3),
    ({"TEST_MODE": "invalid-signature"}, "not valid", 3),
    ({"TEST_ACTUAL": "F" * 40}, "not the configured certificate", 1),
    ({"TEST_TOOL_EXIT": "2"}, "warnings", 1),
    ({"TEST_TOOL_EXIT": "1"}, "rejected the signature", 1),
], 0, 1, 4, 5, 8, 12))
def test_signer_failure_stops_requests_and_ships_no_artifacts(tmp_path, overrides, error, requests):
    result = _run_signer(tmp_path, **overrides)
    output = result.stdout + result.stderr
    assert result.returncode != 0 and error in output, output
    assert "SYNTHETIC_VENDOR_SECRET_MUST_NOT_REACH_LOG" not in output
    # Never retried: a failure stops at the request that failed, leaving neither ZIP nor Setup in dist\.
    assert len(_lines(tmp_path / "java-calls")) == requests
    assert not (tmp_path / ARCHIVE).exists() and not (tmp_path / INSTALLER).exists()
    assert not (tmp_path / "github-output").exists()
    assert not list(tmp_path.glob("es-*"))  # vendor/signed scratch removed on every outcome


def test_pack_windows_setup_validates_payload_and_output_guards(tmp_path):
    payload = tmp_path / "Ouroboros"
    (payload / "_internal").mkdir(parents=True)
    (payload / "bin").mkdir()
    (payload / "Ouroboros.exe").write_bytes(b"launcher")
    (payload / "_internal" / "VERSION").write_text("7.6.1-rc.1\n", encoding="utf-8")
    (payload / "bin" / "ouroboros.cmd").write_bytes(b"@echo off\r\n")
    out_inside = payload / "out"
    res = run_ps(tmp_path, r'''
& "$env:TEST_REPO/scripts/pack_windows_setup.ps1" -PayloadRoot $env:TEST_PAYLOAD -OutputDir $env:TEST_OUT
''', TEST_PAYLOAD=str(payload), TEST_OUT=str(out_inside))
    assert res.returncode != 0 and "outside the payload" in res.stderr
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "existing.txt").write_text("keep", encoding="utf-8")
    res = run_ps(tmp_path, r'''
& "$env:TEST_REPO/scripts/pack_windows_setup.ps1" -PayloadRoot $env:TEST_PAYLOAD -OutputDir $env:TEST_OUT
''', TEST_PAYLOAD=str(payload), TEST_OUT=str(nonempty))
    assert res.returncode != 0 and "new or empty folder" in res.stderr
    (payload / "stray.txt").write_text("stray", encoding="utf-8")
    res = run_ps(tmp_path, r'''
& "$env:TEST_REPO/scripts/pack_windows_setup.ps1" -PayloadRoot $env:TEST_PAYLOAD -OutputDir $env:TEST_OUT
''', TEST_PAYLOAD=str(payload), TEST_OUT=str(tmp_path / "fresh"))
    assert res.returncode != 0 and "not a collected Ouroboros payload" in res.stderr


INNO_HASH_MOCK = r'''
function Get-FileHash {
    param([string]$LiteralPath, [string]$Algorithm)
    $Pins = $env:TEST_PINS | ConvertFrom-Json
    if ($LiteralPath -and (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        $Text = [IO.File]::ReadAllText($LiteralPath)
        if ($Pins.PSObject.Properties.Name -contains $Text) {
            return [pscustomobject]@{ Hash = $Pins.$Text.ToUpperInvariant() }
        }
    }
    Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $LiteralPath -Algorithm $Algorithm
}
'''


def _inno_pins():
    pin = json.loads((ROOT / "packaging" / "windows" / "inno_setup_pin.json").read_text(encoding="utf-8"))
    return pin, json.dumps({
        "pinned-installer": pin["sha256"],
        "pinned-iscc": pin["compilerSha256"]["ISCC.exe"],
        "pinned-iscmplr": pin["compilerSha256"]["ISCmplr.dll"],
    })


def test_download_inno_setup_verifies_pinned_pair_without_executing(tmp_path):
    pin, pins_json = _inno_pins()
    body = INNO_HASH_MOCK + r'''
$TestRoot = (Get-Location).Path
function Invoke-WebRequest {
    param($Uri, $OutFile)
    Add-Content -LiteralPath (Join-Path $TestRoot 'network-calls') -Value $Uri
    [IO.File]::WriteAllText($OutFile, $env:TEST_INSTALLER_BODY)
}
function Get-Item {
    param([string]$LiteralPath)
    if ($LiteralPath -and $LiteralPath.EndsWith($env:TEST_INSTALLER)) {
        return [pscustomobject]@{ Length = [int64]$env:TEST_SIZE }
    }
    Microsoft.PowerShell.Management\Get-Item -LiteralPath $LiteralPath
}
function Start-Process {
    param($FilePath, [switch]$Wait, [switch]$PassThru, [string[]]$ArgumentList)
    Add-Content -LiteralPath (Join-Path $TestRoot 'installer-calls') -Value ($ArgumentList -join ' ')
    $LogArg = @($ArgumentList | Where-Object { $_ -like '/LOG=*' })[0]
    if ($LogArg) { Set-Content -LiteralPath $LogArg.Substring(5).Trim('"') -Value 'SYNTHETIC_INNO_INSTALL_LOG' }
    $DirPath = (@($ArgumentList | Where-Object { $_ -like '/DIR=*' })[0]).Substring(5).Trim('"')
    New-Item -ItemType Directory -Force -Path $DirPath | Out-Null
    if ($env:TEST_ISCC_BODY) { [IO.File]::WriteAllText((Join-Path $DirPath 'ISCC.exe'), $env:TEST_ISCC_BODY) }
    if ($env:TEST_DLL_BODY) { [IO.File]::WriteAllText((Join-Path $DirPath 'ISCmplr.dll'), $env:TEST_DLL_BODY) }
    [pscustomobject]@{ ExitCode = 0 }
}
$Returned = @(& "$env:TEST_REPO/scripts/download_inno_setup.ps1" -Destination $env:TEST_DEST)
ConvertTo-Json -InputObject $Returned | Set-Content -LiteralPath (Join-Path $TestRoot 'returned.json')
'''
    env = dict(
        RUNNER_TEMP=str(tmp_path),
        TEST_PINS=pins_json,
        TEST_INSTALLER=pin["installer"],
        TEST_SIZE=str(pin["sizeBytes"]),
        TEST_INSTALLER_BODY="pinned-installer",
        TEST_ISCC_BODY="pinned-iscc",
        TEST_DLL_BODY="pinned-iscmplr",
    )

    # Fresh staging verifies outer installer and staged (ISCC.exe, ISCmplr.dll) pair without executing ISCC.
    dest = tmp_path / "inno-fresh"
    res = run_ps(tmp_path, body, TEST_DEST=str(dest), **env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert json.loads((tmp_path / "returned.json").read_text()) == [str(dest / "ISCC.exe")]
    assert not list(tmp_path.glob("is-*"))

    # Cached pinned pair is reused without network or installer calls.
    for marker in ("network-calls", "installer-calls", "returned.json"):
        (tmp_path / marker).unlink()
    res = run_ps(tmp_path, body, TEST_DEST=str(dest), **env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert not (tmp_path / "network-calls").exists() and not (tmp_path / "installer-calls").exists()
    assert json.loads((tmp_path / "returned.json").read_text()) == [str(dest / "ISCC.exe")]

    # Cached mismatching companion DLL or CLI is refused with its actual SHA-256 and no network call.
    for filename, bad_bytes, good_bytes in (
        ("ISCmplr.dll", b"wrong-dll", b"pinned-iscmplr"),
        ("ISCC.exe", b"wrong-iscc", b"pinned-iscc"),
    ):
        (dest / filename).write_bytes(bad_bytes)
        res = run_ps(tmp_path, body, TEST_DEST=str(dest), **env)
        assert res.returncode != 0 and filename in res.stderr, res.stdout + res.stderr
        assert hashlib.sha256(bad_bytes).hexdigest() in res.stderr
        assert not (tmp_path / "network-calls").exists()
        (dest / filename).write_bytes(good_bytes)

    # Outer installer digest mismatch stops before Start-Process; post-install pair mismatch prints install.log.
    res = run_ps(tmp_path, body, TEST_DEST=str(tmp_path / "bad-outer"),
                 **{**env, "TEST_INSTALLER_BODY": "wrong-installer"})
    assert res.returncode != 0 and "not the pinned" in res.stderr, res.stdout + res.stderr
    assert not (tmp_path / "installer-calls").exists()
    res = run_ps(tmp_path, body, TEST_DEST=str(tmp_path / "bad-staged-dll"),
                 **{**env, "TEST_DLL_BODY": "wrong-dll"})
    assert res.returncode != 0 and hashlib.sha256(b"wrong-dll").hexdigest() in res.stderr, res.stdout + res.stderr
    assert "SYNTHETIC_INNO_INSTALL_LOG" in res.stdout and not list(tmp_path.glob("is-*"))


def test_pack_windows_setup_verifies_pinned_pair_before_packing(tmp_path):
    _, pins_json = _inno_pins()
    payload = tmp_path / "Ouroboros"
    (payload / "_internal").mkdir(parents=True)
    (payload / "bin").mkdir()
    (payload / "Ouroboros.exe").write_bytes(b"launcher")
    (payload / "_internal" / "VERSION").write_text("7.6.1-rc.1\n", encoding="utf-8")
    (payload / "bin" / "ouroboros.cmd").write_bytes(b"@echo off\r\n")
    empty_env = tmp_path / "empty-env"
    empty_env.mkdir()

    def stage_pair(name, iscc_bytes, dll_bytes):
        folder = tmp_path / f"inno-{name}"
        folder.mkdir()
        iscc = folder / "ISCC.exe"
        iscc.write_bytes(iscc_bytes)
        if dll_bytes is not None:
            (folder / "ISCmplr.dll").write_bytes(dll_bytes)
        return iscc

    good = stage_pair("good", b"pinned-iscc", b"pinned-iscmplr")
    wrong_cli = stage_pair("wrong-cli", b"wrong-iscc", b"pinned-iscmplr")
    wrong_dll = stage_pair("wrong-dll", b"pinned-iscc", b"wrong-dll")
    missing_dll = stage_pair("missing-dll", b"pinned-iscc", None)

    # Only the verified compiler path is aliased for the final compile step; unverified candidates stay inert.
    mock_pack = INNO_HASH_MOCK + r'''
$TestRoot = (Get-Location).Path
function Invoke-MockIscc {
    Add-Content -LiteralPath (Join-Path $TestRoot 'iscc-calls') -Value (ConvertTo-Json -InputObject @($args) -Compress)
    $OutArg = @($args | Where-Object { $_ -like '/O*' })[0].Substring(2)
    $FileArg = @($args | Where-Object { $_ -like '/F*' })[0].Substring(2)
    [IO.File]::WriteAllText((Join-Path $OutArg "$FileArg.exe"), 'mock-compiled-setup')
    $global:LASTEXITCODE = 0
}
Set-Alias -Name (Resolve-Path -LiteralPath $env:TEST_GOOD_ISCC).ProviderPath -Value Invoke-MockIscc
'''
    env = {
        "TEST_PINS": pins_json,
        "TEST_PAYLOAD": str(payload),
        "TEST_GOOD_ISCC": str(good),
        "TEST_WRONG_CLI": str(wrong_cli),
        "TEST_WRONG_DLL": str(wrong_dll),
        "TEST_MISSING_DLL": str(missing_dll),
        "LOCALAPPDATA": str(empty_env),
        "ProgramFiles": str(empty_env),
        "ProgramFiles(x86)": str(empty_env),
    }

    # Explicit -Compiler refuses mismatching CLI, mismatching companion DLL, and missing companion before packing.
    for key, expected in (
        ("TEST_WRONG_CLI", hashlib.sha256(b"wrong-iscc").hexdigest()),
        ("TEST_WRONG_DLL", hashlib.sha256(b"wrong-dll").hexdigest()),
        ("TEST_MISSING_DLL", "companion ISCmplr.dll is missing"),
    ):
        out = tmp_path / f"out-{key}"
        res = run_ps(tmp_path, mock_pack + f'''
& "$env:TEST_REPO/scripts/pack_windows_setup.ps1" -PayloadRoot $env:TEST_PAYLOAD -OutputDir $env:TEST_OUT -Compiler $env:{key}
''', TEST_OUT=str(out), **env)
        assert res.returncode != 0 and expected in res.stderr, res.stdout + res.stderr
        assert not list(out.glob("*.exe")) and not (tmp_path / "iscc-calls").exists()

    # Explicit -Compiler accepts the pinned pair and invokes ISCC once for packing.
    out_good = tmp_path / "out-good"
    res = run_ps(tmp_path, mock_pack + r'''
& "$env:TEST_REPO/scripts/pack_windows_setup.ps1" -PayloadRoot $env:TEST_PAYLOAD -OutputDir $env:TEST_OUT -Compiler $env:TEST_GOOD_ISCC
''', TEST_OUT=str(out_good), **env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (out_good / "Ouroboros-Setup-7.6.1-rc.1-windows-x64.exe").read_bytes() == b"mock-compiled-setup"
    calls = _lines(tmp_path / "iscc-calls")
    assert len(calls) == 1 and json.loads(calls[0])[0] == "/Qp"
    (tmp_path / "iscc-calls").unlink()

    # Discovery skips unverified candidates without executing them, packs with the pinned pair, and reports rejections if none match.
    discovery_mock = mock_pack + r'''
function Get-Command {
    param($Name, [switch]$All, $ErrorAction)
    if ($Name -eq 'ISCC.exe') {
        $Items = @($env:TEST_CANDIDATES | ConvertFrom-Json)
        return @(foreach ($Item in $Items) { [pscustomobject]@{ Source = $Item } })
    }
    Microsoft.PowerShell.Core\Get-Command @PSBoundParameters
}
& "$env:TEST_REPO/scripts/pack_windows_setup.ps1" -PayloadRoot $env:TEST_PAYLOAD -OutputDir $env:TEST_OUT
'''
    out_auto = tmp_path / "out-auto"
    res = run_ps(tmp_path, discovery_mock, TEST_OUT=str(out_auto),
                 TEST_CANDIDATES=json.dumps([str(p) for p in (wrong_cli, wrong_dll, missing_dll, good)]), **env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (out_auto / "Ouroboros-Setup-7.6.1-rc.1-windows-x64.exe").read_bytes() == b"mock-compiled-setup"
    assert len(_lines(tmp_path / "iscc-calls")) == 1
    (tmp_path / "iscc-calls").unlink()

    out_none = tmp_path / "out-none"
    res = run_ps(tmp_path, discovery_mock, TEST_OUT=str(out_none), TEST_CANDIDATES=json.dumps([str(wrong_dll)]), **env)
    assert res.returncode != 0 and hashlib.sha256(b"wrong-dll").hexdigest() in res.stderr, res.stdout + res.stderr
    assert not list(out_none.glob("*.exe")) and not (tmp_path / "iscc-calls").exists()


def test_pack_windows_setup_refuses_hidden_entries_its_zip_would_keep(tmp_path):
    """Inno's wildcard Source skips Hidden entries while the ZIP keeps them, so a direct build from such a tree is
    refused upfront, naming the extracted-ZIP remedy, before any compiler is looked for or output written."""
    payload = tmp_path / "dist" / "Ouroboros"
    browsers = payload / "_internal" / ".local-browsers"
    browsers.mkdir(parents=True)
    (payload / "bin").mkdir()
    (payload / "Ouroboros.exe").write_bytes(b"launcher")
    (payload / "_internal" / "VERSION").write_text("7.6.1-rc.1\n", encoding="utf-8")
    (browsers / "browser.dll").write_bytes(b"browser")
    out = tmp_path / "setup-out"
    res = run_ps(tmp_path, r'''
if ($IsWindows) {
    (Get-Item -Force -LiteralPath (Join-Path $env:TEST_PAYLOAD '_internal/.local-browsers')).Attributes = 'Hidden,Directory'
}
& "$env:TEST_REPO/scripts/pack_windows_setup.ps1" -PayloadRoot $env:TEST_PAYLOAD -OutputDir $env:TEST_OUT
''', TEST_PAYLOAD=str(payload), TEST_OUT=str(out))
    assert res.returncode != 0 and "leave out Hidden entries" in res.stderr, res.stdout + res.stderr
    assert ".local-browsers" in res.stderr and "extracted ZIP" in res.stderr
    assert not out.exists(), "refused before any output folder"


def test_smoke_windows_setup_refuses_non_disposable_host(tmp_path):
    setup = tmp_path / "Ouroboros-Setup-7.6.1-rc.1-windows-x64.exe"
    archive = tmp_path / "Ouroboros-7.6.1-rc.1-windows-x64.zip"
    setup.write_bytes(b"mock setup")
    archive.write_bytes(b"mock zip")
    for overrides in (
        {"GITHUB_ACTIONS": "", "RUNNER_ENVIRONMENT": "", "RUNNER_TEMP": ""},
        {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "self-hosted", "RUNNER_TEMP": str(tmp_path)},
        {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "", "RUNNER_TEMP": str(tmp_path)},
        {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_TEMP": ""},
    ):
        res = run_ps(tmp_path, r'''
& "$env:TEST_REPO/scripts/smoke_windows_setup.ps1" -Installer $env:TEST_SETUP -Archive $env:TEST_ARCHIVE
''', TEST_SETUP=str(setup), TEST_ARCHIVE=str(archive), **overrides)
        assert res.returncode != 0 and "-AllowDisposableProfile" in res.stderr, (overrides, res.stdout + res.stderr)

    # Evaluate the real guard function from smoke_windows_setup.ps1 across hosted/self-hosted/OS combinations.
    guard = run_ps(tmp_path, r'''
$Tokens = $null; $Errors = $null
$Ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $env:TEST_REPO 'scripts/smoke_windows_setup.ps1'), [ref]$Tokens, [ref]$Errors)
if ($Errors.Count) { throw ($Errors | Out-String) }
$Fn = $Ast.Find({ param($Node)
    $Node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $Node.Name -eq 'Test-DisposableWindowsCiRunner'
}, $true)
if (-not $Fn) { throw 'Test-DisposableWindowsCiRunner not found' }
Invoke-Expression $Fn.Extent.Text
$Cases = @(
    @{ OnWin = $true;  Act = 'true';  Env = 'github-hosted'; Tmp = '/tmp'; Expect = $true }
    @{ OnWin = $true;  Act = 'true';  Env = 'self-hosted';   Tmp = '/tmp'; Expect = $false }
    @{ OnWin = $true;  Act = 'true';  Env = '';              Tmp = '/tmp'; Expect = $false }
    @{ OnWin = $true;  Act = 'true';  Env = 'github-hosted'; Tmp = '';     Expect = $false }
    @{ OnWin = $true;  Act = 'false'; Env = 'github-hosted'; Tmp = '/tmp'; Expect = $false }
    @{ OnWin = $false; Act = 'true';  Env = 'github-hosted'; Tmp = '/tmp'; Expect = $false }
)
foreach ($Case in $Cases) {
    $env:GITHUB_ACTIONS = $Case.Act
    $env:RUNNER_ENVIRONMENT = $Case.Env
    $env:RUNNER_TEMP = $Case.Tmp
    $Actual = Test-DisposableWindowsCiRunner -OnWindows $Case.OnWin
    if ($Actual -ne $Case.Expect) {
        throw "Guard returned $Actual (expected $($Case.Expect)) for $($Case | ConvertTo-Json -Compress)"
    }
}
''')
    assert guard.returncode == 0, guard.stdout + guard.stderr


def test_smoke_windows_setup_start_exact_process_preserves_spaces_and_handle(tmp_path):
    spaced_dir = tmp_path / "dir with spaces"
    spaced_dir.mkdir()
    ready = spaced_dir / "pythonw busy.ready"
    res = run_ps(tmp_path, r'''
$Tokens = $null; $Errors = $null
$Path = Join-Path $env:TEST_REPO 'scripts/smoke_windows_setup.ps1'
$Ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$Tokens, [ref]$Errors)
if ($Errors.Count) { throw ($Errors | Out-String) }
$Fn = $Ast.Find({ param($Node)
    $Node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $Node.Name -eq 'Start-ExactProcess'
}, $true)
if (-not $Fn) { throw 'Start-ExactProcess not found' }
Invoke-Expression $Fn.Extent.Text
$Text = Get-Content -Raw -LiteralPath $Path
$Match = [regex]::Match($Text, '"(import pathlib,sys,time;[^"]+)"')
if (-not $Match.Success) { throw 'Busy-probe -c snippet not found in smoke_windows_setup.ps1' }
$Code = $Match.Groups[1].Value
$Proc = Start-ExactProcess $env:TEST_PYTHON @('-I', '-B', '-c', $Code, $env:TEST_READY)
try {
    for ($Tick = 0; $Tick -lt 100; $Tick++) {
        if (Test-Path -LiteralPath $env:TEST_READY -PathType Leaf) { break }
        if ($Proc.HasExited) { throw "Exact-argv probe exited early ($($Proc.ExitCode))" }
        Start-Sleep -Milliseconds 100
    }
    if (-not (Test-Path -LiteralPath $env:TEST_READY -PathType Leaf)) { throw 'Ready marker was not written' }
    if ((Get-Content -Raw -LiteralPath $env:TEST_READY) -ne 'up') { throw 'Unexpected ready marker content' }
    if ($Proc.HasExited) { throw 'Probe process exited unexpectedly before Kill' }
    $Proc.Kill()
    if (-not $Proc.WaitForExit(10000)) { throw 'Probe process did not exit after Kill' }
    if (-not $Proc.HasExited) { throw 'Probe handle did not report HasExited after WaitForExit' }
} finally {
    if ($Proc) {
        if (-not $Proc.HasExited) {
            try { $Proc.Kill(); $Proc.WaitForExit(10000) | Out-Null } catch { }
        }
        try { $Proc.Dispose() } catch { }
    }
}
''', TEST_PYTHON=sys.executable, TEST_READY=str(ready))
    assert res.returncode == 0, res.stdout + res.stderr
    assert ready.read_text(encoding="utf-8") == "up"


def test_verifier_checks_every_catalogued_file_of_an_extracted_release(tmp_path):
    for path in CATALOG_PATHS:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(b"signed:" + path.encode())
    body = SIGNER_MOCKS + r'''
$Signed = @(& "$env:TEST_REPO/scripts/verify_windows_signature.ps1" -ExtractedRoot $env:TEST_ROOT -ExpectedThumbprint $env:TEST_EXPECTED)
ConvertTo-Json -InputObject @($Signed | ForEach-Object { [pscustomobject]@{Path = $_.Path; Sha256 = $_.Sha256} }) |
    Set-Content -LiteralPath signed.json
'''
    env = dict(TEST_ROOT=str(tmp_path), TEST_EXPECTED=SIGNER, TEST_ACTUAL=SIGNER, TEST_STATUS="Valid",
               TEST_TOOL_EXIT="0", TEST_PINS="{}")
    result = run_ps(tmp_path, body, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "signed.json").read_text()) == [
        {"Path": path, "Sha256": hashlib.sha256(b"signed:" + path.encode()).hexdigest().upper()}
        for path in CATALOG_PATHS]
    (tmp_path / "signed.json").unlink()
    (tmp_path / CATALOG_PATHS[2]).write_bytes(b"unsigned pythonw")
    refused = run_ps(tmp_path, body, **env)
    assert refused.returncode != 0 and "not valid: NotSigned" in refused.stderr
    (tmp_path / CATALOG_PATHS[2]).unlink()
    missing = run_ps(tmp_path, body, **env)
    assert missing.returncode != 0 and "Windows executable missing" in missing.stderr
    assert not (tmp_path / "signed.json").exists()
