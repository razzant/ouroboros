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
        "pack_windows_archive.ps1", "sign_windows_release.ps1", "verify_windows_signature.ps1")]]
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
    @($args) | ConvertTo-Json | Set-Content -LiteralPath tool-arguments.json
    # Deliberately localized and misleading; only exit status decides.
    Write-Output 'Successfully verified. Проверка. Warnungen.'
    $global:LASTEXITCODE = [int]$env:TEST_TOOL_EXIT
}
${env:ProgramFiles(x86)} = (Get-Location).Path
'''


@pytest.mark.parametrize(("status", "actual", "tool_exit", "error"), [
    ("Valid", SIGNER, "0", None),
    ("Valid", SIGNER.lower(), "0", None),
    ("Valid", SIGNER, "1", "rejected the signature"),
    ("Valid", SIGNER, "2", "warnings"),
    ("Valid", SIGNER, "missing", "SDK signtool.exe is required"),
    ("HashMismatch", SIGNER, "0", "not valid"),
    ("NoCertificate", SIGNER, "0", "not valid"),
    ("Valid", "F" * 40, "0", "not the configured certificate"),
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
    "Ouroboros/_internal/.local-browsers/hidden": b"browser",
    # Look-alikes outside the catalog: venv launcher templates and a DLL its
    # vendor signed stay byte for byte as built.
    "Ouroboros/_internal/python-standalone/Lib/venv/scripts/nt/python.exe": b"venv launcher template",
    "Ouroboros/_internal/python-standalone/Lib/venv/scripts/nt/pythonw.exe": b"venv windowed template",
    "Ouroboros/_internal/vcruntime140.dll": b"vendor-signed: runtime DLL",
}
ARCHIVE = Path("dist") / "Ouroboros-7.6.1-rc.1-windows-x64.zip"
MOCK_SERVICE = r"""
$TestRoot = (Get-Location).Path
function Invoke-WebRequest {
    param($Uri, $OutFile)
    Add-Content -LiteralPath (Join-Path $TestRoot 'network-calls') -Value $Uri
    Copy-Item -LiteralPath (Join-Path $TestRoot 'vendor.zip') -Destination $OutFile
}
function Invoke-MockJava {
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
    ({"ESIGNER_PASSWORD": ""}, "Missing Windows signing configuration"),
    ({"ESIGNER_CERT_SHA1": " "}, "not a 40-hex"),
    ({"ESIGNER_CERT_SHA1": SIGNER + "\n"}, "not a 40-hex"),
    ({"GITHUB_EVENT_NAME": "workflow_dispatch"}, "requires a pushed release tag"),
    ({"GITHUB_REF": "refs/tags/v1.0.0"}, "must match VERSION exactly"),
    ({"GITHUB_OUTPUT": ""}, "requires a step output file"),
    ({"JAVA_HOME_11_X64": ""}, "no maintained Java 11"),
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


def test_signer_signs_exactly_the_catalog_and_reports_both_hashes(tmp_path):
    result = _run_signer(tmp_path)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "SYNTHETIC_VENDOR_SECRET_MUST_NOT_REACH_LOG" not in output
    # One billed request per catalogued file, each for that file alone.
    requested = [line.replace("\\", "/").split("/p/", 1)[1] for line in _lines(tmp_path / "java-calls")]
    assert requested == CATALOG_PATHS
    with zipfile.ZipFile(tmp_path / ARCHIVE) as packed:
        files = {info.filename: packed.read(info) for info in packed.infolist() if not info.is_dir()}
    assert set(files) == set(UNSIGNED_PAYLOAD)
    for name, content in UNSIGNED_PAYLOAD.items():
        assert files[name] == (b"signed:" + content if name in CATALOG_PATHS else content), name
    # Preflight, then each returned and each copied file: three reads per file.
    assert len(_lines(tmp_path / "signature-checks")) == 3 * len(CATALOG_PATHS)
    assert json.loads((tmp_path / "tool-arguments.json").read_text())[:3] == ["verify", "/pa", "/all"]
    (line,) = [line for line in _lines(tmp_path / "github-output") if line.startswith("signed_files=")]
    record = line.split("=", 1)[1]
    pins = {entry["path"]: entry.get("upstreamSha256") for entry in CATALOG["files"]}
    assert json.loads(record) == [{
        "path": path,
        "unsignedSha256": pins[path] or hashlib.sha256(UNSIGNED_PAYLOAD[path]).hexdigest(),
        "sha256": hashlib.sha256(b"signed:" + UNSIGNED_PAYLOAD[path]).hexdigest(),
    } for path in CATALOG_PATHS]
    # The secret-free proof's receipt writer accepts this record for this ZIP.
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
    assert not list(tmp_path.glob("es-*"))


@pytest.mark.parametrize(("change", "error"), [
    *[({path: None}, f"has no {path}") for path in CATALOG_PATHS],
    *[({path: b"rebuilt " + UPSTREAM[path]}, "not the catalogued upstream") for path in CATALOG_PATHS[1:]],
    ({CATALOG_PATHS[0]: b"vendor-signed: launcher"}, "already has an Authenticode signature state (Valid)"),
    ({CATALOG_PATHS[2]: b"vendor-signed: pythonw"}, "already has an Authenticode signature state (Valid)"),
])
def test_signer_refuses_wrong_catalog_input_before_any_request(tmp_path, change, error):
    payload = {name: content for name, content in {**UNSIGNED_PAYLOAD, **change}.items() if content is not None}
    result = _run_signer(tmp_path, payload)
    assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
    assert not (tmp_path / "network-calls").exists() and not (tmp_path / "java-calls").exists()
    assert not (tmp_path / ARCHIVE).exists() and not (tmp_path / "github-output").exists()


@pytest.mark.parametrize(("overrides", "error", "requests"), [
    ({"TEST_MODE": "digest"}, "not the pinned", 0),
    *[({"TEST_MODE": "vendor-failure", "TEST_FAIL_AT": str(n)},
       f"sign exited 1 at signing request {n} of 3 ({CATALOG_PATHS[n - 1]})", n) for n in (1, 2, 3)],
    ({"TEST_MODE": "start-failure", "TEST_FAIL_AT": "2"},
     "could not be started with the runner Java 11 at signing request 2", 2),
    ({"TEST_MODE": "no-output", "TEST_FAIL_AT": "3"}, "wrote no signed file at signing request 3", 3),
    ({"TEST_MODE": "invalid-signature"}, "not valid", 3),
    ({"TEST_ACTUAL": "F" * 40}, "not the configured certificate", 1),
    ({"TEST_TOOL_EXIT": "2"}, "warnings", 1),
    ({"TEST_TOOL_EXIT": "1"}, "rejected the signature", 1),
])
def test_signer_failure_stops_requests_and_ships_no_zip(tmp_path, overrides, error, requests):
    result = _run_signer(tmp_path, **overrides)
    output = result.stdout + result.stderr
    assert result.returncode != 0 and error in output, output
    assert "SYNTHETIC_VENDOR_SECRET_MUST_NOT_REACH_LOG" not in output
    # Never retried: a failure stops at the request that failed.
    assert len(_lines(tmp_path / "java-calls")) == requests
    assert not (tmp_path / ARCHIVE).exists() and not (tmp_path / "github-output").exists()
    assert not list(tmp_path.glob("es-*"))  # vendor/signed scratch removed on every outcome


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
