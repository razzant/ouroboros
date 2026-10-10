"""Execute the release scripts with PowerShell; crypto/SDK calls are mocks.

Runs on any host with pwsh. These tests prove packing and script control flow,
not Windows trust, the vendor service, native SDK behavior or a live signature.
"""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import zipfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PWSH = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(not PWSH, reason="PowerShell is required for script execution")
SIGNER = "AB" * 20


def run_ps(tmp_path, body, **overrides):
    wrapper = tmp_path / "test.ps1"
    wrapper.write_text('$ErrorActionPreference = "Stop"\ntry {\n' + body +
                       '\n} catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 }\n',
                       encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("ESIGNER_") and key != "RUNNER_TEMP"}
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
            "verify", "/pa", "/tw", "/v", "./Ouroboros.exe"]
    else:
        assert error


@pytest.mark.parametrize(("overrides", "error"), [
    ({"ESIGNER_CERT_SHA1": ""}, "Missing Windows signing configuration"),
    ({"ESIGNER_PASSWORD": ""}, "Missing Windows signing configuration"),
    ({"ESIGNER_CERT_SHA1": " "}, "not a 40-hex"),
    ({"ESIGNER_CERT_SHA1": SIGNER + "\n"}, "not a 40-hex"),
    ({"GITHUB_EVENT_NAME": "workflow_dispatch"}, "requires a pushed release tag"),
    ({"GITHUB_REF": "refs/tags/v1.0.0"}, "must match VERSION exactly"),
    ({"JAVA_HOME_11_X64": ""}, "no maintained Java 11"),
])
def test_signer_refuses_broken_configuration_before_network(tmp_path, overrides, error):
    (tmp_path / "VERSION").write_text("7.6.1-rc.1\n")
    env = {name: "synthetic-test-value" for name in (
        "ESIGNER_USERNAME", "ESIGNER_PASSWORD", "ESIGNER_CREDENTIAL_ID", "ESIGNER_TOTP_SECRET")}
    env.update(ESIGNER_CERT_SHA1=SIGNER, GITHUB_EVENT_NAME="push", GITHUB_REF="refs/tags/v7.6.1-rc.1",
               RUNNER_TEMP=str(tmp_path), JAVA_HOME_11_X64="")
    env.update(overrides)
    result = run_ps(tmp_path, r'''
function Invoke-WebRequest { throw 'UNEXPECTED_NETWORK_CALL' }
& "$env:TEST_REPO/scripts/sign_windows_release.ps1" -UnsignedArchive './missing.zip'
''', **env)
    assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
    assert "UNEXPECTED_NETWORK_CALL" not in result.stdout + result.stderr
    assert not (tmp_path / "dist").exists()


@pytest.mark.parametrize(("mode", "error"), [
    ("success", None),
    ("digest", "not the pinned"),
    ("vendor-failure", "sign exited 1"),
    ("start-failure", "could not be started with the runner Java 11"),
    ("no-output", "wrote no signed Ouroboros.exe"),
    ("invalid-signature", "not valid"),
])
def test_signer_mock_service_repackages_only_verified_payload(tmp_path, mode, error):
    (tmp_path / "VERSION").write_text("7.6.1-rc.1\n")
    java = tmp_path / "java11" / "bin" / "java.exe"
    java.parent.mkdir(parents=True)
    java.touch()
    with zipfile.ZipFile(tmp_path / "unsigned.zip", "w") as archive:
        archive.writestr("Ouroboros/Ouroboros.exe", b"unsigned mock PE")
        archive.writestr("Ouroboros/_internal/.local-browsers/hidden", b"browser")
    with zipfile.ZipFile(tmp_path / "vendor.zip", "w") as archive:
        archive.writestr("vendor/jar/code_sign_tool-test.jar", b"mock JAR, never executed")
    env = {name: "synthetic-test-value" for name in (
        "ESIGNER_USERNAME", "ESIGNER_PASSWORD", "ESIGNER_CREDENTIAL_ID", "ESIGNER_TOTP_SECRET")}
    env.update(ESIGNER_CERT_SHA1=SIGNER, GITHUB_EVENT_NAME="push", GITHUB_REF="refs/tags/v7.6.1-rc.1",
               RUNNER_TEMP=str(tmp_path), JAVA_HOME_11_X64=str(java.parent.parent),
               TEST_MODE=mode, TEST_ACTUAL=SIGNER, TEST_TOOL_EXIT="0",
               TEST_VERIFICATION_LOG=str(tmp_path / "signature-checks"),
               TEST_STATUS="HashMismatch" if mode == "invalid-signature" else "Valid")
    result = run_ps(tmp_path, MOCK_VERIFIER + r'''
$TestRoot = (Get-Location).Path
function Invoke-WebRequest {
    param($Uri, $OutFile)
    Copy-Item -LiteralPath (Join-Path $TestRoot 'vendor.zip') -Destination $OutFile
}
function Get-FileHash {
    param($LiteralPath, $Algorithm)
    if ($LiteralPath.EndsWith('CodeSignTool-v1.3.2-windows.zip')) {
        $Hash = if ($env:TEST_MODE -eq 'digest') { '0' * 64 } else {
            '4afc32e8b7f79bbe1de7e4e7049aaad4e0f754357613b9bbec0e3052f06fd36b'
        }
        return [pscustomobject]@{Hash=$Hash}
    }
    Microsoft.PowerShell.Utility\Get-FileHash -LiteralPath $LiteralPath -Algorithm $Algorithm
}
function Invoke-MockJava {
    Set-Content -LiteralPath (Join-Path $TestRoot 'java-called') -Value 'called'
    Write-Output 'SYNTHETIC_VENDOR_SECRET_MUST_NOT_REACH_LOG'
    if ($env:TEST_MODE -eq 'start-failure') { throw 'SYNTHETIC_VENDOR_SECRET_MUST_NOT_REACH_LOG' }
    $global:LASTEXITCODE = 0
    if ($env:TEST_MODE -eq 'vendor-failure') { $global:LASTEXITCODE = 1; return }
    if ($env:TEST_MODE -eq 'no-output') { return }
    $Output = @($args | Where-Object { $_ -like '-output_dir_path=*' })[0].Substring(17)
    [IO.File]::WriteAllBytes((Join-Path $Output 'Ouroboros.exe'), [Text.Encoding]::UTF8.GetBytes('signed mock PE'))
}
Set-Alias -Name (Join-Path $env:JAVA_HOME_11_X64 'bin/java.exe') -Value Invoke-MockJava
& "$env:TEST_REPO/scripts/sign_windows_release.ps1" -UnsignedArchive (Join-Path $TestRoot 'unsigned.zip')
''', **env)
    output = result.stdout + result.stderr
    assert "SYNTHETIC_VENDOR_SECRET_MUST_NOT_REACH_LOG" not in output
    archive = tmp_path / "dist" / "Ouroboros-7.6.1-rc.1-windows-x64.zip"
    if error:
        assert result.returncode != 0 and error in output, output
        assert not archive.exists()
    else:
        assert result.returncode == 0, output
        with zipfile.ZipFile(archive) as packed:
            assert packed.read("Ouroboros/Ouroboros.exe") == b"signed mock PE"
            assert packed.read("Ouroboros/_internal/.local-browsers/hidden") == b"browser"
        assert len((tmp_path / "signature-checks").read_text().splitlines()) == 2
    assert (tmp_path / "java-called").exists() == (mode != "digest")
    assert not list(tmp_path.glob("es-*"))  # vendor/signed scratch removed on every outcome
