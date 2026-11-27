"""Workflow and script contract for configuration-selected Windows release signing.

ESIGNER_CERT_SHA1 unset: no job binds the signing Environment and the build's
ZIP is proven and published as explicitly unsigned. Set: signing is required and
any signing failure blocks the whole release. These checks read the parsed
workflow and the scripts; they are not a Windows signature, a run of the
PowerShell scripts, or a test of the protected Environment's settings.
"""
from fnmatch import fnmatch
from itertools import product
from pathlib import Path
import json
import re

import yaml
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGURED = "vars.ESIGNER_CERT_SHA1 != ''"
UNCONFIGURED = "vars.ESIGNER_CERT_SHA1 == ''"
SECRETS = ("ESIGNER_USERNAME", "ESIGNER_PASSWORD", "ESIGNER_CREDENTIAL_ID", "ESIGNER_TOTP_SECRET")


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def workflow_jobs():
    return yaml.safe_load(source(".github/workflows/ci.yml"))["jobs"]


def step(job, name):
    return next(item for item in job["steps"] if item.get("name") == name)


def step_index(job, name):
    return next(i for i, item in enumerate(job["steps"]) if item.get("name") == name)


def uploads(job):
    # Reusable-workflow jobs have uses/with rather than a steps list.
    return [item for item in job.get("steps", []) if str(item.get("uses", "")).startswith("actions/upload-artifact@")]


def test_build_always_hands_over_the_unsigned_zip_with_its_digest():
    build = workflow_jobs()["build"]
    assert "environment" not in build
    assert "secrets.ESIGNER_" not in yaml.safe_dump(build)
    assert build["outputs"]["windows_archive_sha256"] == (
        "${{ matrix.os == 'windows-latest' && steps.release_asset.outputs.sha256 || '' }}"
    )
    windows_build = step(build, "Build Windows executable")
    assert windows_build["run"] == ".\\build_windows.ps1" and "env" not in windows_build
    locate = step(build, "Locate final release archive")
    assert "if" not in locate and locate["id"] == "release_asset"
    handoff = step(build, "Transfer unsigned Windows archive")
    assert handoff["if"] == "matrix.os == 'windows-latest'"
    assert handoff["with"]["name"] == "windows-unsigned-archive"
    assert handoff["with"]["path"] == "${{ steps.release_asset.outputs.path }}"
    assert handoff["with"]["overwrite"] is True
    assert step_index(build, "Locate final release archive") < step_index(build, "Transfer unsigned Windows archive")
    # The Windows release asset itself comes only from windows-proof.
    for name in ("Upload build artifact", "Record packaged artifact smoke", "Attest build provenance"):
        assert step(build, name)["if"] == "matrix.os != 'windows-latest'"


def test_signing_job_runs_only_when_configured_and_alone_holds_the_secrets():
    jobs = workflow_jobs()
    signing = jobs["windows-sign"]
    assert CONFIGURED in signing["if"]
    assert "needs.release-preflight.result == 'success'" in signing["if"]
    assert "needs.build.result == 'success'" in signing["if"]
    assert signing["environment"] == "windows-release-signing"
    assert signing["permissions"] == {"contents": "read", "artifact-metadata": "write"}
    assert signing["outputs"] == {
        "archive_sha256": "${{ steps.release_asset.outputs.sha256 }}",
        "installer_sha256": "${{ steps.installer_asset.outputs.sha256 }}",
        "signed_files": "${{ steps.sign.outputs.signed_files }}",
        "signed_installer": "${{ steps.sign.outputs.signed_installer }}",
    }
    # The Environment and its secrets reach this one step of this one job.
    bound = [name for name, job in jobs.items() if "environment" in job]
    assert bound == ["windows-sign"]
    sign = step(signing, "Sign, verify and package Windows executables")
    assert sign["id"] == "sign"
    assert {key: sign["env"][key] for key in SECRETS} == {
        key: f"${{{{ secrets.{key} }}}}" for key in SECRETS
    }
    assert sign["env"]["ESIGNER_CERT_SHA1"] == "${{ vars.ESIGNER_CERT_SHA1 }}"
    for name, job in jobs.items():
        for item in job.get("steps", []):
            if item is not sign:
                assert "secrets.ESIGNER_" not in yaml.safe_dump(item), (name, item.get("name"))
    # The unsigned ZIP is checked against the build's digest before extraction.
    check = step(signing, "Check unsigned archive digest before extraction")
    assert check["env"]["EXPECTED_SHA256"] == "${{ needs.build.outputs.windows_archive_sha256 }}"
    assert "--directory unsigned --expect-sha256 \"$EXPECTED_SHA256\"" in check["run"]
    stage_inno = step(signing, "Stage pinned Inno Setup compiler")
    assert stage_inno["id"] == "inno_setup"
    assert "env" not in stage_inno and "ESIGNER_" not in yaml.safe_dump(stage_inno)
    assert "download_inno_setup.ps1" in stage_inno["run"]
    assert sign["env"]["UNSIGNED_ARCHIVE"] == f"${{{{ steps.{check['id']}.outputs.path }}}}"
    assert sign["env"]["INNO_COMPILER"] == "${{ steps.inno_setup.outputs.compiler }}"
    assert "-InnoCompiler $env:INNO_COMPILER" in sign["run"]
    locate_setup = step(signing, "Locate final Windows installer")
    assert locate_setup["id"] == "installer_asset"
    assert "--directory dist --suffix .exe --github-output" in locate_setup["run"]
    order = [step_index(signing, name) for name in (
        "Download unsigned Windows archive (never a release asset)",
        "Check unsigned archive digest before extraction",
        "Stage pinned Inno Setup compiler",
        "Sign, verify and package Windows executables",
        "Locate final release archive",
        "Locate final Windows installer",
        "Transfer signed Windows archive (not yet release proof)",
    )]
    assert order == sorted(order)
    (handoff,) = uploads(signing)
    assert handoff["with"]["name"] == "windows-signed-archive"
    assert {line.strip() for line in handoff["with"]["path"].splitlines() if line.strip()} == {
        "${{ steps.release_asset.outputs.path }}",
        "${{ steps.installer_asset.outputs.path }}",
    }
    assert handoff["with"]["overwrite"] is True


def test_proof_job_proves_both_modes_without_secrets():
    jobs = workflow_jobs()
    proof = jobs["windows-proof"]
    assert "environment" not in proof
    assert "secrets." not in yaml.safe_dump(proof)
    assert set(proof["needs"]) == {"build", "windows-sign"}
    assert "needs.build.result == 'success'" in proof["if"]
    assert f"(({UNCONFIGURED} && github.repository != 'razzant/ouroboros') || needs.windows-sign.result == 'success')" in proof["if"]
    assert proof["env"]["WINDOWS_SIGNER_THUMBPRINT"] == "${{ vars.ESIGNER_CERT_SHA1 }}"
    download = step(proof, "Download final Windows archive")
    assert download["with"]["name"] == (
        f"${{{{ {CONFIGURED} && 'windows-signed-archive' || 'windows-unsigned-archive' }}}}"
    )
    check = step(proof, "Check final archive digest before extraction")
    assert check["env"] == {
        "SIGNED_SHA256": "${{ needs.windows-sign.outputs.archive_sha256 }}",
        "UNSIGNED_SHA256": "${{ needs.build.outputs.windows_archive_sha256 }}",
    }
    assert '--expect-sha256 "$EXPECTED_SHA256"' in check["run"]
    check_setup = step(proof, "Check final installer digest before execution")
    assert check_setup["if"] == CONFIGURED
    assert check_setup["env"] == {
        "SIGNED_INSTALLER_SHA256": "${{ needs.windows-sign.outputs.installer_sha256 }}",
    }
    assert '--suffix .exe --expect-sha256 "$SIGNED_INSTALLER_SHA256"' in check_setup["run"]
    smoke = step(proof, "Smoke final Windows archive")
    assert step_index(proof, check["name"]) < step_index(proof, smoke["name"])
    assert smoke["run"].index("Expand-Archive") < smoke["run"].index("verify_windows_signature.ps1")
    assert "if ($env:WINDOWS_SIGNER_THUMBPRINT)" in smoke["run"]
    # Every catalogued executable, not just the launcher.
    assert "verify_windows_signature.ps1 -ExtractedRoot $SmokeRoot" in smoke["run"]
    # The windowless interpreter runs isolated, reports through a file and is
    # waited for with a bound, so a stuck GUI process cannot hang the proof.
    probe = smoke["run"][smoke["run"].index("python-standalone\\pythonw.exe"):]
    assert "-I -B -c" in probe and "WaitForExit(60000)" in probe and "$Process.Kill()" in probe
    assert "$Process.ExitCode -ne 0" in probe and "Get-Content -Raw -LiteralPath $Probe" in probe
    record = step(proof, "Record packaged artifact smoke")
    assert record["env"] == {
        "SIGNER_THUMBPRINT": "${{ steps.smoke_windows.outputs.signer_thumbprint }}",
        "SIGNER_PUBLISHER": "${{ steps.smoke_windows.outputs.signer_publisher }}",
        "SIGNED_FILES": "${{ needs.windows-sign.outputs.signed_files }}",
    }
    assert '--authenticode-files "$SIGNED_FILES"' in record["run"]
    for check_name in ("authenticode_signer", "timestamp", "signed_payload_archive_match"):
        assert f"--check {check_name}" in record["run"]

    # The archive smoke's expanded copy goes once its SBOM is written, before the installer build and smoke
    # expand theirs: only that folder, and a part still in use stays rather than failing the release.
    release = step(proof, "Release the archive smoke's expanded copy")
    sbom = "Generate CycloneDX SBOM from final Windows payload"
    assert step_index(proof, sbom) < step_index(proof, release["name"]) < step_index(proof, "Build unsigned Windows installer")
    assert release["env"] == {"EXPANDED": "${{ steps.smoke_windows.outputs.sbom_path }}"}
    assert '$SmokeRoot = Join-Path $env:RUNNER_TEMP "ouroboros-windows-$env:GITHUB_RUN_ATTEMPT"' in smoke["run"]
    assert '$Expanded = Join-Path $env:RUNNER_TEMP "ouroboros-windows-$env:GITHUB_RUN_ATTEMPT"' in release["run"]
    assert 'if ($env:EXPANDED -ne $Expanded) { throw' in release["run"]
    assert "Remove-Item -LiteralPath $Expanded -Recurse -Force -ErrorAction SilentlyContinue" in release["run"]

    # One pinned compiler per proof job, staged secret-free in both modes: the unsigned pack and the
    # installer smoke's two-version update (step 4b) use the same one.
    inno = step(proof, "Stage pinned Inno Setup compiler")
    assert "if" not in inno and "env" not in inno
    assert "download_inno_setup.ps1" in inno["run"] and 'Value "compiler=$Iscc"' in inno["run"]
    unsigned_setup = step(proof, "Build unsigned Windows installer")
    assert unsigned_setup["if"] == f"{UNCONFIGURED} && github.repository != 'razzant/ouroboros'"
    assert step_index(proof, inno["name"]) < step_index(proof, unsigned_setup["name"])
    assert "download_inno_setup.ps1" not in unsigned_setup["run"]
    assert '-Compiler "${{ steps.inno_setup.outputs.compiler }}"' in unsigned_setup["run"]
    assert "pack_windows_setup.ps1" in unsigned_setup["run"]
    assert "locate --directory dist --suffix .exe" in unsigned_setup["run"]

    smoke_setup = step(proof, "Smoke final Windows installer")
    assert step_index(proof, check_setup["name"]) < step_index(proof, smoke_setup["name"])
    assert step_index(proof, unsigned_setup["name"]) < step_index(proof, smoke_setup["name"])
    assert step_index(proof, "Install digest-pinned Syft") < step_index(proof, smoke_setup["name"])
    assert smoke_setup["env"]["INSTALLER_PATH"] == (
        "${{ steps.signed_installer_asset.outputs.path || steps.unsigned_installer_asset.outputs.path }}"
    )
    assert smoke_setup["env"]["INNO_COMPILER"] == "${{ steps.inno_setup.outputs.compiler }}"
    # Workflow delegates the heavy Windows installer smoke to the standalone script, with no inline C#.
    assert "smoke_windows_setup.ps1" in smoke_setup["run"]
    assert "SHGetPropertyStoreFromParsingName" not in smoke_setup["run"]
    for expected_arg in (
        "-Installer $env:INSTALLER_PATH",
        "-Archive $env:ARCHIVE_PATH",
        "-ExpectedThumbprint $env:WINDOWS_SIGNER_THUMBPRINT",
        "-ZipSignerThumbprint $env:ZIP_SIGNER_THUMBPRINT",
        "-ZipSignerPublisher $env:ZIP_SIGNER_PUBLISHER",
        "-SyftPath $env:SYFT_PATH",
        "-SbomOutput 'dist/sbom-windows-setup-x64.cdx.json'",
        "-GitHubOutput $env:GITHUB_OUTPUT",
        "-Compiler $env:INNO_COMPILER",
    ):
        assert expected_arg in smoke_setup["run"], expected_arg

    record_setup = step(proof, "Record Windows installer smoke")
    assert record_setup["env"] == {
        "INSTALLER_PATH": "${{ steps.signed_installer_asset.outputs.path || steps.unsigned_installer_asset.outputs.path }}",
        "SIGNER_THUMBPRINT": "${{ steps.smoke_windows_setup.outputs.signer_thumbprint }}",
        "SIGNER_PUBLISHER": "${{ steps.smoke_windows_setup.outputs.signer_publisher }}",
        "SIGNED_INSTALLER": "${{ needs.windows-sign.outputs.signed_installer }}",
    }
    assert '--proof-id windows-setup-x64' in record_setup["run"]
    assert '--authenticode-installer "$SIGNED_INSTALLER"' in record_setup["run"]
    assert "--check installer_busy_upgrade_refused" in record_setup["run"]
    for check_name in ("authenticode_signer", "timestamp", "signed_installer_match", "installed_payload_signatures"):
        assert f"--check {check_name}" in record_setup["run"]

    attest_setup = step(proof, "Attest Windows installer build provenance")
    attest_setup_sbom = step(proof, "Attest Windows installer SBOM")
    assert attest_setup["with"]["subject-path"] == (
        "${{ steps.signed_installer_asset.outputs.path || steps.unsigned_installer_asset.outputs.path }}"
    )
    assert attest_setup_sbom["with"]["sbom-path"] == "dist/sbom-windows-setup-x64.cdx.json"

    (release_upload,) = uploads(proof)
    assert release_upload["with"]["name"] == "ouroboros-windows-latest"
    assert release_upload["with"]["overwrite"] is True
    uploaded_paths = {line.strip() for line in release_upload["with"]["path"].splitlines() if line.strip()}
    assert uploaded_paths == {
        "${{ steps.release_asset.outputs.path }}",
        "${{ steps.signed_installer_asset.outputs.path || steps.unsigned_installer_asset.outputs.path }}",
        "dist/release-smoke-windows-x64.json",
        "dist/release-smoke-windows-setup-x64.json",
        "dist/sbom-windows-x64.cdx.json",
        "dist/sbom-windows-setup-x64.cdx.json",
    }


def test_release_consumes_only_proven_assets_and_states_the_signing_mode():
    jobs = workflow_jobs()
    release = jobs["release"]
    assert "windows-proof" in release["needs"]
    assert "needs.windows-proof.result == 'success'" in release["if"]
    download = step(release, "Download release artifacts (exclude unsigned build handoff)")
    pattern = download["with"]["pattern"]
    # Every artifact the release pattern admits is a proof-job upload; the
    # signing handoffs never match it.
    names = {item["with"]["name"] for job in jobs.values() for item in uploads(job)}
    assert {"windows-unsigned-archive", "windows-signed-archive"} <= names
    assert not fnmatch("windows-unsigned-archive", pattern)
    assert not fnmatch("windows-signed-archive", pattern)
    assert fnmatch("ouroboros-windows-latest", pattern)
    assemble = step(release, "Assemble release proof capsule and notes")
    assert assemble["env"]["WINDOWS_SIGNER_THUMBPRINT"] == "${{ vars.ESIGNER_CERT_SHA1 }}"
    assert '--windows-signer-thumbprint "$WINDOWS_SIGNER_THUMBPRINT"' in assemble["run"]


def test_one_packer_serves_local_unsigned_and_signed_archives():
    packer = source("scripts/pack_windows_archive.ps1")
    build = source("build_windows.ps1")
    signing = source("scripts/sign_windows_release.ps1")
    # Keep the explanation of Compress-Archive's hidden-file defect; reject
    # command invocations, not comments describing why we avoid the command.
    assert not re.search(r"(?im)^\s*(?:&\s*)?Compress-Archive\b", build + signing + packer)
    assert "OUROBOROS_WINDOWS_DEFER_ARCHIVE" not in build
    assert '& "$PSScriptRoot\\scripts\\pack_windows_archive.ps1" -PayloadRoot "dist\\Ouroboros"' in build
    assert '& "$PSScriptRoot\\pack_windows_archive.ps1" -PayloadRoot' in signing
    assert "CreateFromDirectory(" in packer and "CompressionLevel]::Optimal, $true)" in packer
    # The roundtrip audit compares hidden files and directories, not just names.
    assert "Get-ChildItem -LiteralPath $Root -Recurse -Force" in packer
    assert "'directory'" in packer and "Get-FileHash" in packer
    assert "[IO.Path]::GetTempPath()" in packer  # usable outside a CI runner


def test_signing_script_uses_maintained_java_and_judges_the_outcome_itself():
    signing = source("scripts/sign_windows_release.ps1")
    # Credentials are copied to script-local variables and scrubbed from Env:ESIGNER_*
    # before any external download, Inno Setup compiler, packer or verifier runs.
    assert signing.index("Get-ChildItem Env:ESIGNER_*") < signing.index("Invoke-WebRequest")
    assert "Remove-Item -LiteralPath ('Env:' + $SecretName) -ErrorAction Stop" in signing
    # A set but malformed thumbprint fails before anything is downloaded.
    assert signing.index("$SignerCertSha1 -notmatch '^[a-fA-F0-9]{40}\\z'") < signing.index("Invoke-WebRequest")
    assert "4afc32e8b7f79bbe1de7e4e7049aaad4e0f754357613b9bbec0e3052f06fd36b" in signing
    assert signing.index("is not the pinned") < signing.index("& $Java -jar")
    assert "$env:JAVA_HOME_11_X64" in signing
    assert signing.index("$env:CODE_SIGN_TOOL_PATH = $Jar.Directory.Parent.FullName") < signing.index("& $Java -jar")
    assert "-Filter 'java.exe'" not in signing  # not the bundled 2019 JDK
    assert "GITHUB_EVENT_NAME -ne 'push'" in signing and "refs/tags/v$Version" in signing
    # Vendor output is discarded, never printed, parsed or redacted into the log.
    invocation = signing[signing.index("& $Java -jar"):signing.index("$SignExit = $LASTEXITCODE")]
    assert invocation.rstrip().endswith("*> $null")
    assert "$_" not in signing[signing.index("& $Java -jar"):signing.index("} finally { Pop-Location }")]
    assert "CodeSignTool sign exited $SignExit" in signing
    assert "exited 0 but wrote no signed file" in signing
    # Provenance, payload shape and the unsigned state of every catalogued file are
    # checked before any tool download or the first billed request.
    preflight = signing.index("foreach ($Entry in $Catalog.files)")
    for marker in ("-ne 'NotSigned'", "$Entry.upstreamSha256", "_internal\\VERSION"):
        assert preflight < signing.index(marker) < signing.index("Invoke-WebRequest"), marker
    order = [signing.index(marker) for marker in (
        "Get-ChildItem Env:ESIGNER_*",
        "Invoke-WebRequest",
        "Join-Path $PSScriptRoot 'download_inno_setup.ps1'",
        "function Invoke-CodeSignRequest",
        "-output_dir_path=$Output",
        "-Executable $SignedFile",
        "Copy-Item -LiteralPath $SignedFile",
        "-Executable $DestinationFile",
        "foreach ($Target in $Targets)",
        "pack_windows_archive.ps1",
        "pack_windows_setup.ps1",
        "$SetupStatus = (Get-AuthenticodeSignature -LiteralPath $UnsignedSetup).Status",
        "Invoke-CodeSignRequest -SourceFile $UnsignedSetup",
        "'signed_files='",
        "'signed_installer='",
    )]
    assert order == sorted(order)


def test_windows_installer_iss_uses_in_place_upgrade_and_fail_closed_os_lock_checks():
    iss = source("packaging/windows/ouroboros.iss")
    # Never recursively wipe destination directories, pre-uninstall before upgrade,
    # enumerate WMI processes, or close running apps in silent mode.
    assert "[InstallDelete]" not in iss
    assert "filesandordirs" not in iss.lower()
    assert "deltree" not in iss.lower()
    assert "Exec(" not in iss
    assert "WbemScripting" not in iss and "Win32_Process" not in iss
    assert "CloseApplications=no" in iss
    assert "RestartApplications=no" in iss
    assert "AllowCancelDuringInstall=no" in iss
    assert "DisableReadyPage=yes" not in iss
    # Fixed per-user install path, Start menu AUMID matching desktop identity, ordinary launch checkbox.
    assert "AppId={{6FB7704B-C2C5-4252-A5D0-BEC473FD007D}" in iss
    assert "PrivilegesRequired=lowest" in iss
    assert "DefaultDirName={localappdata}\\Programs\\Ouroboros" in iss
    assert "DisableDirPage=yes" in iss
    assert "UsePreviousAppDir=no" in iss
    assert "AlwaysShowDirOnReadyPage=yes" in iss
    assert 'Name: "{userprograms}\\Ouroboros"; Filename: "{app}\\Ouroboros.exe"; AppUserModelID: "com.ouroboros.agent"' in iss
    assert 'Filename: "{app}\\Ouroboros.exe"; Description: "{cm:LaunchProgram,Ouroboros}"; Flags: nowait postinstall skipifsilent' in iss
    assert "SetupIconFile=..\\..\\assets\\icon.ico" in iss
    assert "UninstallDisplayIcon={app}\\Ouroboros.exe" in iss
    for asset in ("assets/icon.ico", "packaging/windows/wizard.png", "packaging/windows/wizard-small.png"):
        assert (ROOT / asset).is_file(), asset
    # Ownership is verified via HKCU Uninstall metadata + existing unins*.exe/.dat, never guessed from path.
    assert "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{6FB7704B-C2C5-4252-A5D0-BEC473FD007D}_is1" in iss
    assert "function IsDirEmpty(" in iss
    assert "function IsOwnedInstallDir(" in iss
    assert "RegQueryStringValue(HKEY_CURRENT_USER, UninstallRegKey, 'Inno Setup: App Path'" in iss
    assert "RegQueryStringValue(HKEY_CURRENT_USER, UninstallRegKey, 'UninstallString'" in iss
    assert "WildcardMatch(Lowercase(ExtractFileName(UninstExe)), 'unins*.exe')" in iss
    assert "FileExists(ChangeFileExt(UninstExe, '.dat'))" in iss
    # Fail-closed OS lock probe covers Ouroboros.exe, python.exe, GUI-handoff pythonw.exe and Python DLLs.
    assert "OpenExclusiveReadWrite = $0012;" in iss
    assert "TFileStream.Create(FilePath, OpenExclusiveReadWrite)" in iss
    assert "Ouroboros.exe" in iss
    assert "_internal\\python-standalone\\python.exe" in iss
    assert "_internal\\python-standalone\\pythonw.exe" in iss
    assert "python3*.dll" in iss
    # English-source unprefixed CustomMessages only (DEVELOPMENT 02); Inno's built-in Russian.isl stays in [Languages].
    custom_messages = iss[iss.index("[CustomMessages]"):iss.index("[Files]")]
    for key in ("AppRunning", "AppStillRunning", "UninstallAppRunning",
                "OnlyCanonicalDir", "ForeignInstallDir", "ForeignShortcut",
                "InstallOrUninstallRunning"):
        assert re.search(rf"(?m)^{key}=", custom_messages), key
    assert not re.search(r"(?m)^(?:en|ru)\.", custom_messages)
    assert not re.search(r"[\u0400-\u04ff]", iss)
    # PrepareToInstall checks canonical path -> exclusive mutex -> file collision -> empty/new (refusing
    # the owner's own Start shortcut there) -> owned install -> busy check (including unins*.dat) and start holds.
    prep = iss[iss.index("function PrepareToInstall"):iss.index("procedure CurStepChanged")]
    order = [prep.index(marker) for marker in (
        "OnlyCanonicalDir",
        "TryAcquireChanging",
        "FileExists(AppDir) and not DirExists(AppDir)",
        "not DirExists(AppDir) or IsDirEmpty(AppDir)",
        "ExpandConstant('{userprograms}\\Ouroboros.lnk')",
        "CustomMessage('ForeignShortcut')",
        "IsOwnedInstallDir(AppDir, UninstDat)",
        "HoldStartPrograms(AppDir, UninstDat, CustomMessage('AppRunning'))",
    )]
    assert order == sorted(order)
    # A busy or held file is asked about again (Retry/Cancel), never waited out on a timer.
    assert "Sleep(" not in iss


def test_setup_keeps_both_entries_from_starting_while_it_changes_the_installation():
    """Setup and Uninstall acquire one named mutex exclusively before their checks and release it once nothing
    changes the installation; a losing Uninstall aborts immediately without a dialog so it frees unins*.dat.
    Ouroboros.exe and the shipped CLI consult the mutex in their own processes (behavior:
    test_launcher_windows_identity.py, scripts/smoke_windows_setup.ps1 step 11). The holds and trailing [Files]
    entries keep Inno from leaving a held binary behind."""
    from ouroboros.launcher_windows_runtime import _SETUP_CHANGING

    iss = source("packaging/windows/ouroboros.iss")
    code = iss[iss.index("[Code]"):]
    assert f"SetupChangingName = '{_SETUP_CHANGING}';" in code
    acquire = code[code.index("function TryAcquireChanging"):code.index("procedure ClearChanging")]
    assert "ErrorAlreadyExists" in acquire and "DllGetLastError" in acquire and "CloseHandle(Handle)" in acquire
    assert "SuppressibleMsgBox" not in acquire
    prep = code[code.index("function PrepareToInstall"):code.index("procedure CurStepChanged")]
    uninstall = code[code.index("function InitializeUninstall"):code.index("procedure CurUninstallStepChanged")]
    for change in (prep, uninstall):
        assert change.index("TryAcquireChanging") < change.index("HoldStartPrograms(")
    assert "SuppressibleMsgBox" not in uninstall[:uninstall.index("HoldStartPrograms(")]
    setup_release = code[code.index("procedure CurStepChanged"):code.index("function InitializeUninstall")]
    uninst_release = code[code.index("procedure CurUninstallStepChanged"):]
    assert "ReleaseChangeState;" in setup_release and "procedure DeinitializeSetup" in setup_release
    assert "usPostUninstall" in uninst_release and "ReleaseChangeState;" in code[code.index("procedure DeinitializeUninstall"):]
    held = re.findall(r"Result := '([^']+)'", code[code.index("function StartProgram"):code.index("procedure ReleaseHold(")])
    entries = [line for line in iss[iss.index("[Files]"):iss.index("[Icons]")].splitlines() if line.startswith("Source:")]
    assert entries[0].split('Excludes: "', 1)[1].split('"', 1)[0].split(",") == ["\\" + path for path in held]
    assert [entry.split("; BeforeInstall: ", 1)[1] for entry in entries[1:]] == [
        f"ReleaseHeldProgram('{path}')" for path in reversed(held)]


def test_wizard_art_is_the_brand_mark_without_a_backplate_in_each_theme():
    """The dynamic wizard shows the icon's snake and eye on its own light or dark image field: dark ink in light
    mode, light ink in dark mode (Inno's *DynamicDark fields), with no tile behind the mark, its hole open."""
    from PIL import Image

    def luminance(rgb):
        rgb = [c / 255 for c in rgb]
        r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast(a, b):
        high, low = sorted((luminance(a), luminance(b)), reverse=True)
        return (high + 0.05) / (low + 0.05)

    iss = source("packaging/windows/ouroboros.iss")
    # field: white for the light image area, Inno's documented WizardImageBackColorDynamicDark default for dark.
    for directive, name, field in (("WizardImageFile", "wizard.png", (255, 255, 255)),
                                   ("WizardSmallImageFile", "wizard-small.png", (255, 255, 255)),
                                   ("WizardImageFileDynamicDark", "wizard-dark.png", (0x2E, 0x3A, 0x3F)),
                                   ("WizardSmallImageFileDynamicDark", "wizard-small-dark.png", (0x2E, 0x3A, 0x3F))):
        assert f"{directive}={name}\n" in iss
        with Image.open(ROOT / "packaging" / "windows" / name) as image:
            assert image.mode == "RGBA", name
            pixels = image.load()
            left, top, right, bottom = image.getchannel("A").getbbox()
            mark = [pixels[x, y] for y in range(top, bottom) for x in range(left, right)]
            # A backplate would fill the mark's own box and its corners; the ring leaves both open.
            assert sum(p[3] >= 128 for p in mark) / len(mark) < 0.75, f"{name}: a filled box behind the mark"
            assert [pixels[x, y][3] for x in (left, right - 1) for y in (top, bottom - 1)] == [0, 0, 0, 0], name
            assert pixels[(left + right) // 2, (top + bottom) // 2][3] == 0, f"{name}: the hole is open"
            solid = sorted((p[:3] for p in mark if p[3] == 255), key=luminance)
            ink = solid[len(solid) // 2]  # the body: the eye is a small share at one end
            assert contrast(ink, field) >= 4.5, f"{name}: ink {ink} unreadable on {field}"


def test_smoke_windows_setup_script_enforces_disposable_guard_aumid_and_safe_cleanup():
    smoke = source("scripts/smoke_windows_setup.ps1")
    # Windows GitHub-hosted runner or explicit throwaway-profile switch required; never runs blindly on self-hosted or workstations.
    assert "-AllowDisposableProfile" in smoke
    # Uses Shell.Application ExtendedProperty('System.AppUserModel.ID') with zero embedded C#.
    assert "Shell.Application" in smoke
    assert "ExtendedProperty('System.AppUserModel.ID')" in smoke
    assert "com.ouroboros.agent" in smoke
    assert "Add-Type -TypeDefinition" not in smoke and "SHGetPropertyStoreFromParsingName" not in smoke
    # Cleanup must NEVER recursively delete $InstallDir or $UserDataDir (preserves unknown files), and the
    # synthetic profile is seeded only where no real one exists.
    assert "Remove-DirOnlyIfEmpty $InstallDir" in smoke
    assert "Remove-Item -LiteralPath $InstallDir -Recurse" not in smoke
    assert "Remove-Item -LiteralPath $UserDataDir -Recurse" not in smoke
    assert smoke.index("if (Test-Path -LiteralPath $UserDataDir) {") < smoke.index("$ProfileBefore = Get-TreeManifest")
    assert smoke.index("$ProfileBefore = Get-TreeManifest") < smoke.index("$InstallProc = Invoke-SilentSetup")


def test_inno_setup_pin_downloader_and_packer_enforce_digest_and_payload_contract():
    pin = json.loads(source("packaging/windows/inno_setup_pin.json"))
    assert pin["schemaVersion"] == 1
    assert re.fullmatch(r"\d+\.\d+\.\d+", pin["version"])
    assert pin["installer"].endswith(".exe")
    assert pin["url"].startswith("https://github.com/jrsoftware/issrc/releases/download/")
    assert isinstance(pin["sizeBytes"], int) and pin["sizeBytes"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])

    downloader = source("scripts/download_inno_setup.ps1")
    assert downloader.index("Invoke-WebRequest") < downloader.index("$Size -ne $Pin.sizeBytes -or $Digest -ne $Pin.sha256")
    assert downloader.index("$Size -ne $Pin.sizeBytes -or $Digest -ne $Pin.sha256") < downloader.index("Start-Process")
    assert "'/CURRENTUSER', '/PORTABLE=1'" in downloader

    packer = source("scripts/pack_windows_setup.ps1")
    assert "Ouroboros-Setup-$Version-windows-x64" in packer
    assert "_internal\\VERSION" in packer
    assert "@('Ouroboros.exe', '_internal', 'bin')" in packer
    assert "[String]::Equals($Output, $Payload, [StringComparison]::OrdinalIgnoreCase)" in packer
    assert "packaging\\windows\\ouroboros.iss" in packer


def test_verifier_uses_signtool_exit_codes_and_runs_outside_ci():
    verification = source("scripts/verify_windows_signature.ps1")
    assert "Get-AuthenticodeSignature -LiteralPath $Executable" in verification
    assert "$Signature.Status -ne 'Valid'" in verification
    assert "verify /pa /all /tw /v $Executable" in verification
    switch = verification[verification.index("switch ($LASTEXITCODE)"):]
    assert "0 { }" in switch and "2 {" in switch and "default {" in switch
    for english in ("Successfully verified", "Number of warnings"):
        assert english not in verification
    assert "RUNNER_TEMP" not in verification
    assert "Thumbprint = $Actual" in verification and "Publisher  =" in verification


def test_signing_catalog_names_three_executables_of_the_pinned_python():
    catalog = json.loads(source("scripts/windows_signing_catalog.json"))
    paths = [entry["path"] for entry in catalog["files"]]
    assert paths == ["Ouroboros/Ouroboros.exe", "Ouroboros/_internal/python-standalone/python.exe",
                     "Ouroboros/_internal/python-standalone/pythonw.exe"]
    pinned = {entry["path"].rsplit("/", 1)[1]: entry.get("upstreamSha256") for entry in catalog["files"]}
    assert pinned["Ouroboros.exe"] is None  # built per run; its unsigned hash is recorded, not pinned
    assert all(re.fullmatch(r"[0-9a-f]{64}", pinned[name]) for name in ("python.exe", "pythonw.exe"))
    # venv launcher templates are not release-signed (docs and notes say so).
    assert not any("venv" in path for path in paths)
    # A Python upgrade that changes the downloader pin must restate the catalog.
    downloader = source("scripts/download_python_standalone.ps1")
    pins = dict(re.findall(r'^\$(Release|PyVersion|Sha256|Platform) = "([^"]+)"$', downloader, re.MULTILINE))
    assert catalog["pythonUpstream"] == {
        "release": pins["Release"], "pythonVersion": pins["PyVersion"], "archiveSha256": pins["Sha256"],
        "archive": f"cpython-{pins['PyVersion']}+{pins['Release']}-{pins['Platform']}-install_only_stripped.tar.gz",
    }
    smoke = step(workflow_jobs()["windows-proof"], "Smoke final Windows archive")["run"]
    for path in paths[1:]:
        assert "$SmokeRoot\\" + path.replace("/", "\\") in smoke


def expression_value(expression, context):
    """Evaluate the boolean/string subset used by these workflow gates.

    Tokenize literals separately so substitutions cannot change their contents.
    This reads the actual YAML expressions, including grouping and negation.
    """
    expression = expression.strip().removeprefix("${{").removesuffix("}}").strip()
    tokens = re.findall(r"'(?:[^']|'')*'|[A-Za-z_][\w.-]*|&&|\|\||!=|==|[!(),]", expression)
    assert re.sub(r"\s+", "", expression) == re.sub(r"\s+", "", "".join(tokens))
    functions = {"always": lambda: True, "cancelled": lambda: context.get("cancelled", False),
                 "startsWith": lambda value, prefix: value.startswith(prefix)}
    converted = []
    for token in tokens:
        if token.startswith("'") or token in functions or token in ("(", ")", ",", "==", "!="):
            converted.append(token)
        elif token in {"&&", "||", "!"}:
            converted.append({"&&": "and", "||": "or", "!": "not"}[token])
        else:
            assert token.startswith(("github.", "needs.", "vars.", "matrix.", "steps.")), token
            converted.append(repr(context.get(token, "")))
    return eval(" ".join(converted), {"__builtins__": {}, **functions})


@pytest.mark.parametrize("repository", ["razzant/ouroboros", "fork-owner/ouroboros"])
@pytest.mark.parametrize(("ref", "certificate"), [
    ("refs/tags/v7.6.0", ""),
    ("refs/tags/v7.6.0", " "),
    pytest.param("refs/tags/v7.6.1-rc.1", "A" * 40, id='refs/tags/v7.6.1-rc.1-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'),
    ("refs/heads/ouroboros", "malformed"),
])
def test_windows_job_control_flow(repository, certificate, ref):
    jobs = workflow_jobs()
    # An explicit status function is needed: otherwise a skipped signing job
    # implicitly skips unsigned proof on forks, even if its own predicate is true.
    for name in ("windows-sign", "windows-proof", "release"):
        assert "always()" in jobs[name]["if"]
        assert jobs[name].get("continue-on-error", False) is False
    is_fork = repository != "razzant/ouroboros"
    for build, preflight, sign, cancelled in product(
        ("success", "failure", "skipped", "cancelled"),
        ("success", "failure"), ("success", "failure", "skipped", "cancelled"), (False, True),
    ):
        context = {
            "github.ref": ref, "github.repository": repository,
            "vars.ESIGNER_CERT_SHA1": certificate, "cancelled": cancelled,
            **{f"needs.{name}.result": "success" for name in jobs["release"]["needs"]},
            "needs.android-build.result": "skipped",  # optional, even on unsigned forks
            "needs.build.result": build, "needs.release-preflight.result": preflight,
            "needs.release-preflight.outputs.tag_valid": "true", "needs.windows-sign.result": sign,
        }
        ready = ref.startswith("refs/tags/v") and not cancelled and build == "success"
        assert bool(expression_value(jobs["windows-sign"]["if"], context)) == (
            ready and bool(certificate) and preflight == "success")
        proof = expression_value(jobs["windows-proof"]["if"], context)
        assert bool(proof) == (ready and ((not certificate and is_fork) or sign == "success"))
        context["needs.windows-proof.result"] = "success" if proof else "skipped"
        assert bool(expression_value(jobs["release"]["if"], context)) == (
            ready and bool(proof) and preflight == "success")
    context["cancelled"] = False
    context["needs.build.result"] = context["needs.release-preflight.result"] = "success"
    context["needs.release-preflight.outputs.tag_valid"] = "false"
    assert not expression_value(jobs["windows-sign"]["if"], context)
    download = step(jobs["windows-proof"], "Download final Windows archive")
    assert expression_value(download["with"]["name"], context) == (
        "windows-signed-archive" if certificate else "windows-unsigned-archive")
    unsigned_setup = step(jobs["windows-proof"], "Build unsigned Windows installer")
    assert bool(expression_value(unsigned_setup["if"], context)) == (not certificate and is_fork)


@pytest.mark.parametrize(("repository", "cert", "should_pass", "detail"), [
    ("razzant/ouroboros", "", False, "Official razzant/ouroboros releases require ESIGNER_CERT_SHA1"),
    ("Razzant/Ouroboros", "", False, "Official razzant/ouroboros releases require ESIGNER_CERT_SHA1"),
    ("razzant/ouroboros", "not-a-sha1", False, "not a 40-hex SHA-1 thumbprint"),
    ("razzant/ouroboros", "A" * 40, True, "tag_valid=true"),
    ("fork-owner/ouroboros", "", True, "tag_valid=true"),
    ("fork-owner/ouroboros", " ", False, "not a 40-hex SHA-1 thumbprint"),
    ("fork-owner/ouroboros", "a" * 40, True, "tag_valid=true"),
])
def test_release_preflight_enforces_official_windows_signing_policy(tmp_path, repository, cert, should_pass, detail):
    import os
    import subprocess
    import sys

    preflight = step(workflow_jobs()["release-preflight"], "Validate tag matches VERSION")
    assert preflight["env"]["ESIGNER_CERT_SHA1"] == "${{ vars.ESIGNER_CERT_SHA1 }}"
    script = preflight["run"].split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    (tmp_path / "VERSION").write_text("7.12.2\n", encoding="utf-8")
    gh_out = tmp_path / "gh_output.txt"
    gh_out.write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "GITHUB_REF_NAME": "v7.12.2",
        "GITHUB_REPOSITORY": repository,
        "GITHUB_OUTPUT": str(gh_out),
        "ESIGNER_CERT_SHA1": cert,
    }
    proc = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    if should_pass:
        assert proc.returncode == 0, proc.stderr
        assert detail in gh_out.read_text(encoding="utf-8")
    else:
        assert proc.returncode != 0
        assert detail in proc.stderr
        assert "tag_valid=false" in gh_out.read_text(encoding="utf-8")


def test_release_stops_for_each_required_proof_failure():
    release = workflow_jobs()["release"]
    context = {"github.ref": "refs/tags/v7.6.0",
               **{f"needs.{name}.result": "success" for name in release["needs"]}}
    assert expression_value(release["if"], context)
    for name in set(release["needs"]) - {"android-build"}:
        for result in ("failure", "skipped", "cancelled"):
            assert not expression_value(release["if"], {**context, f"needs.{name}.result": result})


def test_release_push_trigger_includes_prereleases():
    # BaseLoader retains YAML's literal "on" key instead of YAML 1.1 boolean coercion.
    triggers = yaml.load(source(".github/workflows/ci.yml"), Loader=yaml.BaseLoader)["on"]
    for tag in ("v7.6.0", "v7.6.1-rc.1"):
        assert any(fnmatch(tag, pattern) for pattern in triggers["push"]["tags"])
