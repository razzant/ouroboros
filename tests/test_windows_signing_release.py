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
    assert signing["outputs"] == {"archive_sha256": "${{ steps.release_asset.outputs.sha256 }}"}
    # The Environment and its secrets reach this one step of this one job.
    bound = [name for name, job in jobs.items() if "environment" in job]
    assert bound == ["windows-sign"]
    sign = step(signing, "Sign, verify and package Windows executable")
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
    assert sign["env"]["UNSIGNED_ARCHIVE"] == f"${{{{ steps.{check['id']}.outputs.path }}}}"
    order = [step_index(signing, name) for name in (
        "Download unsigned Windows archive (never a release asset)",
        "Check unsigned archive digest before extraction",
        "Sign, verify and package Windows executable",
        "Locate final release archive",
        "Transfer signed Windows archive (not yet release proof)",
    )]
    assert order == sorted(order)
    (handoff,) = uploads(signing)
    assert handoff["with"]["name"] == "windows-signed-archive"
    assert handoff["with"]["overwrite"] is True


def test_proof_job_proves_both_modes_without_secrets():
    jobs = workflow_jobs()
    proof = jobs["windows-proof"]
    assert "environment" not in proof
    assert "secrets." not in yaml.safe_dump(proof)
    assert set(proof["needs"]) == {"build", "windows-sign"}
    assert "needs.build.result == 'success'" in proof["if"]
    assert f"({UNCONFIGURED} || needs.windows-sign.result == 'success')" in proof["if"]
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
    smoke = step(proof, "Smoke final Windows archive")
    assert step_index(proof, check["name"]) < step_index(proof, smoke["name"])
    assert smoke["run"].index("Expand-Archive") < smoke["run"].index("verify_windows_signature.ps1")
    assert "if ($env:WINDOWS_SIGNER_THUMBPRINT)" in smoke["run"]
    record = step(proof, "Record packaged artifact smoke")
    assert record["env"] == {
        "SIGNER_THUMBPRINT": "${{ steps.smoke_windows.outputs.signer_thumbprint }}",
        "SIGNER_PUBLISHER": "${{ steps.smoke_windows.outputs.signer_publisher }}",
    }
    for check_name in ("authenticode_signer", "timestamp", "signed_payload_archive_match"):
        assert f"--check {check_name}" in record["run"]
    (release_upload,) = uploads(proof)
    assert release_upload["with"]["name"] == "ouroboros-windows-latest"
    assert release_upload["with"]["overwrite"] is True


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
    # A set but malformed thumbprint fails before anything is downloaded.
    assert signing.index("ESIGNER_CERT_SHA1 -notmatch '^[a-fA-F0-9]{40}\\z'") < signing.index("Invoke-WebRequest")
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
    assert "exited 0 but wrote no signed Ouroboros.exe" in signing
    order = [signing.index(marker) for marker in (
        "-output_dir_path=$Signed",
        "-Executable $SignedExecutable",
        "Copy-Item -LiteralPath $SignedExecutable",
        "-Executable $Executable",
        "pack_windows_archive.ps1",
    )]
    assert order == sorted(order)


def test_verifier_uses_signtool_exit_codes_and_runs_outside_ci():
    verification = source("scripts/verify_windows_signature.ps1")
    assert "Get-AuthenticodeSignature -LiteralPath $Executable" in verification
    assert "$Signature.Status -ne 'Valid'" in verification
    assert "verify /pa /tw /v $Executable" in verification
    switch = verification[verification.index("switch ($LASTEXITCODE)"):]
    assert "0 { }" in switch and "2 {" in switch and "default {" in switch
    for english in ("Successfully verified", "Number of warnings"):
        assert english not in verification
    assert "RUNNER_TEMP" not in verification
    assert "Thumbprint = $Actual" in verification and "Publisher  =" in verification


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


@pytest.mark.parametrize("certificate", ["", "A" * 40, "malformed", " "])
@pytest.mark.parametrize("ref", ["refs/tags/v7.6.0", "refs/tags/v7.6.1-rc.1", "refs/heads/ouroboros"])
def test_windows_job_control_flow(certificate, ref):
    jobs = workflow_jobs()
    # An explicit status function is needed: otherwise a skipped signing job
    # implicitly skips unsigned proof, even if its own predicate is true.
    for name in ("windows-sign", "windows-proof", "release"):
        assert "always()" in jobs[name]["if"]
        assert jobs[name].get("continue-on-error", False) is False
    for build, preflight, sign, cancelled in product(
        ("success", "failure", "skipped", "cancelled"),
        ("success", "failure"), ("success", "failure", "skipped", "cancelled"), (False, True),
    ):
        context = {
            "github.ref": ref, "vars.ESIGNER_CERT_SHA1": certificate, "cancelled": cancelled,
            **{f"needs.{name}.result": "success" for name in jobs["release"]["needs"]},
            "needs.android-build.result": "skipped",  # optional, even on unsigned forks
            "needs.build.result": build, "needs.release-preflight.result": preflight,
            "needs.release-preflight.outputs.tag_valid": "true", "needs.windows-sign.result": sign,
        }
        ready = ref.startswith("refs/tags/v") and not cancelled and build == "success"
        assert bool(expression_value(jobs["windows-sign"]["if"], context)) == (
            ready and bool(certificate) and preflight == "success")
        proof = expression_value(jobs["windows-proof"]["if"], context)
        assert bool(proof) == (ready and (not certificate or sign == "success"))
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
