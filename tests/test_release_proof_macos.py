"""macOS notarization in the release proof: receipt, evidence and notes.

v7.7.0 shipped a DMG whose app Gatekeeper rejected while every signature
verified. Configuration selects the macOS state as it does for Windows
Authenticode (test_release_proof.py): a signed stable build must prove both
stapled tickets and Gatekeeper acceptance of the copied-out app.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.test_release_proof import REPO, _assemble_windows, _fixture_release, _run_cli, release_proof


@pytest.mark.parametrize("notarized", [False, True])
def test_macos_notarization_state_reaches_evidence_and_notes(tmp_path: Path, notarized: bool):
    release_dir, version_file, readme = _fixture_release(tmp_path, notarized=notarized)
    _, notes = _assemble_windows(tmp_path, release_dir, version_file, readme, signer="")
    evidence = json.loads((release_dir / "release-evidence.json").read_text(encoding="utf-8"))
    macos = next(row for row in evidence["artifacts"] if row["proofId"] == "macos-arm64")

    if notarized:
        assert macos["notarization"] == {"status": "notarized"}
        assert "Apple notarized both the app and the DMG" in notes
        assert "this DMG is not notarized" not in notes
    else:
        assert macos["notarization"] == {"status": "not_notarized"}
        assert "this DMG is not notarized" in notes and "Open Anyway" in notes
        assert "Apple notarized" not in notes


@pytest.mark.parametrize(
    ("notarized", "mutate"),
    [
        # A notarized claim needs every staple and Gatekeeper proof.
        (True, "drop-gatekeeper"),
        (True, "drop-staple"),
        # An unnotarized DMG may not claim any of them.
        (False, "claim-gatekeeper"),
        (False, "drop-state"),
        (False, "unknown-state"),
    ],
)
def test_macos_notarization_state_must_be_consistent(tmp_path: Path, notarized: bool, mutate: str):
    release_dir, version_file, readme = _fixture_release(tmp_path, notarized=notarized)
    path = release_dir / "release-smoke-macos-arm64.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if mutate == "drop-gatekeeper":
        receipt["checks"].remove("gatekeeper_app")
    elif mutate == "drop-staple":
        receipt["checks"].remove("stapled_app")
    elif mutate == "claim-gatekeeper":
        receipt["checks"].append("gatekeeper_app")
    elif mutate == "drop-state":
        del receipt["notarization"]
    else:
        receipt["notarization"] = {"status": "stapled"}
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="consistent notarization state"):
        _assemble_windows(tmp_path, release_dir, version_file, readme, signer="")
    assert not (release_dir / "release-evidence.json").exists()


def test_macos_receipt_records_notarization_state(tmp_path: Path):
    dmg = tmp_path / "Ouroboros-6.87.5.dmg"
    dmg.write_bytes(b"dmg")
    common = ["record-smoke", "--proof-id", "macos-arm64", "--artifact", str(dmg),
              "--commit", "a" * 40, "--tag", "v6.87.5", "--check", "packaged_cli_help"]
    plain, notarized = tmp_path / "plain.json", tmp_path / "notarized.json"
    assert _run_cli(*common, "--output", str(plain)).returncode == 0
    assert json.loads(plain.read_text(encoding="utf-8"))["notarization"] == {"status": "not_notarized"}
    assert _run_cli(*common, "--output", str(notarized),
                    "--macos-notarization", "notarized").returncode == 0
    assert json.loads(notarized.read_text(encoding="utf-8"))["notarization"] == {"status": "notarized"}
    # An empty workflow output never becomes a state.
    assert _run_cli(*common, "--output", str(tmp_path / "empty.json"),
                    "--macos-notarization", "").returncode != 0


@pytest.mark.parametrize("observed", ["notarized", "not_notarized", ""])
def test_workflow_records_the_observed_macos_notarization(tmp_path: Path, observed: str):
    """Execute the YAML receipt writer for the macOS shard with each smoke outcome."""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("workflow shell execution requires bash")
    jobs = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    step = next(step for step in jobs["build"]["steps"]
                if step.get("name") == "Record packaged artifact smoke")
    dmg = tmp_path / "Ouroboros-6.87.5.dmg"
    dmg.write_bytes(b"dmg")
    (tmp_path / "dist").mkdir()
    script = (step["run"]
              .replace("python scripts/release_proof.py", " ".join(map(shlex.quote, [
                  Path(sys.executable).as_posix(), (REPO / "scripts/release_proof.py").as_posix()])))
              .replace("${{ matrix.proof_id }}", "macos-arm64")
              .replace("${{ steps.release_asset.outputs.path }}", dmg.as_posix())
              .replace("${{ steps.smoke_macos.outputs.macos_notarization }}", observed))
    result = subprocess.run([bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script],
                            cwd=tmp_path, capture_output=True, text=True,
                            env=dict(os.environ, GITHUB_SHA="a" * 40, GITHUB_REF_NAME="v6.87.5"))

    if not observed:
        # A smoke that reported nothing never becomes an unnotarized release.
        assert result.returncode != 0
        return
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "dist" / "release-smoke-macos-arm64.json").read_text(encoding="utf-8"))
    assert receipt["notarization"] == {"status": observed}
    claimed = release_proof.NOTARIZATION_SMOKE_CHECKS & set(receipt["checks"])
    assert claimed == (release_proof.NOTARIZATION_SMOKE_CHECKS if observed == "notarized" else set())
    assert release_proof._notarization_state(receipt, receipt["checks"]) == {"status": observed}


OK = "accepted\nsource=Notarized Developer ID"


@pytest.mark.parametrize(
    ("prerelease", "signing", "dmg_rc", "app_rc", "dmg_spctl", "app_spctl", "expected"),
    [
        ("false", "true", 0, 0, OK, OK, "notarized"),
        # A service or ticket error during validation stops the release; it is
        # never relabelled as an unnotarized build (it was notarized and stapled).
        ("false", "true", 68, 0, OK, OK, None),
        # v7.7.0: the DMG passes, the app copied out of it is rejected.
        ("false", "true", 0, 0, OK, "rejected\nsource=Unnotarized Developer ID", None),
        ("false", "true", 0, 0, OK, "accepted\nsource=Developer ID", None),
        ("false", "true", 0, 65, OK, OK, None),  # the app lacks its own ticket
        # Gatekeeper switched off on the runner accepts everything, so it proves nothing.
        ("false", "true", 0, 0, "accepted\noverride=security disabled", OK, None),
        # Prereleases build unsigned by policy; forks without a certificate too.
        ("true", "true", 0, 0, "", "", "not_notarized"),
        ("false", "false", 0, 0, "", "", "not_notarized"),
    ],
)
def test_smoke_decides_notarization_from_configuration_and_proves_it(
    tmp_path: Path, prerelease: str, signing: str, dmg_rc: int, app_rc: int,
    dmg_spctl: str, app_spctl: str, expected: str | None,
):
    """Execute the final-DMG smoke's notarization block with per-target fakes."""
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("workflow shell execution requires bash")
    jobs = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    run = next(step for step in jobs["build"]["steps"]
               if step.get("name") == "Smoke final macOS DMG")["run"]
    block = run[run.index('DMG="${{ steps.release_asset.outputs.path }}"'):]
    block = (block.replace("${{ steps.release_asset.outputs.path }}", (tmp_path / "x.dmg").as_posix())
             .replace("${{ needs.release-preflight.outputs.is_prerelease }}", prerelease))
    fake = tmp_path / "bin"
    fake.mkdir()
    fakes = {
        "xcrun": 'echo "xcrun $*" >> "$CALLS"\n'
                 'case "$*" in *Ouroboros.app*) exit "$APP_RC" ;; esac\nexit "$DMG_RC"\n',
        "spctl": 'echo "spctl $*" >> "$CALLS"\n'
                 'case "$*" in *"--type exec"*) out="$APP_SPCTL" ;; *) out="$DMG_SPCTL" ;; esac\n'
                 'printf \'%s\\n\' "$out"\ncase "$out" in rejected*) exit 3 ;; esac\n',
    }
    for tool, body in fakes.items():
        # LF on every OS: bash rejects CRLF scripts on the Windows shards.
        (fake / tool).write_text("#!/bin/sh\n" + body, encoding="utf-8", newline="\n")
        (fake / tool).chmod(0o755)
    output, calls = tmp_path / "output", tmp_path / "calls"
    env = dict(os.environ, PATH=f"{fake}{os.pathsep}{os.environ['PATH']}", HAS_APPLE_SIGNING=signing,
               SBOM_ROOT=str(tmp_path / "sbom"), RUNNER_TEMP=str(tmp_path), GITHUB_OUTPUT=str(output),
               CALLS=str(calls), DMG_RC=str(dmg_rc), APP_RC=str(app_rc),
               DMG_SPCTL=dmg_spctl, APP_SPCTL=app_spctl)
    result = subprocess.run([bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", block],
                            cwd=tmp_path, env=env, capture_output=True, text=True)

    recorded = output.read_text(encoding="utf-8") if output.exists() else ""
    if expected is None:
        assert result.returncode != 0 and "macos_notarization" not in recorded
    else:
        assert result.returncode == 0, result.stderr
        assert recorded == f"macos_notarization={expected}\n"
    probed = calls.read_text(encoding="utf-8") if calls.exists() else ""
    if expected == "not_notarized":
        assert probed == ""
    if expected == "notarized":
        assert probed.count("stapler validate") == 2 and probed.count("spctl --assess") == 2


ENTRY = "$MOUNT/Ouroboros.app/Contents/MacOS/ouroboros-shell"
PLAIN_PYTHON = "$MOUNT/Ouroboros.app/Contents/Frameworks/python-standalone/bin/python3"


def test_final_dmg_smoke_imports_the_desktop_modules_in_the_app_entry_without_a_window(tmp_path: Path):
    """The source launcher runs in the C app entry (packaging/macos/ouroboros_shell.c), not in plain python3:
    the smoke runs that entry on an import-only ``-c``, never launcher.py or a window, and proves its base is
    the bundled plain Python. The workflow's own check runs here over stub desktop modules; only a macOS CI
    build executes the native entry itself."""
    jobs = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    run = next(step for step in jobs["build"]["steps"] if step.get("name") == "Smoke final macOS DMG")["run"]
    lines = [line for line in run.replace("\\\n", " ").splitlines()
             if ENTRY in line and not line.lstrip().startswith("#")]
    assert len(lines) == 1, "one entry invocation"
    argv = shlex.split(lines[0])
    assert argv[:5] == ["HOME=$HOME_DIR", ENTRY, "-I", "-B", "-c"], "isolated, no user site, a command only"
    code, args = argv[5], argv[6:]
    assert args == [ENTRY, PLAIN_PYTHON]
    assert "import os, sys, webview, objc, AppKit, WebKit, PyObjCTools.AppHelper;" in code
    for launch in ("launcher", "webview.start", "create_window", "main("):
        assert launch not in code
    assert run.index(lines[0].split(" -I ", 1)[0]) < run.index("codesign --verify"), "before the signing proofs"

    stubs = tmp_path / "stubs"
    for module in ("webview/__init__.py", "PyObjCTools/__init__.py", "PyObjCTools/AppHelper.py",
                   "objc.py", "AppKit.py", "WebKit.py"):
        (stubs / module).parent.mkdir(parents=True, exist_ok=True)
        (stubs / module).write_text("", encoding="utf-8")
    python, base = sys.executable, getattr(sys, "_base_executable", sys.executable)

    def check(entry: str, plain: str) -> subprocess.CompletedProcess:
        return subprocess.run([python, "-B", "-c", code, entry, plain], cwd=stubs, capture_output=True, text=True,
                              env=dict(os.environ, PYTHONPATH=str(stubs)))

    assert check(python, base).returncode == 0, check(python, base).stderr
    other = str(stubs / "objc.py")
    for entry, plain in ((python, other), (other, base)):
        refused = check(entry, plain)
        assert refused.returncode != 0 and "AssertionError" in refused.stderr, (entry, plain)
