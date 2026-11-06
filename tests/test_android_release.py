"""Android source delivery and release-proof boundaries, without device access."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tarfile
from pathlib import Path

import pytest

from ouroboros.tools.release_sync import (
    DESKTOP_DOWNLOAD_IDS,
    RELEASE_ASSET_TEMPLATES,
    VERSION_CARRIER_SPANS,
    release_asset_download_url,
    release_asset_name,
    version_carrier_desyncs,
)


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("android_release", REPO / "scripts/build_android_release.py")
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_android_assets_share_the_release_registry_without_changing_desktop_carriers():
    assert release_asset_name("android-arm64", "7.1.0-rc.1") == "Ouroboros-7.1.0-rc.1-android-arm64.tar.gz"
    assert release_asset_name("android-apk", "7.1.0-rc.1") == "Ouroboros-7.1.0-rc.1-android.apk"
    assert len(RELEASE_ASSET_TEMPLATES) == 9
    assert len(DESKTOP_DOWNLOAD_IDS) == 7
    assert not any("android" in span.carrier_id for span in VERSION_CARRIER_SPANS)
    references = "".join(
        f"[download-{key}]: {release_asset_download_url(key, '7.1.0')}\n"
        for key in DESKTOP_DOWNLOAD_IDS
    )
    assert version_carrier_desyncs("7.1.0", download_readme_text=references) == []


def _archive_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    output = tmp_path / "dist"
    for path in (source, stage, output):
        path.mkdir()
    for name in builder.REQUIRED_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("first-party source\n", encoding="utf-8")
    # A local credential/cache is not part of git's tracked source inventory.
    (source / "android/private.keystore").write_bytes(b"never publish")
    monkeypatch.setattr(builder, "ROOT", source)
    monkeypatch.setattr(builder, "run", lambda *_args, **_kwargs: "\0".join(builder.REQUIRED_FILES))

    def bundle(_root, bundle_path, manifest_path, **kwargs):
        assert kwargs["source_branch"] == "ouroboros"
        bundle_path.write_bytes(b"git bundle fixture")
        manifest_path.write_text(json.dumps({
            "source_sha": "a" * 40, "release_tag": "v7.1.0",
            "bundle_sha256": builder.RELEASE["sha256_file"](bundle_path),
        }))

    monkeypatch.setitem(builder.BUNDLE, "build_bundle", bundle)
    apk = output / "Ouroboros-7.1.0-android.apk"
    apk.write_bytes(b"verified APK fixture")
    identity = {"packageName": "ai.ouroboros.android", "versionName": "7.1.0",
                "versionCode": 12, "signerSha256": "b" * 64}
    args = argparse.Namespace(out=output, apk=apk, source_branch="ouroboros",
                              commit="a" * 40, tag="v7.1.0", version_code=12)
    archive = builder.create_archive(args, stage, "7.1.0", identity)
    return args, archive, identity


def test_archive_carries_only_tracked_source_and_binds_every_delivered_byte(tmp_path, monkeypatch):
    args, archive, identity = _archive_fixture(tmp_path, monkeypatch)
    with tarfile.open(archive) as handle:
        names = handle.getnames()
        assert not any("private.keystore" in name for name in names)
        assert "Ouroboros-Android/docs/ANDROID_RECOVERY.md" in names
        manifest = json.load(handle.extractfile("Ouroboros-Android/android_release_manifest.json"))
    assert set(manifest["files"]) == set(builder.REQUIRED_FILES) | {"repo.bundle", "repo_bundle_manifest.json"}
    assert manifest["sourceCommit"] == args.commit
    assert manifest["referenceApk"] == {"name": args.apk.name, **builder.file_record(args.apk), **identity}


@pytest.mark.parametrize("tamper", [False, True])
def test_final_archive_inspection_checks_bundle_and_installer_before_receipt(tmp_path, monkeypatch, tamper):
    args, archive, identity = _archive_fixture(tmp_path, monkeypatch)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return args.commit + "\n" if argv[:2] == ["git", "rev-parse"] else ""

    monkeypatch.setattr(builder, "run", run)
    monkeypatch.setattr(builder, "apk_identity", lambda *_args: identity)
    if tamper:
        args.apk.write_bytes(b"different APK after packaging")
        with pytest.raises(ValueError, match="Final Android APK differs"):
            builder.inspect_archive(args, archive, tmp_path / "extracted")
    else:
        assert builder.inspect_archive(args, archive, tmp_path / "extracted") == identity
        assert any(argv[:2] == ["git", "clone"] for argv in calls)
        assert any(
            Path(argv[1]).name == "install.py"
            and Path(argv[1]).parent.name == "android"
            and argv[-1] == "--help"
            for argv in calls
        )


def test_release_builder_refuses_missing_key_before_source_or_compiler_work(tmp_path, monkeypatch):
    monkeypatch.setattr(builder.sys, "argv", [
        "build_android_release.py", "--sdk", str(tmp_path), "--java-home", str(tmp_path),
        "--keystore", str(tmp_path / "absent.keystore"),
        "--keystore-pass-file", str(tmp_path / "absent.password"),
        "--version-code", "12", "--source-branch", "ouroboros",
        "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work"),
    ])
    monkeypatch.setattr(builder, "run", lambda *_args, **_kwargs: pytest.fail("no build may run without signing input"))
    with pytest.raises(SystemExit) as exc:
        builder.main()
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


def test_android_ci_is_fork_safe_and_experimental_for_publication():
    workflow = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    validation = workflow.split("  android-test:", 1)[1].split("  android-build:", 1)[0]
    release = workflow.split("  android-build:", 1)[1].split("  release-preflight:", 1)[0]
    publication = workflow.split("  release:\n", 1)[1]
    assert "secrets." not in validation
    assert "--create-development-key" in validation
    assert "python -m pytest android/tests tests/test_android_release.py" in validation
    assert "--create-development-key" not in release
    assert "if: startsWith(github.ref, 'refs/tags/v')" in release
    assert "needs: [android-test, android-emulator-smoke, release-preflight]" in release
    assert "publisher signing credentials are required" in release
    assert "android-build" in next(line for line in publication.splitlines() if "needs:" in line)
    assert "secrets." not in release.split("- name: Generate Android source", 1)[1]
    assert "always() && !cancelled()" in publication
    assert "needs.android-build.result == 'success'" not in publication
    assert "needs.build.result == 'success'" in publication
    assert "needs.skill-smoke.result == 'success'" in publication
    assert "fromJSON(steps.release_proof.outputs.files_json)" in publication
    assert "--android-build-result" in publication
    assert "--android-attestation-result" in publication
    assert "continue-on-error" not in publication
    assert "draft: true" in publication


def test_android_ci_has_representative_emulator_matrix_without_calling_it_device_qualification():
    import yaml

    workflow = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    smoke = workflow.split("  android-emulator-smoke:", 1)[1].split("  # The publisher key", 1)[0]
    assert "api-level: [26, 29, 30, 33, 36]" in smoke
    assert "needs: android-test" in smoke
    assert "needs.android-test.outputs.android_changed == 'true'" in smoke
    assert "startsWith(github.ref, 'refs/tags/v')" in smoke
    assert "adb install -r" in smoke
    assert "dumpsys package ai.ouroboros.android" in smoke
    assert "SELinux" not in smoke
    steps = yaml.safe_load(workflow)["jobs"]["android-emulator-smoke"]["steps"]
    setup = next(step for step in steps if step.get("uses", "").startswith("android-actions/setup-android@"))
    assert setup["with"]["packages"].split() == ["platform-tools"]


@pytest.mark.parametrize("job_name", ["android-test", "android-build"])
@pytest.mark.skipif(os.name == "nt", reason="Exercises the Ubuntu workflow's POSIX Bash step")
def test_android_sdk_cache_uses_the_exported_runner_path(tmp_path, job_name):
    import shutil
    import subprocess
    import yaml

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("the workflow's Bash runner is unavailable")
    jobs = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    steps = jobs[job_name]["steps"]
    cache_index = next(i for i, step in enumerate(steps) if step.get("uses", "").startswith("actions/cache@"))
    exporter = next(step["run"] for step in steps[:cache_index]
                    if "ANDROID_HOME=" in step.get("run", "") and "GITHUB_ENV" in step["run"])
    sdk = tmp_path / "SDK from runner"
    for relative in ("platforms/android-36", "build-tools/36.0.0"):
        (sdk / relative).mkdir(parents=True)
    env_file = tmp_path / "job-env"
    result = subprocess.run([bash, "-c", exporter], capture_output=True, text=True,
                            env={**os.environ, "ANDROID_HOME": str(sdk), "GITHUB_ENV": str(env_file)})
    assert result.returncode == 0, result.stderr
    exported = dict(line.split("=", 1) for line in env_file.read_text(encoding="utf-8").splitlines())
    assert exported["ANDROID_HOME"] == str(sdk)
    paths = steps[cache_index]["with"]["path"].splitlines()
    assert {path.replace("${{ env.ANDROID_HOME }}", exported["ANDROID_HOME"]) for path in paths} == {
        str(sdk / "platforms/android-36"), str(sdk / "build-tools/36.0.0"),
    }


def test_android_smoke_requirements_do_not_claim_a_device_was_tested():
    checks = builder.RELEASE["REQUIRED_SMOKE_CHECKS"]
    assert checks["android-arm64"] == {"embedded_repo_bundle", "android_source_manifest", "usb_installer_help"}
    assert checks["android-apk"] == {"apk_signature", "apk_package_version"}


def test_default_host_build_uses_the_shared_root_asset_path():
    source = (REPO / "android/host/build.py").read_text(encoding="utf-8")
    assert 'source.parents[1] / "assets" / "icon_1024.png"' in source
    assert (REPO / "assets/icon_1024.png").is_file()


@pytest.mark.parametrize(("event", "path", "ref", "expected"), [
    ("pull_request", "docs/readme.md", "refs/pull/1/merge", False),
    ("pull_request", "ouroboros/core.py", "refs/pull/1/merge", False),
    ("pull_request", "android/host/change.java", "refs/pull/1/merge", True),
    ("schedule", "android/host/change.java", "refs/heads/main", False),
])
@pytest.mark.skipif(os.name == "nt", reason="Exercises the Ubuntu workflow's POSIX Bash step")
def test_android_emulator_selection_uses_event_diff(tmp_path, event, path, ref, expected):
    import os
    import shutil
    import subprocess
    import yaml

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("the workflow's Bash runner is unavailable")
    jobs = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    script = next(step["run"] for step in jobs["android-test"]["steps"] if step.get("id") == "android_changes")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    changed = tmp_path / path
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("change", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "change"], cwd=tmp_path, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    output = tmp_path / "output"
    env = {**os.environ, "GITHUB_SHA": head, "GITHUB_OUTPUT": str(output),
           "PR_BASE": base if event == "pull_request" else "", "PR_HEAD": head if event == "pull_request" else "",
           "PUSH_BASE": base if event == "push" else ""}
    result = subprocess.run([bash, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    detected = output.read_text(encoding="utf-8").strip().split("=", 1)[1]
    expression = jobs["android-emulator-smoke"]["if"]
    for key, value in {"github.event_name": event, "github.ref": ref, "inputs.scope": "",  # no dispatch: inputs are empty
                       "needs.android-test.outputs.android_changed": detected}.items():
        expression = expression.replace(key, repr(value))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    assert eval(expression, {"__builtins__": {}}, {"startsWith": str.startswith}) is expected


@pytest.mark.parametrize(("failed_job", "result", "cancelled", "expected"), [
    ("android-build", "failure", False, True),
    ("android-build", "cancelled", False, True),
    ("android-build", "skipped", False, True),
    *[(job, "failure", False, False) for job in ("build", "release-preflight", "marker-guards",
       "ui-smoke", "docker-ui-smoke", "docker-portable-test", "skill-smoke")],
    (None, "success", True, False),
])
def test_release_requires_desktop_gates_and_respects_workflow_cancel(failed_job, result, cancelled, expected):
    import re
    import yaml

    job = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]["release"]
    expression = job["if"].removeprefix("${{").removesuffix("}}")
    for name in job["needs"]:
        expression = expression.replace(f"needs.{name}.result", repr(result if name == failed_job else "success"))
    expression = expression.replace("github.ref", repr("refs/tags/v7.0.0"))
    expression = re.sub(r"!(?!=)", "not ", expression).replace("&&", " and ").replace("||", " or ")
    assert eval(f"({expression.strip()})", {"__builtins__": {}}, {
        "always": lambda: True, "cancelled": lambda: cancelled, "startsWith": str.startswith,
    }) is expected


@pytest.mark.parametrize(("failed_platform", "expected"), [("android-apk", 0), ("macos-arm64", 1), (None, 0)])
@pytest.mark.skipif(os.name == "nt", reason="Executes the Ubuntu release step with a POSIX gh fixture")
def test_attestation_command_failure_excludes_android_but_stops_desktop(tmp_path, failed_platform, expected):
    import os
    import shutil
    import subprocess
    import sys
    import yaml

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("the workflow's Bash runner is unavailable")
    workflow = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    script = next(step["run"] for step in workflow["jobs"]["release"]["steps"] if step.get("id") == "verify_artifacts")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "python").symlink_to(sys.executable)
    gh = binaries / "gh"
    gh.write_text(f"#!{sys.executable}\nimport os,sys,pathlib\n"
                  "with open(os.environ['GH_CALL_LOG'],'a',encoding='utf-8') as stream:\n"
                  "    stream.write(pathlib.Path(sys.argv[3]).name+'\\n')\n"
                  "raise SystemExit(1 if pathlib.Path(sys.argv[3]).name == os.environ['FAIL_ARTIFACT'] else 0)\n",
                  encoding="utf-8")
    gh.chmod(0o755)
    version = (REPO / "VERSION").read_text(encoding="utf-8").strip()
    output, calls = tmp_path / "output", tmp_path / "calls"
    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", ""),
           "RUNNER_TEMP": str(tmp_path), "GITHUB_OUTPUT": str(output), "ANDROID_BUILD_RESULT": "success",
           "GITHUB_REPOSITORY": "example/source", "GITHUB_SHA": "a" * 40, "GITHUB_REF": "refs/tags/v" + version,
           "GH_CALL_LOG": str(calls), "FAIL_ARTIFACT": release_asset_name(failed_platform, version) if failed_platform else ""}
    result = subprocess.run([bash, "-c", script], cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
    if expected == 0:
        outcome = "failure" if failed_platform else "success"
        assert output.read_text(encoding="utf-8").strip() == "android_result=" + outcome
        observed = calls.read_text(encoding="utf-8").splitlines()
        assert {release_asset_name(key, version) for key in DESKTOP_DOWNLOAD_IDS} <= set(observed)


@pytest.mark.parametrize("api", [26, 29, 36])
@pytest.mark.skipif(os.name == "nt", reason="Exercises the Ubuntu emulator setup's Bash step")
def test_emulator_image_selection_uses_google_apis_only_for_oreo(tmp_path, api):
    import shutil
    import subprocess
    import yaml

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("the workflow's Bash runner is unavailable")
    jobs = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    script = next(step["run"] for step in jobs["android-emulator-smoke"]["steps"]
                  if step.get("name") == "Install emulator API and compiler")
    script = script.replace("${{ matrix.api-level }}", str(api))
    sdk = tmp_path / "sdk"
    (sdk / "emulator").mkdir(parents=True)
    executable = sdk / "emulator/emulator"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    setup = ('sdkmanager() { printf "%s\\n" "$@" > "$RUNNER_TEMP/sdk-args"; }\n'
             'avdmanager() { cat >/dev/null; printf "%s\\n" "$@" > "$RUNNER_TEMP/avd-args"; }\n')
    result = subprocess.run([bash, "-c", setup + script], capture_output=True, text=True,
                            env={**os.environ, "RUNNER_TEMP": str(tmp_path), "ANDROID_HOME": str(sdk),
                                 "GITHUB_ENV": str(tmp_path / "job-env")})
    assert result.returncode == 0, result.stderr
    expected = "system-images;android-26;google_apis;x86" if api == 26 else f"system-images;android-{api};default;x86_64"
    for name in ("sdk-args", "avd-args"):
        args = (tmp_path / name).read_text(encoding="utf-8").splitlines()
        assert [arg for arg in args if arg.startswith("system-images;")] == [expected]


@pytest.mark.parametrize(("case", "expected_exit", "attempts"), [
    ("recover", 0, 3), ("dump_error", 7, 5), ("stale", 9, 5),
])
@pytest.mark.skipif(os.name == "nt", reason="Exercises the Ubuntu capture and EXIT trap in Bash")
def test_emulator_capture_retries_fresh_xml_and_preserves_failure_evidence(tmp_path, case, expected_exit, attempts):
    import shutil
    import subprocess
    import yaml

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("the workflow's Bash runner is unavailable")
    jobs = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))["jobs"]
    script = next(step["run"] for step in jobs["android-emulator-smoke"]["steps"]
                  if step.get("name") == "Boot emulator and smoke install/start")
    capture = "capture_native_ui() {" + script.split("capture_native_ui() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
    trap = next(line for line in script.splitlines() if line.startswith("trap "))
    assertion = next(line for line in script.splitlines() if line.startswith("grep -Fq 'text="))
    smoke = tmp_path / "android-smoke"
    screenshots = smoke / "screenshots"
    screenshots.mkdir(parents=True)
    old_xml = '<hierarchy><node text="Ouroboros access to your phone"/></hierarchy>'
    (tmp_path / "remote.xml").write_text(old_xml, encoding="utf-8")
    (screenshots / "01-access-setup.xml").write_text(old_xml, encoding="utf-8")
    (tmp_path / "count").write_text("0", encoding="utf-8")
    (smoke / "emulator.log").write_text("emulator evidence\n", encoding="utf-8")
    # Command doubles exercise the real workflow shell, without starting adb or a device.
    commands = r'''
set -euo pipefail
timeout() { shift; "$@"; }
sleep() { :; }
kill() { :; }
adb() {
  case "$1 $2" in
    "shell rm") rm -f "$REMOTE_XML" ;;
    "shell uiautomator")
      count=$(( $(cat "$COUNT") + 1 )); printf '%s' "$count" > "$COUNT"
      case "$CASE" in
        recover) if [ "$count" -lt 3 ]; then return 0; fi ;;
        dump_error) return 7 ;;
        stale) return 0 ;;
        crash_dialog) printf '<hierarchy><node text="Ouroboros has stopped"/></hierarchy>' > "$REMOTE_XML"; return 0 ;;
      esac
      printf '<hierarchy><node text="Ouroboros access to your phone"/></hierarchy>' > "$REMOTE_XML"
      ;;
    "pull /data/local/tmp/obo-native-ui.xml")
      if [ ! -s "$REMOTE_XML" ]; then return 9; fi
      cp "$REMOTE_XML" "$3" ;;
    "exec-out screencap") printf 'screenshot command executed' ;;
    "logcat -d") printf 'full logcat evidence' ;;
    *) return 88 ;;
  esac
}
'''
    env = {**os.environ, "CASE": case, "COUNT": str(tmp_path / "count"),
           "REMOTE_XML": str(tmp_path / "remote.xml"), "RUNNER_TEMP": str(tmp_path),
           "SCREENSHOTS": str(screenshots), "LOG": str(smoke / "emulator.log"), "emulator_pid": "fixture"}
    result = subprocess.run([bash, "-c", commands + trap + "\n" + capture
                             + "capture_native_ui 01-access-setup\n" + assertion],
                            env=env, capture_output=True, text=True)
    assert result.returncode == expected_exit, result.stderr
    assert int((tmp_path / "count").read_text(encoding="utf-8")) == attempts
    assert (screenshots / "01-access-setup.png").read_bytes() == b"screenshot command executed"
    assert (smoke / "logcat.txt").read_text(encoding="utf-8") == "full logcat evidence"
    if case in {"stale", "dump_error"}:
        assert not (screenshots / "01-access-setup.xml").exists()
