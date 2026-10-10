"""The managed CLI toolchain witness's own logic, under the ordinary suite.

Only what needs no network: the PATH scrub, the process rebinding done before the
runtime is imported, the refusals that precede any install, the harness receipt and
doctor verdicts, daemon cleanup, process custody, CI wiring, and Node repair's
directory transaction under injected sharing refusals. Real downloads, Node/npm
execution and the pinned Codex install belong to the `toolchain` and Windows
`consumer` CI jobs. Local repair tests use archive fixtures, not an executable Node.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest
import yaml

from ouroboros.claudexor_runtime import ClaudexorRuntimeManager

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_witness():
    script = REPO_ROOT / "scripts" / "claudexor_toolchain_smoke.py"
    spec = importlib.util.spec_from_file_location("claudexor_toolchain_smoke", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


witness = _load_witness()


def _tool(directory: pathlib.Path, name: str) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"{name}.cmd" if os.name == "nt" else name)
    path.write_text("", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_path_scrub_drops_every_node_and_npm_provider(tmp_path):
    node_dir, npm_dir, clean = tmp_path / "node", tmp_path / "npm", tmp_path / "clean"
    _tool(node_dir, "node")
    _tool(npm_dir, "npx")
    _tool(clean, "git")
    value = os.pathsep.join([str(node_dir), str(clean), str(npm_dir)])

    kept, dropped = witness.scrubbed_path(value)

    assert kept == str(clean)
    assert dropped == [str(node_dir), str(npm_dir)]


def test_isolation_rebinds_home_and_drops_overrides_before_runtime_imports(tmp_path, monkeypatch):
    node_dir, clean = tmp_path / "ambient-node", tmp_path / "clean"
    _tool(node_dir, "npm")
    clean.mkdir()
    monkeypatch.setattr(witness.os, "environ", {
        "PATH": os.pathsep.join([str(node_dir), str(clean)]),
        "HOME": str(tmp_path / "operator"),
        "OUROBOROS_DATA_DIR": str(tmp_path / "live-data"),
        "OUROBOROS_BUNDLE_DIR": str(tmp_path / "bundle"),
        "CLAUDEXOR_CONFIG_DIR": str(tmp_path / "live-engine"),
        "CLAUDEXOR_CODEX_BIN": str(tmp_path / "codex.exe"),
        "npm_config_prefix": str(tmp_path / "npm-prefix"),
        "NODE_OPTIONS": "--require=hook.js",
        "OPENAI_API_KEY": "fixture-secret",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    })
    root = tmp_path / "isolated"

    dropped = witness.isolate_environment(root)

    env = witness.os.environ
    assert dropped == [str(node_dir)] and env["PATH"] == str(clean)
    assert env["HOME"] == env["USERPROFILE"] == str(root / "home")
    assert env["LOCALAPPDATA"].startswith(str(root / "home"))
    # No Ouroboros root is set: production derives app/data from the disposable home.
    assert not [key for key in env if key.startswith(("OUROBOROS_", "CLAUDEXOR_"))]
    assert not {"npm_config_prefix", "NODE_OPTIONS", "OPENAI_API_KEY"} & set(env)
    assert env["GITHUB_STEP_SUMMARY"] == str(tmp_path / "summary.md")


@pytest.mark.parametrize("ambient", ("path", "override"))
def test_witness_refuses_ambient_node_or_override_before_the_runtime(tmp_path, monkeypatch, ambient):
    node_dir = tmp_path / "ambient"
    _tool(node_dir, "node")
    env = {"PATH": str(node_dir) if ambient == "path" else str(tmp_path)}
    if ambient == "override":
        env["CLAUDEXOR_CODEX_BIN"] = str(tmp_path / "codex.exe")
    monkeypatch.setattr(witness.os, "environ", env)

    with pytest.raises(witness.WitnessFailure) as excinfo:
        witness.run_witness(tmp_path, [])

    assert excinfo.value.code == (
        "ambient_node_present" if ambient == "path" else "ambient_override_present")


def test_witness_expects_npm_where_the_official_distribution_and_manager_keep_it():
    for node in (pathlib.Path("cx/node-standalone/node.exe"), pathlib.Path("cx/node-standalone/bin/node")):
        assert witness.official_npm_cli(node) == ClaudexorRuntimeManager._managed_npm_cli(node)
    assert witness.official_npm_cli(pathlib.Path("n/node.exe")).parent.parent.parent == pathlib.Path(
        "n/node_modules")


def test_main_requires_an_empty_root(tmp_path):
    (tmp_path / "stale-cache").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit):
        witness.main(["--root", str(tmp_path)])


@pytest.mark.serial
def test_isolated_start_ignores_ambient_pythonpath_before_script_scrub(tmp_path):
    injected = tmp_path / "injected"
    injected.mkdir()
    marker = tmp_path / "sitecustomize-ran"
    (injected / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n", encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-I", str(REPO_ROOT / "scripts" / "claudexor_toolchain_smoke.py"), "--help"],
        env={**os.environ, "PYTHONPATH": str(injected)},
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "--root" in result.stdout
    assert not marker.exists()


def test_a_refusal_is_named_and_the_limits_always_reach_the_summary(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(witness.os, "environ", {"GITHUB_STEP_SUMMARY": str(summary)})
    monkeypatch.setattr(witness, "isolate_environment", lambda _root: [])

    def refuse(_root, _dropped, _harness=None):
        raise witness.WitnessFailure("runtime_node_archive_invalid", "fixture refusal")

    monkeypatch.setattr(witness, "run_witness", refuse)

    assert witness.main(["--root", str(tmp_path / "root")]) == 1
    written = summary.read_text(encoding="utf-8")
    assert "FAILED" in written and "`runtime_node_archive_invalid`" in written
    assert "No vendor harness was installed" in written


def test_platform_gate_runs_the_toolchain_witness_on_windows_pull_requests():
    path = REPO_ROOT / ".github" / "workflows" / "claudexor-platform-gate.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    for event in ("push", "pull_request"):
        assert "scripts/claudexor_toolchain_smoke.py" in triggers[event]["paths"]
    assert triggers["pull_request"]["branches"] == ["ouroboros"]
    job = workflow["jobs"]["toolchain"]
    assert "if" not in job
    assert "windows-latest" in job["strategy"]["matrix"]["os"]
    steps = job["steps"]
    assert not [step for step in steps if "setup-node" in str(step.get("uses", ""))]
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    assert "python -I scripts/claudexor_toolchain_smoke.py --root" in commands
    # The witness proves the toolchain only; it neither pre-seeds a harness nor claims one.
    assert "CLAUDEXOR_CODEX_BIN" not in commands and "harness install" not in commands
    assert "--harness-install" not in commands
    assert "npm " not in commands


def test_platform_gate_runs_the_real_codex_install_on_windows_without_credentials_or_node():
    path = REPO_ROOT / ".github" / "workflows" / "claudexor-platform-gate.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["consumer"]
    # Unconditional (pushes and PRs), Windows only, and its caveat is in the check name.
    assert "if" not in job and "strategy" not in job
    assert job["runs-on"] == "windows-latest"
    assert "no login, no model task" in job["name"]
    steps = job["steps"]
    assert steps[0]["with"]["persist-credentials"] is False
    assert not [step for step in steps if "setup-node" in str(step.get("uses", ""))]
    assert not [step for step in steps if step.get("env")]
    commands = "\n".join(str(step["run"]) for step in steps if "run" in step)
    assert commands == (
        'python -I scripts/claudexor_toolchain_smoke.py --root "$RUNNER_TEMP/cx-consumer" '
        "--harness-install codex"
    )
    assert "secrets." not in str(job)


def _receipt(binary: pathlib.Path, **overrides):
    receipt = {
        "ok": True, "dryRun": False, "exitCode": 0, "target": "local", "harness": "codex",
        "command": "npm install --global --prefix ~/.claudexor/node @openai/codex@1.2.3",
        "installLocation": "~/.claudexor/node/node_modules/@openai/codex/.../bin",
        "installedBinary": str(binary), "installedVersion": "1.2.3", "pinnedVersion": "1.2.3",
        "verification": "release_verified",
    }
    receipt.update(overrides)
    return receipt


def _image(toolchain: pathlib.Path, name: str = "codex.exe") -> pathlib.Path:
    image = toolchain / "node_modules" / "@openai" / "codex" / "vendor" / "bin" / name
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_text("", encoding="utf-8")
    return image


def _node_entry(toolchain: pathlib.Path, *, string_bin=False, name="codex.js") -> pathlib.Path:
    package = toolchain / "node_modules" / "@openai" / "codex"
    entry = package / "bin" / name
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("#!/usr/bin/env node\nconsole.log('codex-cli 1.2.3');\n", encoding="utf-8")
    (package / "package.json").write_text(json.dumps({
        "name": "@openai/codex", "version": "1.2.3",
        "bin": f"bin/{name}" if string_bin else {"codex": f"bin/{name}"},
    }), encoding="utf-8")
    return entry


def test_the_receipt_must_name_the_release_verified_native_image_in_the_disposable_home(tmp_path):
    toolchain = tmp_path / "home" / ".claudexor" / "node"
    image = _image(toolchain)
    assert witness.verify_install_receipt(_receipt(image), "codex", toolchain, windows=True) == image

    shim = _image(toolchain, "codex.cmd")
    outside = tmp_path / "operator" / "codex.exe"
    outside.parent.mkdir()
    outside.write_text("", encoding="utf-8")
    for receipt, windows in (
        (_receipt(shim), True),  # an npm shim is never the launcher on Windows
        (_receipt(outside), True),  # never the owner's (or any other) home
        (_receipt(image, verification="deterministic_only"), True),
        (_receipt(image, installedVersion="1.2.4"), True),
        (_receipt(toolchain / "missing.exe"), True),
        (_receipt(image, harness="claude"), True),  # the production contract itself
        ({**_receipt(image), "extra": 1}, False),
    ):
        with pytest.raises(witness.WitnessFailure) as excinfo:
            witness.verify_install_receipt(receipt, "codex", toolchain, windows=windows)
        assert excinfo.value.code == "harness_receipt_invalid"


@pytest.mark.parametrize("string_bin", [False, True])
def test_windows_receipt_accepts_the_exact_npm_declared_node_entry(tmp_path, string_bin):
    toolchain = tmp_path / "home" / ".claudexor" / "node"
    entry = _node_entry(toolchain, string_bin=string_bin)
    assert witness.verify_install_receipt(_receipt(entry), "codex", toolchain, windows=True) == entry


@pytest.mark.parametrize("fault", ["bin", "version", "package", "header", "cmd", "ps1"])
def test_windows_node_receipt_requires_package_binding_and_node_source(tmp_path, fault):
    toolchain = tmp_path / "home" / ".claudexor" / "node"
    entry = _node_entry(toolchain, name=f"codex.{fault}" if fault in {"cmd", "ps1"} else "codex.js")
    manifest = entry.parent.parent / "package.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if fault == "bin":
        metadata["bin"] = {"codex": "bin/another.js"}
    elif fault == "version":
        metadata["version"] = "1.2.30"
    elif fault == "package":
        metadata["name"] = "other-package"
    elif fault == "header":
        entry.write_text("#!/bin/sh\necho codex-cli 1.2.3\n", encoding="utf-8")
    manifest.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(witness.WitnessFailure) as excinfo:
        witness.verify_install_receipt(_receipt(entry), "codex", toolchain, windows=True)
    assert excinfo.value.code == "harness_receipt_invalid"


@pytest.mark.parametrize("kind", ["native", "node"])
def test_the_receipt_executable_must_resolve_inside_the_toolchain(tmp_path, kind):
    toolchain = tmp_path / "home" / ".claudexor" / "node"
    image = _image(toolchain) if kind == "native" else _node_entry(toolchain)
    operator = tmp_path / "operator" / "bin"
    operator.mkdir(parents=True)
    (operator / image.name).write_text("", encoding="utf-8")
    links = toolchain / "links"
    (links / "file").mkdir(parents=True)
    inside, file_escape, dir_escape = links / image.name, links / "file" / image.name, links / "dir"
    try:
        inside.symlink_to(image)  # npm's own bin links stay inside the toolchain
        file_escape.symlink_to(operator / image.name)
        dir_escape.symlink_to(operator, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this host cannot create symlinks: {exc}")

    assert witness.verify_install_receipt(_receipt(inside), "codex", toolchain, windows=True) == inside
    for escape in (file_escape, dir_escape / image.name):
        with pytest.raises(witness.WitnessFailure) as excinfo:
            witness.verify_install_receipt(_receipt(escape), "codex", toolchain, windows=True)
        assert excinfo.value.code == "harness_receipt_invalid"
        # Lexically beneath the toolchain; only the effective path gives it away.
        assert "resolves outside" in str(excinfo.value) and "not beneath" not in str(excinfo.value)


# A vendor CLI that leaves a detached grandchild behind, then exits or hangs.
_SPAWNS_A_GRANDCHILD = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],"
    " stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
    "open(sys.argv[1], 'w').write(str(child.pid))\n"
    "print('codex-cli 1.2.3', flush=True)\n"
    "if sys.argv[2] == 'hang':\n"
    "    time.sleep(600)\n"
)


@pytest.mark.serial
@pytest.mark.parametrize("mode", ["exit", "hang"])
def test_a_vendor_probe_leaves_no_process_behind(tmp_path, mode):
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.process_containment import pid_is_zombie

    pid_file = tmp_path / "grandchild.pid"
    argv = [sys.executable, "-c", _SPAWNS_A_GRANDCHILD, pid_file, mode]
    if mode == "exit":
        assert witness._contained_run(argv, "harness_direct_failed", dict(os.environ),
                                      timeout=60) == "codex-cli 1.2.3"
    else:
        with pytest.raises(witness.WitnessFailure) as excinfo:
            witness._contained_run(argv, "harness_by_name_failed", dict(os.environ), timeout=5)
        assert excinfo.value.code == "harness_by_name_failed"
        assert "process tree reaped" in str(excinfo.value)
    grandchild = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while pid_is_alive(grandchild) and not pid_is_zombie(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not pid_is_alive(grandchild) or pid_is_zombie(grandchild)


@pytest.mark.parametrize("kind", ["native", "node"])
@pytest.mark.parametrize("version_output", ["codex-cli 1.2.3", "codex-cli 1.2.30"])
def test_vendor_launch_uses_managed_transport_and_engine_doctor_in_custody(
        tmp_path, monkeypatch, kind, version_output):
    import ouroboros.claudexor_daemon as owned
    import ouroboros.config as config

    toolchain = tmp_path / ".claudexor" / "node"
    node = tmp_path / "managed-node" / "node.exe"
    image = (toolchain / "bin" / "codex.exe" if kind == "native" else
             toolchain / "node_modules" / "@openai" / "codex" / "bin" / "codex.js")

    def install(_harness):
        if kind == "node":
            assert _node_entry(toolchain) == image
        else:
            image.parent.mkdir(parents=True)
            image.write_text("", encoding="utf-8")

    def uncontained(*_args, **_kwargs):
        raise AssertionError("a vendor executable ran outside process custody")

    probes, doctors = [], []

    def contained(argv, code, env, timeout=0):
        probes.append((list(argv), code))
        assert env["PATH"].split(os.pathsep)[0] == str(node.parent)
        return version_output

    def doctor(*args):
        doctors.append(args)
        return {"daemon_stop": "stopped"}

    monkeypatch.setattr(witness.os, "environ", {"HOME": str(tmp_path)})
    monkeypatch.setattr(owned, "install_missing_harness_cli", install)
    monkeypatch.setattr(config, "get_claudexor_harness_install_timeout_sec", lambda: 1)
    monkeypatch.setattr(witness, "_cli_json", lambda *_args, **_kwargs: _receipt(image))
    monkeypatch.setattr(witness, "_run", uncontained)
    monkeypatch.setattr(witness, "_contained_run", contained)
    monkeypatch.setattr(witness, "doctor_witness", doctor)

    if version_output == "codex-cli 1.2.30":
        with pytest.raises(witness.WitnessFailure) as excinfo:
            witness.harness_install_witness("codex", node, ["node", "cli"], {}, windows=True)
        assert excinfo.value.code == "harness_direct_failed" and not doctors
    else:
        facts = witness.harness_install_witness("codex", node, ["node", "cli"], {}, windows=True)
        assert facts["direct_version"] == version_output
        assert facts["by_name_probe"] == "engine_doctor"
        assert doctors == [("codex", ["node", "cli"], {}, image, "1.2.3")]

    expected = [image, "--version"] if kind == "native" else [node, image, "--version"]
    assert probes == [(expected, "harness_direct_failed")]


def test_doctor_must_resolve_the_harness_to_the_receipts_image_and_pin(tmp_path):
    image = tmp_path / "Codex.exe"

    def report(status="pass", detail=f"codex-cli 1.2.3 at {str(image).upper()}"):
        return {"harnesses": [{"id": "claude", "checks": []}, {
            "id": "codex", "status": "unavailable",
            "checks": [{"id": "installed", "status": status, "detail": detail},
                       {"id": "native_session", "status": "fail", "detail": "not logged in"}],
        }]}

    # Not logged in is reported, not asserted: only the installed row is required.
    assert witness.doctor_installed_check(report(), "codex", image, "1.2.3")["status"] == "unavailable"
    for broken in (report(status="fail"), report(detail=f"codex-cli 1.2.2 at {image}"),
                   report(detail=f"codex-cli 1.2.30 at {image}"),
                   report(detail="codex-cli 1.2.3 at C:\\other\\codex.exe"), {"harnesses": []}, {}):
        with pytest.raises(witness.WitnessFailure) as excinfo:
            witness.doctor_installed_check(broken, "codex", image, "1.2.3")
        assert excinfo.value.code == "harness_doctor_unresolved"


class _OwnedDaemon:
    def __init__(self, outcomes=("stopped",)):
        self.outcomes, self.stops = list(outcomes), 0

    def stop_outcome(self):
        self.stops += 1
        return self.outcomes[min(self.stops, len(self.outcomes)) - 1]


class _Gateway:
    def __init__(self, rows):
        self.rows, self.closed = rows, False

    def harnesses(self):
        return self.rows

    def close(self):
        self.closed = True


def _owned_seam(monkeypatch, image, *, start=None, outcomes=("stopped",), cli_fails=False):
    """Fake only the production owned-daemon seam and the engine CLI around the doctor."""
    import ouroboros.claudexor_daemon as owned

    rows = [{"id": "codex", "status": "unavailable", "checks": [
        {"id": "installed", "status": "pass", "detail": f"codex-cli 1.2.3 at {image}"}]}]
    daemon, gateway, verbs = _OwnedDaemon(outcomes), _Gateway(rows), []

    def ensure():
        if start is not None:
            raise start
        return gateway

    def fake_cli(argv, code, env, timeout=0):
        verbs.append(tuple(argv[2:]))
        if cli_fails:
            raise witness.WitnessFailure(code, "fixture")
        return {"harnesses": rows}

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: daemon)
    monkeypatch.setattr(owned, "ensure_owned_gateway", ensure)
    monkeypatch.setattr(witness, "_cli_json", fake_cli)
    return daemon, gateway, verbs


def test_the_doctor_rides_the_owned_daemon_and_stops_it(tmp_path, monkeypatch):
    image = tmp_path / "codex.exe"
    daemon, gateway, verbs = _owned_seam(monkeypatch, image)

    facts = witness.doctor_witness("codex", ["node", "cli"], {}, image, "1.2.3")

    assert facts["daemon_stop"] == "stopped" and daemon.stops == 1 and gateway.closed
    assert facts["daemon_harnesses"]["status"] == facts["cli_doctor"]["status"] == "unavailable"
    # The full CLI doctor is independently exercised by Claudexor's Windows
    # release smoke; the witness itself still selects and verifies codex below.
    assert verbs == [("doctor", "--json")]


def test_a_failed_doctor_still_stops_the_owned_daemon(tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    image = tmp_path / "codex.exe"
    daemon, _gateway, _verbs = _owned_seam(monkeypatch, image, cli_fails=True)
    with pytest.raises(witness.WitnessFailure) as excinfo:
        witness.doctor_witness("codex", ["node", "cli"], {}, image, "1.2.3")
    assert excinfo.value.code == "harness_doctor_failed" and daemon.stops == 1

    daemon, _gateway, verbs = _owned_seam(
        monkeypatch, image, start=ClaudexorUnavailable("daemon_start_failed", "fixture"))
    with pytest.raises(witness.WitnessFailure) as excinfo:
        witness.doctor_witness("codex", ["node", "cli"], {}, image, "1.2.3")
    assert excinfo.value.code == "daemon_start_failed" and daemon.stops == 1 and verbs == []


def test_an_unconfirmed_stop_settles_only_to_nothing_left_alive(tmp_path, monkeypatch):
    image = tmp_path / "codex.exe"
    monkeypatch.setattr(witness.time, "sleep", lambda _sec: None)
    # A group member outliving the confirmed shutdown by a moment: the re-check finds nothing.
    daemon, _gateway, _verbs = _owned_seam(
        monkeypatch, image, outcomes=("unconfirmed", "unconfirmed", "nothing_to_stop"))
    facts = witness.doctor_witness("codex", ["node", "cli"], {}, image, "1.2.3")
    assert facts["daemon_stop"] == "nothing_to_stop" and daemon.stops == 3

    monkeypatch.setattr(witness, "STOP_SETTLE_SEC", 0)
    _owned_seam(monkeypatch, image, outcomes=("unconfirmed",))
    with pytest.raises(witness.WitnessFailure) as excinfo:
        witness.doctor_witness("codex", ["node", "cli"], {}, image, "1.2.3")
    assert excinfo.value.code == "doctor_daemon_not_stopped"


def test_a_preseeded_engine_toolchain_is_refused_before_any_install(tmp_path, monkeypatch):
    (tmp_path / ".claudexor" / "node").mkdir(parents=True)
    monkeypatch.setattr(witness.os, "environ", {"HOME": str(tmp_path)})

    def never(*_args, **_kwargs):
        raise AssertionError("no CLI verb may run over a pre-seeded toolchain")

    monkeypatch.setattr(witness, "_cli_json", never)
    with pytest.raises(witness.WitnessFailure) as excinfo:
        witness.harness_install_witness("codex", tmp_path / "node", ["node", "cli"], {})
    assert excinfo.value.code == "harness_preinstalled"


@pytest.mark.serial
def test_cli_json_keeps_the_engines_typed_refusal_code(tmp_path):
    refusal = json.dumps({"ok": False, "dryRun": True, "code": "unsupported_platform",
                          "refusal": "--target local is not supported on Windows"})
    script = f"import sys; print({refusal!r}); sys.exit(1)"
    with pytest.raises(witness.WitnessFailure) as excinfo:
        witness._cli_json([sys.executable, "-c", script, "install"], "harness_install_refused",
                          dict(os.environ))
    assert excinfo.value.code == "unsupported_platform"
    assert "not supported on Windows" in str(excinfo.value)
    with pytest.raises(witness.WitnessFailure) as excinfo:
        witness._cli_json([sys.executable, "-c", "print('not json')", "x"], "fallback_code",
                          dict(os.environ))
    assert excinfo.value.code == "fallback_code"


def test_the_harness_summary_carries_its_own_limits(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(witness.os, "environ", {"GITHUB_STEP_SUMMARY": str(summary)})
    monkeypatch.setattr(witness, "isolate_environment", lambda _root: [])
    seen = []

    def refuse(_root, _dropped, harness=None):
        seen.append(harness)
        raise witness.WitnessFailure("unsupported_platform", "fixture refusal")

    monkeypatch.setattr(witness, "run_witness", refuse)

    assert witness.main(["--root", str(tmp_path / "root"), "--harness-install", "codex"]) == 1
    written = summary.read_text(encoding="utf-8")
    assert seen == ["codex"] and "`codex` local install" in written
    assert "`unsupported_platform`" in written and "No login, OAuth, account or model task" in written
    assert "No vendor harness was installed" not in written
    with pytest.raises(SystemExit):
        witness.main(["--root", str(tmp_path / "other"), "--harness-install", "claude"])


@pytest.mark.serial
@pytest.mark.parametrize("fault", [
    "none", "displacement", "promotion", "rollback",
    "persistent_displacement", "persistent_promotion", "non_permission",
])
def test_node_repair_directory_transaction_preserves_success_and_failure(tmp_path, monkeypatch, fault):
    """The real repair transaction retries shared-handle refusals, including undo."""
    import zipfile
    from ouroboros import claudexor_runtime as runtime, platform_layer as platform, utils
    from tests.test_claudexor_runtime_delivery import _archive, _data_plane, _node_artifacts, _pin, NODE_VERSION

    _data_plane(monkeypatch, tmp_path)
    distribution = f"node-v{NODE_VERSION}-win-x64"
    archive = tmp_path / f"{distribution}.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"{distribution}/node.exe", b"node fixture\n")
        bundle.writestr(f"{distribution}/node_modules/npm/bin/npm-cli.js", b"npm fixture\n")
    pin = _pin(_archive(tmp_path / "closure.tar.gz"),
               node_artifacts=_node_artifacts(exact_key="win32-x64", exact_archive=archive))
    manager = runtime.ClaudexorRuntimeManager(pin)
    monkeypatch.setattr(platform, "embedded_node_candidates",
                        lambda base: [pathlib.Path(base) / "node-standalone" / "node.exe"])
    monkeypatch.setattr(platform, "probe_node_version", lambda path: NODE_VERSION if pathlib.Path(path).is_file() else "")
    artifact = pin.node_artifacts["win32-x64"]
    manager._promote_node(pin, "win32-x64", artifact, archive, include_npm=True)
    root = runtime.managed_node_dir(pin, "win32-x64")
    npm_cli = root / "node-standalone" / "node_modules" / "npm" / "bin" / "npm-cli.js"
    npm_cli.unlink()  # The same deliberate missing-entry repair as the CI witness.
    old_tree = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    original_archive = archive.read_bytes()
    real_replace = runtime.os.replace
    calls = {"displacement": 0, "promotion": 0, "rollback": 0}
    denied = PermissionError(13, "fixture Windows sharing refusal")
    denied.winerror = 5
    original_failure = OSError("fixture final promotion failure")

    def sharing_replace(src, dst):
        src, dst = pathlib.Path(src), pathlib.Path(dst)
        phase = ("displacement" if src == root and dst.name.startswith(".old-") else
                 "promotion" if src.name.startswith(".tmp-") and dst == root else
                 "rollback" if src.name.startswith(".old-") and dst == root else "")
        if phase:
            calls[phase] += 1
            if phase == "promotion" and fault in {"rollback", "non_permission"}:
                raise original_failure
            if fault == f"persistent_{phase}" or (fault == phase and calls[phase] <= 2):
                raise denied
        return real_replace(src, dst)

    monkeypatch.setattr(runtime.os, "replace", sharing_replace)
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)
    fails = fault in {"rollback", "non_permission", "persistent_displacement", "persistent_promotion"}
    if fails:
        with pytest.raises(runtime.ClaudexorRuntimeError) as caught:
            manager._promote_node(pin, "win32-x64", artifact, archive, include_npm=True)
        expected = denied if fault.startswith("persistent_") else original_failure
        assert caught.value.code == "runtime_node_install_failed"
        assert caught.value.__cause__ is expected
        assert str(expected) in str(caught.value)
        assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == old_tree
        assert not npm_cli.exists()
    else:
        manager._promote_node(pin, "win32-x64", artifact, archive, include_npm=True)
        assert npm_cli.read_bytes() == b"npm fixture\n"
        assert (root / "node-standalone" / "node.exe").read_bytes() == b"node fixture\n"
        assert json.loads((root / "managed-node.json").read_text(encoding="utf-8"))["schema_version"] == 2
    if fault.startswith("persistent_"):
        assert calls[fault.removeprefix("persistent_")] == utils._REPLACE_RETRY_ATTEMPTS
    elif fault in calls:
        assert calls[fault] == 3
    if fault == "non_permission":
        assert calls["promotion"] == calls["rollback"] == 1
    assert archive.read_bytes() == original_archive
    assert not list(root.parent.glob(".tmp-*")) and not list(root.parent.glob(".old-*"))
