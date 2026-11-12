"""The managed CLI toolchain witness's own logic, under the ordinary suite.

Only what needs no network: the PATH scrub, the process rebinding done before the
runtime is imported, the refusals that precede any install, the harness receipt and
doctor verdicts, daemon cleanup, process custody, CI wiring, and Node repair's
directory transaction under injected sharing refusals. Real downloads, Node/npm
execution and the vendor Codex install belong to the `toolchain` and Windows
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
        assert {"scripts/claudexor_toolchain_smoke.py", "ouroboros/harness_maintenance.py",
                "ouroboros/harness_program_facts.py"} <= set(triggers[event]["paths"])
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
        'python -I scripts/claudexor_toolchain_smoke.py --root "$env:RUNNER_TEMP/cx-consumer" '
        "--harness-install codex"
    )
    # The vendor installer resolves a bare `tar`: the witness starts from PowerShell, where
    # Windows's own tools come first, and not from the workflow's default Git Bash.
    assert workflow["defaults"]["run"]["shell"] == "bash"
    assert [step["shell"] for step in steps if "run" in step] == ["pwsh"]
    assert "secrets." not in str(job)


def _receipt(binary: pathlib.Path, **overrides):
    from tests.test_harness_maintenance_transport import OPERATION, PROGRAM
    return {**OPERATION, "harness": "codex", "state": "succeeded", "phase": "settled",
            "termination": "confirmed", "after": {**PROGRAM, "path": str(binary), "realPath": str(binary),
                "version": "1.2.3", "versionOutput": "codex-cli 1.2.3"}, **overrides}


def _image(home: pathlib.Path, name: str = "codex.exe") -> pathlib.Path:
    image = home / ".local" / "bin" / name
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_text("", encoding="utf-8")
    return image


def test_the_receipt_must_name_a_selected_program_in_the_disposable_home(tmp_path):
    home = tmp_path / "home"
    image = _image(home)
    assert witness.verify_install_receipt(_receipt(image), "codex", home, windows=True) == image
    outside = _image(tmp_path / "operator")
    for receipt in (_receipt(_image(home, "codex.cmd")), _receipt(outside), _receipt(home / "missing.exe"),
                    _receipt(image, harness="other-program"), _receipt(image, state="failed"),
                    _receipt(image, termination="unconfirmed"), _receipt(image, after=None),
                    _receipt(image, after={**_receipt(image)["after"], "version": None})):
        with pytest.raises(witness.WitnessFailure) as failure:
            witness.verify_install_receipt(receipt, "codex", home, windows=True)
        assert failure.value.code == "harness_receipt_invalid"


def test_receipt_links_must_resolve_inside_the_disposable_home(tmp_path):
    home = tmp_path / "home"
    image, outside = _image(home), _image(tmp_path / "operator")
    inside, escape = home / "inside.exe", home / "escape.exe"
    try:
        inside.symlink_to(image)
        escape.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this host cannot create symlinks: {exc}")
    assert witness.verify_install_receipt(_receipt(inside), "codex", home, windows=True) == inside
    with pytest.raises(witness.WitnessFailure, match="not a direct executable"):
        witness.verify_install_receipt(_receipt(escape), "codex", home, windows=True)


_SPAWNS_A_GRANDCHILD = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'],"
    " stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
    "open(sys.argv[1], 'w').write(str(child.pid))\n"
    "print('codex-cli 1.2.3', flush=True)\n"
    "if sys.argv[2] == 'hang':\n"
    "    time.sleep(600)\n"
)


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


@pytest.mark.parametrize(("version_output", "kind"), [("codex-cli 1.2.3", "native"), ("codex-cli 1.2.30", "node")])
def test_vendor_maintenance_uses_shared_service_then_probes_in_custody(tmp_path, monkeypatch, kind, version_output):
    from ouroboros import harness_maintenance as maintenance

    node = tmp_path / "managed-node" / "node.exe"
    image = _image(tmp_path, "codex.exe" if kind == "native" else "codex.js")
    daemon, gateway, _verbs = _owned_seam(monkeypatch, image)
    installs, probes, doctors = [], [], []
    monkeypatch.setattr(witness.os, "environ", {"HOME": str(tmp_path)})
    monkeypatch.setattr(maintenance, "install_for_connect", lambda harness: installs.append(harness) or _receipt(image))
    monkeypatch.setattr(witness, "_run", lambda *_a, **_kw: pytest.fail("uncontained vendor probe"))

    def contained(argv, code, env, timeout=0):
        probes.append((list(argv), code))
        assert env["PATH"].split(os.pathsep)[0] == str(node.parent)
        return version_output

    monkeypatch.setattr(witness, "_contained_run", contained)
    monkeypatch.setattr(witness, "doctor_witness", lambda *args: doctors.append(args) or {"daemon_stop": "stopped"})
    if kind == "node":
        with pytest.raises(witness.WitnessFailure) as failure:
            witness.harness_install_witness("codex", node, ["node", "cli"], {}, windows=True)
        assert failure.value.code == "harness_direct_failed" and not doctors
    else:
        facts = witness.harness_install_witness("codex", node, ["node", "cli"], {}, windows=True)
        assert facts["selected_version"] == "1.2.3" and facts["operation"] == _receipt(image)
        assert doctors == [("codex", ["node", "cli"], {}, image, "1.2.3")]
    expected = [image, "--version"] if kind == "native" else [node, image, "--version"]
    assert probes == [(expected, "harness_direct_failed")]
    assert installs == ["codex"] and daemon.stops == 1 and gateway.closed


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

    def maintenance_harnesses(self, names, **kwargs):
        from tests.test_harness_maintenance_transport import ENTRY, PROGRAM
        return {"harnesses": [{**ENTRY, "harness": names[0], "selection": {**PROGRAM, "path": None},
            "actions": {**ENTRY["actions"], "install": {"available": True, "reason": None, "commands": []}}}]}

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


@pytest.mark.parametrize("refusal", ["present", "unsupported", "install_failed"])
def test_witness_refusals_stop_the_engine_without_running_a_probe(tmp_path, monkeypatch, refusal):
    from ouroboros import harness_maintenance as maintenance
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    daemon, gateway, _verbs = _owned_seam(monkeypatch, tmp_path / "codex.exe")
    inventory = gateway.maintenance_harnesses(["codex"])
    row = inventory["harnesses"][0]
    if refusal == "present":
        row["selection"]["path"] = str(tmp_path / "codex.exe")
    elif refusal == "unsupported":
        row["actions"] = {**row["actions"], "install": {"available": False, "reason": "vendor refuses", "commands": []}}
    gateway.maintenance_harnesses = lambda *_a, **_kw: inventory
    installs = []

    def install(harness):
        installs.append(harness)
        raise ClaudexorUnavailable("vendor_failure", "fixture failed install")

    monkeypatch.setattr(maintenance, "install_for_connect", install)
    monkeypatch.setattr(witness, "_contained_run", lambda *_a, **_kw: pytest.fail("no probe after refusal"))
    with pytest.raises(witness.WitnessFailure) as failure:
        witness.harness_install_witness("codex", tmp_path / "node", ["node", "cli"], {})
    expected = {"present": "harness_preinstalled", "unsupported": "harness_install_refused", "install_failed": "vendor_failure"}
    assert failure.value.code == expected[refusal]
    assert installs == (["codex"] if refusal == "install_failed" else [])
    assert daemon.stops == 1 and gateway.closed


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
    assert witness.main(["--root", str(tmp_path / "other"), "--harness-install", "future-program"]) == 1
    assert seen == ["codex", "future-program"]


@pytest.mark.parametrize("fault", [
    "none", "persistent_promotion", "non_permission",
])
def test_node_repair_directory_transaction_preserves_success_and_failure(tmp_path, monkeypatch, fault):
    """Unique generation publication retries sharing refusals, never displaces a live root.

    Displacement/rollback injection is obsolete: the repaired flow does neither.
    The old damaged tree must remain byte-identical on success AND failure.
    """
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
    node = pathlib.Path(manager._resolve_managed_node(pin, "win32-x64", require_npm=True))
    root = node.parent.parent
    npm_cli = root / "node-standalone" / "node_modules" / "npm" / "bin" / "npm-cli.js"
    npm_cli.unlink()  # The same deliberate missing-entry repair as the CI witness.
    old_tree = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    original_archive = archive.read_bytes()
    real_replace = runtime.os.replace
    calls = {"promotion": 0}
    denied = PermissionError(13, "fixture Windows sharing refusal")
    denied.winerror = 5
    original_failure = OSError("fixture final promotion failure")

    def sharing_replace(src, dst):
        src, dst = pathlib.Path(src), pathlib.Path(dst)
        phase = "promotion" if src.name.startswith(".tmp-") and ".g-" in dst.name else ""
        if phase:
            calls[phase] += 1
            if fault == "non_permission":
                raise original_failure
            if fault == f"persistent_{phase}" or (fault == phase and calls[phase] <= 2):
                raise denied
        return real_replace(src, dst)

    monkeypatch.setattr(runtime.os, "replace", sharing_replace)
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)
    fails = fault in {"non_permission", "persistent_promotion"}
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
        selected = pathlib.Path(manager._resolve_managed_node(pin, "win32-x64", require_npm=True))
        assert selected != node and selected.read_bytes() == b"node fixture\n"
        assert manager._managed_npm_cli(selected).read_bytes() == b"npm fixture\n"
        assert json.loads((selected.parent.parent / "managed-node.json").read_text())["schema_version"] == 2
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == old_tree
    assert not npm_cli.exists()
    if fault.startswith("persistent_"):
        assert calls[fault.removeprefix("persistent_")] == utils._REPLACE_RETRY_ATTEMPTS
    elif fault in calls:
        assert calls[fault] == 3
    if fault == "non_permission":
        assert calls["promotion"] == 1
    assert archive.read_bytes() == original_archive
    assert not list(root.parent.glob(".tmp-*")) and not list(root.parent.glob(".old-*"))
