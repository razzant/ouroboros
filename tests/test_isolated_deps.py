from __future__ import annotations

import json
import os
import subprocess

import pytest

from ouroboros.marketplace.install_specs import normalize_install_specs
from ouroboros.marketplace.isolated_deps import (
    DEPS_STATE_FILENAME,
    FINGERPRINT_FILENAME,
    _installer_env,
    _run,
    augment_env_for_skill_deps,
    isolated_env_dir,
    read_deps_state,
)
from ouroboros.skill_loader import skill_state_dir


def _manager(**behaviour):
    """A runtime-manager double with the real ``ClaudexorRuntimeManager`` surface (``spec_set``): a call to, or a
    configured behaviour for, a method the real class no longer has fails here instead of passing on a hand fake."""
    from unittest import mock

    from ouroboros.claudexor_runtime import ClaudexorRuntimeManager

    manager = mock.create_autospec(ClaudexorRuntimeManager, instance=True, spec_set=True)
    manager.pin = None
    manager.pin_error = ""
    for name, value in behaviour.items():
        getattr(manager, name).side_effect = value
    return manager


def test_installer_env_scrubs_secret_keys_and_uses_isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("USERPROFILE", "/host/profile")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = _installer_env(tmp_path / ".ouroboros_env")
    assert "OPENAI_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert env["HOME"].startswith(str(tmp_path))
    assert env["USERPROFILE"].startswith(str(tmp_path))
    assert env["APPDATA"].startswith(str(tmp_path))
    assert env["LOCALAPPDATA"].startswith(str(tmp_path))
    assert env["PIP_CACHE_DIR"].startswith(str(tmp_path))
    assert env["npm_config_cache"].startswith(str(tmp_path))


def test_installer_env_node_emergency_prepends_bundled_node_dir(monkeypatch, tmp_path):
    """Emergency only (PATH node dead + healthy bundled selected): the node
    installer env PATH gains the bundled-node dir so npm's `#!/usr/bin/env node`
    shebang resolves the working runtime. Python installs never get it."""
    from ouroboros import node_runtime

    monkeypatch.setenv("PATH", "/usr/bin")
    bundle_bin = str(tmp_path / "bundle" / "bin")
    monkeypatch.setattr(
        node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: bundle_bin
    )

    node_env = _installer_env(tmp_path / ".ouroboros_env", ecosystem="node")
    assert node_env["PATH"] == os.pathsep.join([bundle_bin, "/usr/bin"])

    python_env = _installer_env(tmp_path / ".ouroboros_env", ecosystem="python")
    assert python_env["PATH"] == "/usr/bin"


def test_installer_env_node_healthy_system_is_byte_identical(monkeypatch, tmp_path):
    from ouroboros import node_runtime

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(
        node_runtime, "skill_node_emergency_path_dir", lambda timeout_sec=10: ""
    )

    node_env = _installer_env(tmp_path / ".ouroboros_env", ecosystem="node")
    baseline = _installer_env(tmp_path / ".ouroboros_env")
    assert node_env == baseline
    assert node_env["PATH"] == "/usr/bin"


def test_install_node_package_prepares_managed_npm_when_path_has_none(monkeypatch, tmp_path):
    from ouroboros.marketplace import isolated_deps
    from ouroboros import claudexor_runtime

    node = tmp_path / "managed" / "bin" / "node"
    npm_cli = tmp_path / "managed" / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
    node.parent.mkdir(parents=True)
    npm_cli.parent.mkdir(parents=True)
    node.touch()
    npm_cli.touch()
    prepared = []
    commands = []

    def ensure_cli_command():  # the reviewed closure with its managed Node/npm toolchain: [node, cli]
        prepared.append(True)
        return [str(node), str(tmp_path / "managed" / "cli.js")]

    def managed_npm_cli(selected_node):
        assert selected_node == node
        return npm_cli

    manager = _manager(ensure_cli_command=ensure_cli_command, _managed_npm_cli=managed_npm_cli)

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return {"returncode": 0}

    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: None)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: manager)
    monkeypatch.setattr(isolated_deps, "_run", fake_run)
    isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1)

    assert prepared == [True]
    cmd, kwargs = commands[0]
    assert cmd[:3] == [str(node), str(npm_cli), "install"]
    assert "--ignore-scripts" in cmd
    assert kwargs["env"]["PATH"].split(os.pathsep)[0] == str(node.parent)
    assert kwargs["env"]["npm_config_cache"].startswith(str(tmp_path))
    assert kwargs["env"]["npm_config_prefix"].startswith(str(tmp_path))


@pytest.mark.parametrize("path_node", [False, True])
def test_prepared_managed_node_serves_skill_readiness_and_child_path(monkeypatch, tmp_path, path_node):
    from types import SimpleNamespace
    from ouroboros import claudexor_runtime, node_runtime
    from ouroboros.marketplace import isolated_deps
    from ouroboros.tools.skill_exec import _resolve_runtime_binary
    from ouroboros.tools.skill_preflight import _resolve_runtime

    node = tmp_path / "managed" / "bin" / "node"
    npm_cli = tmp_path / "managed" / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
    node.parent.mkdir(parents=True)
    npm_cli.parent.mkdir(parents=True)
    node.touch()
    node.chmod(0o755)
    npm_cli.touch()
    prepared, probes, events = [], [], []

    def ensure_cli_command():
        events.append("prepare")
        prepared.append(True)
        return [str(node), str(tmp_path / "managed" / "cli.js")]

    def reviewed_node(*, prepared=False, allow_install=False):
        assert prepared and not allow_install  # passive lookup uses metadata, never another probing resolver
        if not events or "prepare" not in events:
            raise claudexor_runtime.ClaudexorRuntimeError("runtime_node_unavailable", "not prepared")
        return str(node)

    manager = _manager(ensure_cli_command=ensure_cli_command, reviewed_node=reviewed_node,
                       _managed_npm_cli=lambda selected: npm_cli)
    manager.pin = SimpleNamespace(node_version="24.22.3")
    manager.planned_node.return_value = str(node)
    manager._resolve_node_toolchain.side_effect = AssertionError("no duplicate toolchain probe")
    monkeypatch.setattr(node_runtime._platform, "_resolve_bundled_payload", lambda candidates: None)
    monkeypatch.setattr(node_runtime._platform, "bootstrap_process_path", lambda: [])
    monkeypatch.setattr(node_runtime, "_NODE_HEALTH_MEMO", {})
    monkeypatch.setattr(node_runtime, "_probe_node_version_outcome",
                        lambda path, timeout_sec=10: probes.append(path) or ("24.22.3", ""))
    from ouroboros import claudexor_node_seed
    monkeypatch.setattr(claudexor_node_seed, "bundled_node", lambda *a, **k: "")
    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: "/older/node" if path_node and name == "node" else None)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: manager)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert node_runtime.select_skill_node_runtime()[0] == ""
    assert prepared == [] and probes == []  # a passive lookup neither downloads nor probes PATH under the pin

    monkeypatch.setattr(isolated_deps, "_run", lambda cmd, **kwargs: {"returncode": 0})
    isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1,
                                       review_check=lambda: events.append("review"))
    assert events == ["review", "prepare"] and prepared == [True]
    assert node_runtime.select_skill_node_runtime() == (str(node), "bundled")  # existing facade provenance
    assert _resolve_runtime_binary("node") == (str(node), "")
    assert _resolve_runtime("node") == (str(node), "")
    assert node_runtime.skill_node_emergency_path_dir() == str(node.parent)
    child_env = {"PATH": str(tmp_path / "empty")}
    node_runtime.prepend_skill_node_emergency_path(child_env)
    assert child_env["PATH"].split(os.pathsep)[0] == str(node.parent)
    assert prepared == [True] and probes == [str(node)]  # reused through the existing health memo


def test_missing_npm_does_not_prepare_before_review(monkeypatch, tmp_path):
    from ouroboros import claudexor_runtime
    from ouroboros.marketplace import isolated_deps

    manager = _manager(ensure_cli_command=AssertionError("unreviewed preparation"))
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: manager)
    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: None)

    def refused():
        raise RuntimeError("fresh executable skill review required")

    with pytest.raises(RuntimeError, match="fresh executable skill review"):
        isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1, review_check=refused)
    manager.ensure_cli_command.assert_not_called()


def test_install_node_package_reports_managed_npm_preparation_failure(monkeypatch, tmp_path):
    import pytest

    from ouroboros import claudexor_runtime
    from ouroboros.marketplace import isolated_deps

    manager = _manager(ensure_cli_command=claudexor_runtime.ClaudexorRuntimeError(
        "runtime_node_install_failed", "reviewed Node archive unavailable"))

    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: None)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: manager)
    with pytest.raises(RuntimeError, match="managed Node/npm preparation failed: reviewed Node archive unavailable"):
        isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1)


def test_install_node_package_rejects_incomplete_managed_npm(monkeypatch, tmp_path):
    import pytest

    from ouroboros import claudexor_runtime
    from ouroboros.marketplace import isolated_deps

    manager = _manager(
        ensure_cli_command=lambda: [str(tmp_path / "managed" / "bin" / "node"), str(tmp_path / "managed" / "cli.js")],
        _managed_npm_cli=lambda selected_node: tmp_path / "managed" / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js")

    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: None)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: manager)
    with pytest.raises(RuntimeError, match="prepared npm CLI is missing"):
        isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1)


def test_normalize_install_specs_rejects_vcs_urls_and_expands_packages():
    auto, manual, warnings = normalize_install_specs([
        {"kind": "pip", "package": "git+https://example.com/pkg.git"},
        {"kind": "npm", "packages": ["left-pad", "@scope/pkg"]},
    ])
    assert [item["package"] for item in auto] == ["left-pad", "@scope/pkg"]
    assert manual and "git+https" in manual[0]["package"]
    assert warnings


def test_augment_env_exposes_python_venv_and_node_path(tmp_path):
    skill_dir = tmp_path / "skill"
    py_bin = skill_dir / ".ouroboros_env" / "python" / ("Scripts" if os.name == "nt" else "bin")
    py_bin.mkdir(parents=True)
    (py_bin / ("python.exe" if os.name == "nt" else "python")).write_text("", encoding="utf-8")
    node_modules = skill_dir / ".ouroboros_env" / "node" / "node_modules"
    node_modules.mkdir(parents=True)
    env = augment_env_for_skill_deps({"PATH": "/usr/bin"}, skill_dir)
    assert str(py_bin) in env["PATH"]
    assert env["VIRTUAL_ENV"].startswith(str(skill_dir / ".ouroboros_env" / "python"))
    assert env["NODE_PATH"] == str(node_modules)


def test_read_deps_state_verifies_live_env_fingerprint(tmp_path):
    drive_root = tmp_path / "drive"
    skill_dir = tmp_path / "skill"
    state_dir = skill_state_dir(drive_root, "skill")
    state_dir.mkdir(parents=True, exist_ok=True)
    state = {"status": "installed", "specs_hash": "abc"}
    (state_dir / DEPS_STATE_FILENAME).write_text(json.dumps(state), encoding="utf-8")

    assert read_deps_state(drive_root, "skill")["status"] == "installed"
    assert read_deps_state(drive_root, "skill", skill_dir)["status"] == "missing"

    env_dir = isolated_env_dir(skill_dir)
    env_dir.mkdir(parents=True)
    (env_dir / FINGERPRINT_FILENAME).write_text(json.dumps(state), encoding="utf-8")

    assert read_deps_state(drive_root, "skill", skill_dir)["status"] == "installed"


def test_run_discards_unbounded_installer_output(monkeypatch, tmp_path):
    # Load process subclasses before replacing Popen with a function.
    from tests.test_tracked_tree_containment import fake_tree_read

    captured = {}

    class Proc:
        returncode = 0
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            captured["timeout"] = timeout
            return 0

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        proc = Proc()
        fake_tree_read(monkeypatch, proc, remains=False)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    result = _run(["tool"], cwd=tmp_path, env={}, timeout_sec=1)
    assert captured["stdout"] is subprocess.DEVNULL
    assert captured["stderr"] is subprocess.PIPE
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["timeout"] == 1
    assert "stdout_tail" not in result
    assert "stderr_tail" not in result


def test_failed_python_install_persists_stderr_tail(monkeypatch, tmp_path):
    from ouroboros.marketplace import isolated_deps

    def fake_run(cmd, **kwargs):
        if "venv" in cmd:
            bin_dir = tmp_path / "skill" / ".ouroboros_env" / "python" / ("Scripts" if os.name == "nt" else "bin")
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / ("python.exe" if os.name == "nt" else "python")).write_text("", encoding="utf-8")
            return {"returncode": 0}
        return {"returncode": 1, "stderr_tail": "ERROR: no matching distribution found for a2a-sdk"}

    monkeypatch.setattr(isolated_deps, "_run", fake_run)
    try:
        isolated_deps.install_isolated_dependencies(
            tmp_path,
            "skill",
            tmp_path / "skill",
            [{"kind": "pip", "package": "a2a-sdk"}],
        )
    except RuntimeError as exc:
        assert "no matching distribution" in str(exc)
    else:
        raise AssertionError("install should fail")

    state = json.loads((skill_state_dir(tmp_path, "skill") / DEPS_STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert "pip install failed" in state["error"]
    assert "no matching distribution" in state["error"]


def test_python_and_npm_install_commands_disable_build_scripts(monkeypatch, tmp_path):
    from ouroboros.marketplace import isolated_deps
    from ouroboros import claudexor_runtime, node_runtime

    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (False, ""))  # a host the pin does not govern
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "venv" in cmd:
            bin_dir = tmp_path / ".ouroboros_env" / "python" / ("Scripts" if os.name == "nt" else "bin")
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / ("python.exe" if os.name == "nt" else "python")).write_text("", encoding="utf-8")
        return {"returncode": 0}

    manager = _manager(ensure_cli_command=AssertionError("managed Node must not be prepared when npm is on PATH"))
    manager._resolve_node_toolchain.side_effect = AssertionError("no secondary probing resolver")

    monkeypatch.setattr(isolated_deps, "_run", fake_run)
    # Source installations need not have a bundled Node.
    monkeypatch.setattr(node_runtime._platform, "resolve_bundled_node", lambda: None)
    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: "/usr/bin/npm" if name == "npm" else None)
    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: manager)
    isolated_deps._install_python_packages(["wheelpkg"], tmp_path / ".ouroboros_env", 1)
    isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1)
    assert any("--only-binary=:all:" in cmd for cmd in calls)
    assert any("--ignore-scripts" in cmd for cmd in calls)
    assert any(cmd[:2] == ["/usr/bin/npm", "install"] for cmd in calls)
    assert not any("freeze" in cmd for cmd in calls)


def test_python_packages_are_batched_into_one_pip_invocation(monkeypatch, tmp_path):
    from ouroboros.marketplace import isolated_deps

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "venv" in cmd:
            bin_dir = tmp_path / "skill" / ".ouroboros_env" / "python" / ("Scripts" if os.name == "nt" else "bin")
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / ("python.exe" if os.name == "nt" else "python")).write_text("", encoding="utf-8")
        return {"returncode": 0}

    monkeypatch.setattr(isolated_deps, "_run", fake_run)
    isolated_deps.install_isolated_dependencies(
        tmp_path,
        "skill",
        tmp_path / "skill",
        [
            {"kind": "pip", "package": "firstpkg"},
            {"kind": "pip", "package": "secondpkg"},
        ],
    )
    pip_installs = [cmd for cmd in calls if "pip" in cmd and "install" in cmd]
    assert len(pip_installs) == 1
    assert "firstpkg" in pip_installs[0]
    assert "secondpkg" in pip_installs[0]


def _governing_pin(monkeypatch, events, *, prepared: str = "", planned: str = "/planned/node"):
    """A pin that governs this host; ``prepared`` names the reviewed Node its selector finds, else it is unprepared.
    Requests for the pin's Node are recorded in ``events`` as ``"request"``."""
    from ouroboros import node_runtime, platform_layer, server_boot

    selection = (prepared, "bundled") if prepared else ("", f"pinned:24.22.3:not_prepared (planned {planned})")
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (True, planned))
    # platform_layer first: its PEP 562 re-export reads node_runtime's attribute, so patched second it would record
    # this stand-in as the value to restore and leave it behind for the next test.
    for owner in (platform_layer, node_runtime):
        monkeypatch.setattr(owner, "select_skill_node_runtime", lambda timeout_sec=10: selection)
    monkeypatch.setattr(platform_layer, "resolve_bundled_node", lambda: prepared or None)
    monkeypatch.setattr(platform_layer, "bootstrap_process_path", lambda: [])
    monkeypatch.setattr(server_boot, "request_pinned_node", lambda root=None: events.append("request"))


def _recording_npm(monkeypatch, calls):
    from ouroboros.marketplace import isolated_deps

    monkeypatch.setattr(isolated_deps, "_run", lambda cmd, **kwargs: calls.append((cmd, kwargs["env"].get("PATH")))
                        or {"returncode": 0})
    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: "/usr/bin/npm" if name == "npm" else None)


def test_a_node_install_under_an_unprepared_pin_waits_after_its_review_check(monkeypatch, tmp_path):
    """R8 (parent's scope decision, 15 Nov): with the pin's Node unprepared, the JS skills' npm installer ran
    `npm install` on whatever Node PATH held. It now waits: after its review check it asks once for the pin's Node and
    fails with the reason; npm never starts. A refused review check asks nothing."""
    import pytest

    from ouroboros.marketplace import isolated_deps
    from ouroboros.node_runtime import NodeRuntimeUnavailable

    events, calls = [], []
    _governing_pin(monkeypatch, events)
    _recording_npm(monkeypatch, calls)
    with pytest.raises(NodeRuntimeUnavailable, match=r"no reviewed Node for this install: pinned:24\.22\.3"):
        isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1,
                                            review_check=lambda: events.append("review"))
    assert events == ["review", "request"] and calls == []

    def refused():
        events.append("review")
        raise RuntimeError("install declarations require a fresh executable skill review")

    events.clear()
    with pytest.raises(RuntimeError, match="fresh executable skill review"):
        isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1, review_check=refused)
    assert events == ["review"] and calls == []


def test_a_node_install_under_a_prepared_pin_runs_npm_with_that_node_first(monkeypatch, tmp_path):
    """npm itself runs on the pin's Node, and its children (an npm script's `#!/usr/bin/env node`) find that Node first
    on PATH. A launcher whose npm CLI is not found is refused before any spawn, as it could choose its own Node."""
    import pytest

    from ouroboros.marketplace import isolated_deps
    from ouroboros.node_runtime import NpmLauncherRefused

    prepared = tmp_path / "pinned" / "bin" / "node"
    host = tmp_path / "host"
    cli = host / "node_modules" / "npm" / "bin" / "npm-cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("// npm\n", encoding="utf-8")
    events, calls = [], []
    _governing_pin(monkeypatch, events, prepared=str(prepared))
    _recording_npm(monkeypatch, calls)
    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: str(host / "npm") if name == "npm" else None)
    monkeypatch.setenv("PATH", "/usr/bin")
    isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1)
    [(command, path)] = calls
    assert command[:3] == [str(prepared), str(cli), "install"] and events == []
    assert path == os.pathsep.join([str(prepared.parent), "/usr/bin"])  # npm's `#!/usr/bin/env node` finds the pin's
    monkeypatch.setattr(isolated_deps.shutil, "which", lambda name: str(tmp_path / "shims" / "npm"))
    with pytest.raises(NpmLauncherRefused, match="would choose its own Node"):
        isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1)
    assert len(calls) == 1 and events == []  # refused before npm, and no new request for the pin's Node


def _stub_node(where, label):
    where.mkdir(parents=True, exist_ok=True)
    node = where / "node"
    node.write_text('#!/bin/sh\necho "ran:%s:$*" >&2\n' % label, encoding="utf-8")  # a stand-in, never a real Node
    node.chmod(0o755)
    return node


def test_a_node_install_runs_npm_on_the_pins_node_not_the_node_its_launcher_picks(monkeypatch, tmp_path):
    """The official Windows ``npm.cmd`` runs the ``node.exe`` beside it, PATH's only when there is none, so an older
    Node install on PATH kept running ``npm install`` under a prepared pin. Real processes: a launcher with that rule
    (POSIX stand-in, labelled stub Nodes) and the installer's own ``_run``; the pin's Node now runs the launcher's CLI,
    for ``npm rebuild`` too."""
    import sys

    import pytest

    from ouroboros.marketplace import isolated_deps

    if sys.platform == "win32":
        pytest.skip("POSIX stand-in for npm.cmd")
    pinned = _stub_node(tmp_path / "pinned" / "bin", "pinned")
    host = tmp_path / "host"
    _stub_node(host, "host")
    cli = host / "node_modules" / "npm" / "bin" / "npm-cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("// npm\n", encoding="utf-8")
    launcher = host / "npm"
    launcher.write_text('#!/bin/sh\nd=$(dirname "$0")\nexec "$d/node" "$d/node_modules/npm/bin/npm-cli.js" "$@"\n',
                        encoding="utf-8")
    launcher.chmod(0o755)
    events = []
    _governing_pin(monkeypatch, events, prepared=str(pinned))
    monkeypatch.setenv("PATH", os.pathsep.join([str(host), "/usr/bin", "/bin"]))
    [result] = isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 30)
    assert result["returncode"] == 0 and result["cmd"][:2] == [str(pinned), str(cli)]
    assert result["stderr_tail"].startswith(f"ran:pinned:{cli} install --ignore-scripts --prefix ")
    from ouroboros import node_runtime

    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (False, ""))
    [unpinned] = isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 30)
    assert unpinned["cmd"][:1] == [str(launcher)] and unpinned["stderr_tail"].startswith(f"ran:host:{cli} install")


def test_a_bare_npm_step_runs_on_the_pins_node_and_keeps_its_launcher_without_a_pin(monkeypatch, tmp_path):
    from ouroboros import node_runtime
    from ouroboros.marketplace import isolated_deps

    pinned = str(tmp_path / "pinned" / "bin" / "node")
    host = tmp_path / "host"
    cli = host / "node_modules" / "npm" / "bin" / "npm-cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("// npm\n", encoding="utf-8")
    launcher = host / ("npm.cmd" if os.name == "nt" else "npm")  # the launcher `which` finds there (PATHEXT on Windows)
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher.chmod(0o755)
    events, ran = [], []
    _governing_pin(monkeypatch, events, prepared=pinned)
    monkeypatch.setattr(isolated_deps, "_run", lambda cmd, **kwargs: ran.append(cmd) or {"returncode": 0})
    env_root = tmp_path / ".ouroboros_env"
    env_root.mkdir()
    env = {"PATH": str(host)}
    isolated_deps._run_install_step({"argv": ["npm", "run", "build"]}, env_root, env, 1)
    # The author's own path stays the author's.
    isolated_deps._run_install_step({"argv": [str(launcher), "run", "build"]}, env_root, env, 1)
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (False, ""))
    isolated_deps._run_install_step({"argv": ["npm", "run", "build"]}, env_root, env, 1)
    # Windows `which` spells the PATHEXT suffix its own way (`npm.CMD`): the same file, compared as the OS does.
    assert [[os.path.normcase(cmd[0]), *cmd[1:]] for cmd in ran] == [
        [os.path.normcase(pinned), str(cli), "run", "build"], [os.path.normcase(str(launcher)), "run", "build"],
        [os.path.normcase(str(launcher)), "run", "build"]]


@pytest.mark.parametrize("complete", [False, True], ids=["node-only", "full"])
def test_a_readiness_check_runs_the_npm_version_the_skill_installed_on_the_pins_node(monkeypatch, tmp_path, complete):
    """A spec installing ``npm@11.19.1`` checks it with a bare ``npm --version``: the dependency PATH finds the npm
    generated in ``node/node_modules/.bin`` (``npm.cmd`` and ``npm`` on Windows). Under a ready Node-only pin that
    launcher was refused on Windows before any spawn; it now runs ``../npm/bin/npm-cli.js`` on the pin's Node, and
    under a complete pin too: the pin chooses the Node, not the npm version the skill installed."""
    from ouroboros.marketplace import isolated_deps

    gen = tmp_path / "gen" / "node-standalone"
    pinned = gen / "bin" / "node"
    if complete:
        managed = gen / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
        managed.parent.mkdir(parents=True)
        managed.write_text("// the pin's npm\n", encoding="utf-8")
    skill_dir = tmp_path / "skill"
    env_root = isolated_deps.isolated_env_dir(skill_dir)
    modules = env_root / "node" / "node_modules"
    cli = modules / "npm" / "bin" / "npm-cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("// npm 11.19.1\n", encoding="utf-8")
    (modules / ".bin").mkdir()
    for name in ("npm", "npm.cmd"):  # npm's generated launchers, neither a link
        (modules / ".bin" / name).write_text("#!/bin/sh\n", encoding="utf-8")
        (modules / ".bin" / name).chmod(0o755)
    events, ran = [], []
    _governing_pin(monkeypatch, events, prepared=str(pinned))
    monkeypatch.setattr(isolated_deps, "_run", lambda cmd, **kwargs: ran.append(cmd) or {"returncode": 0})
    env = isolated_deps.augment_env_for_skill_deps({"PATH": str(tmp_path / "empty")}, skill_dir)
    isolated_deps._run_install_step({"argv": ["npm", "--version"]}, env_root, env, 1)
    assert ran == [[str(pinned), str(cli), "--version"]] and events == []


def test_without_a_governing_pin_a_node_install_keeps_its_path_node(monkeypatch, tmp_path):
    """No pin, or a host the pin has no archive for: today's behaviour, npm on the PATH Node, nothing asked."""
    from ouroboros import node_runtime
    from ouroboros.marketplace import isolated_deps

    events, calls = [], []
    _governing_pin(monkeypatch, events)
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (False, ""))
    _recording_npm(monkeypatch, calls)
    monkeypatch.setenv("PATH", "/usr/bin")
    isolated_deps._install_node_package("left-pad", tmp_path / ".ouroboros_env", 1)
    assert [path for _command, path in calls] == ["/usr/bin"] and events == []


def test_a_work_domain_install_resolves_its_own_node(monkeypatch, tmp_path):
    """A managed work domain's container resolves its own runtime: the host pin is not read, no host path given."""
    from ouroboros import node_runtime
    from ouroboros.marketplace import isolated_deps

    monkeypatch.setattr(isolated_deps.work_domain, "is_work", lambda path: True)
    monkeypatch.setattr(node_runtime, "pinned_node_wait", lambda: (_ for _ in ()).throw(AssertionError("pin read")))
    assert isolated_deps._wait_for_reviewed_node(tmp_path / ".ouroboros_env") is None


def test_a_node_ecosystem_check_waits_for_the_pins_node_and_a_python_one_does_not(monkeypatch, tmp_path):
    import pytest

    from ouroboros.marketplace import isolated_deps
    from ouroboros.node_runtime import NodeRuntimeUnavailable

    events, ran = [], []
    _governing_pin(monkeypatch, events)
    monkeypatch.setattr(isolated_deps, "_run_install_step", lambda step, *a, **k: ran.append(step["argv"])
                        or {"returncode": 0})
    env_root = tmp_path / ".ouroboros_env"
    env_root.mkdir()
    node_spec = {"kind": "npm", "package": "x", "check": {"argv": ["node", "-e", "1"]}}
    with pytest.raises(NodeRuntimeUnavailable):
        isolated_deps._install_actions(node_spec, env_root, tmp_path / "cache", 1, [])
    assert ran == [] and events == ["request"]
    python_spec = {"kind": "pip", "package": "y", "check": {"argv": ["python3", "-c", "1"]}}
    assert isolated_deps._install_actions(python_spec, env_root, tmp_path / "cache", 1, [])["executable_ready"] is True
    assert ran == [["python3", "-c", "1"]] and events == ["request"]


def test_a_waiting_node_install_is_recorded_as_a_failed_install_with_its_reason(monkeypatch, tmp_path):
    """The existing failure record: the next dependency pass retries it."""
    import pytest

    from ouroboros.marketplace import isolated_deps

    events, calls = [], []
    _governing_pin(monkeypatch, events)
    _recording_npm(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="no reviewed Node for this install"):
        isolated_deps.install_isolated_dependencies(tmp_path, "skill", tmp_path / "skill",
                                                    [{"kind": "npm", "package": "left-pad"}])
    state = json.loads((skill_state_dir(tmp_path, "skill") / DEPS_STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["status"] == "failed" and state["error"].startswith("NodeRuntimeUnavailable: no reviewed Node")
    assert calls == [] and events == ["request"]


def test_a_node_install_step_asks_for_the_pins_node_only_after_its_review_check(monkeypatch, tmp_path):
    """R8c (parent): ``_install_actions`` asked for the pin's Node before its own review check. A node step or check is
    now reviewed first: a refused review asks nothing and runs nothing; an accepted one asks once, then waits. The
    step and the check keep their own re-checks, and a non-node spec keeps exactly the checks it had."""
    import pytest

    from ouroboros.marketplace import isolated_deps
    from ouroboros.node_runtime import NodeRuntimeUnavailable

    events, ran = [], []
    _governing_pin(monkeypatch, events)
    monkeypatch.setattr(isolated_deps, "_run_install_step", lambda step, *a, **k: ran.append(step["argv"])
                        or {"returncode": 0})
    env_root = tmp_path / ".ouroboros_env"
    env_root.mkdir()
    node_spec = {"kind": "npm", "package": "x", "steps": [{"argv": ["node", "build.js"]}],
                 "check": {"argv": ["node", "-e", "1"]}}

    def refused():
        events.append("review")
        raise RuntimeError("install declarations require a fresh executable skill review")

    def reviewed():
        events.append("review")

    with pytest.raises(RuntimeError, match="fresh executable skill review"):
        isolated_deps._install_actions(node_spec, env_root, tmp_path / "cache", 1, [], review_check=refused)
    assert events == ["review"] and ran == []  # refused: nothing asked, nothing run
    events.clear()
    with pytest.raises(NodeRuntimeUnavailable):
        isolated_deps._install_actions(node_spec, env_root, tmp_path / "cache", 1, [], review_check=reviewed)
    assert events == ["review", "request"] and ran == []  # reviewed, one request, then the wait

    events.clear()
    _governing_pin(monkeypatch, events, prepared=str(tmp_path / "pinned" / "bin" / "node"))
    assert isolated_deps._install_actions(node_spec, env_root, tmp_path / "cache", 1, [],
                                          review_check=reviewed)["executable_ready"] is True
    assert events == ["review", "review", "review"] and ran == [["node", "build.js"], ["node", "-e", "1"]]
    events.clear()
    ran.clear()
    python_spec = {"kind": "pip", "package": "y", "steps": [{"argv": ["python3", "build.py"]}],
                   "check": {"argv": ["python3", "-c", "1"]}}
    isolated_deps._install_actions(python_spec, env_root, tmp_path / "cache", 1, [], review_check=reviewed)
    assert events == ["review", "review"] and ran == [["python3", "build.py"], ["python3", "-c", "1"]]
