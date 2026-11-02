"""App/source handoff without a native window, permission query or Git mutation.

The real launcher control flow runs with process/UI boundaries replaced. A fresh
plain-Python child reads generation sentinels; this does not qualify NSBundle,
WebKit, signing or notification delivery on macOS.
"""
from __future__ import annotations

import atexit
import io
import json
import logging
import os
import pathlib
import subprocess
import sys
import tarfile
import threading
import types

import pytest

from ouroboros import launcher_bootstrap as boot
from ouroboros import platform_layer as platform
from ouroboros.launcher_bootstrap import source_launcher_importable

_CHECKOUT_SHA = boot.checkout_sha  # the real Git read; the launch fixture replaces it


class Handoff(BaseException):
    def __init__(self, executable, argv, env):
        self.executable, self.argv, self.env = executable, argv, env


class FrozenContinues(BaseException):
    pass


@pytest.fixture
def launch(monkeypatch, tmp_path):
    import launcher

    monkeypatch.setattr(platform, "IS_WINDOWS", False)  # select POSIX Python paths even on a Windows host
    monkeypatch.setattr(launcher, "_shutdown_event", threading.Event())
    seed = tmp_path / "Ouroboros app.app" / "Contents" / "Frameworks"
    seed.mkdir(parents=True)
    entry = seed.parent / "MacOS" / "ouroboros-shell"
    entry.parent.mkdir()
    entry.write_text("test entry\n")
    entry.chmod(0o755)
    python = platform.embedded_python_candidates(seed)[0]
    python.parent.mkdir(parents=True)
    python.write_text("test body Python\n")
    (seed / "VERSION").write_text("7.7.0\n")
    (seed / boot.BUNDLE_MANIFEST_NAME).write_text(json.dumps({"source_sha": "bundle-generation"}))
    repo = tmp_path / "managed source"
    repo.mkdir()
    calls = []
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key, value in {"IS_MACOS": True, "IS_WINDOWS": False, "REPO_DIR": repo,
                       "DATA_DIR": tmp_path / "data", "SETTINGS_PATH": tmp_path / "data/settings.json",
                       "_headless": False, "_external_ui": False, "_external_seed_bundle": None,
                       "_runtime_venv": None, "_launch_argv": [], "APP_VERSION": "7.7.0",
                       "EMBEDDED_PYTHON": str(python)}.items():
        monkeypatch.setattr(launcher, key, value)
    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)
    monkeypatch.setattr(launcher.sys, "_MEIPASS", str(seed), raising=False)
    monkeypatch.setattr(launcher, "__file__", str(repo / "launcher.py"))
    monkeypatch.setattr(boot, "_LOADED_CHECKOUT_SHA", "")
    monkeypatch.setattr(boot, "checkout_sha", lambda root: "updated-generation")
    monkeypatch.setattr(atexit, "register", lambda *a: None)
    monkeypatch.setitem(sys.modules, "webview", types.ModuleType("webview"))
    monkeypatch.setattr(launcher, "_detect_headless", lambda: None)
    monkeypatch.setattr(launcher, "acquire_pid_lock", lambda: calls.append("lock") or True)
    monkeypatch.setattr(launcher, "release_pid_lock", lambda: calls.append("release"))
    monkeypatch.setattr(launcher, "stop_tray_before_exit", lambda release: (calls.append("teardown"), release()))
    monkeypatch.setattr(launcher, "restore_tray_after_handoff", lambda: calls.append("restore"))
    monkeypatch.setattr(launcher, "check_git", lambda: calls.append("git") or True)
    background = types.SimpleNamespace(_open_pending=False, panic=threading.Event())
    background.listen = lambda path: calls.append("listen") or background
    monkeypatch.setattr(launcher, "Background", lambda *args, **kwargs: background)
    monkeypatch.setattr(launcher, "bootstrap_repo", lambda: calls.append("bootstrap") or True)
    monkeypatch.setattr(boot, "source_launcher_importable", lambda *a, **kw: calls.append("probe") or True)
    monkeypatch.setattr(launcher, "launcher_sources_changed", lambda *a, **kw: True)

    def stop_before_server():
        calls.append("frozen-server-boundary")
        raise FrozenContinues()

    def execve(executable, argv, env):
        calls.append("exec")
        raise Handoff(executable, argv, env)

    monkeypatch.setattr(launcher, "_prepare_first_run_settings", stop_before_server)
    monkeypatch.setattr(launcher.os, "execve", execve)
    yield types.SimpleNamespace(module=launcher, seed=seed, entry=entry, python=python, repo=repo,
                                calls=calls, background=background)


def test_cold_changed_helpers_handoff_after_bootstrap_before_server_or_window(launch):
    l = launch
    with pytest.raises(Handoff) as stopped:
        l.module.main(["--launch-intent", "automatic", "-psn_0_123"])
    assert l.calls == ["lock", "listen", "git", "bootstrap", "probe", "teardown", "release", "exec"]
    assert stopped.value.executable == str(l.entry)
    assert stopped.value.argv == [str(l.entry), str(l.repo / "launcher.py"), "--launch-intent", "automatic",
                                  "-psn_0_123", "--seed-bundle", str(l.seed)]
    assert stopped.value.env[boot.LAUNCHER_HANDOFF_ENV] == "1"
    assert stopped.value.env["PYTHONUSERBASE"] == str(l.module.DATA_DIR / "state/python-userbase")
    assert "PYTHONHOME" not in stopped.value.env and "CFProcessPath" not in stopped.value.env
    assert boot._LOADED_CHECKOUT_SHA == "bundle-generation", "frozen helpers are never labelled updated-generation"


def test_panic_during_cold_change_detection_cannot_replace_the_launcher(launch, monkeypatch):
    def changed(*a, **kw):
        launch.module._shutdown_event.set()
        return True

    monkeypatch.setattr(launch.module, "launcher_sources_changed", changed)
    try:
        launch.module.main([])
    except (Handoff, FrozenContinues):
        pass
    assert "exec" not in launch.calls and "frozen-server-boundary" not in launch.calls, \
        "Panic during the change check must fence both successor and local server"


def test_macos_fixture_on_a_simulated_windows_host(request, monkeypatch):
    import launcher  # import before simulating the host; no Windows native API is available here

    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    l = request.getfixturevalue("launch")
    assert l.python == l.seed / "python-standalone/bin/python3", "the fixture must create macOS Python on Windows too"
    with pytest.raises(Handoff) as stopped:
        launcher.main([])
    assert stopped.value.executable == str(l.entry)


@pytest.mark.parametrize("reason", ["same-generation", "server-only", "imports-broken", "legacy-bundle", "source"])
def test_cold_fallback_and_source_never_loop_into_frozen(launch, monkeypatch, reason):
    l = launch
    argv = []
    if reason in ("same-generation", "server-only"):
        monkeypatch.setattr(l.module, "launcher_sources_changed", lambda *a, **kw: False)
    elif reason == "imports-broken":
        monkeypatch.setattr(boot, "source_launcher_importable", lambda *a, **kw: False)
    elif reason == "legacy-bundle":
        l.entry.unlink()
    else:
        monkeypatch.setattr(l.module.sys, "frozen", False)
        monkeypatch.setattr(l.module.sys, "executable", str(l.entry))
        monkeypatch.setattr(l.module, "launcher_sources_changed", boot.launcher_sources_changed)
        argv = ["--seed-bundle", str(l.seed)]
    with pytest.raises(FrozenContinues):
        l.module.main(argv)
    assert l.calls[-1] == "frozen-server-boundary"
    assert "exec" not in l.calls and "release" not in l.calls and "teardown" not in l.calls
    assert l.module.EMBEDDED_PYTHON == str(l.python), "the app entry is never the body interpreter"


def test_automatic_panic_and_single_instance_loss_stop_before_bootstrap(launch, monkeypatch):
    l = launch
    marker = l.module.DATA_DIR / "state" / "panic_stop.flag"
    marker.parent.mkdir(parents=True)
    marker.write_text("panic")
    l.module.main(["--launch-intent", "automatic"])
    assert l.calls == ["lock"] and marker.read_text() == "panic"
    l.calls.clear()
    monkeypatch.setattr(l.module, "acquire_pid_lock", lambda: False)
    l.module.main(["--launch-intent", "automatic"])
    assert l.calls == []


def test_open_during_automatic_boot_survives_cold_exec(launch):
    launch.background._open_pending = True
    with pytest.raises(Handoff) as stopped:
        launch.module.main(["--launch-intent", "automatic"])
    options = boot.parse_launch_options(stopped.value.argv[2:])
    assert options.launch_intent == "owner"


def test_shell_and_body_interpreters_are_separate_even_with_runtime_venv(launch, monkeypatch):
    l = launch
    monkeypatch.setattr(l.module.sys, "frozen", False)
    monkeypatch.setattr(l.module, "_external_seed_bundle", l.seed)
    monkeypatch.setattr(l.module, "_launch_argv", ["--seed-bundle", str(l.seed), "--runtime-venv=/persistent/venv",
                                                 "--no-ui", "--launch-intent=automatic"])
    monkeypatch.setattr(l.module, "EMBEDDED_PYTHON", "/persistent/venv/bin/python")
    with pytest.raises(Handoff) as stopped:
        l.module._exec_source_launcher()
    command = stopped.value.argv
    assert command[:2] == [str(l.entry), str(l.repo / "launcher.py")]
    assert "--runtime-venv=/persistent/venv" in command and "--no-ui" in command
    l.entry.unlink()  # old app: self-update still works, with the disclosed identity limitation
    with pytest.raises(Handoff) as stopped:
        l.module._exec_source_launcher()
    legacy = stopped.value.argv
    assert legacy[0] == "/persistent/venv/bin/python"
    assert not any(arg.startswith("--runtime-venv") for arg in legacy)


@pytest.mark.parametrize("frozen", [True, False])
def test_runtime_venv_preparation_keeps_plain_python_as_its_base(launch, monkeypatch, tmp_path, frozen):
    l = launch
    seen = []
    venv = types.SimpleNamespace(python=str(tmp_path / "venv/bin/python"), bin_dir=tmp_path / "venv/bin", ready=True)

    def prepare(path, **kwargs):
        seen.append(kwargs["base_python"])
        return venv

    monkeypatch.setattr(l.module.sys, "frozen", frozen)
    monkeypatch.setattr(l.module.sys, "_base_executable", "/plain/base/python", raising=False)
    monkeypatch.setattr(l.module, "prepare_runtime_venv", prepare)
    argv = ["--runtime-venv", str(tmp_path / "venv")]
    if not frozen:
        monkeypatch.setattr(l.module.sys, "executable", str(l.entry))
        argv += ["--seed-bundle", str(l.seed)]
    with pytest.raises(Handoff) as stopped:
        l.module.main(argv)
    assert seen == [str(l.python) if frozen else "/plain/base/python"]
    assert l.module.EMBEDDED_PYTHON == venv.python
    assert stopped.value.executable == str(l.entry)
    assert boot.parse_launch_options(stopped.value.argv[2:]).runtime_venv == tmp_path / "venv"


@pytest.mark.parametrize("acquired", [True, False])
def test_signed_entry_exec_refusal_restores_only_after_reacquiring_the_lock(launch, monkeypatch, acquired):
    l = launch
    commands = []

    def refuse(executable, argv, env):
        l.calls.append("refused")
        commands.append((executable, argv, env))
        raise PermissionError("entry refused")

    monkeypatch.setattr(l.module.os, "execve", refuse)
    monkeypatch.setattr(l.module, "acquire_pid_lock", lambda: l.calls.append("reacquire") or acquired)
    monkeypatch.setattr(l.module.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    if acquired:
        l.module._exec_source_launcher()
    else:
        with pytest.raises(SystemExit) as stopped:
            l.module._exec_source_launcher()
        assert stopped.value.code == 1
    assert l.calls == ["probe", "teardown", "release", "refused", "reacquire"] + (["restore"] if acquired else [])
    executable, argv, env = commands[0]
    assert executable == argv[0] == str(l.entry)
    assert argv[1:] == [str(l.repo / "launcher.py"), "--seed-bundle", str(l.seed)]
    assert env["PYTHONUSERBASE"] == str(l.module.DATA_DIR / "state/python-userbase")


def _lifecycle(l, monkeypatch, *, returncode=42, landed=lambda: None, waited=lambda: None):
    """The real lifecycle loop with its process, port and sync boundaries replaced; a second start stops it.
    ``landed`` is what the server lands before it exits ``returncode``."""
    module = l.module
    for event in ("_shutdown_event", "_agent_restart_requested", "_agent_exit_before_ready"):
        monkeypatch.setattr(module, event, threading.Event())
    monkeypatch.setattr(module, "_launch_intent", "owner")
    monkeypatch.setattr(module, "_pre_generation_cleanup", lambda: [])
    monkeypatch.setattr(module, "_clear_port_file", lambda: True)
    monkeypatch.setattr(module, "update_external_host", lambda *a: {})
    monkeypatch.setattr(module, "_await_generation_endpoint", lambda *a, **kw: 1)
    monkeypatch.setattr(module, "_update_server_process_record_port", lambda *a: None)
    monkeypatch.setattr(module, "_wait_for_server", lambda *a, **kw: True)
    monkeypatch.setattr(module, "_cleanup_recorded_server_group_for_pid", lambda *a: None)
    monkeypatch.setattr(module, "_sync_existing_repo_from_bundle", lambda: l.calls.append("sync"))
    monkeypatch.setattr(module, "_install_deps", lambda: l.calls.append("deps") or True)
    monkeypatch.setattr(module.time, "sleep", lambda *a: None)
    monkeypatch.setitem(sys.modules, "ouroboros.delegate_recovery",
                        types.SimpleNamespace(acknowledge_observed_restart_exit=lambda *a, **kw: True))

    def start(port):
        if "server" in l.calls:
            raise FrozenContinues()
        l.calls.append("server")
        landed()
        return types.SimpleNamespace(pid=4242, returncode=returncode, wait=waited)

    monkeypatch.setattr(module, "start_agent", start)


@pytest.mark.parametrize("mode", ["helper-change", "server-only", "source", "legacy", "panic", "tray-panic"])
def test_lifecycle_restart_and_panic_keep_the_existing_owners(launch, monkeypatch, mode):
    l = launch
    module = l.module
    if mode == "source":
        monkeypatch.setattr(module.sys, "frozen", False)
        monkeypatch.setattr(module, "_external_seed_bundle", l.seed)
        monkeypatch.setattr(module, "_launch_argv", ["--seed-bundle", str(l.seed)])
    elif mode == "legacy":
        l.entry.unlink()
    monkeypatch.setattr(module, "launcher_sources_changed", lambda *a, **kw: mode != "server-only")
    panic = mode in {"panic", "tray-panic"}
    _lifecycle(l, monkeypatch, returncode=module.PANIC_EXIT_CODE if panic else 42,
               waited=lambda: module._shutdown_event.set() if mode == "tray-panic" else None)
    if panic:
        monkeypatch.setitem(sys.modules, "ouroboros.process_scope",
                            types.SimpleNamespace(complete_pending_panic_sweep=lambda *a, **kw: None))
        monkeypatch.setattr(module, "request_tray_cleanup", lambda: l.calls.append("tray"))
        monkeypatch.setattr(module, "_kill_orphaned_children", lambda **kw: l.calls.append("cleanup"))
        monkeypatch.setattr(module, "stop_tray_before_exit", lambda release, **kw: release())
        monkeypatch.setattr(module.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
        with pytest.raises(SystemExit):
            module.agent_lifecycle_loop()
        assert l.calls == ["server", "tray", "cleanup", "release"]
        assert module._shutdown_event.is_set()
    elif mode == "server-only":
        with pytest.raises(FrozenContinues):
            module.agent_lifecycle_loop()
        assert l.calls == ["server", "sync", "deps"]
    else:
        with pytest.raises(Handoff) as stopped:
            module.agent_lifecycle_loop()
        assert stopped.value.executable == str(l.python if mode == "legacy" else l.entry)
        assert l.calls[:3] == ["server", "sync", "deps"] and l.calls[-2:] == ["release", "exec"]


@pytest.mark.parametrize("stop_at", ["sleep", "sync", "deps", "change-check"])
def test_panic_during_exit_42_cannot_handoff(launch, monkeypatch, stop_at):
    l = launch
    _lifecycle(l, monkeypatch, returncode=42)

    def stop(*a, **kw):
        l.module._shutdown_event.set()
        return True

    target, name = {
        "sleep": (l.module.time, "sleep"), "sync": (l.module, "_sync_existing_repo_from_bundle"),
        "deps": (l.module, "_install_deps"), "change-check": (l.module, "launcher_sources_changed"),
    }[stop_at]
    monkeypatch.setattr(target, name, stop)
    try:
        l.module.agent_lifecycle_loop()
    except (Handoff, FrozenContinues):
        pass
    assert "exec" not in l.calls and l.calls.count("server") == 1, "Panic must outlive an in-flight exit-42 handoff"


def test_other_frozen_platforms_report_the_seed_as_loaded_source(launch, monkeypatch):
    """Shell facts read before main cannot mistake the newer checkout for frozen imported code."""
    monkeypatch.setattr(launch.module, "IS_MACOS", False)
    facts = launch.module._shell_facts()
    assert facts["source_revision"] == "bundle-generation" and facts["macos_source_entry"] is None
    assert facts["bundle_version"] == "7.7.0", "the installed seed is still reported as installed"


def _git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=repo,
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("changed", ["ouroboros/process_scope.py", "supervisor/queue.py", "server.py", None])
@pytest.mark.parametrize("macos", [True, False])
def test_cold_open_counts_packaged_helpers_the_frozen_launcher_has_not_imported_yet(launch, monkeypatch, changed,
                                                                                macos):
    """Real Git and the real detector. A frozen launcher imports more helpers from its seed later (Panic sweep,
    restart acknowledgement, server records, Claudexor custody): none is loaded at the cold check, so it compares
    every packaged tree. A server.py-only or no update keeps the frozen launcher."""
    l = launch
    monkeypatch.setattr(l.module, "IS_MACOS", macos)
    monkeypatch.setattr(boot, "checkout_sha", _CHECKOUT_SHA)
    monkeypatch.setattr(l.module, "launcher_sources_changed", boot.launcher_sources_changed)
    for rel in ("launcher.py", "server.py", "ouroboros/process_scope.py", "supervisor/queue.py"):
        (l.repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (l.repo / rel).write_text("GEN = 1\n")
    _git(l.repo, "init", "-q")
    _git(l.repo, "add", "-A")
    _git(l.repo, "commit", "-q", "-m", "seed")
    (l.seed / boot.BUNDLE_MANIFEST_NAME).write_text(json.dumps({"source_sha": _git(l.repo, "rev-parse", "HEAD")}))
    if changed:
        (l.repo / changed).write_text("GEN = 2\n")
        _git(l.repo, "commit", "-q", "-am", "update")
    if changed in ("ouroboros/process_scope.py", "supervisor/queue.py"):
        with pytest.raises(Handoff) as stopped:
            l.module.main([])
        assert stopped.value.executable == str(l.entry if macos else l.python)
        assert l.calls[-3:] == ["teardown", "release", "exec"], "before any server or normal window"
        assert "probe" in l.calls, "every packaged platform probes imports before losing the frozen recovery shell"
    else:
        with pytest.raises(FrozenContinues):
            l.module.main([])
        assert "probe" not in l.calls and "exec" not in l.calls


def test_missing_plain_python_does_not_run_the_app_entry_as_server(launch, monkeypatch):
    launch.python.unlink()
    monkeypatch.setattr(launch.module.sys, "executable", str(launch.entry))
    with pytest.raises(RuntimeError, match="bundled Python is missing"):
        boot.find_embedded_python(launch.seed, macos=True)


def test_bundle_and_loaded_versions_stay_separate_after_source_update(launch, monkeypatch):
    l = launch
    monkeypatch.setattr(l.module.sys, "frozen", False)
    monkeypatch.setattr(l.module.sys, "executable", str(l.entry))
    monkeypatch.setattr(l.module, "_external_seed_bundle", l.seed)
    monkeypatch.setattr(l.module, "APP_VERSION", "8.0.0")
    assert l.module._shell_facts() == {"shell_version": "8.0.0", "bundle_version": "7.7.0",
                                      "source_launcher": True, "source_revision": "updated-generation",
                                      "macos_source_entry": True}
    monkeypatch.setattr(boot, "checkout_sha", lambda repo: "next-not-yet-loaded")
    assert l.module._shell_facts()["source_revision"] == "updated-generation"


def test_source_shell_spawns_body_with_plain_python_and_installed_app_version(launch, monkeypatch):
    l = launch
    monkeypatch.delenv("OUROBOROS_AGENT_PYTHON", raising=False)
    monkeypatch.setattr(l.module.sys, "frozen", False)
    monkeypatch.setattr(l.module.sys, "executable", str(l.entry))
    monkeypatch.setattr(l.module, "_external_seed_bundle", l.seed)
    monkeypatch.setattr(l.module, "APP_VERSION", "8.0.0")
    monkeypatch.setattr(l.module, "_load_settings", lambda: {})
    monkeypatch.setattr(l.module, "_apply_settings_to_env", lambda settings: None)
    # The launcher process never imports server.py, whose import exports OUROBOROS_AGENT_PYTHON into this
    # pytest process when an earlier test in the same worker imported it (an order-dependent full-battery red).
    monkeypatch.delenv("OUROBOROS_AGENT_PYTHON", raising=False)
    seen = {}

    def popen(cmd, **kwargs):
        seen.update(cmd=cmd, **kwargs)
        raise FrozenContinues()  # before a child, pipe or output thread exists

    monkeypatch.setattr(l.module, "_hidden_popen", popen)
    with pytest.raises(FrozenContinues):
        l.module.start_agent()
    assert seen["cmd"] == [str(l.python), str(l.repo / "server.py")]
    assert seen["env"]["OUROBOROS_APP_VERSION"] == "7.7.0"
    assert seen["env"]["PYTHONPATH"] == str(l.repo)
    assert seen["env"].get("OUROBOROS_AGENT_PYTHON") in (None, str(l.python))
    assert "PYTHONHOME" not in seen["env"] and "CFProcessPath" not in seen["env"]


@pytest.mark.parametrize("changed", ["ouroboros/launcher_background.py", "launcher.py", "server.py"])
def test_cold_detector_compares_bundle_generation_and_only_loaded_sources(launch, monkeypatch, changed):
    l = launch
    from ouroboros import launcher_background

    monkeypatch.setattr(launcher_background, "__file__", str(l.seed / "ouroboros/launcher_background.pyc"))
    boot.remember_loaded_checkout(l.repo, bundle_dir=l.seed)

    def git_diff(cmd, **kwargs):
        assert cmd[:5] == ["git", "diff", "--quiet", "bundle-generation", "updated-generation"]
        assert "launcher.py" in cmd and "ouroboros/launcher_background.py" in cmd
        return types.SimpleNamespace(returncode=1 if changed in cmd[6:] else 0)

    monkeypatch.setattr(boot.subprocess, "run", git_diff)
    assert boot.launcher_sources_changed(l.repo, bundle_dir=l.seed) is (changed != "server.py")


def test_cold_and_warm_commands_load_updated_helper_in_fresh_python(launch, monkeypatch):
    """Sentinels catch preserving profile while accidentally retaining frozen helpers."""
    l = launch
    python = sys.executable  # real test Python; the entry itself needs native qualification
    (l.repo / "helper.py").write_text("GENERATION = 'updated-helper-2'\n")
    (l.repo / "launcher.py").write_text(
        "import helper, json, sys\nprint(json.dumps([helper.GENERATION, sys.argv[1:]]))\n")
    monkeypatch.setitem(sys.modules, "helper", types.SimpleNamespace(GENERATION="frozen-helper-1"))
    with pytest.raises(Handoff) as cold:
        l.module.main(["--no-ui"])
    monkeypatch.setattr(l.module.sys, "frozen", False)
    monkeypatch.setattr(l.module, "_external_seed_bundle", l.seed)
    monkeypatch.setattr(l.module, "_launch_argv", cold.value.argv[2:])
    with pytest.raises(Handoff) as warm:
        l.module._exec_source_launcher()
    assert cold.value.argv == warm.value.argv
    for command in (cold.value.argv, warm.value.argv):
        child = subprocess.run([python, *command[1:]], env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                               capture_output=True, text=True, check=True)
        gen, args = json.loads(child.stdout)
        assert gen == "updated-helper-2" and args == command[2:]
    assert sys.modules["helper"].GENERATION == "frozen-helper-1"


@pytest.mark.parametrize("outcome", [0, 1, "timeout"])
def test_import_probe_uses_plain_python_and_has_no_main_or_native_entry(launch, outcome):
    l = launch
    seen = []

    def run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(cmd, 30)
        return types.SimpleNamespace(returncode=outcome, stdout="", stderr="missing source dependency")

    context = boot.BootstrapContext(l.seed, l.repo, l.module.DATA_DIR, l.module.SETTINGS_PATH,
                                    str(l.python), "7.7.0", run, lambda *a: None, logging.getLogger("probe"))
    assert source_launcher_importable(context, desktop=True) is (outcome == 0)
    cmd, kwargs = seen[0]
    assert cmd == [str(l.python), "-c", "import launcher; import webview"]
    assert kwargs["timeout"] == 30 and kwargs["cwd"] == str(l.repo)
    assert kwargs["env"]["PYTHONPATH"] == str(l.repo)
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_frozen_handoff_probe_uses_shell_python_even_when_body_is_in_a_venv(launch, monkeypatch):
    seen = {}
    monkeypatch.setattr(launch.module, "EMBEDDED_PYTHON", "/body/venv/bin/python")
    monkeypatch.setattr(boot, "source_launcher_importable", lambda context, **kw: seen.update(kw) or True)
    with pytest.raises(Handoff):
        launch.module._exec_source_launcher()
    assert seen["python"] == str(launch.python)


@pytest.mark.parametrize("frozen", [True, False])
@pytest.mark.parametrize("source", ["old-339fce", "no-protocol", "compatible-local", "future-protocol",
                                    "damaged-helper"])
def test_entry_handoff_requires_source_capability_in_a_real_plain_python_probe(launch, monkeypatch, frozen, source):
    """The actual old launcher imports successfully but cannot host its body under the entry.

    No commits, native entry or launcher main in the child. The current source can
    load a local helper without ancestry to the seed; damage keeps the running shell.
    A clone without that commit (a shallow one) still proves the refusal: the current
    launcher without its protocol declaration is the same entry-unaware source.
    """
    l = launch
    root = pathlib.Path(__file__).resolve().parents[1]
    if source == "old-339fce":
        archive = subprocess.run(["git", "archive", "339fce215", "launcher.py", "ouroboros", "VERSION"],
                                 cwd=root, capture_output=True)
        if archive.returncode:
            pytest.skip("pre-entry 339fce215 is not in this clone")
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tree:
            tree.extractall(l.repo)
    else:
        code = (root / "launcher.py").read_text(encoding="utf-8")
        if source == "no-protocol":
            assert code.count("\nMACOS_SOURCE_ENTRY_PROTOCOL = 1\n") == 1
            code = code.replace("\nMACOS_SOURCE_ENTRY_PROTOCOL = 1\n", "\n")
        if source == "future-protocol":
            code += "\nMACOS_SOURCE_ENTRY_PROTOCOL = 999\n"
        code += "\nimport local_helper\n"
        (l.repo / "launcher.py").write_text(code, encoding="utf-8")
        (l.repo / "local_helper.py").write_text(
            "raise ImportError('damaged helper')\n" if source == "damaged-helper" else
            "from pathlib import Path\nPath(__file__).with_suffix('.loaded').write_text('local-helper-2')\n",
            encoding="utf-8")

    before = (l.repo / "launcher.py").read_bytes()
    real_python = sys.executable
    probes = []

    def run(cmd, **kwargs):
        assert cmd[0] == str(l.python), "the probe must select plain bundled Python"
        assert kwargs["env"]["PYTHONPATH"] == str(l.repo)
        if source != "old-339fce":
            # Keep the local launcher first; share its unchanged package without
            # requiring Windows symlink privileges or copying it for every case.
            kwargs["env"]["PYTHONPATH"] += os.pathsep + str(root)
        result = subprocess.run([real_python, *cmd[1:]], **kwargs)
        probes.append(result)
        return result

    monkeypatch.setattr(l.module, "_hidden_run", run)
    monkeypatch.setattr(boot, "source_launcher_importable", source_launcher_importable)
    monkeypatch.setattr(l.module.sys, "frozen", frozen)
    if not frozen:
        monkeypatch.setattr(l.module, "_external_seed_bundle", l.seed)
        monkeypatch.setattr(l.module, "_launch_argv", ["--no-ui", "--seed-bundle", str(l.seed)])
        monkeypatch.setattr(l.module, "_external_ui", True)
    invoke = (lambda: l.module.main(["--no-ui"])) if frozen else l.module._exec_source_launcher
    if source == "compatible-local":
        with pytest.raises(Handoff) as stopped:
            invoke()
        assert stopped.value.executable == str(l.entry)
        assert (l.repo / "local_helper.loaded").read_text() == "local-helper-2"
        assert len(probes) == 1 and probes[0].returncode == 0
    else:
        if frozen:
            with pytest.raises(FrozenContinues):
                invoke()
        else:
            invoke()
        assert "teardown" not in l.calls and "release" not in l.calls and "exec" not in l.calls
        assert len(probes) == 1 and probes[0].returncode != 0
        assert ("damaged helper" if source == "damaged-helper" else "macOS source entry protocol 1") in probes[0].stderr
    assert (l.repo / "launcher.py").read_bytes() == before, "old source remains recoverable"
    assert l.module.EMBEDDED_PYTHON == str(l.python)
    assert platform.pip_install_target_args(str(l.python)) == ["--user"]


@pytest.mark.parametrize("kind", ["outside-app", "symlink", "non-executable"])
def test_source_entry_requires_a_real_executable_in_the_app(launch, kind, tmp_path):
    l = launch
    if kind == "non-executable" and sys.platform == "win32":
        pytest.skip("Windows access checks do not model POSIX executable bits")
    if kind == "outside-app":
        assert boot.macos_source_entry(tmp_path / "Frameworks") is None
    else:
        if kind == "symlink":
            l.entry.unlink()
            try:
                l.entry.symlink_to(l.python)
            except OSError as exc:
                if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
                    pytest.skip("Windows symlink privilege is unavailable")
                raise
        else:
            l.entry.chmod(0o644)
        assert boot.macos_source_entry(l.seed) is None


@pytest.mark.parametrize("visible", [True, False])
def test_legacy_macos_attention_never_runs_audio(monkeypatch, visible):
    monkeypatch.setattr(platform, "IS_MACOS", True)
    monkeypatch.setattr(platform.subprocess, "run", lambda *a, **kw: pytest.fail("no audio subprocess"))
    shown = []
    answer = platform.request_native_attention((lambda: shown.append(True)) if visible else None, sound=True)
    assert answer.get("sound_played", False) is False and shown == ([True] if visible else [])


def test_windows_legacy_attention_sound_is_preserved(monkeypatch):
    played = []
    monkeypatch.setattr(platform, "IS_MACOS", False)
    monkeypatch.setattr(platform, "IS_WINDOWS", True)
    monkeypatch.setitem(sys.modules, "winsound", types.SimpleNamespace(MB_ICONASTERISK=64, MessageBeep=played.append))
    assert platform.request_native_attention(sound=True)["sound_played"] is True
    assert played == [64]
