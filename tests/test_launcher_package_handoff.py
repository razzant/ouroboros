"""A launcher whose packages an Update switched hands itself over before it does ordinary work with them.

A launcher process keeps the modules it imported. When an Update renames a new package generation
into the one path a fresh entry loads (``body_switch._select_environment``), that process would go
on with cached modules of the old generation and import the rest from the new one. So the exit 42
that follows the switch reaches a fresh launcher through the existing source handoff
(``exec_source_launcher``); if that handoff does not happen the launcher starts no generation itself.
A bare interpreter's launcher counts its private environment as such a path, and its dependency
install never writes that interpreter's own packages. The servers it starts are told so
(``OUROBOROS_LAUNCHER_PACKAGE_HANDOFF``); an older launcher tells nothing, and an Update then
refuses to switch packages under it (``update_prepare.launcher_refusal``).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import types

import pytest

import ouroboros.launcher_bootstrap as bootstrap_module
from ouroboros import runtime_venv


def _swap(fixed: pathlib.Path, adoption: str = "a1") -> None:
    """The switch's two renames: the serving packages aside, the prepared generation into the fixed path."""
    generation = fixed.with_name(f"{fixed.name}.update-c1")
    generation.mkdir()
    os.rename(fixed, fixed.with_name(f"{fixed.name}.previous-{adoption}"))
    os.rename(generation, fixed)


def test_a_launcher_on_a_user_site_or_a_venv_tells_its_loaded_packages_from_the_switched_ones(tmp_path, monkeypatch):
    data, repo = tmp_path / "data", tmp_path / "repo"
    repo.mkdir()
    userbase, venv = data / "state" / "python-userbase", tmp_path / "runtime venv"
    userbase.mkdir(parents=True)
    venv.mkdir()
    monkeypatch.setenv("PYTHONUSERBASE", str(userbase))
    monkeypatch.setattr("site.ENABLE_USER_SITE", True)  # a launcher whose site added its user base (not -s)
    for kind, prefix, fixed in (("venv", str(venv), venv), ("user_site", str(tmp_path / "base"), userbase)):
        monkeypatch.setattr(sys, "prefix", prefix)
        monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
        loaded = bootstrap_module.package_environment(data, repo)
        assert loaded[0] == kind and loaded[1:] == (os.lstat(fixed).st_dev, os.lstat(fixed).st_ino)
        monkeypatch.setattr(bootstrap_module, "_LOADED_CHECKOUT_SHA", "a" * 40)  # recorded with the loaded checkout
        monkeypatch.setattr(bootstrap_module, "_LOADED_PACKAGES", loaded)
        assert bootstrap_module.loaded_packages_replaced(repo, data) is False
        _swap(fixed)
        assert bootstrap_module.loaded_packages_replaced(repo, data) is True
    monkeypatch.setattr(bootstrap_module, "_LOADED_CHECKOUT_SHA", "")
    assert bootstrap_module.loaded_packages_replaced(repo, data) is False  # nothing recorded: never a relaunch


def test_the_launcher_records_its_packages_with_the_checkout_it_loaded(tmp_path, monkeypatch):
    """One record of what the process loaded (``remember_loaded_checkout``): the commit and, with it, the packages; a
    fingerprint left without its commit (another process's, say) is never consulted."""
    import subprocess

    data, repo = tmp_path / "data", tmp_path / "repo"
    userbase = data / "state" / "python-userbase"
    userbase.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty",
                    "-m", "loaded"], check=True)
    monkeypatch.setenv("PYTHONUSERBASE", str(userbase))
    monkeypatch.setattr("site.ENABLE_USER_SITE", True)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "base"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.setattr("ouroboros.config.DATA_DIR", data)
    monkeypatch.setattr(bootstrap_module, "_LOADED_CHECKOUT_SHA", "")
    monkeypatch.setattr(bootstrap_module, "_LOADED_PACKAGES", ("user_site", 0, 0))  # left over, without its commit
    assert bootstrap_module.loaded_packages_replaced(repo, data) is False

    assert bootstrap_module.remember_loaded_checkout(repo)

    assert bootstrap_module._LOADED_PACKAGES == bootstrap_module.package_environment(data, repo)
    assert bootstrap_module.loaded_packages_replaced(repo, data) is False
    _swap(userbase)
    assert bootstrap_module.loaded_packages_replaced(repo, data) is True


def _bare_private(tmp_path: pathlib.Path, repo: pathlib.Path, *, bound: bool) -> tuple:
    """A bare interpreter (``bare``, the same real file as this test's own) and the private environment an Update
    selected for ``repo`` on it, as preparation and the switch leave it."""
    bare = tmp_path / "system" / "bin" / "python3"
    bare.parent.mkdir(parents=True)
    bare.symlink_to(os.path.realpath(sys.executable))
    data = tmp_path / "data"
    private = data / "state" / runtime_venv.PRIVATE_NAME
    (private / "bin").mkdir(parents=True)
    (private / "bin" / "python").symlink_to(bare)
    (private / "pyvenv.cfg").write_text(f"home = {bare.parent}\ninclude-system-site-packages = true\n")
    if bound:
        (private / runtime_venv.PRIVATE_MARKER).write_text(json.dumps({"schema_version": 1, "candidate_id": "c1",
                                                                       "repo_dir": os.path.realpath(repo)}))
    return bare, data, private


@pytest.mark.skipif(sys.platform == "win32", reason="the POSIX layout (bin/python linked to the interpreter)")
def test_a_bare_launchers_private_environment_is_its_switched_packages_and_its_install_target(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements-runtime.lock").write_text("toydep==2.0\n")
    bare, data, private = _bare_private(tmp_path, repo, bound=False)
    other = tmp_path / "other"
    other.mkdir()
    assert runtime_venv.private_entry_interpreter(data, repo, str(bare)) == ""  # nothing selected yet
    (private / runtime_venv.PRIVATE_MARKER).write_text(json.dumps({"repo_dir": os.path.realpath(repo)}))
    assert runtime_venv.private_entry_interpreter(data, repo, str(bare)) == str(private / "bin" / "python")
    assert runtime_venv.private_entry_interpreter(data, other, str(bare)) == ""  # another checkout runs as it is
    assert runtime_venv.private_entry_interpreter(data, repo, str(private / "bin" / "python")) == ""  # already there

    # The launcher started bare, before the Update selected that environment: its packages were the interpreter's own.
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setenv("PYTHONUSERBASE", "")
    monkeypatch.setattr(bootstrap_module, "_LOADED_CHECKOUT_SHA", "a" * 40)
    monkeypatch.setattr(bootstrap_module, "_LOADED_PACKAGES", ("system",))
    monkeypatch.setattr(runtime_venv, "private_entry_interpreter",
                        lambda data_dir, repo_dir, python: str(private / "bin" / "python"))  # this process's own entry
    assert bootstrap_module.package_environment(data, repo)[0] == "private"
    assert bootstrap_module.loaded_packages_replaced(repo, data) is True
    monkeypatch.undo()

    seen, warned = [], []
    context = bootstrap_module.BootstrapContext(
        bundle_dir=tmp_path / "no-bundle", repo_dir=repo, data_dir=data, settings_path=data / "settings.json",
        embedded_python=str(bare), app_version="0", save_settings=lambda _changes: None, log=types.SimpleNamespace(
            warning=lambda *a, **k: warned.append(a[-1]), info=lambda *a, **k: None),
        hidden_run=lambda cmd, **_kw: seen.append(cmd) or types.SimpleNamespace(returncode=0, stdout=b"", stderr=b""))
    assert bootstrap_module.install_deps(context) is True
    # Its dependency install goes where its next entry runs: the private venv's own pip, never the system's packages.
    assert seen[-1] == [str(private / "bin" / "python"), "-m", "pip", "install", "-r",
                        str(repo / "requirements-runtime.lock")]
    # Selected, then its marker gone or another checkout's: refused and visible, never the interpreter's packages.
    installs = len(seen)
    (private / runtime_venv.PRIVATE_MARKER).unlink()
    assert bootstrap_module.install_deps(context) is False and len(seen) == installs
    (private / runtime_venv.PRIVATE_MARKER).write_text(json.dumps({"repo_dir": os.path.realpath(other)}))
    assert bootstrap_module.install_deps(context) is False and len(seen) == installs
    assert all("does not serve" in str(reason) for reason in warned[-2:])
    from supervisor import update_prepare

    update_prepare.retain_fact("private_environment", {"root": str(private)}, data)  # an adopted Update's record
    shutil.rmtree(private)  # selected, then gone: refused all the same
    assert bootstrap_module.install_deps(context) is False and len(seen) == installs and "is missing" in str(warned[-1])
    update_prepare.retain_fact("private_environment", None, data)  # never selected: the interpreter's own, as before
    assert bootstrap_module.install_deps(context) is True and seen[-1][0] == str(bare)


def _lifecycle(monkeypatch, tmp_path, *, importable: bool):
    """The real exit-42 cycle of ``launcher.agent_lifecycle_loop`` over one generation whose entry switched the
    packages; process starts, the readiness waits and POSIX ``execve`` are stood in, nothing native runs."""
    import launcher
    from ouroboros import platform_layer

    class Handover(BaseException):
        pass

    class NextGeneration(BaseException):
        pass

    calls, switched = [], []
    process = types.SimpleNamespace(pid=12345, returncode=launcher.RESTART_EXIT_CODE, wait=lambda: None)

    def start_agent(port):
        calls.append("core")
        if calls.count("core") > 1:
            raise NextGeneration()
        switched.append(True)  # this generation's entry renamed the new packages in, then exited 42
        return process

    def execve(executable, argv, env):
        calls.append(("exec", argv[1:2]))
        raise Handover()

    monkeypatch.setattr(launcher, "REPO_DIR", tmp_path)
    monkeypatch.setattr(platform_layer, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "IS_MACOS", False)
    monkeypatch.setattr(launcher, "_launch_argv", ["--no-ui"])
    monkeypatch.setattr(launcher, "_external_seed_bundle", None)
    monkeypatch.setattr(launcher, "_external_ui", True)
    monkeypatch.setattr(launcher.sys, "frozen", False, raising=False)
    monkeypatch.setattr(launcher, "loaded_packages_replaced", lambda _repo: bool(switched))
    monkeypatch.setattr(launcher, "launcher_sources_changed", lambda *_a, **_kw: False)  # a lock-only Update
    monkeypatch.setattr(launcher, "_pre_generation_cleanup", lambda: [])
    monkeypatch.setattr(launcher, "update_external_host", lambda *a: calls.append("native") or {})
    monkeypatch.setattr(launcher, "start_agent", start_agent)
    monkeypatch.setattr(launcher, "_await_generation_endpoint", lambda *a, **kw: 1)
    monkeypatch.setattr(launcher, "_update_server_process_record_port", lambda *a: None)
    monkeypatch.setattr(launcher, "_wait_for_server", lambda *a, **kw: True)
    monkeypatch.setattr(launcher, "_cleanup_recorded_server_group_for_pid", lambda *a: None)
    monkeypatch.setattr(launcher, "_sync_existing_repo_from_bundle", lambda: calls.append("source"))
    monkeypatch.setattr(launcher, "_install_deps", lambda: calls.append("deps") or True)
    monkeypatch.setattr(launcher, "release_pid_lock", lambda: calls.append("release"))
    monkeypatch.setattr(launcher, "stop_tray_before_exit", lambda release: release())
    monkeypatch.setattr(launcher.time, "sleep", lambda *a: None)
    monkeypatch.setattr(bootstrap_module.os, "execve", execve)
    monkeypatch.setattr(bootstrap_module, "source_launcher_importable", lambda *a, **kw: importable)
    launcher._shutdown_event.clear()
    launcher._agent_restart_requested.clear()
    return launcher, calls, Handover, NextGeneration


def test_after_a_switch_of_its_packages_the_launcher_hands_over_to_a_fresh_one(tmp_path, monkeypatch):
    launcher, calls, Handover, _next = _lifecycle(monkeypatch, tmp_path, importable=True)
    try:
        with pytest.raises(Handover):
            launcher.agent_lifecycle_loop()
    finally:
        launcher._shutdown_event.clear()
    # The hand-over comes first: the bundle sync and the dependency cycle are the fresh launcher's work, never this
    # process's, which still holds the old packages' modules and would import the new tree's helpers beside them.
    assert calls == ["native", "core", "release", ("exec", [str(tmp_path / "launcher.py")])]


def test_a_refused_hand_over_starts_no_generation_on_packages_the_launcher_did_not_load(tmp_path, monkeypatch, caplog):
    launcher, calls, _handover, _next = _lifecycle(monkeypatch, tmp_path, importable=False)
    try:
        launcher.agent_lifecycle_loop()  # returns: the cycle stops instead of starting the next generation itself
    finally:
        launcher._shutdown_event.clear()
    assert calls == ["native", "core"]  # no sync, no install and no generation by the stale process: a reopen recovers
    assert "reopen Ouroboros" in caplog.text


def test_the_server_a_current_launcher_starts_is_told_it_hands_itself_over(monkeypatch):
    """Only the launcher's own child environment carries the mark: a launcher it hands over to (or an older one a
    rollback brings) never inherits it from this process's environment."""
    import launcher

    source = pathlib.Path(launcher.__file__).read_text(encoding="utf-8")
    start = source[source.index("def start_agent"):source.index("def stop_agent")]
    assert 'env["OUROBOROS_LAUNCHER_PACKAGE_HANDOFF"] = "1"' in start
    assert "OUROBOROS_LAUNCHER_PACKAGE_HANDOFF" not in os.environ
    from supervisor import update_prepare

    assert update_prepare.LAUNCHER_HANDOFF_ENV == "OUROBOROS_LAUNCHER_PACKAGE_HANDOFF"
