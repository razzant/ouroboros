"""PID 1 as a live spawner: a bare container server versus an init adoption.

A child's parent PID alone cannot tell them apart, so the spawner names itself
(``OUROBOROS_PARENT_PID``) to plain Python entrypoints, and a multiprocessing
child reads its own parent and start method instead of the env.
"""

import multiprocessing
import os
import pathlib
import subprocess
import sys
import time
import types

import pytest

from ouroboros import process_custody
from ouroboros.process_custody import PARENT_PID_ENV, _orphaned_at_start

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="lifeline is POSIX-only")


def _clean_env(named=None):
    env = {key: value for key, value in os.environ.items() if key != PARENT_PID_ENV}
    if named is not None:
        env[PARENT_PID_ENV] = named
    return env


@pytest.mark.parametrize("named,ppid,orphaned", [
    (None, 4242, False), (None, 1, True), (None, 0, True),  # legacy spawner
    ("1", 1, False),  # bare container server
    ("4242", 1, True),  # init adopted us before the lifeline started
    ("4242", 777, True),  # ... or a subreaper did
    ("4242", 4242, False),
    ("junk", 1, True), ("junk", 4242, False),  # unreadable name: legacy rule
])
def test_plain_child_trusts_only_its_named_spawner(monkeypatch, named, ppid, orphaned):
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: None)
    if named is None:
        monkeypatch.delenv(PARENT_PID_ENV, raising=False)
    else:
        monkeypatch.setenv(PARENT_PID_ENV, named)
    assert _orphaned_at_start(ppid) is orphaned


class _Spawned(multiprocessing.process.BaseProcess):
    _start_method = "spawn"


class _Forked(multiprocessing.process.BaseProcess):
    _start_method = "fork"


class _ForkServed(multiprocessing.process.BaseProcess):
    _start_method = "forkserver"


@pytest.mark.parametrize("kind,parent_pid,ppid,orphaned", [
    (_Spawned, 1, 1, False), (_Forked, 1, 1, False),  # PID 1 created us directly
    (_ForkServed, 1, 1, True),  # ppid is the forkserver: PID 1 is an adoption
    (_Spawned, 4242, 1, True),  # spawner was not PID 1: adopted
    (multiprocessing.process.BaseProcess, 1, 1, True),  # unknown method: refuse
    (_ForkServed, 4242, 5150, False),
])
def test_multiprocessing_child_ignores_the_env_name(monkeypatch, kind, parent_pid, ppid, orphaned):
    monkeypatch.setenv(PARENT_PID_ENV, "1")  # an inherited name must not rescue anything
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: types.SimpleNamespace(pid=parent_pid))
    monkeypatch.setattr(multiprocessing, "current_process", lambda: object.__new__(kind))
    assert _orphaned_at_start(ppid) is orphaned


def test_unreadable_multiprocessing_state_keeps_the_legacy_refusal(monkeypatch):
    def broken():
        raise RuntimeError("no parent view")

    monkeypatch.setenv(PARENT_PID_ENV, "1")
    monkeypatch.setattr(multiprocessing, "parent_process", broken)
    assert _orphaned_at_start(1) is True
    assert _orphaned_at_start(4242) is False


@_POSIX_ONLY
@pytest.mark.serial
@pytest.mark.parametrize("named,faked_ppid,survives", [
    (None, True, False), ("4242", True, False), ("junk", True, False), ("1", True, True),
    ("999999999", False, False),  # a real parent that is not the named spawner
    ("<self>", False, True),  # the real parent named itself
])
def test_plain_lifeline_arms_only_for_its_named_parent(named, faked_ppid, survives):
    # getppid is faked only inside the child: never in pytest, which a firing
    # lifeline would group-kill.
    src = ("import os, sys; sys.path.insert(0, %r);%s"
           "from ouroboros.process_custody import start_parent_lifeline;"
           "start_parent_lifeline(poll_sec=60, label='test'); print('armed', flush=True)"
           ) % (str(REPO_ROOT), " os.getppid = lambda: 1;" if faked_ppid else "")
    env = _clean_env(str(os.getpid()) if named == "<self>" else named)
    done = subprocess.run([sys.executable, "-c", src], env=env, capture_output=True,
                          text=True, timeout=30, start_new_session=True)
    assert ("armed" in done.stdout and done.returncode == 0) is survives, done


def _process_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return not state.strip() or state.strip().startswith("Z")


@_POSIX_ONLY
@pytest.mark.serial
@pytest.mark.parametrize("start_method,arms", [
    ("spawn", True),
    pytest.param("fork", True, marks=pytest.mark.skipif(sys.platform != "linux", reason="fork is Linux-only here")),
    ("forkserver", False),
])
def test_pid1_multiprocessing_child_lives_only_when_created_directly(tmp_path, start_method, arms):
    """Seen from the child, parent PID 1 with multiprocessing parent PID 1 is a
    direct spawn/fork child of a bare PID 1 supervisor; under forkserver the same
    view means the forkserver died and PID 1 adopted us. The real sentinel still
    ends an admitted child when its supervisor dies."""
    from tests._shared import reap_test_process_group

    script = tmp_path / "supervisor.py"
    script.write_text(
        "import multiprocessing as mp, os, pathlib, sys, time, types\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "def child(armed):\n"
        "    real = mp.parent_process()\n"
        "    mp.parent_process = lambda: types.SimpleNamespace(pid=1, sentinel=real.sentinel)\n"
        "    os.getppid = lambda: 1\n"
        "    from ouroboros.process_custody import start_parent_lifeline\n"
        "    start_parent_lifeline(poll_sec=0.2, label='test')\n"
        "    pathlib.Path(armed).write_text('armed', encoding='utf-8', newline='\\n')\n"
        "    time.sleep(60)\n"
        "if __name__ == '__main__':\n"
        f"    proc = mp.get_context({start_method!r}).Process(target=child, args=(sys.argv[2],))\n"
        "    proc.start()\n"
        "    pathlib.Path(sys.argv[1]).write_text(str(proc.pid), encoding='utf-8', newline='\\n')\n"
        "    proc.join(60)\n",
        encoding="utf-8", newline="\n",
    )
    pid_file, armed = tmp_path / "child_pid", tmp_path / "armed"
    # An inherited name of PID 1 must not rescue a forkserver adoption.
    supervisor = subprocess.Popen([sys.executable, str(script), str(pid_file), str(armed)],
                                  env=_clean_env("1"), start_new_session=True)
    try:
        deadline = time.time() + 60
        while not pid_file.exists() and time.time() < deadline:
            time.sleep(0.1)
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        if not arms:
            while time.time() < deadline and not _process_gone(child_pid):
                time.sleep(0.1)
            assert _process_gone(child_pid) and not armed.exists()
            return
        while not armed.exists() and time.time() < deadline:
            time.sleep(0.1)
        assert armed.exists(), "a direct PID-1 child was killed as an orphan"
        supervisor.kill()  # the sentinel, not the faked ppid, must still end it
        supervisor.wait(timeout=10)
        deadline = time.time() + 15
        while time.time() < deadline and not _process_gone(child_pid):
            time.sleep(0.2)
        assert _process_gone(child_pid), f"{start_method} child outlived its supervisor"
    finally:
        reap_test_process_group(supervisor)


class _SpawnCaptured(Exception):
    pass


def _capture_spawn(monkeypatch, captured):
    def fake_spawn_supervised(cmd, **kwargs):
        captured.update(kwargs["env"])
        raise _SpawnCaptured()

    monkeypatch.setattr(process_custody, "spawn_supervised", fake_spawn_supervised)


def test_maintenance_job_names_its_spawner(tmp_path, monkeypatch):
    from ouroboros.startup_historical_audit import spawn_maintenance_job

    monkeypatch.setenv(PARENT_PID_ENV, "1")  # a leftover in our own env never leaks through
    captured = {}
    _capture_spawn(monkeypatch, captured)
    with pytest.raises(_SpawnCaptured):
        spawn_maintenance_job(tmp_path, REPO_ROOT, "maintenance")
    assert captured[PARENT_PID_ENV] == str(os.getpid())


def test_query_data_child_names_its_spawner(tmp_path, monkeypatch):
    from ouroboros.query_data_sql import run_sql

    monkeypatch.setenv(PARENT_PID_ENV, "1")
    captured = {}
    _capture_spawn(monkeypatch, captured)
    ctx = types.SimpleNamespace(drive_root=str(tmp_path), task_id="t")
    with pytest.raises(_SpawnCaptured):
        run_sql(ctx, tmp_path, db="current", query="select 1", max_rows=1, max_bytes=100,
                snapshot=False, timeout_sec=1)
    assert captured[PARENT_PID_ENV] == str(os.getpid())


def test_extension_runner_child_names_its_spawner(tmp_path, monkeypatch):
    from ouroboros import extension_process_runner as runner

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs["env"])
        raise _SpawnCaptured()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    with pytest.raises(_SpawnCaptured):
        with runner._child_process({"skill_name": "skill"}, skill_dir=tmp_path, drive_root=tmp_path,
                                   repo_dir=REPO_ROOT, env={PARENT_PID_ENV: "1", "PATH": "/bin"}):
            pass
    assert captured[PARENT_PID_ENV] == str(os.getpid())
    assert captured["PATH"] == "/bin"


def test_companion_names_the_server_over_any_descriptor_value(tmp_path, monkeypatch):
    from ouroboros.extension_companion import CompanionDescriptor, CompanionSupervisor, init_server_process_pid

    init_server_process_pid()
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs["env"])
        raise _SpawnCaptured()

    monkeypatch.setattr("ouroboros.extension_companion.subprocess.Popen", fake_popen)
    descriptor = CompanionDescriptor(skill_name="skill", name="daemon", command=["python", "-c", "0"],
                                     cwd=tmp_path, env={PARENT_PID_ENV: "1"})
    with pytest.raises(_SpawnCaptured):
        CompanionSupervisor(tmp_path).start(descriptor)
    assert captured[PARENT_PID_ENV] == str(os.getpid())
