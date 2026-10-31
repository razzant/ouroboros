"""Cold-launcher Panic must capture a live worker's marked, task-bound skill child.

Uses the real worker spawn, private stop socket, parent lifeline and skill spawn.
The injected 0.3 s census delay exercises the launcher hard-exit race. No live
installation, endpoint or engine is used; existing ProcessContainer custody
also covers setup failures before a child can publish its identity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

MODULE = "tests.test_launcher_panic_worker_owned"
TASK = "panic-skill-fixture"


def _write(root, name, value):
    path = root / f"{name}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def _read_when_ready(root, name, timeout=30):
    path = root / f"{name}.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(.02)
    raise AssertionError(f"fixture did not publish {path}")


def _identity(pid):
    from ouroboros.platform_layer import process_start_time

    birth = process_start_time(pid)
    assert birth, f"fixture could not measure its child {pid}"
    return {"pid": pid, "birth": birth, "pgid": os.getpgid(pid)}


def _process_state(identity):
    from ouroboros.platform_layer import pid_is_alive, process_start_time
    from ouroboros.process_containment import pid_is_zombie

    pid = identity["pid"]
    if not pid_is_alive(pid) or pid_is_zombie(pid):
        return "exited"
    birth = process_start_time(pid)
    return "unknown" if not birth else "alive" if birth == identity["birth"] else "reused"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX separate sessions; Windows uses a Job")
def test_cold_launcher_panic_stops_task_bound_skill_with_slow_census(tmp_path):
    from ouroboros.process_census import is_scope_name
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.test_environment import isolated_environment, retain_tree

    repo = Path(__file__).resolve().parents[1]
    # A fresh source prevents live scope markers, engine homes or credentials
    # from reaching this disposable installation. Container custody is inherited.
    env = isolated_environment(tmp_path, repo, source={"PATH": os.environ.get("PATH", os.defpath)})
    assert not any(is_scope_name(key) or key.startswith("CLAUDEXOR_") for key in env)
    root = Path(env["OUROBOROS_DATA_DIR"])
    container, launcher = ProcessContainer(), None
    try:
        with (root / "launcher.log").open("w") as output:
            launcher = container.spawn([sys.executable, "-m", MODULE, "launcher"],
                                       cwd=repo, env=env, stdout=output, stderr=output)
            code = launcher.wait(timeout=60)
        logs = "\n".join(path.read_text(errors="replace") for path in root.glob("*.log"))
        assert code == 99, logs
        assert (root / "state" / "panic_stop.flag").read_bytes() == b"panic", logs
        facts = _read_when_ready(root, "work")
        assert facts["task_groups"] and not facts["foreground_records"], facts
        assert facts["task_groups"][0]["leader_pid"] == facts["skill"]["pid"], facts
        scopes = _read_when_ready(root, "skill-scopes")
        assert any(key.startswith("OURO_INSTALL_SCOPE_") for key in scopes), scopes
        assert any(key.startswith("OURO_TASK_SCOPE_") for key in scopes), scopes
        known = {"server": _read_when_ready(root, "server"),
                 "worker": facts["worker"], "skill": facts["skill"]}
        # Observed liveness, separate from the request receipt and later cleanup.
        deadline = time.monotonic() + 6
        while True:
            states = {name: _process_state(row) for name, row in known.items()}
            if all(state in {"exited", "reused"} for state in states.values()) or time.monotonic() >= deadline:
                break
            time.sleep(.1)
        _write(root, "outcome-before-cleanup", {"identities": known, "states": states})
        assert all(state in {"exited", "reused"} for state in states.values()), f"{states}\n{logs}"
    finally:
        # Includes forkserver/resource-tracker and failures before work.json.
        problem = container.reap()
        if launcher is not None:
            launcher.wait(timeout=5)
        _write(root, "cleanup", {"problem": problem})
        if problem:
            retain_tree(tmp_path, problem)
        assert not problem, problem


def _worker(wid, in_q, out_q, repo_dir, drive_root, custody_session_id="", stop_socket=None):
    """The real worker's custody/lifeline prologue, then one skill invocation."""
    import contextlib
    from ouroboros.platform_layer import create_new_session
    from ouroboros.process_custody import adopt_session_id, start_parent_lifeline
    from supervisor.worker_process import request_worker_owned_stops
    from ouroboros import owner_pause, tool_custody
    from ouroboros.owned_shutdown import executor_record_paths
    from ouroboros.process_scope import task_scope_record
    from ouroboros.tools import shell_process, skill_exec

    os.environ["OUROBOROS_IN_WORKER"] = "1"
    adopt_session_id(custody_session_id)
    create_new_session()
    start_parent_lifeline(label=f"worker-{wid}", stop_socket=stop_socket,
                          before_exit=lambda: request_worker_owned_stops(drive_root))
    root = Path(drive_root)
    binding = {"drive_root": str(root), "root_task_id": TASK, "task_id": TASK, "launch_operation_id": ""}
    # Only tool-row bookkeeping and Pause admission are replaced; process
    # registry, task markers, group recording and the lifeline remain real.
    tool_custody.invocation_binding = lambda: dict(binding)
    tool_custody.external_handoff = lambda kind: (lambda: None)
    owner_pause.operation_start = contextlib.nullcontext

    def spawned():
        proc = next(iter(shell_process._active_subprocesses))
        _write(root, "work", {"worker": _identity(os.getpid()), "skill": _identity(proc.pid),
                             "task_groups": (task_scope_record(root, TASK) or {}).get("groups"),
                             "foreground_records": [str(p) for p in executor_record_paths(root, "foreground")]})

    code = ("import json,os,pathlib,time; "
            "pathlib.Path('skill-scopes.tmp').write_text(json.dumps([k for k in os.environ "
            "if k.startswith(('OURO_INSTALL_SCOPE_', 'OURO_TASK_SCOPE_'))])); "
            "pathlib.Path('skill-scopes.tmp').replace('skill-scopes.json'); time.sleep(90)")
    skill_exec._run_skill_subprocess([sys.executable, "-I", "-S", "-c", code], cwd=str(root),
                                    env=dict(os.environ), timeout_sec=95,
                                    stdout_cap=65536, stderr_cap=65536, on_spawn=spawned)
    threading.Event().wait(60)


def _server():
    import importlib
    import multiprocessing
    from ouroboros.config import DATA_DIR
    from ouroboros.process_scope import plant_installation_scope
    from supervisor import worker_process

    plant_installation_scope(DATA_DIR)
    worker_process.worker_main = importlib.import_module(MODULE)._worker
    ctx = multiprocessing.get_context("forkserver" if sys.platform.startswith("linux") else "spawn")
    queues = ctx.Queue(), ctx.Queue()  # held while the real worker rebuilds their semaphores
    worker = worker_process.spawn_worker_process(ctx, 0, *queues, os.getcwd(), str(DATA_DIR))
    worker.join(timeout=90)
    os._exit(3)


def _launcher():
    import logging
    from ouroboros import launcher_background as background, process_scope
    from ouroboros.config import DATA_DIR
    from ouroboros.platform_layer import subprocess_new_group_kwargs
    from supervisor import state

    logging.basicConfig(level=logging.INFO)
    root = Path(DATA_DIR)
    state.init(root)
    state.init_state()
    with (root / "server.log").open("w") as output:
        server = subprocess.Popen([sys.executable, "-m", MODULE, "server"],
                                  stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                  **subprocess_new_group_kwargs())
    _write(root, "server", _identity(server.pid))
    _read_when_ready(root, "work")
    _read_when_ready(root, "skill-scopes")
    census = process_scope.census

    def slow_census(*args, **kwargs):
        time.sleep(.3)  # Explicit fault injection, matching the adjudicated skill regression.
        return census(*args, **kwargs)

    process_scope.census = slow_census
    background.indicator_class = lambda: None

    def unexpected():
        raise AssertionError("no ordinary exit or port lookup")

    bg = background.Background(unexpected, unexpected, threading.Event(), panic_child=lambda: (server, None))
    bg.request_panic()
    threading.Event().wait(20)  # the actual emergency owner persists, then hard-exits
    os._exit(1)


if __name__ == "__main__":
    {"launcher": _launcher, "server": _server}[sys.argv[1]]()
