"""Scope metadata follows the real launch boundary, not the wrapper's outcome."""
from __future__ import annotations

import io
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import process_scope as ps
from ouroboros import workspace_executor as we
from ouroboros.tools import shell_process, skill_exec
from tests.test_task_scope_lifetime import _alive, _bound, _end_pids, _gone

pytestmark = pytest.mark.serial


def _launch(kind, root, *, on_spawn=None):
    command = [sys.executable, "-c", "print('ok')"]
    if kind == "shell":
        return shell_process._tracked_subprocess_run(command, cwd=str(root),
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    if kind == "executor":
        return we._execute_local(we.ExecutorRef(kind="local", executor_id="local", network="host", mappings=()),
                                 command, root, 5, drive_root=root)
    return skill_exec._run_skill_subprocess(command, cwd=str(root), env={}, timeout_sec=5,
                                             stdout_cap=1024, stderr_cap=1024, on_spawn=on_spawn)


@pytest.mark.parametrize("kind", ["shell", "executor", "skill"])
@pytest.mark.parametrize("error_type", [TimeoutError, PermissionError])
def test_scope_record_failure_is_proven_before_any_process_spawn(tmp_path, monkeypatch, kind, error_type):
    def unavailable(*args, **kwargs):
        raise error_type("scope metadata unavailable")

    def forbidden(*args, **kwargs):
        pytest.fail("Popen must not be reached after scope preparation failed")

    monkeypatch.setattr(ps, "task_scope_marker", unavailable)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(skill_exec, "Popen", forbidden)
    with _bound(tmp_path, "before-spawn"), pytest.raises(error_type) as caught:
        _launch(kind, tmp_path)
    assert caught.value.process_not_started is True


@pytest.mark.parametrize("kind", ["shell", "executor", "skill"])
def test_scope_preparation_precedes_the_external_duty_that_the_spawn_retains(tmp_path, monkeypatch, kind):
    """Through the registry's real invocation: the task marker is prepared BEFORE the external duty
    (``tool_custody.external_handoff``), so a refused preparation starts nothing and retains no duty,
    while an actual spawn carries the marker, finds its duty retained and its receipt discharges it."""
    from tests.test_batch4_producer_custody import _registry
    from tests.test_invocation_ownership import claims, owner, run

    registry, _, _ = _registry(tmp_path, monkeypatch)
    command, real_popen, seen = [sys.executable, "-c", "print('ok')"], subprocess.Popen, []
    pending = lambda: any(row.get("external_pending") for row in claims(tmp_path).values())  # noqa: E731

    def spawn(*args, **kwargs):
        if list(args[0]) == command:
            seen.append((pending(), dict(kwargs.get("env") or {})))
        return real_popen(*args, **kwargs)

    def refused(*args, **kwargs):
        raise PermissionError("scope metadata unavailable")

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(skill_exec, "Popen", spawn)
    registry.override_handler("knowledge_read", lambda *_a, **_kw: str(_launch(kind, tmp_path)))
    with monkeypatch.context() as scoped:
        scoped.setattr(ps, "task_scope_marker", refused)
        with owner(registry):
            assert run(registry)["is_error"]
    assert seen == [] and not pending(), "a refused preparation starts nothing and retains no duty"
    with owner(registry):
        assert not run(registry)["is_error"]
    marker = ps.task_scope_record(tmp_path, "root")["marker"]
    assert seen and all(held and env.get(marker) == ps.SCOPE_VALUE for held, env in seen)
    assert not pending(), "the spawn's own receipt discharged its duty"


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("task", ["manual", "t-scoped", ""], ids=["manual", "task", "unbound"])
def test_the_foreground_receipt_follows_the_invocation_not_the_task_scope(tmp_path, monkeypatch, task):
    """``run_command`` under b814's real invocation: the foreground receipt (binding drive root AND task
    id) discharges the command's own external duty while it still runs. ``manual`` is a valid task id,
    not a lifetime label. Unbound keeps no duty, receipt or marker."""
    # The test owns only the launch it creates. An outer task runner may put
    # its own marker in this pytest process's environment.
    for marker in tuple(os.environ):
        if marker.startswith(ps.TASK_SCOPE_PREFIX):
            monkeypatch.delenv(marker)
    import threading
    import time

    from ouroboros.owner_pause import _TOOL_OPERATION
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
    from ouroboros.tool_custody import invocation_outcome
    from ouroboros.tools.shell import _run_shell
    from tests.test_run_command_schema_truth import _ctx

    ctx, release = _ctx(tmp_path, task), tmp_path / "release"
    command = ["sh", "-c", f"while [ ! -e '{release}' ]; do sleep 0.05; done; echo released"]
    if task:
        write_task_result(tmp_path, task, STATUS_RUNNING, result="running")
        ctx.task_lifecycle_bound = True

    def duties():
        claims = ((load_task_result(tmp_path, task) or {}).get("launch_handoffs") or {}) if task else {}
        return [kind for claim in claims.values() for kind in (claim.get("external_pending") or {}).values()]

    real_popen, spawned, result = subprocess.Popen, [], {}

    def spawn(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        if list(args[0]) == command:
            spawned.append((proc, duties(), dict(kwargs.get("env") or os.environ)))
        return proc

    def call():
        token = _TOOL_OPERATION.set((ctx, "run_command", invocation_outcome(ctx, "run_command", "op-shell"))) if task else None
        try:
            result["text"] = _run_shell(ctx, command, timeout_sec=30)
        finally:
            if token is not None:
                _TOOL_OPERATION.reset(token)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 15
        while not spawned and time.monotonic() < deadline:
            time.sleep(0.02)
        assert spawned, "the command started"
        proc, at_spawn, env = spawned[0]
        while duties() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        running, held = proc.poll() is None, duties()
        receipts = sorted((tmp_path / "state" / "workspace_executor_processes").glob("foreground-*.json"))
    finally:
        release.touch()
        worker.join(timeout=30)
    assert running and "released" in result.get("text", ""), result
    assert at_spawn == (["shell_process"] if task else []), "the command's own duty is retained at spawn"
    assert len(receipts) == (1 if task else 0), "a bound launch has a durable foreground receipt while it runs"
    assert held == [], "that receipt, not the command's exit, discharged the duty"
    markers = [key for key in env if key.startswith(ps.TASK_SCOPE_PREFIX)]
    record = ps.task_scope_record(tmp_path, task) if task else None
    if task:
        assert markers == [record["marker"]] and record["groups"], "a task launch stays task-scoped"
    else:
        assert markers == [] and record is None, "an unbound launch carries no task scope"


@pytest.mark.parametrize("kind", ["shell", "executor", "skill"])
def test_an_error_after_popen_is_not_relabelled_as_no_process(tmp_path, monkeypatch, kind):
    error = OSError("failure after child creation")

    def fail(*args, **kwargs):
        raise error

    proc = SimpleNamespace(pid=0, args=["fixture"], returncode=0, stdout=io.BytesIO(), stderr=io.BytesIO(),
                           poll=lambda: 0, wait=lambda **kwargs: 0, communicate=fail)
    # A hashable fake child is sufficient here: the consumer must preserve the phase.
    class Child:
        pass
    child = Child()
    child.__dict__.update(proc.__dict__)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(skill_exec, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(skill_exec, "_kill_process_group", lambda proc: None)
    monkeypatch.setattr(we, "_register_process", lambda *args, **kwargs: None)
    with _bound(tmp_path, "after-spawn"), pytest.raises(OSError) as caught:
        _launch(kind, tmp_path, on_spawn=fail)
    assert caught.value is error
    assert not getattr(error, "process_not_started", False)


@pytest.mark.skipif(os.name == "nt", reason="scope marker discovery is POSIX-only; Windows retains its existing tree controls")
def test_scrubbed_skill_extension_and_companion_envs_keep_only_installation_scope(tmp_path, monkeypatch):
    from ouroboros import extension_companion, extension_loader, extension_process_runner

    install = ps.mint_marker(ps.INSTALL_SCOPE_PREFIX)
    task = ps.mint_marker(ps.TASK_SCOPE_PREFIX)
    run = ps.mint_marker(ps.RUN_SCOPE_PREFIX)
    for marker in (install, task, run):
        monkeypatch.setenv(marker, "1")
    monkeypatch.setenv("UNRELATED_PROVIDER_SECRET", "must-not-propagate")
    monkeypatch.setattr(extension_loader, "mint_skill_token", lambda *args, **kwargs: "")
    environments = [
        skill_exec._scrub_env([], tmp_path / "state", "fixture"),
        extension_process_runner._child_env(drive_root=tmp_path, repo_dir=tmp_path / "repo", skill_name="fixture",
                                            skill_dir=tmp_path / "skill", env_allowlist=[], granted_keys=[]),
        extension_companion._companion_base_env(),
    ]
    for env in environments:
        assert env[install] == "1"
        assert task not in env and run not in env and "UNRELATED_PROVIDER_SECRET" not in env


@pytest.mark.skipif(os.name == "nt", reason="the detached-member census is POSIX-only")
@pytest.mark.parametrize("task_id", ["skill-owner", "manual"])
def test_a_task_skill_script_has_the_same_terminal_lifetime_as_an_own_command(tmp_path, task_id):
    from ouroboros.task_results import write_task_result

    source = ("import subprocess,sys; "
              "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],"
              "start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
              "print(p.pid,flush=True)")
    with _bound(tmp_path, task_id):
        code, stdout, _, overflow = skill_exec._run_skill_subprocess(
            [sys.executable, "-c", source], cwd=str(tmp_path),
            env=skill_exec._scrub_env([], tmp_path / "state", "fixture"), timeout_sec=10,
            stdout_cap=1024, stderr_cap=1024)
    pid = int(stdout.strip())
    try:
        assert code == 0 and not overflow and _alive(pid), "script completion is not the task end"
        record = ps.task_scope_record(tmp_path, task_id)
        assert record and record["groups"], "the native script group has an owner too"
        rows = ps.census([record["marker"]])
        assert record["marker"] in rows[pid].markers, "a detached child retains the task marker"
        write_task_result(tmp_path, task_id, "completed", result="done")
        ps.wait_for_cleanups(15)
        assert _gone(pid)
        assert ps.task_scope_record(tmp_path, task_id) is None
    finally:
        _end_pids(pid)
        ps.wait_for_cleanups(15)


@pytest.mark.skipif(os.name == "nt", reason="the identity reader is POSIX-only")
def test_known_only_recovery_does_not_read_a_whole_environment_census(tmp_path, monkeypatch):
    from ouroboros.utils import atomic_write_json

    pid, birth = 424242, "Sat Oct 10 20:10:23 2026"
    path = ps.panic_sweep_pending_path(tmp_path)
    atomic_write_json(path, {"known_members": [{"pid": pid, "birth": birth}], "excluded": []})
    monkeypatch.setattr(ps, "census", lambda *args: pytest.fail("known-only recovery must use addressed identities"))
    monkeypatch.setattr(ps, "read_identities", lambda pids: {p: ps._Identity("absent") for p in pids})
    outcome = ps.complete_pending_panic_sweep(tmp_path)
    assert outcome["scope"] == "known_members" and outcome["pending_retired"] is True
    assert outcome["captured_gone"] == [pid]
