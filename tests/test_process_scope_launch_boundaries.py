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


@pytest.mark.parametrize(("error_type", "kind"), [
    pytest.param(TimeoutError, "shell", id='TimeoutError-shell'),
    pytest.param(TimeoutError, "skill", id='TimeoutError-skill'),
    pytest.param(PermissionError, "executor", id='PermissionError-executor'),
])
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
        return (shell_process.contained_popen(real_popen, *args, **kwargs)
                if kind != "executor" and (kwargs.get("start_new_session") or os.name == "nt")
                else real_popen(*args, **kwargs))

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
        proc = (shell_process.contained_popen(real_popen, *args, **kwargs)
                if (kwargs.get("start_new_session") or os.name == "nt") else real_popen(*args, **kwargs))
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
    from tests.test_tracked_tree_containment import fake_tree_read
    fake_tree_read(monkeypatch, child, remains=False)
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


# ------------------------------------------------------------------ memory-gated foreground commands (#3792)

class _FakeLease:
    """A leaf stand-in: ``place`` is where the real one moves the shim; ``finish`` its typed outcome."""

    def __init__(self, on_place=None, facts=None):
        self.placed, self.finished, self.on_place, self.facts = [], [], on_place, facts

    def place(self, pid):
        self.placed.append(pid)
        if self.on_place is not None:
            self.on_place(pid)

    def watch(self, stop=None):
        return stop or (lambda: None)

    def finish(self, pid):
        self.finished.append(pid)
        return self.facts


def _gated(monkeypatch, lease):
    from ouroboros import command_memory

    monkeypatch.setattr(command_memory, "begin", lambda *_: lease)
    return command_memory


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
def test_gated_shell_command_runs_after_placement_and_keeps_its_own_argv(tmp_path, monkeypatch):
    lease = _FakeLease(facts={"victim_in_leaf": True, "leaf_local_oom": 1, "commands_local_oom": 0, "unknown": False})
    _gated(monkeypatch, lease)
    command = ["/bin/sh", "-c", "echo gated"]
    res = shell_process._tracked_subprocess_run(command, cwd=str(tmp_path), stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, text=True, timeout=10, memory=True)
    assert (res.args, res.returncode, res.stdout) == (command, 0, "gated\n")  # the shim never shows
    assert len(lease.placed) == 1 and lease.finished == lease.placed
    assert res.memory_oom["victim_in_leaf"] is True


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
def test_memory_is_off_by_default_and_unsupported_layout_takes_the_old_path(tmp_path, monkeypatch):
    from ouroboros import command_memory

    monkeypatch.setattr(command_memory, "begin", lambda *_: pytest.fail("memory is opt-in per call"))
    res = shell_process._tracked_subprocess_run(["/bin/sh", "-c", "echo old"], cwd=str(tmp_path),
                                                stdout=subprocess.PIPE, text=True, timeout=10)
    assert res.stdout == "old\n" and not hasattr(res, "memory_oom")
    monkeypatch.setattr(command_memory, "begin", lambda *_: None)  # probe unavailable
    res = shell_process._tracked_subprocess_run(["/bin/sh", "-c", "echo old"], cwd=str(tmp_path),
                                                stdout=subprocess.PIPE, text=True, timeout=10, memory=True)
    assert res.stdout == "old\n"


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("failure", ["move", "stop"])
def test_a_failed_move_or_a_stop_before_the_gate_never_runs_the_argv(tmp_path, monkeypatch, failure):
    from ouroboros import command_memory

    ran = tmp_path / "ran"

    def on_place(_pid):
        if failure == "move":
            raise command_memory.CommandMemoryRefused("move_failed", "no group")
        monkeypatch.setattr(shell_process, "_panic_requested", True)  # Stop lands between placement and release

    lease = _FakeLease(on_place)
    _gated(monkeypatch, lease)
    expected = command_memory.CommandMemoryRefused if failure == "move" else shell_process.CommandAdmissionClosed
    from ouroboros import tool_custody
    from ouroboros.owned_shutdown import executor_record_paths

    discharged = []
    monkeypatch.setattr(tool_custody, "external_handoff", lambda kind: lambda: discharged.append(kind))

    with _bound(tmp_path, "t-refused"), pytest.raises(expected) as caught:
        shell_process._tracked_subprocess_run(["/usr/bin/touch", str(ran)], cwd=str(tmp_path),
                                              stdout=subprocess.PIPE, timeout=10, memory=True)
    assert caught.value.process_not_started is True and not ran.exists() and lease.finished
    assert not executor_record_paths(tmp_path, "foreground")  # a refused gate leaves no receipt behind
    assert discharged  # the exited shim's external-handoff duty is confirmed, not left owed


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
def test_gated_foreground_receipt_carries_its_launch_identity(tmp_path, monkeypatch):
    from ouroboros.process_census import is_launch_marker

    seen = {}
    real_register = we._register_process

    def register(drive_root, payload, **options):
        seen.update(payload, options=options)
        return real_register(drive_root, payload, **options)

    monkeypatch.setattr(we, "_register_process", register)
    _gated(monkeypatch, _FakeLease())
    with _bound(tmp_path, "t-gated"):
        shell_process._tracked_subprocess_run(["/bin/sh", "-c", "true"], cwd=str(tmp_path), stdout=subprocess.PIPE,
                                              timeout=10, memory=True)
    assert seen["record_type"] == "foreground" and is_launch_marker(seen["scope_marker"]) and seen["host_birth"]
    assert seen["options"] == {"already_started": False}  # a held argv: a failed write is not a started command's


def test_foreground_launch_record_is_a_launch_row_with_its_own_purpose():
    from ouroboros import process_custody as pc
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, mint_marker

    marker = mint_marker(LAUNCH_SCOPE_PREFIX)
    record = {"record_type": "foreground", "executor_type": "local", "id": "foreground", "host_pid": 4242,
              "scope_marker": marker, "host_birth": "b", "task_id": "t1"}
    assert we._recorded_launch(record) or os.name == "nt"
    row = we._launch_row(record)
    assert row["purpose"] == "foreground_command:foreground" and row["scope"] == "task" and row["owner_task"] == "t1"
    assert pc._owned_service_row(row) or os.name == "nt"
    service = we._launch_row({**record, "record_type": "service", "name": "web"})
    assert service["purpose"] == "workspace_service:web"  # services keep theirs
    from ouroboros.launch_custody import _KEPT_PURPOSES

    assert "foreground_command:" not in _KEPT_PURPOSES  # a command row is never retained as a service


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
def test_gated_local_executor_publishes_oom_facts_and_refuses_before_argv(tmp_path, monkeypatch):
    from ouroboros import command_memory

    ref = we.ExecutorRef(kind="local", executor_id="local", network="host", mappings=())
    lease = _FakeLease(facts={"victim_in_leaf": True, "leaf_local_oom": 1, "commands_local_oom": 0, "unknown": False})
    _gated(monkeypatch, lease)
    res = we._execute_local(ref, ["/bin/sh", "-c", "echo gated"], tmp_path, 10, drive_root=tmp_path, memory=True)
    assert res.stdout == "gated\n" and res.memory_oom["victim_in_leaf"] is True and lease.finished
    ran = tmp_path / "ran"
    refusing = _FakeLease(lambda _pid: (_ for _ in ()).throw(command_memory.CommandMemoryRefused("move_failed", "x")))
    _gated(monkeypatch, refusing)
    with pytest.raises(command_memory.CommandMemoryRefused):
        we._execute_local(ref, ["/usr/bin/touch", str(ran)], tmp_path, 10, drive_root=tmp_path, memory=True)
    assert not ran.exists() and not we._FOREGROUND


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("body", ["#!/bin/sh\necho up > \"$1\"\nsleep 60\n", "#!/bin/sh\nx=1\necho up > \"$1\"\nsleep 60\necho $x\n"],
                         ids=["shebang", "multiline"])
@pytest.mark.parametrize("wait", [True, False], ids=["stop", "panic_settle"])
def test_a_stop_ends_a_gated_command_whose_argv_changed_after_exec(tmp_path, monkeypatch, body, wait):
    """The record was registered while the pid was still the gate shim; after exec its command differs, and the
    launch identity (marker + birth), not the recorded command hash, is what the measured stop trusts."""
    import threading
    import time

    from ouroboros.owned_shutdown import executor_record_paths

    script = tmp_path / "job.sh"
    script.write_text(body)
    script.chmod(0o755)
    ready = tmp_path / "ready"  # written by the script itself: the shim has exec'd it
    _gated(monkeypatch, _FakeLease())
    outcome = {}

    def run():
        with _bound(tmp_path, "t-stop"):
            outcome["res"] = shell_process._tracked_subprocess_run(
                [str(script), str(ready)], cwd=str(tmp_path), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, memory=True)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not (ready.exists() and executor_record_paths(tmp_path, "foreground")):
            time.sleep(0.05)
        assert ready.exists()
        assert executor_record_paths(tmp_path, "foreground")
        stopped = we.kill_all_foreground(tmp_path, wait=wait)  # wait=False: Panic's settle, which reads no census
    finally:
        worker.join(60)  # the command's own timeout: a slow run cannot leak its thread past the test
    assert not worker.is_alive() and stopped and stopped[0]["cleanup_dispatched"] is True
    assert outcome["res"].returncode != 0


def _entry(entry, tmp_path, command):
    ref = we.ExecutorRef(kind="local", executor_id="local", network="host", mappings=())
    if entry == "tracked":
        return shell_process._tracked_subprocess_run(command, cwd=str(tmp_path), stdout=subprocess.PIPE, timeout=10, memory=True)
    return we._execute_local(ref, command, tmp_path, 10, drive_root=tmp_path, memory=True)


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
def test_gate_refusal_keeps_measured_oom_without_claiming_the_argv_ran(tmp_path, monkeypatch, entry):
    import time

    from ouroboros import command_memory
    from ouroboros.tools.process_facts import consume_last_process_facts

    oom = {"victim_in_leaf": True, "leaf_local_oom": 1, "commands_local_oom": 0, "unknown": False}
    lease = _FakeLease(facts=oom)
    _gated(monkeypatch, lease)
    monkeypatch.setattr(command_memory, "stage_gate", lambda *_: None)
    monkeypatch.setattr(command_memory, "release_gate", lambda *_: None)
    monkeypatch.setattr(command_memory, "exec_verdict", lambda *_: command_memory.GATE_ENDED)
    ran = tmp_path / "ran"
    with _bound(tmp_path, "t-preexec-oom"), pytest.raises(command_memory.CommandMemoryRefused) as refused:
        _entry(entry, tmp_path, ["/usr/bin/touch", str(ran)])
    assert refused.value.process_not_started and refused.value.code == "gate_ended" and not ran.exists()
    assert refused.value.memory_oom == oom and lease.finished == lease.placed
    if entry == "tracked":
        shell_process._publish_unfinished_process_facts(None, time.monotonic(), spawn_error=refused.value)
    else:
        shell_process.start_refused_result(None, refused.value, tool="verify_and_record", where=str(tmp_path),
                                          started_ts=time.monotonic())
    facts = consume_last_process_facts()
    assert facts["pre_exec_failure"] == "CommandMemoryRefused" and facts["memory_oom"] == oom and "exit_code" not in facts


@pytest.mark.parametrize("isolating", [False, True])
def test_public_local_executor_deadline_includes_candidate_setup(tmp_path, monkeypatch, isolating):
    import time
    from types import SimpleNamespace

    from ouroboros import body_candidate, command_memory

    clock = [time.monotonic()]
    monkeypatch.setattr(we.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(command_memory, "_now", lambda: clock[0])

    def prepare(*args, **kwargs):
        clock[0] += 2
        return {}

    monkeypatch.setattr(body_candidate, "executor_environment", prepare)
    monkeypatch.setattr(we, "map_host_path", lambda _, path: path)
    seen = []

    def begin(deadline):
        seen.append(deadline)
        if isolating:
            command_memory.check_deadline(deadline)

    monkeypatch.setattr(command_memory, "begin", begin)
    monkeypatch.setattr(we.subprocess, "Popen", lambda *a, **k: pytest.fail("argv spawned after its deadline"))
    ref = we.ExecutorRef(kind="local", executor_id="local", network="host", mappings=())
    with pytest.raises(command_memory.CommandMemoryRefused) as refused:
        we.execute(SimpleNamespace(drive_root=None), ["not-run"], tmp_path, 1, executor=ref, memory=True)
    assert refused.value.code == "deadline" and seen == [clock[0] - 1]


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
@pytest.mark.parametrize("failure", ["record_write", "index_publication"])
def test_unwritable_custody_keeps_the_command_in_its_leaf_with_the_old_host_unconfirmed_fact(tmp_path, monkeypatch, entry, failure):
    """The gated registration fails (full disk): the placed leaf is kept and the command runs in it, once, with the usual
    ``host_unconfirmed`` latch of a failed write; the external-handoff duty stays owed until the command ends, and nothing is
    refused, so ``rm`` and ``du`` still work on a full disk."""
    from ouroboros import command_memory, owned_shutdown, tool_custody

    ran = tmp_path / "ran"
    lease = _FakeLease()
    _gated(monkeypatch, lease)
    latched, discharged, released = [], [], []
    monkeypatch.setattr(tool_custody, "retain_unconfirmed_host_operation", lambda reason: latched.append(reason))
    monkeypatch.setattr(tool_custody, "external_handoff", lambda kind: lambda: discharged.append(kind))
    real_release = command_memory.release_gate
    monkeypatch.setattr(command_memory, "release_gate", lambda proc, env=None: released.append(list(discharged)) or real_release(proc, env))
    if failure == "record_write":
        monkeypatch.setattr(we, "atomic_write_json", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    else:
        monkeypatch.setattr(owned_shutdown, "record_executor_process", lambda *a, **k: False)
    with _bound(tmp_path, "t-nocustody"):
        res = _entry(entry, tmp_path, ["/usr/bin/touch", str(ran)])
    assert res.returncode == 0 and ran.exists()  # the argv ran once, in the leaf
    assert len(lease.placed) == 1 and lease.finished == lease.placed  # the one placed leaf, finished when the command ended
    assert released == [[]] and len(discharged) == 1  # the duty was owed at the release and discharged only at the end
    assert latched == ["executor_custody_write_failed"]  # exactly one latch, the ungated path's own fact
    assert not we._FOREGROUND


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
def test_an_exec_verdict_that_never_arrives_leaves_the_released_command_running_with_an_unconfirmed_fact(tmp_path, monkeypatch, entry):
    """Unknown is not failed: the released command is settled by the ordinary wait (it ran once, its exit code is its own),
    and ``host_unconfirmed`` says its start was never confirmed: not a refused 'never started' check."""
    from ouroboros import command_memory, tool_custody

    ran = tmp_path / "ran"
    lease = _FakeLease()
    _gated(monkeypatch, lease)
    latched = []
    monkeypatch.setattr(tool_custody, "retain_unconfirmed_host_operation", lambda reason: latched.append(reason))
    monkeypatch.setattr(command_memory, "exec_verdict", lambda *_: command_memory.EXEC_TIMEOUT)
    with _bound(tmp_path, "t-exec-unconfirmed"):
        res = _entry(entry, tmp_path, ["/usr/bin/touch", str(ran)])
    assert res.returncode == 0 and ran.exists()
    assert latched == ["exec_unconfirmed"] and len(lease.placed) == 1 and lease.finished == lease.placed


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
def test_the_ungated_fallback_carries_the_absolute_deadline(tmp_path, monkeypatch, entry):
    """No support in this layout: the old start follows, but the gate's preparation (here a 0.4 s lock wait) still counts
    against the command's bound: a spent bound refuses before any Popen, a partly spent one leaves only the remainder."""
    import time

    from ouroboros import command_memory

    monkeypatch.setattr(command_memory, "begin", lambda *_: time.sleep(0.4))  # returns None: the old path
    waited = []
    real = shell_process.communicate_or_stop
    monkeypatch.setattr(shell_process, "communicate_or_stop", lambda proc, timeout, *a: waited.append(timeout) or real(proc, timeout, *a))
    ref = we.ExecutorRef(kind="local", executor_id="local", network="host", mappings=())

    def run(timeout):
        if entry == "tracked":
            return shell_process._tracked_subprocess_run(["/usr/bin/touch", str(tmp_path / "ran")], cwd=str(tmp_path),
                                                         stdout=subprocess.PIPE, timeout=timeout, memory=True)
        return we._execute_local(ref, ["/usr/bin/touch", str(tmp_path / "ran")], tmp_path, timeout, drive_root=tmp_path, memory=True)

    with _bound(tmp_path, "t-deadline"):
        with pytest.raises(command_memory.CommandMemoryRefused) as refused:
            run(0.2)
        assert refused.value.code == "deadline" and refused.value.process_not_started
        assert not (tmp_path / "ran").exists() and not waited
        assert run(30).returncode == 0
    assert (tmp_path / "ran").exists() and 0 < waited[0] < 30 - 0.3  # only what the preparation left


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
def test_a_bare_program_name_is_exec_d_from_the_path_the_child_would_search(tmp_path, monkeypatch):
    """The gate's helper execs the argv as asked (``$0`` is the argv's own; a relative PATH entry is searched against the
    child's cwd); an exec the kernel refuses is a typed not-started refusal that runs nothing."""
    tool = tmp_path / "bin" / "tool"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\necho \"found $0\"\n")
    tool.chmod(0o755)
    lease = _FakeLease()
    _gated(monkeypatch, lease)
    res = shell_process._tracked_subprocess_run(["tool"], cwd=str(tmp_path), env={"PATH": "bin"}, stdout=subprocess.PIPE, text=True,
                                                timeout=10, memory=True)
    assert res.stdout == "found bin/tool\n" and res.args == ["tool"]  # the kernel's script $0, as Popen's search gives it
    with pytest.raises(FileNotFoundError) as caught:
        shell_process._tracked_subprocess_run(["missing"], cwd=str(tmp_path), env={"PATH": "bin"}, stdout=subprocess.PIPE, timeout=10, memory=True)
    assert caught.value.process_not_started is True and len(lease.placed) == 2  # the helper's exec failed: never a started command


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
def test_a_begun_stop_door_keeps_the_gate_shut(tmp_path, monkeypatch, entry):
    lease = _FakeLease()
    _gated(monkeypatch, lease)
    monkeypatch.setattr(shell_process, "stop_begun", lambda: True)  # this generation's stop door has begun
    with _bound(tmp_path, "t-stopdoor"), pytest.raises(shell_process.CommandAdmissionClosed) as caught:
        _entry(entry, tmp_path, ["/usr/bin/touch", str(tmp_path / "ran")])
    assert caught.value.process_not_started is True and not (tmp_path / "ran").exists() and lease.finished and not we._FOREGROUND


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
@pytest.mark.parametrize("control", ["pause", "store"])
def test_pause_or_the_store_is_not_a_not_started_command_and_keeps_its_typed_result(tmp_path, monkeypatch, entry, control):
    """``OwnerPauseRefused``/``StoreError`` before the release are control exceptions: never stamped ``process_not_started``
    (the registry publishes OWNER_PAUSE_NOT_STARTED / TOOL_ADMISSION_UNAVAILABLE), and the argv never ran."""
    import contextlib

    from ouroboros import owner_pause
    from ouroboros.store import StoreError

    refusal = owner_pause.OwnerPauseRefused("owner_pause") if control == "pause" else StoreError("busy")

    @contextlib.contextmanager
    def refused():
        raise refusal
        yield

    lease = _FakeLease()
    _gated(monkeypatch, lease)
    monkeypatch.setattr(owner_pause, "operation_start", refused)
    with _bound(tmp_path, "t-pause"), pytest.raises(type(refusal)) as caught:
        _entry(entry, tmp_path, ["/usr/bin/touch", str(tmp_path / "ran")])
    assert getattr(caught.value, "process_not_started", None) is None and not (tmp_path / "ran").exists() and lease.finished


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
def test_pause_during_gated_verify_preparation_is_the_typed_blocked_result_not_a_no_effect_refusal(tmp_path, monkeypatch):
    import contextlib

    from ouroboros import owner_pause
    from ouroboros.tools.tool_result import launch_refusal_result
    from ouroboros.tools.verify import _verify_and_record
    from tests.test_process_signal_observability import _verify_ctx

    @contextlib.contextmanager
    def paused():
        raise owner_pause.OwnerPauseRefused("owner_pause")
        yield

    ctx, drive = _verify_ctx(tmp_path)
    _gated(monkeypatch, _FakeLease())
    monkeypatch.setattr(owner_pause, "operation_start", paused)
    with _bound(drive, "t"), pytest.raises(owner_pause.OwnerPauseRefused) as caught:
        _verify_and_record(ctx, contract_kind="explicit_command", check=["sh", "-c", "true"])
    typed = launch_refusal_result(caught.value, not_started=True)  # what the registry publishes for it
    assert typed.code == "OWNER_PAUSE_NOT_STARTED" and typed.status == "blocked"
    assert not list(drive.rglob("verification_receipts.jsonl"))  # and no receipt


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
def test_the_gate_shell_starts_without_the_commands_selected_environment(tmp_path, monkeypatch, entry):
    """A selected variable (a profiling library) is applied to the TARGET after placement and admission, not to the
    shim that waits in the service's group: the shim's own environment is its markers and a default PATH."""
    from ouroboros import command_memory

    captured = []
    real_popen = subprocess.Popen

    class Spy(real_popen):
        def __init__(self, argv, *args, **kwargs):
            if command_memory._ENV_HELPER in list(argv):  # only the gate helper: other helpers start under their own envs
                captured.append(dict(kwargs.get("env") or {}))
            super().__init__(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", Spy)
    monkeypatch.setenv("SELECTED_PROFILER_LIB", "/nonexistent/lib.so")
    lease = _FakeLease()
    _gated(monkeypatch, lease)
    with _bound(tmp_path, "t-env"):
        if entry == "tracked":
            res = shell_process._tracked_subprocess_run(["/bin/sh", "-c", 'printf %s "$SELECTED_PROFILER_LIB"'], cwd=str(tmp_path),
                                                        env={**os.environ, "SELECTED_PROFILER_LIB": "/nonexistent/lib.so"},
                                                        stdout=subprocess.PIPE, text=True, timeout=10, memory=True)
        else:
            res = _entry(entry, tmp_path, ["/bin/sh", "-c", 'printf %s "$SELECTED_PROFILER_LIB"'])
    assert res.stdout == "/nonexistent/lib.so"  # the target has it
    assert len(captured) == 1
    markers = {key for key in captured[0] if key != "PATH"}
    assert markers and all(key.endswith("_SCOPE_" + key.rsplit("_", 1)[1]) for key in markers)  # scope markers only
    assert "SELECTED_PROFILER_LIB" not in captured[0] and captured[0]["PATH"] == os.defpath


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize(("entry", "failing"), [("local_executor", "handoff"), ("tracked", "tracked_start")])
def test_every_failure_after_the_leaf_exists_finishes_its_lease(tmp_path, monkeypatch, entry, failing):
    """Both gated paths hold the same lease lifetime: a failure while the invocation prepares (its external duty cannot be
    taken, or the stop door refuses the tracked start) leaves no empty leaf behind."""
    from ouroboros import tool_custody

    lease = _FakeLease()
    _gated(monkeypatch, lease)
    if failing == "handoff":
        monkeypatch.setattr(tool_custody, "external_handoff", lambda kind: (_ for _ in ()).throw(RuntimeError("invocation closed")))
        expected = RuntimeError
    else:
        monkeypatch.setattr(shell_process, "_panic_requested", True)
        expected = shell_process.CommandAdmissionClosed
    with _bound(tmp_path, "t-lease"), pytest.raises(expected):
        _entry(entry, tmp_path, ["/usr/bin/touch", str(tmp_path / "ran")])
    assert lease.finished and not (tmp_path / "ran").exists()


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
def test_a_missing_executable_keeps_the_typed_not_started_refusal_behind_the_gate(tmp_path, monkeypatch, entry):
    ref = we.ExecutorRef(kind="local", executor_id="local", network="host", mappings=())
    command = ["/nonexistent/3792-prog"]
    lease = _FakeLease()
    _gated(monkeypatch, lease)
    from ouroboros import tool_custody

    discharged = []
    monkeypatch.setattr(tool_custody, "external_handoff", lambda kind: lambda: discharged.append(kind))
    with _bound(tmp_path, "t-missing"), pytest.raises(FileNotFoundError) as caught:
        if entry == "tracked":
            shell_process._tracked_subprocess_run(command, cwd=str(tmp_path), stdout=subprocess.PIPE, timeout=10, memory=True)
        else:
            we._execute_local(ref, command, tmp_path, 10, drive_root=tmp_path, memory=True)
    assert caught.value.process_not_started is True and lease.finished and lease.placed  # the placed helper's exec failed: never a bare 127
    assert discharged  # no process exists: the external-handoff duty is discharged, not left owed


def _run_entry(entry, tmp_path, command, timeout, stop=None):
    ref = we.ExecutorRef(kind="local", executor_id="local", network="host", mappings=())
    if entry == "tracked":
        return shell_process._tracked_subprocess_run(command, cwd=str(tmp_path), stdout=subprocess.PIPE, timeout=timeout,
                                                     memory=True, stop=stop)
    return we._execute_local(ref, command, tmp_path, timeout, drive_root=tmp_path, memory=True, stop=stop)


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
@pytest.mark.parametrize("gated", [True, False])
def test_the_one_absolute_deadline_is_read_in_the_final_admission_after_all_preparation(tmp_path, monkeypatch, entry, gated):
    """Review witness: an 80 ms bound and 180 ms of preparation after the old early check still ran the command. The bound
    starts at entry and is read inside the final admission: slow placement (gated) or slow maintenance and environment
    preparation (ungated) refuse it before the argv, typed ``deadline``."""
    import time

    from ouroboros import command_memory, process_scope

    ran = tmp_path / "ran"
    if gated:
        _gated(monkeypatch, _FakeLease(on_place=lambda _pid: time.sleep(0.3)))
    else:
        monkeypatch.setattr(command_memory, "begin", lambda *_: None)
        if entry == "tracked":
            monkeypatch.setattr(shell_process, "maintain_tracked_children", lambda: time.sleep(0.3))
        else:
            real = process_scope.task_launch_target
            monkeypatch.setattr(process_scope, "task_launch_target", lambda *a: time.sleep(0.3) or real(*a))
    with _bound(tmp_path, "t-final-deadline"), pytest.raises(command_memory.CommandMemoryRefused) as refused:
        _run_entry(entry, tmp_path, ["/usr/bin/touch", str(ran)], 0.08)
    assert refused.value.code == "deadline" and refused.value.process_not_started and not ran.exists()


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
def test_a_turn_stop_already_effective_at_placement_never_releases_the_gate(tmp_path, monkeypatch, entry):
    """Review witness: a direct-chat turn's Stop now, effective when the helper was placed, was read only by the later wait,
    so the argv ran. It is read before the release: a typed refusal, nothing ran. False or unknown admits."""
    ran = tmp_path / "ran"
    lease = _FakeLease()
    _gated(monkeypatch, lease)
    with _bound(tmp_path, "t-stop-at-placement"), pytest.raises(shell_process.CommandAdmissionClosed) as refused:
        _run_entry(entry, tmp_path, ["/usr/bin/touch", str(ran)], 10, stop=lambda: True)
    assert refused.value.process_not_started and not ran.exists() and lease.finished
    with _bound(tmp_path, "t-stop-unknown"):
        assert _run_entry(entry, tmp_path, ["/usr/bin/touch", str(ran)], 10, stop=lambda: None).returncode == 0
    assert ran.exists()


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
@pytest.mark.parametrize("entry", ["tracked", "local_executor"])
def test_an_oom_that_coincides_with_a_timeout_rides_on_the_timeout(tmp_path, monkeypatch, entry):
    """Review witness: the leaf's OOM count was read at cleanup but kept only on a completed result. A timeout carries it
    too, beside (not merged into) the host's own kill."""
    facts = {"victim_in_leaf": True, "leaf_local_oom": 1, "commands_local_oom": 0, "unknown": False, "pid": 1}
    _gated(monkeypatch, _FakeLease(facts=facts))
    with _bound(tmp_path, "t-oom-timeout"), pytest.raises(subprocess.TimeoutExpired) as ended:
        _run_entry(entry, tmp_path, ["/bin/sh", "-c", "sleep 5"], 0.5)
    assert ended.value.memory_oom["victim_in_leaf"] is True
    from ouroboros.tools.shell_process import _oom_note

    assert "separately, memory: the kernel recorded an out-of-memory kill" in _oom_note(ended.value)
