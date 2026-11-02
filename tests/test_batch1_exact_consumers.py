"""Narrow counterexamples from the frozen Batch1 review; isolated consumers only."""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

import pytest

from supervisor import state, state_initialization
from tests import test_schedule_occurrence as schedule_fixtures
from tests import test_state_authority as state_fixtures
from tests._shared import stop_socket_sharer, wait_test_child_stop

root, _prior, _write = state_fixtures.root, state_fixtures._prior, state_fixtures._write
q, _rows = schedule_fixtures.q, schedule_fixtures._rows

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("writer", ["update", "save", "init"])
def test_missing_controls_survive_all_existing_state_writers(root, writer):
    state.save_state(_prior())
    raw = json.loads(state.STATE_PATH.read_bytes())
    for key in ("owner_external_id", "owner_external_chat_id", "evolution_owner_stopped"):
        raw.pop(key)
    _write(state.STATE_PATH, raw)
    if writer == "update":
        state.update_state(lambda live: live.update(last_owner_message_at="now"))
    elif writer == "save":
        state.save_state({**raw, "owner_external_id": None})
    else:
        state.init_state()
    state.init(root)  # cold read, not a process-local recovery flag
    for key in ("owner_external_id", "owner_external_chat_id", "evolution_owner_stopped"):
        assert state.control_value(state.load_state(), key) == (False, None)
    state.update_state(lambda live: live.update(evolution_owner_stopped=True),
                       confirm=("evolution_owner_stopped",))
    assert state.control_value(state.load_state(), "evolution_owner_stopped") == (True, True)
    assert state.control_value(state.load_state(), "owner_external_id") == (False, None)


@pytest.mark.parametrize("changed", [False, True])
def test_interrupted_legacy_adoption_resumes_only_exact_source(root, monkeypatch, changed):
    original = _write(state.STATE_PATH, _prior())
    real = state.atomic_write_text
    monkeypatch.setattr(state, "atomic_write_text", lambda *_: False)
    assert state.init_state().quality == "unavailable"
    identity = state_initialization.read_witness(root)[1]["initialization_id"]
    assert state.STATE_PATH.read_bytes() == original
    if changed:
        _write(state.STATE_PATH, _prior(session_id="different"))
    monkeypatch.setattr(state, "atomic_write_text", real)
    observed = state.init_state()
    assert (observed.quality == "unavailable") is changed
    assert state_initialization.read_witness(root)[1]["initialization_id"] == identity
    if not changed:
        assert state.load_state()["session_id"] == "old-session"
        assert state.control_value(state.load_state(), "owner_external_id") == (True, 99)


@pytest.mark.parametrize("zero_write", [False, True])
def test_corrupt_primary_is_not_replaced_until_full_copy(root, monkeypatch, zero_write):
    state.save_state(_prior())
    corrupt = b'{"truncated":' + b'x' * 4096
    state.STATE_PATH.write_bytes(corrupt)
    real = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: 0 if zero_write else real(fd, data[:17]))
    if zero_write:
        with pytest.raises(state.StateUnavailable, match="corrupt_primary_unpreserved"):
            state.update_state(lambda st: st.update(message_offset=2))
        assert state.STATE_PATH.read_bytes() == corrupt
    else:
        state.update_state(lambda st: st.update(message_offset=2))
        [copy] = list((root / "state").glob("state.corrupt-*.json"))
        assert copy.read_bytes() == corrupt


def test_context_uses_the_addressed_witness_authority(root):
    from ouroboros.context import _drive_state_section

    state.save_state(_prior())
    witness = state_initialization.witness_path(root)
    raw = json.loads(witness.read_bytes())
    raw["phase"] = "pending"
    _write(witness, raw)
    rendered = _drive_state_section(SimpleNamespace(drive_path=lambda p: root / p))
    assert '"evolution_mode_enabled": {\n  "status": "unknown"' in rendered
    assert "initialization_incomplete" in rendered
    raw["phase"] = "complete"
    _write(witness, raw)
    assert '"evolution_mode_enabled": true' in _drive_state_section(SimpleNamespace(drive_path=lambda p: root / p))


def test_consciousness_recovers_unknown_but_never_overrides_live_stop(root, monkeypatch):
    from ouroboros.consciousness import BackgroundConsciousness

    state.save_state(_prior())
    primary, backup = state.STATE_PATH.read_bytes(), state.STATE_LAST_GOOD_PATH.read_bytes()
    state.STATE_PATH.unlink()
    state.STATE_LAST_GOOD_PATH.unlink()
    clock = BackgroundConsciousness(root, root / "repo", lambda: 7, now=100)
    monkeypatch.setattr(clock, "live_wake", lambda: "")
    assert clock.tick(now=100) == "disabled"
    state.STATE_PATH.write_bytes(primary)
    state.STATE_LAST_GOOD_PATH.write_bytes(backup)
    assert clock.tick(now=100) == "not_due"
    clock.stop()
    assert clock.tick(now=100) == "disabled"


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_real_gateway_producer_admits_without_injected_intent(q, monkeypatch, project, legacy):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import schedules
    from ouroboros.gateway.schedules import api_schedules_upsert
    from ouroboros.projects_registry import create_project

    monkeypatch.setattr(schedules, "request_drive_root", lambda _: q.root)
    if project:
        create_project(q.root, "room", name="Folderless room")
    if legacy:
        from tests.test_schedule_occurrence import _row
        _row(q, "api", project_id="room" if project else "")
    app = Starlette(routes=[Route('/api/schedules', api_schedules_upsert, methods=['POST'])])
    with TestClient(app) as client:
        response = client.post('/api/schedules', json={"id": "api", "trigger": {
            "type": "once", "run_at": "2000-01-01T00:00:00Z"}, "task": {
            "type": "task", "text": "Do the scheduled work", "chat_id": 1,
            **({"project_id": "room"} if project else {})}})
        assert response.status_code == 200, response.text
    q.queue.check_scheduled_tasks()
    if legacy:
        assert not q.pending
        assert _rows(q)["api"]["hold"]["reason"] == "resource_intent_unknown"
        return
    [task] = q.pending
    assert task["metadata"]["resource_intent"]["kind"] == ("room_default" if project else "system_repo")
    assert not _rows(q)["api"].get("hold")


def test_cli_schedule_add_uses_the_gateway_producer(q, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import cli
    from ouroboros.gateway import schedules
    from supervisor import queue_schedules

    monkeypatch.setattr(schedules, "request_drive_root", lambda _: q.root)
    app = Starlette(routes=[Route('/api/schedules', schedules.api_schedules_upsert, methods=['POST'])])
    with TestClient(app) as client:
        # Only HTTP transport is in-process: real parser, CLI body, gateway and store.
        monkeypatch.setattr(cli, '_client', lambda _: SimpleNamespace(
            request=lambda method, path, body: client.request(method, path, json=body).json()))
        assert cli.main(['schedule', 'add', '--name', 'CLI work', '--cron', '* * * * *', 'Do work']) == 0
    [record] = _rows(q).values()
    record['next_run_at'] = '2000-01-01T00:00:00Z'  # advance to the producer's due occurrence
    queue_schedules._write_scheduled_tasks({'tasks': [record]}, q.root)
    q.queue.check_scheduled_tasks()
    [task] = q.pending
    assert task['metadata']['resource_intent'] == {'kind': 'system_repo'}


def test_history_reloads_late_settlement_over_frozen_summary(tmp_path):
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.post_task_synthesis import task_tool_metrics
    from ouroboros.tool_call_log import append_call_row
    from ouroboros.utils import append_jsonl

    metrics = task_tool_metrics({"tool_calls": [{"tool": "run_command", "is_error": True}]})
    append_jsonl(tmp_path / "logs/chat.jsonl", {"type": "task_summary", "direction": "out",
        "role": "assistant", "chat_id": 1, "task_id": "t", "text": "Done", "ts": "2026-09-27T01:00:00Z", **metrics})
    for event in ("tool_call_started", "tool_call_timeout", "tool_call"):
        assert append_call_row({}, tmp_path / "logs", {"type": event, "task_id": "t",
            "invocation_id": "call", "tool": "run_command", "is_error": event == "tool_call_timeout",
            "status": "ok" if event == "tool_call" else "timeout"})["task_log"]
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    summary = next(row for row in json.loads(response.body)["messages"] if row.get("task_id") == "t")
    observations = summary["tool_evidence"]["observations"]
    assert [row["fact"] for row in observations] == ["started", "wait_ended", "settled"]
    assert observations[-1]["status"] == "ok" and summary["tool_errors"] == 1  # frozen historical wait count


def test_history_stamps_replay_evidence_on_a_latest_accounting_wait_checkpoint(tmp_path):
    from ouroboros._usage_wait import _hold
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.tool_call_log import append_call_row
    from ouroboros.utils import append_jsonl

    append_jsonl(tmp_path / "logs/progress.jsonl", {"type": "send_message", "role": "assistant", "chat_id": 1,
        "task_id": "t", "content": "Reading the file", "ts": "2000-01-01T00:00:00Z"})
    for event in ("tool_call_started", "tool_call"):
        assert append_call_row({}, tmp_path / "logs", {"type": event, "task_id": "t",
            "invocation_id": "call", "tool": "read_file", "status": "ok"})["task_log"]
    owner = SimpleNamespace(event_queue=None, task_id="t", drive_root=str(tmp_path), task={"chat_id": 1})
    _hold(owner, "entered", time.monotonic(), "wait-1")  # the upstream producer's progress projection
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    latest = [row for row in json.loads(response.body)["messages"] if row.get("task_id") == "t"][-1]
    assert latest["system_type"] == "task_checkpoint" and latest["is_progress"] is True
    assert [row["fact"] for row in latest["tool_evidence"]["observations"]] == ["started", "settled"]


def test_review_reference_producer_history_and_chat_reconnect_keep_tool_evidence(tmp_path):
    import shutil
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.tool_call_log import append_call_row
    from ouroboros.tools.plan_review_references import _emit_review_reference
    from ouroboros.utils import append_jsonl
    from tests.group_run import run_in_group

    task_id = "turn-a"
    append_jsonl(tmp_path / "logs/progress.jsonl", {"type": "send_message", "role": "assistant",
        "task_id": task_id, "chat_id": 1, "content": "Reading the file", "ts": "2000-01-01T00:00:00Z"})
    for invocation, event in [("before-review", "tool_call_started"), ("before-review", "tool_call_timeout"),
                              ("before-review", "tool_call"), ("next", "tool_call_started")]:
        assert append_call_row({}, tmp_path / "logs", {"type": event, "task_id": task_id,
            "invocation_id": invocation, "tool_call_id": "reused-provider-id", "tool": "read_file",
            "is_error": event == "tool_call_timeout", "status": "ok"})["task_log"]
    owner = SimpleNamespace(drive_root=tmp_path, current_chat_id=1, task_metadata={}, event_queue=None)
    _emit_review_reference(owner, task_id, {"asks": [{"ask_id": "a1"}]}, surface="plan_review", fingerprint="plan-1")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body)
    latest = payload["messages"][-1]
    assert latest["system_type"] == "review_reference" and latest["is_progress"] is True
    assert latest["presentation_owner_task_id"] == task_id
    assert not any(row.get("system_type") == "task_summary" for row in payload["messages"])
    observations = latest["tool_evidence"]["observations"]
    assert [row["fact"] for row in observations] == ["started", "wait_ended", "settled", "started"]
    assert len({row["key"] for row in observations}) == 2 and observations[2]["status"] == "ok"
    history = tmp_path / "history.json"
    history.write_bytes(response.body)
    node = shutil.which("node")
    assert node, "Node is required for the real Chat consumer proof"
    result = run_in_group([node, "--test", "--test-name-pattern=reference carriers",
        "web/tests/chat_activity_block.test.js"], cwd=pathlib.Path(__file__).resolve().parents[1],
        env={**os.environ, "OURO_TEST_TOOL_HISTORY": str(history)}, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_osworld_readers_count_unfinished_calls_once_and_keep_legacy(tmp_path):
    from devtools.benchmarks.osworld import run_cu_bridge_agent as bridge
    from ouroboros.extension_loader import extension_name_prefix

    prefix = extension_name_prefix(bridge.SKILL_NAME)
    root = tmp_path / 'state/headless_tasks/t/data/logs'
    root.mkdir(parents=True)
    rows = [{"type": t, "task_id": "t", "tool": prefix + "screenshot", "invocation_id": "i"}
            for t in ("tool_call_started", "tool_call_timeout")]
    rows += [{"type": "tool_call", "tool": prefix + "remote_exec", "args": {"cmd": "pwd"}}] * 2
    (root / 'tools.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    counts = bridge._collect_budget_counters(tmp_path, {}, "t")
    assert (counts["skill_tool_calls"], counts["screenshots"], counts["remote_exec_calls"]) == (3, 1, 2)
    trace = bridge._gate_tool_trace(tmp_path, "t")
    assert len(trace) == 3 and trace[0]["state"] == "wait_ended"
    assert trace[0]["is_error"] is None and not trace[0]["settled"]


# The command child writes its own pid and parent pid atomically before sleeping.
_COMMAND_CHILD = ('import os,time,pathlib; t=pathlib.Path("child.pid.tmp"); '
                  't.write_text(f"{os.getpid()} {os.getppid()}"); t.replace("child.pid"); time.sleep(60)')
_PUBLISHED, _HELD = "batch1-fixture:command-child-published", "batch1-fixture:command-child-held"


def _is_command_child(popen):
    return isinstance(popen.args, (list, tuple)) and _COMMAND_CHILD in popen.args


def _pooled_test_entry(*args):
    """Real worker_main and registry, with only model/extension setup replaced.

    The exact command's real communicate() first reports on this worker's own out_q
    whether its Popen is in shell_process._active_subprocesses: the published-child ACK.
    """
    import subprocess

    from ouroboros import agent, extension_loader
    from ouroboros.tools import shell_process
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from supervisor.worker_process import worker_main

    communicate = subprocess.Popen.communicate

    def acknowledged_communicate(popen, *a, **kw):
        if _is_command_child(popen):
            try:
                args[2].put((_PUBLISHED, popen.pid, popen in shell_process._active_subprocesses))
            except Exception:
                pass  # no ACK: the parent fails on its own deadline and never requests a stop
        return communicate(popen, *a, **kw)

    subprocess.Popen.communicate = acknowledged_communicate

    class CommandAgent:
        def handle_task(self, task):
            import logging

            from ouroboros import process_custody

            root = pathlib.Path(args[4])
            workspace = root / 'workspace'
            workspace.mkdir(exist_ok=True)
            ctx = ToolContext(repo_dir=root / 'repo', drive_root=root / 'data',
                              workspace_root=workspace, workspace_mode='external', task_id='pooled')
            from ouroboros.task_results import write_task_result
            write_task_result(ctx.drive_root, 'pooled', 'running', root_task_id='pooled')
            registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
            registry.set_context(ctx)
            shell_process._subprocess_lock.acquire()  # cleanup must not gate requests
            handler = logging.StreamHandler()
            handler.acquire()  # persistence/logging must not gate the worker's own exit
            process_custody.log.addHandler(handler)
            process_custody.log.setLevel(logging.WARNING)
            result = registry.execute('run_command', {'cmd': [sys.executable, '-c', _COMMAND_CHILD]})
            (root / 'command-result.txt').write_text(str(result))
            return []

    agent.make_agent = lambda **_: CommandAgent()
    extension_loader.reload_all = lambda *_a, **_k: None
    import ouroboros.safety
    ouroboros.safety.check_safety = lambda *_a, **_k: (True, '')
    worker_main(*args)


def _held_publication_entry(*args):
    """Hold the exact command's real Popen() return, so the child runs before publication."""
    import subprocess

    from ouroboros.tools import shell_process

    release, init = pathlib.Path(args[4]) / 'release-publication', subprocess.Popen.__init__

    def held_init(popen, *a, **kw):
        init(popen, *a, **kw)
        if _is_command_child(popen):
            args[2].put((_HELD, popen.pid, popen in shell_process._active_subprocesses,
                         shell_process._spawning_subprocesses))
            deadline = time.monotonic() + 30  # bounded if the parent never releases
            while not release.exists() and time.monotonic() < deadline:
                time.sleep(.01)

    subprocess.Popen.__init__ = held_init
    _pooled_test_entry(*args)


def _observe_command_child(proc, outgoing, workspace, seen, need, timeout=20.0):
    """Record the child's own marker and the worker's tagged fixture events until `need` is seen."""
    import queue

    from ouroboros import platform_layer as platform

    marker, result, deadline = workspace / 'child.pid', workspace.parent / 'command-result.txt', \
        time.monotonic() + timeout
    while not need <= seen.keys():
        assert time.monotonic() < deadline, ('no published command child', sorted(need - seen.keys()), seen,
                                             result.read_text() if result.exists() else '')
        assert proc.is_alive(), (proc.exitcode, seen)
        if 'pid' not in seen and marker.exists():
            seen['pid'], seen['launcher'] = map(int, marker.read_text().split())
            seen['birth'] = platform.process_start_time(seen['pid'])
        try:
            event = outgoing.get(timeout=.01)
        except queue.Empty:
            continue
        if isinstance(event, tuple) and event[:1] in ((_PUBLISHED,), (_HELD,)):  # skip worker events
            seen['ack' if event[0] == _PUBLISHED else 'held'] = event[1:]
    return seen


def _is_marker_child(pid, seen):
    # Direct child, or the Windows venv python.exe redirector that runs the interpreter as its child.
    return pid == seen['pid'] or (os.name == 'nt' and pid == seen['launcher'])


def _await_published_command_child(proc, outgoing, workspace, seen, timeout=20.0):
    """The child's marker AND its registry ACK: the marker alone can precede publication."""
    _observe_command_child(proc, outgoing, workspace, seen, {'pid', 'ack'}, timeout)
    pid, registered = seen['ack']
    assert registered and _is_marker_child(pid, seen), ('exact command not published', seen)
    return seen['pid']


def test_actual_pooled_worker_requests_separate_session_command_before_owner_exit(tmp_path, monkeypatch):
    import multiprocessing

    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.process_containment import pid_is_zombie
    from supervisor import worker_pool_lifecycle, worker_process

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setattr(worker_process, 'worker_main', _pooled_test_entry)
    ctx = multiprocessing.get_context('spawn')
    incoming, outgoing = ctx.Queue(), ctx.Queue()
    proc = worker_process.spawn_worker_process(ctx, 0, incoming, outgoing, tmp_path, tmp_path)
    seen = {}
    try:
        incoming.put({'id': 'pooled', 'type': 'task'})
        child_pid = _await_published_command_child(proc, outgoing, tmp_path / 'workspace', seen)
        if os.name != 'nt':
            assert os.getpgid(child_pid) == child_pid != os.getpgid(proc.pid)
        started = time.monotonic()
        receipt = worker_pool_lifecycle.kill_worker_tree(proc.pid, panic_process=proc)
        assert receipt['requested'] and receipt['scope'] == 'worker_owners'
        assert time.monotonic() - started < .5
        proc.join(timeout=5)
        assert not proc.is_alive()
        verdict = wait_test_child_stop(child_pid)
        assert verdict["child_stopped"], (receipt, proc.exitcode, seen, verdict)
    finally:
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            child_pid = seen.get('pid', 0)  # the marker's own pid, even when its ACK never came
            if child_pid and pid_is_alive(child_pid) and not pid_is_zombie(child_pid):
                # This test created and continuously observed the child; no unrelated PID search.
                try:
                    os.kill(child_pid, 9)
                except OSError:  # exited after the probe: ESRCH on POSIX, WinError 87/5 on Windows
                    if pid_is_alive(child_pid) and not pid_is_zombie(child_pid):
                        raise  # a refused kill of a still-running child is a real failure
        finally:  # release channels and the socket sharer even when a cleanup step above failed
            proc._ouroboros_stop_socket.close()
            for channel in (incoming, outgoing):
                channel.close()
                channel.cancel_join_thread()
            stop_socket_sharer()


def test_publication_ack_waits_for_the_registry_not_the_child_marker(tmp_path, monkeypatch):
    """The child's own marker can precede publication; only the registry ACK proves it.

    Characterizes, never repairs, the declared unpublished-spawn window: no stop is
    requested here, and an unpublished child is not claimed stoppable.
    """
    import multiprocessing

    from ouroboros import platform_layer as platform
    from ouroboros.process_containment import pid_is_zombie
    from supervisor import worker_process

    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setattr(worker_process, 'worker_main', _held_publication_entry)
    ctx = multiprocessing.get_context('spawn')
    incoming, outgoing = ctx.Queue(), ctx.Queue()
    proc = worker_process.spawn_worker_process(ctx, 0, incoming, outgoing, tmp_path, tmp_path)
    workspace, seen = tmp_path / 'workspace', {}
    try:
        incoming.put({'id': 'pooled', 'type': 'task'})
        _observe_command_child(proc, outgoing, workspace, seen, {'pid', 'held'})
        pid, registered, spawning = seen['held']
        assert _is_marker_child(pid, seen) and platform.pid_is_alive(seen['pid'])
        assert not registered and spawning == 1  # running, marked, yet only a spawn count
        with pytest.raises(AssertionError, match='no published command child'):
            _await_published_command_child(proc, outgoing, workspace, seen, timeout=1)
        assert 'ack' not in seen
        (tmp_path / 'release-publication').touch()
        assert _await_published_command_child(proc, outgoing, workspace, seen) == seen['pid']
        assert seen['ack'][0] == pid
    finally:
        try:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
            child_pid, birth = seen.get('pid', 0), seen.get('birth', '')
            if (birth and platform.pid_is_alive(child_pid) and not pid_is_zombie(child_pid)
                    and platform.process_start_time(child_pid) == birth):
                platform.force_kill_pid(child_pid)  # the marker's own pid AND its birth
            # Identity authorizes the signal, never a claim that cleanup succeeded.
            # Unknown/mismatched birth must still fail if the observed child survives.
            deadline = time.monotonic() + 3
            while (platform.pid_is_alive(child_pid) and not pid_is_zombie(child_pid)
                   and time.monotonic() < deadline):
                time.sleep(.01)
            assert not platform.pid_is_alive(child_pid) or pid_is_zombie(child_pid), seen
        finally:  # a failed death proof must not leak the stop channel, queues or socket sharer
            worker_process.close_worker_stop_channel(proc)
            for channel in (incoming, outgoing):
                channel.close()
                channel.cancel_join_thread()
            stop_socket_sharer()


@pytest.mark.parametrize("platform_name", ["linux", "darwin"])
def test_attached_request_never_turns_held_identity_into_numeric_group(monkeypatch, platform_name):
    from ouroboros import platform_layer as platform
    calls = []
    monkeypatch.setattr(platform, "IS_WINDOWS", False)
    monkeypatch.setattr(platform, "IS_MACOS", platform_name == "darwin")
    # The whole mocked POSIX surface, present on every host (Windows has no getpgid/killpg/SIGKILL).
    monkeypatch.setattr(platform.os, "getpgid", lambda *_: pytest.fail("numeric group lookup loses identity"),
                        raising=False)
    monkeypatch.setattr(platform.os, "kill", lambda *_: pytest.fail("numeric PID cannot signal an attachment"))
    monkeypatch.setattr(platform.os, "killpg", lambda *_: pytest.fail("numeric group cannot signal an attachment"),
                        raising=False)
    monkeypatch.setattr(platform.signal, "SIGKILL", getattr(platform.signal, "SIGKILL", 9), raising=False)
    monkeypatch.setattr(platform.signal, "pidfd_send_signal", lambda *a: calls.append(a), raising=False)
    target = {"pid": 123, "handle": SimpleNamespace(fileno=lambda: 987, close=lambda: None), "pgid": 123}
    receipt = platform.request_process_tree_kill(target)
    assert receipt["requested"] is (platform_name == "linux")
    assert receipt["scope"] == "process"
    assert calls == ([(987, platform.signal.SIGKILL)] if platform_name == "linux" else [])


def test_partial_primary_cannot_register_a_stranger_after_bookkeeping(root, monkeypatch):
    import server
    from supervisor import message_bus
    from tests.test_state_authority import _bridge, _ingress_ctx

    state.save_state(_prior())
    raw = json.loads(state.STATE_PATH.read_bytes())
    raw.pop("owner_external_id")
    raw.pop("owner_external_chat_id")
    _write(state.STATE_PATH, raw)
    state.update_state(lambda live: live.update(message_offset=99))
    replies, panics = [], []
    monkeypatch.setattr(message_bus, "log_chat", lambda *_a, **_k: None)
    monkeypatch.setattr(message_bus, "record_inbound_message", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_execute_panic_stop", lambda *_: panics.append(True))
    server._process_bridge_updates(_bridge('/panic', user=5, chat=5), 0, _ingress_ctx(root, replies, panics))
    assert not panics and any('unknown' in reply for reply in replies)
    assert state.control_value(state.load_state(), 'owner_external_id') == (False, None)
