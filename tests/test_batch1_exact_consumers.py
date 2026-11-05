"""Narrow counterexamples from the frozen Batch1 review; isolated consumers only."""
from __future__ import annotations

import os
import pathlib
import sys
import time
from types import SimpleNamespace

import pytest

from supervisor import state, state_initialization
from tests import test_schedule_occurrence as schedule_fixtures
from tests import test_state_authority as state_fixtures

root, _prior, _write = state_fixtures.root, state_fixtures._prior, state_fixtures._write
q, _rows = schedule_fixtures.q, schedule_fixtures._rows



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
