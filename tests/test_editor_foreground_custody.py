"""File editors are foreground mutators: the loop joins a writer past its soft timeout.

Owner choice (2026-10-10): editor calls stay sequential and an admitted writer keeps
foreground ownership until it settles or its worker is confirmed dead. Exercised here:
the real loop on a held editor (call order, and a negative control without the join);
owner Pause through the real fence while an editor runs; and three ways a hung editor
worker ends, each against a spawned worker process in disposable roots: Stop now
through cancel custody, the task deadline through the real timeout rail and reaper,
and Panic through ``server_control.execute_panic_stop`` with its host-port, daemon
and persistence side channels stubbed and its companion owner a test-only one.
These are fixtures, not a native install.
"""
import json
import multiprocessing
import pathlib
import queue as stdqueue
import threading
import time

import pytest

from tests.test_panic_owned_requests import owners  # noqa: F401 - restores every Panic owner latch

pytestmark = pytest.mark.serial


def _call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _held_edit_text(release, entered):
    from ouroboros.tools.core import _edit_text

    def held(ctx, _resolved_binding=None, **kwargs):
        entered.set()
        assert release.wait(10)
        return _edit_text(ctx, _resolved_binding=_resolved_binding, **kwargs)
    return held


@pytest.mark.parametrize("foreground", [True, False])
def test_an_editor_past_its_soft_timeout_settles_before_the_next_call(tmp_path, monkeypatch, foreground):
    from ouroboros import loop_tool_execution
    from ouroboros.loop_tool_execution import StatefulToolExecutor, handle_tool_calls
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo, data = tmp_path / "repo", tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    repo.mkdir()
    target = repo / "a.txt"
    target.write_text("one\n")
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id="soft-timeout")
    frames = []
    ctx.event_queue = type("Frames", (), {"put_nowait": staticmethod(frames.append)})()
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ctx)
    release, entered = threading.Event(), threading.Event()
    registry.override_handler("edit_text", _held_edit_text(release, entered))
    monkeypatch.setattr(loop_tool_execution, "_get_tool_timeout",
                        lambda _tools, name, _args: 0.05 if name == "edit_text" else 30)
    if not foreground:  # negative control: the class this membership exists to prevent
        monkeypatch.setattr(loop_tool_execution, "FOREGROUND_MUTATIVE_TOOLS", frozenset({"submit_skill_to_hub"}))
    calls = [_call("first", "edit_text", {"path": "a.txt", "old_str": "one", "new_str": "ONE"}),
             _call("second", "edit_batch", {"edits": [{"path": "a.txt", "old_str": "ONE", "new_str": "TWO"}]})]
    if foreground:
        threading.Timer(0.4, release.set).start()  # well past the 0.05 s soft timeout
    messages, stateful = [], StatefulToolExecutor()
    try:
        handle_tool_calls(calls, registry, data / "logs", ctx.task_id, stateful, messages,
                          {"tool_calls": []}, lambda _text: None)
    finally:
        release.set()
        stateful.shutdown()
    first, second = (row["content"] for row in messages)
    payloads = [frame.get("data") or {} for frame in frames]
    if foreground:
        started = next(p for p in payloads if p.get("type") == "tool_call_started" and p.get("tool") == "edit_text")
        assert started["terminal_wait"] is True and started["timeout_sec"] == 0.05
        assert any(p.get("type") == "tool_call_late" for p in payloads)
        assert "TOOL_TIMEOUT" not in first and second.startswith("✅"), (first, second)
        assert target.read_text() == "TWO\n"
    else:
        deadline = time.monotonic() + 10
        while target.read_text() == "one\n" and time.monotonic() < deadline:
            time.sleep(0.02)
        # The abandoned writer lands after the next call already judged stale bytes.
        assert "TOOL_TIMEOUT" in first and "EDIT_BATCH_ERROR" in second, (first, second)
        assert target.read_text() == "ONE\n"


def test_pause_during_a_running_editor_keeps_custody_and_refuses_the_next_launch(tmp_path, monkeypatch):
    from ouroboros.loop_tool_execution import _execute_with_timeout
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import load_task_result
    from supervisor.owner_pause_control import refresh_owner_pause_tree, request_owner_pause
    from tests.test_batch4_producer_custody import _registry

    registry, _queue, workers = _registry(tmp_path, monkeypatch)
    workspace = registry._ctx.workspace_root
    target = workspace / "a.txt"
    target.write_text("one\n")
    release, entered = threading.Event(), threading.Event()
    registry.override_handler("edit_text", _held_edit_text(release, entered))
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    results = []
    first = _call("first", "edit_text", {"path": "a.txt", "old_str": "one", "new_str": "ONE"})
    thread = threading.Thread(target=lambda: results.append(
        _execute_with_timeout(registry, first, logs, timeout_sec=0.05, task_id="root")))
    thread.start()
    try:
        assert entered.wait(10)
        assert request_owner_pause("root", request_id="editor-in-flight")["ok"]
        refresh_owner_pause_tree("root")
        time.sleep(0.2)  # past the soft timeout: the loop still owns the writer
        claims = load_task_result(tmp_path, "root").get("launch_handoffs") or {}
        assert thread.is_alive() and claims and read_fence(tmp_path, "root")["state"] == "requested"
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and "TOOL_TIMEOUT" not in results[0]["result"], results
    assert target.read_text() == "ONE\n"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    second = _call("second", "edit_text", {"path": "a.txt", "old_str": "ONE", "new_str": "TWO"})
    refused = _execute_with_timeout(registry, second, logs, timeout_sec=5, task_id="root")
    assert "NOT STARTED" in refused["result"] and target.read_text() == "ONE\n", refused
    workers.RUNNING.clear()
    refresh_owner_pause_tree("root")
    assert read_fence(tmp_path, "root")["state"] == "paused"


def _hung_editor_worker(root):
    """A worker whose foreground edit_text hangs inside the write itself."""
    from ouroboros import utils
    from ouroboros.loop_tool_execution import _execute_with_timeout
    from ouroboros.model_wait import task_model_wait_scope
    from ouroboros.tools.registry import ToolRegistry

    real_write = utils.write_bytes_atomic

    def hung(path, data, **kwargs):
        if pathlib.Path(path).name == "target.txt":
            (root / "inside").touch()
            time.sleep(120)
        return real_write(path, data, **kwargs)

    utils.write_bytes_atomic = hung
    registry = ToolRegistry(repo_dir=root / "repo", drive_root=root)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    call = _call("hung", "edit_text", {"path": "target.txt", "old_str": "before", "new_str": "after"})
    with task_model_wait_scope(task={"id": "root", "_attempt": 3}, drive_root=root,
                               event_queue=None, worker_slot_held=True):
        _execute_with_timeout(registry, call, root / "logs", timeout_sec=0.05, task_id="root")
    (root / "returned").touch()


@pytest.fixture
def hung_editor(tmp_path, monkeypatch):
    """A RUNNING root whose spawned worker is joined on an edit hung inside its write."""
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import worker_pool_lifecycle
    from tests.test_batch4_producer_custody import _registry

    _unused, queue, workers = _registry(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")  # inherited by the spawned worker
    (tmp_path / "repo").mkdir()
    (tmp_path / "logs").mkdir(exist_ok=True)
    target = tmp_path / "repo" / "target.txt"
    target.write_text("before\n")
    write_task_result(tmp_path, "root", "running", root_task_id="root", chat_id=0)
    workers.RUNNING["root"].update(attempt=3, worker_id=0)
    workers.RUNNING["root"]["task"]["_attempt"] = 3
    child = multiprocessing.get_context("spawn").Process(target=_hung_editor_worker, args=(tmp_path,))
    child.start()
    try:
        slot = workers.Worker(0, child, stdqueue.Queue(), busy_task_id="root")
        monkeypatch.setattr(workers, "WORKERS", {0: slot})
        worker_pool_lifecycle._record_worker_processes()
        end = time.monotonic() + 30
        while not (tmp_path / "inside").exists() and child.is_alive() and time.monotonic() < end:
            time.sleep(0.02)
        assert (tmp_path / "inside").exists()
        time.sleep(0.3)  # several soft timeouts later the loop still joins the writer
        assert child.is_alive() and not (tmp_path / "returned").exists()
        from ouroboros.tool_call_log import replay_evidence
        evidence = replay_evidence(tmp_path, "root")
        [started] = [fact for fact in evidence["observations"] if fact["fact"] == "started"]
        assert started["tool"] == "edit_text" and started["status"] == "unknown"
        assert not load_task_result(tmp_path, "root").get("launch_handoffs")
        assert workers.RUNNING["root"]["worker_id"] == slot.wid
        yield type("HungEditor", (), {"root": tmp_path, "queue": queue, "workers": workers, "child": child,
                                      "slot": slot, "call_key": started["key"], "target": target})
    finally:
        if child.is_alive():
            child.kill()
        child.join(5)
        child.close()


def _assert_unknown_and_never_replayed(root, call_key):
    from ouroboros.task_results import load_task_result
    from ouroboros.tool_call_log import replay_evidence

    row = load_task_result(root, "root")
    assert not row.get("launch_handoffs"), row
    facts = [fact for fact in replay_evidence(root, "root")["observations"] if fact["key"] == call_key]
    assert [(fact["fact"], fact["status"]) for fact in facts] == [("started", "unknown")]
    assert not (root / "returned").exists() and not row.get("retry_task_id")
    return row


def test_stop_now_ends_a_hung_editor_only_by_confirmed_death(hung_editor, monkeypatch):
    from ouroboros.cancel_intents import request_cancel

    h = hung_editor
    monkeypatch.setattr(h.workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    monkeypatch.setattr(h.workers, "respawn_worker", lambda *_a, **_kw: False)
    request_cancel(h.root, "root", source="owner", reason="Owner Stop")
    assert h.queue.cancel_task_custody("root", deliver=False) == h.queue.CANCEL_CANCELLED
    h.child.join(10)
    assert not h.child.is_alive() and not (h.root / "returned").exists()
    assert h.target.read_text() == "before\n"
    _assert_unknown_and_never_replayed(h.root, h.call_key)


def test_deadline_rail_reaps_a_hung_editor_without_false_completion(hung_editor, monkeypatch):
    from supervisor import task_reaper

    h = hung_editor
    jobs = stdqueue.Queue()
    monkeypatch.setattr(h.queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(h.queue, "_reap_queue", jobs)
    monkeypatch.setattr(h.queue, "persist_queue_snapshot", lambda **_kw: None)
    monkeypatch.setattr(h.workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    monkeypatch.setattr(h.workers, "respawn_worker", lambda *_a, **_kw: False)
    monkeypatch.setattr("supervisor.cancel_publication._audit_delegated_runs_on_kill",
                        lambda *_a, **_kw: {"unreconciled": []})
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs", lambda *_a, **_kw: None)
    monkeypatch.setattr(h.workers, "get_event_q", lambda: type("Frames", (), {"put": staticmethod(lambda _f: None)})())
    meta = h.queue.RUNNING["root"]
    meta["task"]["deadline_at"] = "2000-01-01T00:00:00Z"
    meta["started_at"] = meta["last_heartbeat_at"] = time.time() - 1000
    h.queue.enforce_task_timeouts()  # first tick: the real rail only arms finalization
    assert "root" in h.queue.RUNNING and meta.get("finalization_requested_at") and jobs.empty()
    time.sleep(0.2)  # the joined loop cannot read that control; the writer stays owned
    assert h.child.is_alive() and h.slot.busy_task_id == "root"
    assert "root" in h.queue.RUNNING and not (h.root / "returned").exists()
    # Elapse the unchanged grace by back-dating the armed episode (no grace tightening).
    meta["finalization_requested_at"] -= h.queue.FINALIZATION_GRACE_SEC + 1
    h.queue.enforce_task_timeouts()
    job = jobs.get_nowait()
    assert job["deadline_reached"] and "will_retry" not in job and "root" not in h.queue.RUNNING
    task_reaper.reap_timed_out_task(job)
    h.child.join(10)
    assert not h.child.is_alive() and not (h.root / "returned").exists()
    assert h.target.read_text() == "before\n"
    row = _assert_unknown_and_never_replayed(h.root, h.call_key)
    assert row["status"] == "failed" and row["reason_code"] == job["terminal_reason"], row


@pytest.mark.usefixtures("owners")
def test_panic_does_not_wait_for_a_hung_editor_and_completes_nothing(hung_editor, monkeypatch):
    import concurrent.futures
    import logging

    from ouroboros import claudexor_daemon, extension_companion, local_model, server_control, startup_historical_audit
    from ouroboros import platform_layer as platform

    h = hung_editor
    # Panic latches process-wide owners; `owners` restores the command/executor/service
    # latches and this run stops a throwaway audit, so later tests keep their admission.
    monkeypatch.setattr(startup_historical_audit, "audit", startup_historical_audit.HistoricalAudit())
    # Panic signals whatever companion supervisor is installed globally. An owner left
    # installed earlier (stood in here) is swapped for a test-only one and comes back untouched.
    earlier = extension_companion.CompanionSupervisor(h.root / "earlier-companions")
    monkeypatch.setattr(extension_companion, "_GLOBAL_SUPERVISOR", earlier)
    isolated = extension_companion.CompanionSupervisor(h.root / "panic-companions")
    scoped = pytest.MonkeyPatch()  # undone inside the test so the restore itself is asserted
    scoped.setattr(extension_companion, "_GLOBAL_SUPERVISOR", isolated)
    ports = []
    monkeypatch.setattr(platform, "kill_process_on_port", ports.append)  # never the live host's ports
    monkeypatch.setattr(claudexor_daemon, "get_owned_daemon",
                        lambda create=True: None if not create else type(
                            "NoDaemon", (), {"stop_outcome": staticmethod(lambda: {"stopped": True})})())
    monkeypatch.setattr(local_model, "get_manager", lambda create=True: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [h.child])  # only this test's worker
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda _root: None)
    monkeypatch.setattr(server_control.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        started = time.monotonic()
        future = pool.submit(server_control.execute_panic_stop, None, lambda **_kw: None,
                             data_dir=h.root, panic_exit_code=99, log=logging.getLogger("panic-editor-test"))
        with pytest.raises(SystemExit):
            future.result(timeout=10)
        assert time.monotonic() - started < 10  # bounded by Panic's own request phases, not the writer
    finally:
        pool.shutdown(wait=True)
        # Panic never joins its settlement threads before the (stubbed) hard exit; one
        # still running could reach a process-wide owner after it is restored.
        for thread in [t for t in threading.enumerate() if t.name.startswith("panic-")]:
            thread.join(10)
        scoped.undo()
    h.child.join(10)
    assert not h.child.is_alive() and not (h.root / "returned").exists()
    assert h.target.read_text() == "before\n"
    row = _assert_unknown_and_never_replayed(h.root, h.call_key)
    # Panic writes no terminal state; the real call log remains an unknown,
    # never a fabricated success or an automatic replay on the next boot.
    assert row["status"] == "running", row
    assert ports == []  # Panic sweeps no port: the exiting server's own listeners close with it
    assert isolated._panic_requested  # the companion phase ran, against the test-only owner
    assert extension_companion._GLOBAL_SUPERVISOR is earlier and not earlier._panic_requested
