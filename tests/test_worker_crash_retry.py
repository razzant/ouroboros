"""Regression tests for worker crash retry loop fixes.

Covers:
- Retry limit enforced (attempt > QUEUE_MAX_RETRIES → STATUS_FAILED, no requeue)
- Attempt counter incremented before requeue
- Already-completed task is not requeued after crash
- A signal death takes the same one retry (no crash-storm fence, no signal special case)
- Terminal event emitted when retry limit exhausted
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_worker_crash_state(tmp_path, monkeypatch):
    """Crash history is process-global and must not leak between serial tests."""
    import supervisor.workers as workers

    import queue as stdqueue
    import supervisor.queue as q
    import supervisor.state as state
    import supervisor.task_reaper as reaper

    state.init(tmp_path)
    q.init(tmp_path)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "WORKERS", {})
    # Real retry admission appends to PENDING; isolate both aliases so imported
    # consumers cannot leave their retry in the next test's activity census.
    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(q, "PENDING", workers.PENDING)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(q, "RUNNING", workers.RUNNING)
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *_a, **_k: {
        "cost_accounting_status": "unavailable", "cost_final": False,
    })
    monkeypatch.setattr(workers, "get_event_q", lambda: stdqueue.Queue())
    jobs = stdqueue.Queue()
    monkeypatch.setattr(reaper, "reap_queue", jobs)
    monkeypatch.setattr(q, "_reap_queue", jobs)
    monkeypatch.setattr(q, "_ensure_reaper_started", lambda: None)
    yield
    workers._WORKER_POOL_DISABLED_REASON = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_id="abc123", attempt=1, chat_id=1):
    return {
        "id": task_id,
        "type": "task",
        "chat_id": chat_id,
        "text": "test",
        "_attempt": attempt,
    }


def _make_worker(wid=0, alive=False, busy_task_id="abc123", exitcode=-11):
    proc = MagicMock()
    proc.is_alive.return_value = alive
    proc.exitcode = exitcode
    proc.pid = 12345
    w = MagicMock()
    w.wid = wid
    w.proc = proc
    w.busy_task_id = busy_task_id
    # Real Worker defaults reaping=False; without this the MagicMock auto-attr is truthy
    # and the new crash-detector reaping guard would skip the worker.
    w.reaping = False
    w.readiness_exhausted = False
    w.active_capacity = True
    return w


def _run_health_and_reap():
    """Old policy tests must execute the new asynchronous consumer explicitly."""
    import supervisor.queue as q
    import supervisor.workers as W
    import supervisor.worker_health as health
    q.RUNNING = W.RUNNING
    W.ensure_workers_healthy()
    count = 0
    # These policy fixtures mock task-result reads. Real file preparation and
    # CURRENT-readiness composition are covered by the dedicated cases below.
    with patch("ouroboros.headless.prepare_terminal_task_files", return_value={"error": "", "terminal_source_present": False}), \
         patch("ouroboros.headless.terminal_task_files_ready", return_value=True):
        while not q._reap_queue.empty():
            job = q._reap_queue.get_nowait()
            try:
                if job["kind"] == "confirmed_dead_worker":
                    health.recover_confirmed_dead_worker(job)
                    count += 1
                else:
                    raise AssertionError(f"Unexpected job {job['kind']}")
            finally:
                q._reap_queue.task_done()
    assert count > 0, "policy assertion must not pass without running a dead-worker job"


# ---------------------------------------------------------------------------
# Test: attempt counter is incremented before requeue
# ---------------------------------------------------------------------------

def test_attempt_incremented_before_requeue(tmp_path):
    """When a worker dies WITHOUT a crash signal (non-negative exitcode) on
    attempt 1 and QUEUE_MAX_RETRIES=1, the requeued task should have _attempt=2.
    Signal crashes (negative exitcode) are terminal and covered separately."""
    import supervisor.workers as W

    task = _make_task(task_id="t001", attempt=1)
    child_drive = tmp_path / "child-drive"
    task["drive_root"] = str(child_drive)
    task["child_drive_root"] = str(child_drive)
    service_dir = child_drive / "services" / "t001"
    service_dir.mkdir(parents=True)
    (service_dir / "devserver.log").write_text("READY\n", encoding="utf-8")
    worker = _make_worker(busy_task_id="t001", exitcode=1)

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.QUEUE_MAX_RETRIES = 1
    W.WORKERS = {0: worker}
    W.RUNNING = {
        "t001": {
            "task": task,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 1,
        }
    }
    W._LAST_SPAWN_TIME = 0  # Grace period elapsed

    enqueued = []

    import supervisor.queue as sq

    with patch.object(sq, "enqueue_task", side_effect=lambda t, front=False: enqueued.append(dict(t))), \
         patch.object(sq, "persist_queue_snapshot", MagicMock()), \
         patch("supervisor.workers.respawn_worker"), \
         patch("ouroboros.task_results.load_task_result", return_value=None), \
         patch("ouroboros.task_results.write_task_result"):
        _run_health_and_reap()

    assert len(enqueued) == 1, "Task should be requeued once"
    assert enqueued[0]["_attempt"] == 2, f"Expected _attempt=2, got {enqueued[0].get('_attempt')}"
    assert not service_dir.exists(), "Child-drive service logs should be archived after worker death"


def test_worker_main_exits_after_post_bootstrap_task_exception(monkeypatch, tmp_path):
    """A caught task exception must become a non-signal process exit for handoff."""
    import ouroboros.agent as agent_module
    import ouroboros.config as config
    import ouroboros.extension_loader as extension_loader
    import ouroboros.platform_layer as platform_layer
    import ouroboros.process_custody as process_custody
    import ouroboros.utils as utils
    import supervisor.workers as W
    import supervisor.worker_process as WP

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    repo.mkdir()
    (drive / "logs").mkdir(parents=True)

    class InputQueue:
        def __init__(self):
            self.calls = 0

        def get(self):
            self.calls += 1
            return (
                {"id": "child1", "type": "task"}
                if self.calls == 1
                else {"type": "shutdown"}
            )

    class Agent:
        def handle_task(self, _task):
            raise RuntimeError("post-bootstrap context build failed")

    incoming = InputQueue()
    crashes = []
    monkeypatch.setattr(WP, "_bind_worker_repo_root", lambda *_a, **_k: None)
    monkeypatch.setattr(WP, "_prepare_worker_task_runtime", lambda: None)
    monkeypatch.setattr(WP, "_log_worker_crash", lambda *args: crashes.append(args))
    monkeypatch.setattr(platform_layer, "create_new_session", lambda: None)
    monkeypatch.setattr(process_custody, "start_parent_lifeline", lambda **_k: None)
    monkeypatch.setattr(config, "initialize_runtime_mode_baseline", lambda: None)
    monkeypatch.setattr(config, "get_skills_repo_path", lambda: "")
    monkeypatch.setattr(extension_loader, "reload_all", lambda *_a, **_k: None)
    monkeypatch.setattr(agent_module, "make_agent", lambda **_k: Agent())
    monkeypatch.setattr(utils, "set_log_sink", lambda _sink: None)
    monkeypatch.setattr(utils, "get_git_info", lambda _repo: ("test", "sha"))

    W.worker_main(0, incoming, SimpleNamespace(put=lambda _event: None), str(repo), str(drive))

    assert incoming.calls == 1, "the worker must not accept another task after a task crash"
    assert len(crashes) == 1 and crashes[0][2] == "handle_task"


def test_crash_retry_admission_block_terminalizes_task(tmp_path):
    """A fence refusal must not leave an interrupted task claiming it was requeued."""
    import supervisor.queue as sq
    import supervisor.workers as W

    task = _make_task(task_id="t-fenced", attempt=1)
    worker = _make_worker(busy_task_id="t-fenced", exitcode=1)
    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.QUEUE_MAX_RETRIES = 1
    W.WORKERS = {0: worker}
    W.RUNNING = {
        "t-fenced": {
            "task": task,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 1,
        }
    }
    W._LAST_SPAWN_TIME = 0
    writes = []
    terminal = []

    def fake_write(_drive, task_id, status, **kwargs):
        writes.append((task_id, status, kwargs))

    with patch.object(
        sq,
        "enqueue_task",
        return_value={"_admission_blocked": "root_budget_fence"},
    ), patch.object(sq, "persist_queue_snapshot", MagicMock()), patch(
        "supervisor.workers.respawn_worker"
    ), patch(
        "supervisor.workers._emit_task_done_terminal",
        side_effect=lambda *args, **kwargs: terminal.append((args, kwargs)),
    ), patch(
        "ouroboros.task_results.load_task_result", return_value=None
    ), patch(
        "ouroboros.task_results.write_task_result", side_effect=fake_write
    ):
        _run_health_and_reap()

    assert writes[-1][1] == "failed"
    assert writes[-1][2]["reason_code"] == "worker_crash_retry_admission_blocked"
    assert terminal and terminal[-1][0][2] == "failed"


# ---------------------------------------------------------------------------
# Test: retry limit exhausted → STATUS_FAILED, no requeue
# ---------------------------------------------------------------------------

def test_retry_limit_exhausted_marks_failed(tmp_path):
    """When attempt > QUEUE_MAX_RETRIES, task is marked failed — not requeued."""
    import supervisor.workers as W

    task = _make_task(task_id="t002", attempt=2)  # attempt=2 > QUEUE_MAX_RETRIES=1
    worker = _make_worker(busy_task_id="t002", exitcode=-11)

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.QUEUE_MAX_RETRIES = 1
    W.WORKERS = {0: worker}
    W.RUNNING = {
        "t002": {
            "task": task,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 2,
        }
    }
    W._LAST_SPAWN_TIME = 0

    written_results = {}
    enqueued = []

    def fake_write(drive, task_id, status, result="", **kw):
        written_results[task_id] = {"status": status, "result": result}

    import supervisor.queue as sq

    with patch.object(sq, "enqueue_task", side_effect=lambda t, front=False: enqueued.append(dict(t))), \
         patch.object(sq, "persist_queue_snapshot", MagicMock()), \
         patch("supervisor.workers.respawn_worker"), \
         patch("ouroboros.task_results.load_task_result", return_value=None), \
         patch("ouroboros.task_results.write_task_result", side_effect=fake_write), \
         patch("supervisor.workers.get_event_q", return_value=MagicMock()), \
         patch("supervisor.message_bus.get_bridge", return_value=None):
        _run_health_and_reap()

    assert len(enqueued) == 0, "Task should NOT be requeued after limit exhausted"
    assert "t002" in written_results, "Task result should be written"
    assert written_results["t002"]["status"] == "failed", (
        f"Expected 'failed', got {written_results['t002']['status']}"
    )


# ---------------------------------------------------------------------------
# Test: already-completed task is not requeued
# ---------------------------------------------------------------------------

def test_already_completed_task_not_requeued(tmp_path):
    """If a task already has a terminal result (e.g. completed via direct-chat),
    it must NOT be requeued after a worker crash."""
    import supervisor.workers as W

    task = _make_task(task_id="t003", attempt=1)
    worker = _make_worker(busy_task_id="t003", exitcode=-11)

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.QUEUE_MAX_RETRIES = 5  # High limit so it's not the reason for skipping
    W.WORKERS = {0: worker}
    W.RUNNING = {
        "t003": {
            "task": task,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 1,
        }
    }
    W._LAST_SPAWN_TIME = 0

    existing_result = {"status": "completed", "result": "done"}
    enqueued = []

    import supervisor.queue as sq

    with patch.object(sq, "enqueue_task", side_effect=lambda t, front=False: enqueued.append(dict(t))), \
         patch.object(sq, "persist_queue_snapshot", MagicMock()), \
         patch("supervisor.workers.respawn_worker"), \
         patch("supervisor.workers.send_with_budget"), \
         patch("supervisor.workers.load_state", return_value={}), \
         patch("ouroboros.task_results.load_task_result", return_value=existing_result), \
         patch("ouroboros.task_results.write_task_result"):
        _run_health_and_reap()

    assert len(enqueued) == 0, (
        "Task with existing terminal result should NOT be requeued"
    )


# ---------------------------------------------------------------------------
# Test: terminal event emitted when retry limit exhausted
# ---------------------------------------------------------------------------

def test_terminal_event_emitted_on_limit_exhausted(tmp_path):
    """When retry limit is exhausted, a task_done event must be emitted."""
    import supervisor.workers as W
    import queue as _queue

    task = _make_task(task_id="t004", attempt=2, chat_id=42)
    worker = _make_worker(busy_task_id="t004", exitcode=-11)

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.QUEUE_MAX_RETRIES = 1
    W.WORKERS = {0: worker}
    W.RUNNING = {
        "t004": {
            "task": task,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 2,
        }
    }
    W._LAST_SPAWN_TIME = 0

    # Use a real queue to capture events
    event_q = _queue.Queue()

    import supervisor.queue as sq

    with patch.object(sq, "enqueue_task", MagicMock()), \
         patch.object(sq, "persist_queue_snapshot", MagicMock()), \
         patch("supervisor.workers.respawn_worker"), \
         patch("supervisor.workers.get_event_q", return_value=event_q), \
         patch("supervisor.workers.send_with_budget"), \
         patch("supervisor.workers.load_state", return_value={}), \
         patch("ouroboros.task_results.load_task_result", return_value=None), \
         patch("ouroboros.task_results.write_task_result"), \
         patch("supervisor.message_bus.get_bridge", return_value=None):
        _run_health_and_reap()

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    task_done_events = [e for e in events if e.get("type") == "task_done"]
    assert len(task_done_events) >= 1, f"Expected task_done event, got: {events}"
    assert task_done_events[0]["task_id"] == "t004"
    assert task_done_events[0]["status"] == "failed"


# ---------------------------------------------------------------------------
# Test: respawn_worker does NOT reset _LAST_SPAWN_TIME
# ---------------------------------------------------------------------------

def test_respawn_worker_does_not_reset_spawn_time(tmp_path):
    """respawn_worker must not reset _LAST_SPAWN_TIME — only spawn_workers should."""
    import supervisor.workers as W

    original_time = 1000.0  # An old timestamp
    W._LAST_SPAWN_TIME = original_time
    W.DRIVE_ROOT = tmp_path
    W.REPO_DIR = tmp_path

    fake_proc = MagicMock()
    fake_proc.pid = 12345
    fake_queue = MagicMock()

    ctx = MagicMock()
    ctx.Process.return_value = fake_proc
    ctx.Queue.return_value = fake_queue

    with patch("supervisor.workers._get_ctx", return_value=ctx), \
         patch("supervisor.workers.get_event_q", return_value=fake_queue), \
         patch("supervisor.workers._verify_worker_sha_after_spawn"):
        W.respawn_worker(0)

    assert W._LAST_SPAWN_TIME == original_time, (
        f"_LAST_SPAWN_TIME should NOT be reset by respawn_worker, "
        f"but changed from {original_time} to {W._LAST_SPAWN_TIME}"
    )


# ---------------------------------------------------------------------------
# Test: deep_self_review crash emits task_done terminal event
# ---------------------------------------------------------------------------

def test_non_completed_terminal_status_not_requeued(tmp_path):
    """Crash after a task reaches any terminal state (rejected_duplicate, interrupted,
    cancelled) must NOT be requeued — not just 'completed' or 'failed'."""
    import supervisor.workers as W

    task = _make_task(task_id="t005", attempt=1, chat_id=9)
    worker = _make_worker(busy_task_id="t005", exitcode=-11)

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.QUEUE_MAX_RETRIES = 3  # High limit so we don't hit retry exhaustion
    W.WORKERS = {0: worker}
    W.RUNNING = {
        "t005": {
            "task": task,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 1,
        }
    }
    W._LAST_SPAWN_TIME = 0

    enqueued = []
    import supervisor.queue as sq

    # Test truly final terminal statuses (STATUS_INTERRUPTED excluded — it's pre-requeue)
    for terminal_status in ("rejected_duplicate", "cancelled", "failed"):
        enqueued.clear()
        worker.reaping = False
        W.WORKERS = {0: worker}
        W.RUNNING = {"t005": {"task": dict(task), "attempt": 1}}
        existing_result = {"status": terminal_status, "result": "done"}

        with patch.object(sq, "enqueue_task", side_effect=lambda t, front=False: enqueued.append(dict(t))), \
             patch.object(sq, "persist_queue_snapshot", MagicMock()), \
             patch("supervisor.workers.respawn_worker"), \
             patch("supervisor.workers.send_with_budget"), \
             patch("supervisor.workers.load_state", return_value={}), \
             patch("ouroboros.task_results.load_task_result", return_value=existing_result), \
             patch("ouroboros.task_results.write_task_result"):
            _run_health_and_reap()

        assert len(enqueued) == 0, (
            f"Task with terminal status '{terminal_status}' should NOT be requeued, "
            f"but was requeued: {enqueued}"
        )

    # STATUS_INTERRUPTED must NOT prevent requeue (it's written before requeue, not after)
    # Reset state: previous loop iterations consumed t005 from RUNNING/WORKERS
    enqueued.clear()
    task2 = _make_task(task_id="t006", attempt=1, chat_id=9)
    # Non-signal death so the retry path runs; this asserts 'interrupted' status
    # does not block requeue (signal crashes are terminal, tested separately).
    worker2 = _make_worker(busy_task_id="t006", exitcode=1)
    W.WORKERS = {0: worker2}
    W.RUNNING = {
        "t006": {
            "task": task2,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 1,
        }
    }
    interrupted_result = {"status": "interrupted", "result": "retrying"}
    with patch.object(sq, "enqueue_task", side_effect=lambda t, front=False: enqueued.append(dict(t))), \
         patch.object(sq, "persist_queue_snapshot", MagicMock()), \
         patch("supervisor.workers.respawn_worker"), \
         patch("supervisor.workers.send_with_budget"), \
         patch("supervisor.workers.load_state", return_value={}), \
         patch("ouroboros.task_results.load_task_result", return_value=interrupted_result), \
         patch("ouroboros.task_results.write_task_result"):
        _run_health_and_reap()

    assert len(enqueued) == 1, (
        f"Task with 'interrupted' status IS NOT terminal and SHOULD be requeued, "
        f"but got: {enqueued}"
    )
    assert enqueued[0].get("_attempt", 1) == 2, (
        f"Attempt should have incremented to 2, got: {enqueued[0].get('_attempt')}"
    )


def test_a_signal_death_takes_the_same_one_retry_and_reserves_no_handoff(tmp_path, monkeypatch):
    """ARCHITECTURE §5: only a recorded stop terminalizes. A worker killed by a signal (-15 here, the
    05.11 pattern kill) is an unrecorded death like an exit code: attempt 1 is retried once on the same id
    from its saved state (``interrupted``, the signal on the row), with NO delegated handoff reserved: the dead
    attempt's still-running delegated runs are cancelled as an abrupt loss (``reconcile_task_runs``
    through the attach-only gateway, the verdict this caller writes next), while an exit-code death keeps its
    handoff and leaves its runs to it; the retried attempt that dies again is terminal
    (``worker_crash_retry_exhausted``), as is a recorded cancel. ``worker_crash_signal`` stays a readable reason
    code for old rows."""
    import supervisor.queue as q
    import supervisor.workers as W
    from ouroboros import delegate_custody_reconcile, delegate_recovery
    from ouroboros.owner_continue import TECHNICAL_REASON_CODES
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.worker_health import recover_confirmed_dead_worker
    from tests.test_retained_work_recovery import _saved

    handoffs, released = [], []
    monkeypatch.setattr(delegate_recovery, "prepare_handoff",
                        lambda root, task, **kw: handoffs.append(kw["exitcode"]) or {"reserved": kw["exitcode"]})
    monkeypatch.setattr(delegate_custody_reconcile, "reconcile_task_runs",
                        lambda root, task_id, **kw: released.append(
                            (task_id, kw["deliberate_terminal"],
                             kw["gateway_factory"] is delegate_custody_reconcile.attach_only_gateway)) or [])
    enqueued = []
    monkeypatch.setattr(q, "enqueue_task", lambda task, front=False: enqueued.append(dict(task)) or task)
    monkeypatch.setattr(W, "reconstruct_task_cost", lambda *_a, **_k: {"cost_accounting_status": "available"})
    monkeypatch.setattr(W, "send_with_budget", lambda *_a, **_k: None)

    # Fires: -15 with saved work -> attempt 2, the same id, saved state attached, no handoff row.
    job, events = _reserved_job(tmp_path, monkeypatch, exitcode=-15, attempt=1)
    task_id = job["task_id"]
    write_task_result(tmp_path, task_id, "running")
    _saved(tmp_path, task_id, 1)
    recover_confirmed_dead_worker(job)
    assert [t["id"] for t in enqueued] == [task_id] and enqueued[0]["_attempt"] == 2
    assert enqueued[0]["_working_recovery"]["cause"] == "worker_crash"
    assert handoffs == [], "a signal death reserves no delegated handoff"
    assert released == [(task_id, "interrupted", True)], "its open delegated runs end as an abrupt loss"
    row = load_task_result(tmp_path, task_id)
    assert row["status"] == "interrupted" and "signal 15" in row["result"]
    assert events.empty(), "no terminal was fabricated for a retried attempt"

    # Quiet: an exit-code death keeps its handoff reservation.
    W.WORKERS.clear(); W.RUNNING.clear(); q.RUNNING = W.RUNNING
    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=1, attempt=1)
    write_task_result(tmp_path, job["task_id"], "running")
    recover_confirmed_dead_worker(job)
    assert handoffs == [1] and len(enqueued) == 2 and len(released) == 1, "an exit-code death keeps its runs for the handoff"

    # The bound holds: the retried attempt's signal death is terminal, with the signal recorded.
    W.WORKERS.clear(); W.RUNNING.clear(); q.RUNNING = W.RUNNING
    job, events = _reserved_job(tmp_path, monkeypatch, exitcode=-15, attempt=2)
    write_task_result(tmp_path, job["task_id"], "running")
    recover_confirmed_dead_worker(job)
    row = load_task_result(tmp_path, job["task_id"])
    assert row["status"] == "failed" and row["reason_code"] == "worker_crash_retry_exhausted"
    assert row["crash_signal"] == 15 and row["crash_exitcode"] == -15 and len(enqueued) == 2
    assert events.get_nowait()["type"] == "task_done"
    assert {"worker_crash_signal", "worker_crash_retry_exhausted"} <= TECHNICAL_REASON_CODES  # old rows stay Continue-eligible


def test_crash_toast_for_a_bound_task_goes_to_its_project_chat(tmp_path):
    """The crash toast is a DIRECT send, so the durable project binding has to
    win over the chat the task was born in: a task converted into a project
    mid-run would otherwise be told about its crash in Main."""
    import supervisor.workers as W
    import queue as _queue

    from ouroboros.projects_registry import bind_task_to_project

    bind_task_to_project(tmp_path, "sig02", "crash-proj", 5151, origin={"absent": "system"})
    task = _make_task(task_id="sig02", attempt=2, chat_id=1)  # born in Main; its one retry already spent
    worker = _make_worker(busy_task_id="sig02", exitcode=-11)

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.QUEUE_MAX_RETRIES = 1
    W.WORKERS = {0: worker}
    W.RUNNING = {
        "sig02": {
            "task": task,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 2,
        }
    }
    W._LAST_SPAWN_TIME = 0

    import supervisor.queue as sq

    incident_notice = MagicMock()
    with patch.object(sq, "enqueue_task", side_effect=lambda t, front=False: None), \
         patch.object(sq, "persist_queue_snapshot", MagicMock()), \
         patch("supervisor.workers.respawn_worker"), \
         patch("supervisor.workers.load_state", return_value={}), \
         patch("ouroboros.task_results.load_task_result", return_value=None), \
         patch("ouroboros.task_results.write_task_result", side_effect=lambda *a, **k: None), \
         patch("supervisor.workers.get_event_q", return_value=_queue.Queue()), \
         patch("supervisor.workers.send_with_budget", incident_notice), \
         patch("supervisor.message_bus.get_bridge", return_value=None):
        _run_health_and_reap()

    incident_notice.assert_called_once()
    assert incident_notice.call_args[0][0] == 5151


def test_deep_self_review_crash_emits_task_done_event(tmp_path):
    """deep_self_review crash at its last attempt must emit task_done so the UI live card closes."""
    import supervisor.workers as W
    import queue as _queue

    task = _make_task(task_id="dsr01", attempt=2, chat_id=7)
    task["type"] = "deep_self_review"
    worker = _make_worker(busy_task_id="dsr01", exitcode=-11)

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.QUEUE_MAX_RETRIES = 1
    W.WORKERS = {0: worker}
    W.RUNNING = {
        "dsr01": {
            "task": task,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 2,
        }
    }
    W._LAST_SPAWN_TIME = 0

    event_q = _queue.Queue()

    import supervisor.queue as sq

    with patch.object(sq, "enqueue_task", MagicMock()), \
         patch.object(sq, "persist_queue_snapshot", MagicMock()), \
         patch("supervisor.workers.respawn_worker"), \
         patch("supervisor.workers.get_event_q", return_value=event_q), \
         patch("supervisor.workers.send_with_budget"), \
         patch("supervisor.workers.load_state", return_value={}), \
         patch("ouroboros.task_results.write_task_result"), \
         patch("supervisor.message_bus.get_bridge", return_value=None):
        _run_health_and_reap()

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    task_done_events = [e for e in events if e.get("type") == "task_done"]
    assert len(task_done_events) >= 1, (
        f"Expected task_done terminal event for deep_self_review crash, got: {events}"
    )
    assert task_done_events[0]["task_id"] == "dsr01"
    assert task_done_events[0]["status"] == "failed"


def _reserved_job(tmp_path, monkeypatch, *, exitcode=1, attempt=1, child=False):
    import queue as stdqueue
    import supervisor.queue as q
    import supervisor.workers as W

    task = _make_task('saved-terminal', attempt=attempt)
    if child:
        from ouroboros.headless import prepare_task_drive

        drive = prepare_task_drive(tmp_path, task['id'], 'empty')
        task.update(drive_root=str(drive), child_drive_root=str(drive))
    worker = _make_worker(busy_task_id=task['id'], exitcode=exitcode)
    meta = {'task': task, 'attempt': attempt, 'worker_id': 0}
    W.WORKERS = {0: worker}
    W.RUNNING = q.RUNNING = {task['id']: meta}
    events = stdqueue.Queue()
    # The retry limit is a process-global (`supervisor.queue.QUEUE_MAX_RETRIES`);
    # sibling tests assign it directly, so an xdist worker can reach this fixture
    # with 0 and refuse the one retry these scenarios rely on. Pin the default.
    monkeypatch.setattr(W, 'QUEUE_MAX_RETRIES', 1)
    monkeypatch.setattr(W, 'get_event_q', lambda: events)
    monkeypatch.setattr(W, 'respawn_worker', MagicMock())
    monkeypatch.setattr(q, 'persist_queue_snapshot', lambda **_k: None)
    W.ensure_workers_healthy()
    job = q._reap_queue.get_nowait()
    assert job['kind'] == 'confirmed_dead_worker'
    assert job['worker'] is worker and job['meta'] is meta
    assert W.RUNNING[task['id']] is meta
    assert worker.reaping and worker.busy_task_id == task['id']
    return job, events


def test_dead_detection_queues_without_result_reads_or_cost_scans(tmp_path, monkeypatch):
    import supervisor.workers as W

    with patch('ouroboros.task_results.load_task_result', side_effect=AssertionError('on drain')), \
         patch('ouroboros.headless.prepare_terminal_task_files', side_effect=AssertionError('on drain')), \
         patch.object(W, 'reconstruct_task_cost', side_effect=AssertionError('on drain')):
        job, _events = _reserved_job(tmp_path, monkeypatch)
    assert job['attempt'] == 1 and job['exitcode'] == 1
    W.respawn_worker.assert_not_called()


@pytest.mark.parametrize('exitcode', [1, -11])
def test_child_terminal_survives_dead_worker_without_retry(tmp_path, monkeypatch, exitcode):
    import supervisor.queue as q
    import supervisor.workers as W
    from supervisor.worker_health import recover_confirmed_dead_worker
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.observability import persist_call, read_blob_ref

    job, events = _reserved_job(tmp_path, monkeypatch, child=True, exitcode=exitcode, attempt=3)
    child = Path(job['task']['drive_root'])
    ref = persist_call(child, task_id=job['task_id'], call_id='original-response',
                       call_type='tool_call', payload={'result': 'FULL ORIGINAL SOURCE'})
    write_task_result(child, job['task_id'], 'completed', result='Original natural answer',
                      trace_refs={'tool_call_refs': [ref]}, accounted_upper_bound_usd=1.25)
    delivery = MagicMock()
    monkeypatch.setattr('supervisor.terminal_delivery.deliver_miss_lane_outcome', delivery)
    monkeypatch.setattr(q, 'enqueue_task', MagicMock())
    recover_confirmed_dead_worker(job)
    current = load_task_result(tmp_path, job['task_id'], strict=True)
    assert current['status'] == 'completed' and current['result'] == 'Original natural answer'
    assert current['accounted_upper_bound_usd'] == 1.25
    assert current['child_ref_promotion']['pending_refs'] and child.exists()
    promoted = current['trace_refs']['tool_call_refs'][0]['redacted_projection_ref']
    assert read_blob_ref(tmp_path, promoted)['result'] == 'FULL ORIGINAL SOURCE'
    q.enqueue_task.assert_not_called()
    delivery.assert_called_once()
    done = events.get_nowait()
    assert done['type'] == 'task_done' and done['status'] == 'completed'
    assert done['_files_prepared_attempt'] == 3 and done['worker_id'] == 0
    assert W.RUNNING[job['task_id']] is job['meta'], 'normal done ingress owns release'
    W.respawn_worker.assert_called_once_with(0)
    from ouroboros.headless import remove_subagent_task_drive, retry_child_task_refs

    retry_child_task_refs(tmp_path, child, job['task_id'])
    assert remove_subagent_task_drive(tmp_path, job['task_id'], live=lambda _task: False)
    assert not child.exists() and read_blob_ref(tmp_path, promoted)['result'] == 'FULL ORIGINAL SOURCE'
    assert events.empty() and load_task_result(tmp_path, job['task_id'])['accounted_upper_bound_usd'] == 1.25


def test_a_crashed_child_retries_on_its_own_id_under_its_parent_and_root(tmp_path, monkeypatch):
    """The same-id crash retry is the same delegated child, one attempt later, under the same parent
    and root: never re-rooted, never a new id or a timeout successor."""
    import supervisor.queue as q
    from supervisor.worker_health import recover_confirmed_dead_worker
    from ouroboros.task_results import write_task_result

    job, _events = _reserved_job(tmp_path, monkeypatch, child=True, exitcode=1, attempt=1)
    lineage = {'parent_task_id': 'parent-task', 'root_task_id': 'root-task', 'delegation_role': 'subagent'}
    for task in {id(job['task']): job['task'], id(job['meta']['task']): job['meta']['task']}.values():
        task.update(lineage)
    write_task_result(tmp_path, job['task_id'], 'running', **lineage)
    enqueued = []
    monkeypatch.setattr(q, 'enqueue_task', lambda task, front=False: enqueued.append(dict(task)) or task)
    recover_confirmed_dead_worker(job)
    [retry] = enqueued
    assert (retry['id'], retry['_attempt']) == (job['task_id'], 2)
    assert {key: retry.get(key) for key in lineage} == lineage
    assert not retry.get('timeout_retry_from') and retry.get('original_task_id', job['task_id']) == job['task_id']


def test_file_preparation_runs_without_queue_or_lifecycle_lock(tmp_path, monkeypatch):
    import threading
    import supervisor.queue as q
    import supervisor.workers as W
    from ouroboros.task_results import write_task_result
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, _events = _reserved_job(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    failures = []

    def prepare(root, task):
        assert not q._queue_lock._is_owned()
        assert not W._WORKER_LIFECYCLE_LOCK._is_owned()
        entered.set()
        assert release.wait(5)
        write_task_result(root, task['id'], 'completed', result='Natural answer')
        return {'task_id': task['id'], 'terminal_source_present': True, 'error': ''}

    def run():
        try:
            recover_confirmed_dead_worker(job)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr('ouroboros.headless.prepare_terminal_task_files', prepare)
    monkeypatch.setattr('supervisor.terminal_delivery.deliver_miss_lane_outcome', lambda *_a, **_k: None)
    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5)
        assert q._queue_lock.acquire(timeout=1)
        q._queue_lock.release()
        assert W._WORKER_LIFECYCLE_LOCK.acquire(timeout=1)
        W._WORKER_LIFECYCLE_LOCK.release()
        assert W.RUNNING[job['task_id']] is job['meta']
        W.respawn_worker.assert_not_called()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and failures == []


@pytest.mark.parametrize('source_present', [None, True])
def test_unknown_or_uncopied_terminal_retains_same_job(tmp_path, monkeypatch, source_present):
    import supervisor.queue as q
    import supervisor.workers as W
    from ouroboros.task_results import write_task_result
    from supervisor.worker_health import recover_confirmed_dead_worker
    from supervisor.task_reaper import TerminalFileRecoveryPending

    job, events = _reserved_job(tmp_path, monkeypatch, child=True)
    # An early post-task canonical completion is not a copied child body.
    write_task_result(tmp_path, job['task_id'], 'completed', post_task_checkpoint={'status': 'open'})
    monkeypatch.setattr('ouroboros.headless.prepare_terminal_task_files', lambda *_a: {
        'task_id': job['task_id'], 'terminal_source_present': source_present,
        'result': {'status': 'completed'}, 'error': 'fixture I/O failure',
    })
    monkeypatch.setattr(q, 'enqueue_task', MagicMock())
    with pytest.raises(TerminalFileRecoveryPending):
        recover_confirmed_dead_worker(job)
    assert W.RUNNING[job['task_id']] is job['meta'] and job['worker'].reaping
    assert events.empty()
    q.enqueue_task.assert_not_called()
    W.respawn_worker.assert_not_called()


@pytest.mark.parametrize('change', ['worker', 'attempt', 'worker_binding'])
def test_stale_captured_job_cannot_prepare_or_respawn(tmp_path, monkeypatch, change):
    import supervisor.workers as W
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, events = _reserved_job(tmp_path, monkeypatch)
    if change == 'worker':
        W.WORKERS[0] = _make_worker(alive=True, busy_task_id='new-task')
    elif change == 'meta':
        W.RUNNING[job['task_id']] = dict(job['meta'])
    elif change == 'attempt':
        job['meta']['attempt'] = 2
    elif change == 'worker_binding':
        job['meta']['worker_id'] = 7
    else:
        W.DRIVE_ROOT = tmp_path / 'new-generation'
    prepare = MagicMock()
    monkeypatch.setattr('ouroboros.headless.prepare_terminal_task_files', prepare)
    recover_confirmed_dead_worker(job)
    prepare.assert_not_called()
    W.respawn_worker.assert_not_called()
    assert events.empty()


def test_generation_change_during_file_save_cannot_publish_or_respawn(tmp_path, monkeypatch):
    import supervisor.workers as W
    from ouroboros.task_results import write_task_result
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, events = _reserved_job(tmp_path, monkeypatch)
    newer = _make_worker(alive=True, busy_task_id='newer')

    def prepare(root, task):
        write_task_result(root, task['id'], 'completed', result='Old answer')
        W.WORKERS[0] = newer
        return {'terminal_source_present': True, 'error': ''}

    monkeypatch.setattr('ouroboros.headless.prepare_terminal_task_files', prepare)
    recover_confirmed_dead_worker(job)
    assert W.WORKERS[0] is newer and newer.busy_task_id == 'newer'
    W.respawn_worker.assert_not_called()
    assert events.empty()


def test_file_recovery_owner_is_not_enqueued_as_another_dead_job(tmp_path, monkeypatch):
    import supervisor.queue as q
    import supervisor.workers as W
    import supervisor.task_reaper as reaper

    task = _make_task()
    worker = _make_worker()
    W.WORKERS = {0: worker}
    W.RUNNING = q.RUNNING = {task['id']: {'task': task, '_terminal_file_recovery': {'held': True}}}
    retry = MagicMock()
    monkeypatch.setattr(reaper, 'retry_terminal_file_recoveries', retry)
    W.ensure_workers_healthy()
    retry.assert_called_once()
    assert q._reap_queue.empty() and not worker.reaping


def test_unreadable_child_is_unknown_not_a_paid_retry(tmp_path, monkeypatch):
    import supervisor.queue as q
    import supervisor.workers as W
    from supervisor.worker_health import recover_confirmed_dead_worker
    from supervisor.task_reaper import TerminalFileRecoveryPending
    from tests._store_shared import read_result, seed_result, unreadable_result

    job, events = _reserved_job(tmp_path, monkeypatch, child=True)
    root = Path(job['task']['drive_root'])
    seed_result(root, job['task_id'], {'status': 'running', 'result': 'preserved source'})
    switch = unreadable_result(monkeypatch, job['task_id'], root=root)  # the store cannot answer for it
    monkeypatch.setattr(q, 'enqueue_task', MagicMock())
    with pytest.raises(TerminalFileRecoveryPending):
        recover_confirmed_dead_worker(job)
    switch['on'] = False
    assert read_result(root, job['task_id'])['result'] == 'preserved source'
    assert W.RUNNING[job['task_id']] is job['meta'] and job['worker'].reaping
    assert events.empty()
    q.enqueue_task.assert_not_called()
    W.respawn_worker.assert_not_called()


@pytest.mark.parametrize('enqueue_raises', [False, True])
def test_retry_queue_transition_is_atomic_and_failure_retains_binding(tmp_path, monkeypatch, enqueue_raises):
    import supervisor.queue as q
    import supervisor.workers as W
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, _events = _reserved_job(tmp_path, monkeypatch)
    calls = []

    def enqueue(task, front=False):
        assert q._queue_lock._is_owned()
        assert job['task_id'] not in W.RUNNING
        assert job['worker'].reaping and task['_attempt'] == 2 and front
        calls.append(task)
        if enqueue_raises:
            raise RuntimeError('fixture enqueue failure')
        return task

    monkeypatch.setattr(q, 'enqueue_task', enqueue)
    if enqueue_raises:
        with pytest.raises(RuntimeError, match='fixture enqueue failure'):
            recover_confirmed_dead_worker(job)
        assert W.RUNNING[job['task_id']] is job['meta']
        W.respawn_worker.assert_not_called()
    else:
        recover_confirmed_dead_worker(job)
        assert job['task_id'] not in W.RUNNING
        W.respawn_worker.assert_called_once_with(0)
    assert len(calls) == 1


@pytest.mark.parametrize('cause', ['owner_wait', 'inactive_owner', 'evolution_stopped'])
def test_existing_nonretry_reasons_and_cost_projection_survive_handoff(tmp_path, monkeypatch, cause):
    import supervisor.queue as q
    import supervisor.workers as W
    from ouroboros.task_results import load_task_result
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, events = _reserved_job(tmp_path, monkeypatch)
    if cause == 'owner_wait':
        job['meta']['owner_wait'] = {'task_attempt': 1, 'source_ref': {'retained': True}}
    elif cause == 'inactive_owner':
        job['worker'].active_capacity = False
    else:
        job['task']['type'] = job['meta']['task']['type'] = 'evolution'
    monkeypatch.setattr(W, 'load_state', lambda: {'evolution_mode_enabled': False})
    monkeypatch.setattr(W, 'send_with_budget', lambda *_a, **_k: None)
    monkeypatch.setattr(W, 'reconstruct_task_cost', lambda *_a, **_k: {
        'cost_accounting_status': 'available', 'cost_final': False,
        'accounted_upper_bound_usd': 12.5, 'rounds': 7,
    })
    monkeypatch.setattr(q, 'enqueue_task', MagicMock())
    recover_confirmed_dead_worker(job)
    result = load_task_result(tmp_path, job['task_id'], strict=True)
    assert result['reason_code'] == ('evolution_stopped_no_retry' if cause == 'evolution_stopped'
                                     else 'worker_crash_owner_wait')
    assert result['status'] == ('cancelled' if cause == 'evolution_stopped' else 'failed')
    done = events.get_nowait()
    assert done['accounted_upper_bound_usd'] == 12.5 and done['rounds'] == 7
    q.enqueue_task.assert_not_called()


@pytest.mark.parametrize("deleted", [False, True])
@pytest.mark.parametrize("basis_kind", ["prepared", "legacy", "converted_main", "converted_derived"])
def test_real_crash_retry_keeps_prepared_scope_across_registry_activity(tmp_path, monkeypatch, deleted, basis_kind):
    from ouroboros import projects_registry as registry
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import queue as q, workers as w

    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(q, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr(q, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(w, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(w, "respawn_worker", lambda *a, **k: None)
    done = []
    monkeypatch.setattr(w, "_emit_task_done_terminal", lambda *a, **k: done.append((a, k)))
    registry.create_project(tmp_path, "target")
    registry.create_project(tmp_path, "other")
    payload = {**_make_task("crash"), "project_id": "target", "workspace_root": "/synthetic/frozen-resource"}
    if basis_kind.startswith("converted"):
        basis = registry.project_scope_admission(tmp_path, workspace_root=(
            payload["workspace_root"] if basis_kind == "converted_derived" else ""))
        payload.update(project_id=basis["project_id"], _project_admission=basis)
    admitted = q.enqueue_task(payload)
    q.PENDING.clear()
    if basis_kind == "legacy":
        registry.bind_task_to_project(tmp_path, "crash", "target", origin={"absent": "system"})
        admitted.pop("_project_admission")
        admitted.pop("project_id")
    monkeypatch.setattr(w, "PENDING", q.PENDING)
    write_task_result(tmp_path, "crash", "running", result="working")
    worker = _make_worker(busy_task_id="crash", exitcode=1)
    monkeypatch.setattr(w, "WORKERS", {0: worker})
    monkeypatch.setattr(w, "RUNNING", {"crash": {"task": admitted, "attempt": 1,
        "started_at": time.time() - 5, "last_heartbeat_at": time.time() - 5}})
    if basis_kind.startswith("converted"):
        import asyncio
        from ouroboros.gateway.projects import api_project_from_task
        from tests.test_project_lease_ui_conversion import _request

        running_before = copy.deepcopy(admitted)
        response = asyncio.run(api_project_from_task(_request(tmp_path, {
            "task_id": "crash", "id": "target", "name": "Target"})))
        assert response.status_code == 200
        assert admitted == running_before  # conversion does not retarget a running execution
        assert registry.project_id_for_task(tmp_path, "crash") == "target"
    real_enqueue = q.enqueue_task
    def enqueue(payload, **kwargs):
        registry.touch_project(tmp_path, "other")
        if deleted:
            registry.begin_project_deletion(tmp_path, "target")
        return real_enqueue(payload, **kwargs)
    monkeypatch.setattr(q, "enqueue_task", enqueue)
    _run_health_and_reap()
    assert not w.RUNNING
    if deleted:
        assert not q.PENDING and done
        result = load_task_result(tmp_path, "crash")
        assert result["status"] == "failed" and result["reason_code"] == "worker_crash_retry_admission_blocked"
    else:
        [retry] = q.PENDING
        assert retry["id"] == "crash" and retry["_attempt"] == 2
        assert retry["_project_admission"]["project_id"] == "target"
        if basis_kind.startswith("converted"):
            assert retry["_project_admission"] == registry.project_admission_basis(
                "target", registry.get_reserved_project(tmp_path, "target"), frozen=True)
        elif basis_kind != "legacy":
            assert retry["_project_admission"] == admitted["_project_admission"]
        assert retry["workspace_root"] == "/synthetic/frozen-resource"
        assert not done


def _run_worker_main_once(monkeypatch, tmp_path, task, error):
    """Drive the real worker loop over one task whose agent raises ``error`` from handle_task,
    or runs ``error(runner, task)`` when it is a callable."""
    import ouroboros.agent as agent_module
    import ouroboros.config as config
    import ouroboros.extension_loader as extension_loader
    import ouroboros.platform_layer as platform_layer
    import ouroboros.process_custody as process_custody
    import ouroboros.utils as utils
    import supervisor.worker_process as WP

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    crashes, drives = [], []

    class Runner:
        owner_wait_callback = None

        def __init__(self, drive_root, budget_drive_root=None):
            self.env = agent_module.Env(repo_dir=repo, drive_root=Path(drive_root))
            self.budget_drive_root = Path(budget_drive_root or drive_root)
            drives.append(str(drive_root))

        def handle_task(self, task_arg):
            if isinstance(error, BaseException):
                raise error
            return error(self, task_arg)

    items = iter([task, {"type": "shutdown"}])
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "0")  # worker_main sets it; teardown restores the original
    monkeypatch.setattr(WP, "_bind_worker_repo_root", lambda *_a, **_k: None)
    monkeypatch.setattr(WP, "_prepare_worker_task_runtime", lambda: None)
    monkeypatch.setattr(WP, "_log_worker_crash", lambda *args: crashes.append(args[2]))
    monkeypatch.setattr(platform_layer, "create_new_session", lambda: None)
    monkeypatch.setattr(process_custody, "start_parent_lifeline", lambda **_k: None)
    monkeypatch.setattr(config, "initialize_runtime_mode_baseline", lambda: None)
    monkeypatch.setattr(config, "get_skills_repo_path", lambda: "")
    monkeypatch.setattr(extension_loader, "reload_all", lambda *_a, **_k: None)
    monkeypatch.setattr(agent_module, "make_agent",
                        lambda **kwargs: Runner(kwargs["drive_root"], kwargs.get("budget_drive_root")))
    monkeypatch.setattr(utils, "set_log_sink", lambda _sink: None)
    monkeypatch.setattr(utils, "get_git_info", lambda _repo: ("test", "sha"))

    WP.worker_main(0, SimpleNamespace(get=lambda: next(items)), SimpleNamespace(put=lambda _event: None),
                   str(repo), str(tmp_path))

    assert crashes == ["handle_task"]  # the worker still records the crash and exits as before
    return drives


@pytest.mark.parametrize("child", [False, True], ids=["pooled_canonical", "headless_child"])
def test_a_handle_task_exception_is_the_tasks_own_terminal_without_a_crash_retry(tmp_path, monkeypatch, child):
    """Owner decision 1A (11.10.2026): the caught exception ends the task; recovery finishes it."""
    import json
    import supervisor.queue as q
    import supervisor.workers as W
    from ouroboros.task_results import load_task_result
    from supervisor.terminal_delivery import project_terminal_result_event
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, events = _reserved_job(tmp_path, monkeypatch, exitcode=0, child=child)
    drives = _run_worker_main_once(monkeypatch, tmp_path, job["task"],
                                   ValueError("predecessor plan source unavailable"))
    assert drives[-1] == job["task"].get("drive_root", str(tmp_path))  # written by the agent that ran it
    toast = MagicMock()
    monkeypatch.setattr("supervisor.terminal_delivery.deliver_miss_lane_outcome", MagicMock())
    monkeypatch.setattr(W, "send_with_budget", toast)
    monkeypatch.setattr(q, "enqueue_task", MagicMock())

    recover_confirmed_dead_worker(job)

    current = load_task_result(tmp_path, job["task_id"], strict=True)
    assert (current["status"], current["reason_code"]) == ("failed", "task_exception")
    assert current["result"] == "⚠️ Error during processing: ValueError: predecessor plan source unavailable"
    assert current["terminal_time"]["source"] == "executor_terminal" and current["terminal_time"]["occurred_at"]
    assert "platform-level" not in json.dumps(current, ensure_ascii=False)
    # A host-written text: delivered as a System row, never as the model's answer.
    assert current.get("terminal_origin") == "host_notice"
    assert project_terminal_result_event(tmp_path, job["task"], job["task_id"], result_text=current["result"],
                                         terminal_origin=current.get("terminal_origin")).get("role") == "system"
    q.enqueue_task.assert_not_called()
    toast.assert_not_called()
    done = events.get_nowait()
    assert (done["type"], done["status"], done["reason_code"]) == ("task_done", "failed", "task_exception")


@pytest.mark.parametrize("child", [False, True], ids=["pooled_canonical", "headless_child"])
@pytest.mark.parametrize("refusal", ["grant_not_live", "source_removed"])
def test_a_refused_exact_continuation_load_parks_the_same_id_again(tmp_path, monkeypatch, refusal, child):
    """#1196 F4 stays the recovery's window: an owner Resume whose continuation the REAL
    loader refuses at startup is no task_exception terminal; the SAME id returns to its pause.
    A forked child's pause row stays in the budget root, never in the drive it runs on."""
    import queue as stdqueue
    import supervisor.queue as q
    from ouroboros import budget_pause
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
    from supervisor.worker_health import recover_confirmed_dead_worker
    from tests._budget_pause_exact_helpers import _install_queue, _parked

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    task, _row = _parked(tmp_path, monkeypatch, task_id="f4")
    assert queue.resume_budget_paused_task("f4")["ok"] is True
    granted = budget_pause.budget_pause_row(tmp_path, "f4")
    if refusal == "source_removed":
        (task_artifact_dir_path(tmp_path, "f4") / granted["source_ref"]["path"]).unlink()
    else:  # a newer writer revoked the grant before the worker could load it
        budget_pause.set_budget_pause(
            tmp_path, "f4", {**granted, "state": budget_pause.STATE_PAUSED,
                             "grant": {**granted["grant"], "revoked_at": time.time(), "revoke_reason": "newer"}},
            expected_pause_id=str(granted["pause_id"]), expected_state=budget_pause.STATE_RESUME_GRANTED,
            expected_grant_id=str(granted["grant"]["grant_id"]))
    if child:
        from ouroboros.headless import prepare_task_drive

        drive = prepare_task_drive(tmp_path, "f4", "empty")
        task.update(drive_root=str(drive), child_drive_root=str(drive))
    workers.PENDING.remove(task)
    workers.RUNNING["f4"] = {"task": task, "worker_id": 0, "attempt": 1}
    monkeypatch.setattr(workers, "WORKERS", {0: _make_worker(busy_task_id="f4", exitcode=0)})
    events = stdqueue.Queue()
    monkeypatch.setattr(workers, "get_event_q", lambda: events)
    for name in ("respawn_worker", "send_with_budget"):
        monkeypatch.setattr(workers, name, MagicMock())
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda **_k: True)
    monkeypatch.setattr(q, "enqueue_task", MagicMock())
    monkeypatch.setattr("supervisor.terminal_delivery.deliver_miss_lane_outcome", MagicMock())
    refused = []

    def load_continuation(runner, task_arg):  # the agent's order: the RUNNING row, then the exact load
        write_task_result(runner.env.drive_root, "f4", STATUS_RUNNING, result="running")
        ctx = SimpleNamespace(task_id="f4", task_attempt=1, drive_root=runner.env.drive_root,
                              budget_drive_root=runner.budget_drive_root,
                              budget_pause_resume=task_arg["_budget_pause_resume"])
        try:
            budget_pause.load_budget_pause(ctx)
        except Exception as exc:
            refused.append(type(exc).__name__)
            raise
        return []  # a load that did not refuse runs on, and the driver's crash assertion fails

    _run_worker_main_once(monkeypatch, tmp_path, task, load_continuation)
    workers.ensure_workers_healthy()
    recover_confirmed_dead_worker(q._reap_queue.get_nowait())

    done = []
    while not events.empty():
        done.append(events.get_nowait().get("type"))
    observed = {"refused": refused, "status": load_task_result(tmp_path, "f4", strict=True)["status"],
                "pause": budget_pause.budget_pause_row(tmp_path, "f4")["state"],
                "pending": [item["id"] for item in workers.PENDING], "running": list(workers.RUNNING),
                "task_done": "task_done" in done}
    assert observed == {"refused": ["ValueError" if refusal == "grant_not_live" else "FileNotFoundError"],
                        "status": "scheduled", "pause": budget_pause.STATE_PAUSED, "pending": ["f4"],
                        "running": [], "task_done": False}


@pytest.mark.parametrize("pause, returns", [
    ({"state": "paused", "task_attempt": 1}, True),
    ({"state": "resume_granted", "task_attempt": 1, "grant": {"grant_id": "g", "consumed_at": 1.0}}, False),
    ({"state": "resumed", "task_attempt": 1}, False),
    ({"state": "paused", "task_attempt": 2}, False),
    ({}, False),
])
def test_only_a_live_unconsumed_pause_of_this_attempt_keeps_the_death_for_recovery(pause, returns):
    from ouroboros.budget_pause import death_returns_to_pause

    assert death_returns_to_pause(pause, 1) is returns


@pytest.mark.parametrize("attempt, pause, records", [
    (1, {}, True),  # no pause: the exception is the task's own terminal
    (2, {"state": "paused", "task_attempt": 2}, False),  # this attempt's live pause: recovery parks it
    (1, OSError("pause row unreadable"), False),  # unknown pause evidence: recovery decides, as before
])
def test_the_worker_reads_this_attempts_pause_from_the_budget_root_before_recording(tmp_path, monkeypatch,
                                                                                    attempt, pause, records):
    import ouroboros.agent as agent_module
    import ouroboros.budget_pause as budget_pause
    import supervisor.worker_process as WP

    reads, written = [], []

    def pause_row(root, task_id):
        reads.append((str(root), task_id))
        if isinstance(pause, BaseException):
            raise pause
        return dict(pause)

    monkeypatch.setattr(budget_pause, "budget_pause_row", pause_row)
    monkeypatch.setattr(agent_module, "_task_exception_terminal", lambda _env, task, *_a, **_k: written.append(task["id"]))
    runner = SimpleNamespace(env=SimpleNamespace(drive_path=lambda name: tmp_path / "child" / name))

    WP._record_task_exception(runner, {"id": "t", "_attempt": attempt}, ValueError("boom"), tmp_path)

    assert reads == [(str(tmp_path), "t")]  # the pool drive is the budget root unless the task names its own
    assert written == (["t"] if records else [])


def test_a_late_handle_task_exception_never_replaces_an_earlier_terminal(tmp_path, monkeypatch):
    import supervisor.queue as q
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, events = _reserved_job(tmp_path, monkeypatch, exitcode=0)
    write_task_result(tmp_path, job["task_id"], "failed", result="ORIGINAL", reason_code="original_reason")
    before = load_task_result(tmp_path, job["task_id"], strict=True)

    _run_worker_main_once(monkeypatch, tmp_path, job["task"], RuntimeError("late cleanup failure"))

    assert load_task_result(tmp_path, job["task_id"], strict=True) == before
    monkeypatch.setattr("supervisor.terminal_delivery.deliver_miss_lane_outcome", MagicMock())
    monkeypatch.setattr(q, "enqueue_task", MagicMock())
    recover_confirmed_dead_worker(job)
    current = load_task_result(tmp_path, job["task_id"], strict=True)
    assert (current["result"], current["reason_code"]) == ("ORIGINAL", "original_reason")
    q.enqueue_task.assert_not_called()
    assert events.get_nowait()["reason_code"] == "original_reason"


def test_an_unrecorded_handle_task_exception_keeps_the_single_crash_retry(tmp_path, monkeypatch):
    """No terminal (here its write fails) means today's crash policy: one retry."""
    import ouroboros.task_results as task_results
    import supervisor.queue as q
    from supervisor.worker_health import recover_confirmed_dead_worker

    job, _events = _reserved_job(tmp_path, monkeypatch, exitcode=0)
    with monkeypatch.context() as broken:
        broken.setattr(task_results, "write_task_result",
                       lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
        _run_worker_main_once(monkeypatch, tmp_path, job["task"], ValueError("boom"))
    assert task_results.load_task_result(tmp_path, job["task_id"]) is None
    retried = []
    monkeypatch.setattr(q, "enqueue_task", lambda task, front=False: retried.append(dict(task)) or task)

    recover_confirmed_dead_worker(job)

    assert [task["_attempt"] for task in retried] == [2]
    assert task_results.load_task_result(tmp_path, job["task_id"])["status"] == "interrupted"


def test_a_dead_carriers_turn_retries_on_its_lane_never_through_pending(tmp_path, monkeypatch):
    """Review M4: a conversation-lane turn whose carrier exited with no terminal retries once (same id, its next
    attempt) on the LANE, whatever the project pool says: never PENDING, where only project slots are assigned
    and the project crash fuse would hold it. Here the project pool is fenced, and the lane's idle carrier takes
    the retry at once, with the turn's own lane facts."""
    import supervisor.queue as q
    import supervisor.workers as W
    from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
    from supervisor.worker_health import recover_confirmed_dead_worker
    from tests.conversation_carrier_support import dispatched_tasks, install_lane_spare

    task = {**_make_task("lane-turn"), "_is_direct_chat": True,
            "_execution_owner": {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}}
    carrier = _make_worker(busy_task_id=task["id"], exitcode=0)  # it exited: no signal, no terminal written
    carrier.lane = LANE_CONVERSATION
    W.WORKERS[0] = carrier
    W.RUNNING[task["id"]] = {"task": task, "attempt": 1, "worker_id": 0, "lane": LANE_CONVERSATION,
                             "client_message_id": "cm-1", "reply_expected": True, "started_at": time.time()}
    spare = install_lane_spare(W, 90)
    monkeypatch.setattr(W, "QUEUE_MAX_RETRIES", 1)
    monkeypatch.setattr(W, "_WORKER_POOL_DISABLED_REASON", "worker_readiness_exhausted")  # the project fence is down
    monkeypatch.setattr(W, "respawn_worker", MagicMock())
    enqueued = []
    monkeypatch.setattr(q, "enqueue_task", lambda queued, front=False: enqueued.append(queued) or queued)
    W.ensure_workers_healthy()
    job = q._reap_queue.get_nowait()
    with patch("ouroboros.headless.prepare_terminal_task_files", return_value={"error": "", "terminal_source_present": False}):
        recover_confirmed_dead_worker(job)
    assert enqueued == [] and W.PENDING == []
    retry = W.RUNNING[task["id"]]
    assert retry["lane"] == LANE_CONVERSATION and retry["attempt"] == 2 and retry["task"]["_attempt"] == 2
    assert retry["client_message_id"] == "cm-1" and retry["reply_expected"] is True
    assert retry["worker_id"] == spare.wid and spare.busy_task_id == task["id"]
    assert [queued["id"] for queued in dispatched_tasks(spare)] == [task["id"]]


def test_a_signal_killed_presence_carrier_settles_its_host_waiter_and_reads_as_a_settled_loss(tmp_path, monkeypatch):
    """Review C2 on the crash path: a Presence carrier killed by a signal at its LAST attempt (the one retry
    spent) writes no terminal and sends no settlement. Crash recovery records the host's terminal as a reconciled running attempt; its task_done
    settles the Host waiter with a typed retry (the reservation returns), and the resend's replay is the retained
    unknown-outcome refusal: a lost local stack is never projected as a completed, silent answer."""
    import queue as stdqueue

    import supervisor.queue as q
    import supervisor.workers as W
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.presence_executions import PresenceTurnExecutions, PresenceTurnInterrupted
    from ouroboros.presence_runner import presence_turn_replay
    from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.utils import append_jsonl
    from supervisor import events
    from supervisor.worker_health import recover_confirmed_dead_worker

    turn_id = "presence-turn"
    metadata = {"source": "presence", "presence": {"binding_id": "b", "event": {"source_event_id": "e1",
                                                                                 "conversation_key": "k"}}}
    task = {**_make_task(turn_id, attempt=2), "type": "presence", "_is_direct_chat": True, "_presence_turn": True,
            "metadata": metadata, "_execution_owner": {"kind": "presence", "lane": LANE_CONVERSATION,
                                                       "executor": EXECUTOR_WORKER}}
    write_task_result(tmp_path, turn_id, "running", metadata=metadata, _is_direct_chat=True)  # the carrier's barrier
    carrier = _make_worker(busy_task_id=turn_id, exitcode=-9)
    carrier.lane = LANE_CONVERSATION
    W.WORKERS[0] = carrier
    W.RUNNING[turn_id] = {"task": task, "attempt": 2, "worker_id": 0, "lane": LANE_CONVERSATION,
                          "started_at": time.time()}
    monkeypatch.setattr(W, "QUEUE_MAX_RETRIES", 1)
    registry, released = PresenceTurnExecutions(), []
    waiter, _ = registry.start_or_dispatch(turn_id, reserve=lambda: True, release=lambda: released.append(1),
                                           dispatch=lambda execution: None)
    bus = stdqueue.Queue()
    monkeypatch.setattr(W, "get_event_q", lambda: bus)
    monkeypatch.setattr(W, "respawn_worker", MagicMock())
    monkeypatch.setattr(W, "send_with_budget", lambda *a, **k: None)
    W.ensure_workers_healthy()
    job = q._reap_queue.get_nowait()
    with patch("ouroboros.headless.prepare_terminal_task_files", return_value={"error": "", "terminal_source_present": False}):
        recover_confirmed_dead_worker(job)
    row = load_task_result(tmp_path, turn_id)
    assert row["status"] == "failed"
    done = []
    while not bus.empty():
        frame = bus.get_nowait()
        if frame.get("type") == "task_done":
            done.append(frame)
    assert [frame["task_id"] for frame in done] == [turn_id] and turn_id not in W.RUNNING
    events._handle_task_done(done[0], SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=W.RUNNING, WORKERS={},
                                                      append_jsonl=append_jsonl,
                                                      persist_queue_snapshot=lambda reason="": None))
    with pytest.raises(PresenceTurnInterrupted):
        waiter.result.result(timeout=1)
    assert released == [1] and registry.live() == []
    with pytest.raises(PresenceTurnError) as refused:
        presence_turn_replay(tmp_path, turn_id, "k")
    assert refused.value.code == "presence_attempt_outcome_unknown" and refused.value.disposition == "retained"
    assert row.get("status_reconciled_from") == "running"  # the reconciled origin the Presence rule reads
