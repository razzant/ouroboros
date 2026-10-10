"""Regression tests for worker crash retry loop fixes.

Covers:
- Retry limit enforced (attempt > QUEUE_MAX_RETRIES → STATUS_FAILED, no requeue)
- Attempt counter incremented before requeue
- Already-completed task is not requeued after crash
- Crash storm detection works (no grace reset on respawn)
- Terminal event emitted when retry limit exhausted
"""

from __future__ import annotations

import sys
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
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(q, "RUNNING", workers.RUNNING)
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0)
    monkeypatch.setattr(workers, "CRASH_TS", [])
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
    workers.CRASH_TS.clear()
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
                elif job["kind"] == "worker_crash_storm":
                    # The root owns the production deferred-storm dispatcher.
                    with W._WORKER_LIFECYCLE_LOCK, q._queue_lock:
                        same = tuple(W.WORKERS.items()) == job["workers"]
                    if same:
                        W.kill_workers(disable_reason="worker_crash_storm")
                        W.CRASH_TS.clear()
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
# Test: crash storm detection accumulates (grace not reset by respawn)
# ---------------------------------------------------------------------------

def test_crash_storm_detection_accumulates(tmp_path, monkeypatch):
    """After multiple rapid crashes, CRASH_TS should accumulate >= 3 entries
    within 60s when _LAST_SPAWN_TIME is not reset by respawn_worker."""
    import supervisor.workers as W

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    # Restored after the test: a bare assignment leaked 0 into sibling retry
    # scenarios on the same xdist worker (they rely on the default of 1).
    monkeypatch.setattr(W, 'QUEUE_MAX_RETRIES', 0)  # Immediately fail, no retry
    W._LAST_SPAWN_TIME = 0  # Grace already elapsed
    W.CRASH_TS = []
    notices = []

    # Simulate 3 sequential busy crashes
    for i in range(3):
        task = _make_task(task_id=f"storm{i}", attempt=1)
        worker = _make_worker(wid=i, busy_task_id=f"storm{i}", exitcode=-11)
        W.WORKERS = {i: worker}
        W.RUNNING = {
            f"storm{i}": {
                "task": task,
                "started_at": time.time() - 1,
                "last_heartbeat_at": time.time() - 1,
                "attempt": 1,
            }
        }

        import supervisor.queue as sq

        with patch.object(sq, "enqueue_task", MagicMock()), \
             patch.object(sq, "persist_queue_snapshot", MagicMock()), \
             patch.object(sq, "drain_all_pending", return_value=[]), \
             patch("supervisor.workers.respawn_worker"), \
             patch("ouroboros.task_results.load_task_result", return_value=None), \
             patch("ouroboros.task_results.write_task_result"), \
             patch("supervisor.workers.kill_workers"), \
             patch(
                 "supervisor.workers.send_with_budget",
                 side_effect=lambda *args, **kwargs: notices.append((args, kwargs)),
             ), \
             patch("supervisor.workers.load_state", return_value={"owner_chat_id": 1}), \
             patch("supervisor.workers.get_event_q", return_value=MagicMock()), \
             patch("supervisor.message_bus.get_bridge", return_value=None):
            # Only run health check — don't call kill_workers directly
            _run_health_and_reap()

    # After 3 busy crashes, CRASH_TS should have accumulated entries OR
    # storm detection fired (which clears CRASH_TS after kill_workers)
    # The important thing: no infinite requeue happened and the system
    # attempted to detect the storm.
    # We verify CRASH_TS was populated at some point (it may have been cleared
    # by storm detection — that's also correct behavior)
    # The key invariant: _LAST_SPAWN_TIME wasn't reset between iterations
    assert W._LAST_SPAWN_TIME == 0, (
        "respawn_worker should not have reset _LAST_SPAWN_TIME during crash loop"
    )
    storm_notices = [
        (args, kwargs) for args, kwargs in notices
        if kwargs.get("progress_meta", {}).get("task_incident") == "worker_crash_storm"
    ]
    assert len(storm_notices) == 1
    assert storm_notices[0][1]["is_progress"] is True
    assert storm_notices[0][1]["progress_meta"]["toast_once"].startswith(
        "worker-crash-storm:"
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


def test_signal_crash_is_terminal_no_retry(tmp_path):
    """A worker killed by a signal (negative exitcode, e.g. SIGSEGV -11) is a
    deterministic infrastructure crash: it must be marked failed with no retry
    for ANY task type, and emit a task_done so the UI card resolves."""
    import supervisor.workers as W
    import queue as _queue

    task = _make_task(task_id="sig01", attempt=1, chat_id=7)  # ordinary task type
    evolution_tx = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "sig01"}
    task["metadata"] = {"evolution_transaction": evolution_tx}
    worker = _make_worker(busy_task_id="sig01", exitcode=-11)

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.QUEUE_MAX_RETRIES = 1
    W.WORKERS = {0: worker}
    W.RUNNING = {
        "sig01": {
            "task": task,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 1,
        }
    }
    W._LAST_SPAWN_TIME = 0
    W.CRASH_TS = []

    written = {}
    enqueued = []
    event_q = _queue.Queue()

    def fake_write(drive, task_id, status, result="", **kw):
        written[task_id] = {"status": status, **kw}

    import supervisor.queue as sq

    incident_notice = MagicMock()
    with patch.object(sq, "enqueue_task", side_effect=lambda t, front=False: enqueued.append(dict(t))), \
         patch.object(sq, "persist_queue_snapshot", MagicMock()), \
         patch("supervisor.workers.respawn_worker"), \
         patch("supervisor.workers.load_state", return_value={}), \
         patch("ouroboros.task_results.load_task_result", return_value=None), \
         patch("ouroboros.task_results.write_task_result", side_effect=fake_write), \
         patch("supervisor.workers.get_event_q", return_value=event_q), \
         patch("supervisor.workers.send_with_budget", incident_notice), \
         patch("supervisor.message_bus.get_bridge", return_value=None):
        _run_health_and_reap()

    assert len(enqueued) == 0, "Signal crash must NOT be retried"
    assert written.get("sig01", {}).get("status") == "failed"
    assert written["sig01"].get("crash_signal") == 11
    drained = []
    while not event_q.empty():
        drained.append(event_q.get_nowait())
    terminal = next(e for e in drained if e.get("type") == "task_done" and e.get("task_id") == "sig01")
    assert terminal["metadata"]["evolution_transaction"] == evolution_tx
    incident_notice.assert_called_once()
    notice_args, notice_kwargs = incident_notice.call_args
    assert notice_args[0] == 7
    assert notice_kwargs["is_progress"] is True
    assert notice_kwargs["task_id"] == "sig01"
    assert notice_kwargs["progress_meta"] == {
        "task_incident": "worker_crash_signal",
        "toast_once": "sig01:worker_crash_signal:1",
    }


def test_crash_toast_for_a_bound_task_goes_to_its_project_chat(tmp_path):
    """The crash toast is a DIRECT send, so the durable project binding has to
    win over the chat the task was born in: a task converted into a project
    mid-run would otherwise be told about its crash in Main."""
    import supervisor.workers as W
    import queue as _queue

    from ouroboros.projects_registry import bind_task_to_project

    bind_task_to_project(tmp_path, "sig02", "crash-proj", 5151, origin={"absent": "system"})
    task = _make_task(task_id="sig02", attempt=1, chat_id=1)  # born in Main
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
            "attempt": 1,
        }
    }
    W._LAST_SPAWN_TIME = 0
    W.CRASH_TS = []

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
    """deep_self_review crash must emit task_done so the UI live card closes."""
    import supervisor.workers as W
    import queue as _queue

    task = _make_task(task_id="dsr01", attempt=1, chat_id=7)
    task["type"] = "deep_self_review"
    worker = _make_worker(busy_task_id="dsr01", exitcode=-11)

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.WORKERS = {0: worker}
    W.RUNNING = {
        "dsr01": {
            "task": task,
            "started_at": time.time() - 5,
            "last_heartbeat_at": time.time() - 5,
            "attempt": 1,
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


@pytest.mark.parametrize('change', ['worker', 'meta', 'attempt', 'root', 'worker_binding'])
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


def test_storm_jobs_are_after_all_dead_jobs_and_suppress_respawns(tmp_path, monkeypatch):
    import supervisor.queue as q
    import supervisor.workers as W

    W.WORKERS = {i: _make_worker(wid=i, busy_task_id=f'storm-{i}') for i in range(3)}
    W.RUNNING = q.RUNNING = {f'storm-{i}': {'task': _make_task(f'storm-{i}'), 'attempt': 1}
                            for i in range(3)}
    monkeypatch.setattr(W, 'kill_workers', MagicMock())
    monkeypatch.setattr(W, 'load_state', lambda: {})
    W.ensure_workers_healthy()
    jobs = [q._reap_queue.get_nowait() for _ in range(4)]
    assert [job['kind'] for job in jobs] == ['confirmed_dead_worker'] * 3 + ['worker_crash_storm']
    assert all(job['skip_respawn'] for job in jobs[:3])
    assert jobs[-1]['workers'] == tuple(W.WORKERS.items())
    assert len(W.RUNNING) == 3
    W.kill_workers.assert_not_called()


def test_malformed_child_is_unknown_not_a_paid_retry(tmp_path, monkeypatch):
    import supervisor.queue as q
    import supervisor.workers as W
    from supervisor.worker_health import recover_confirmed_dead_worker
    from supervisor.task_reaper import TerminalFileRecoveryPending

    job, events = _reserved_job(tmp_path, monkeypatch, child=True)
    path = Path(job['task']['drive_root']) / 'task_results' / (job['task_id'] + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{malformed preserved source', encoding='utf-8')
    monkeypatch.setattr(q, 'enqueue_task', MagicMock())
    with pytest.raises(TerminalFileRecoveryPending):
        recover_confirmed_dead_worker(job)
    assert path.read_text(encoding='utf-8') == '{malformed preserved source'
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


def test_storm_honors_saved_children_before_pool_stop(tmp_path, monkeypatch):
    import queue as stdqueue
    import supervisor.queue as q
    import supervisor.workers as W
    from ouroboros.task_results import write_task_result, load_task_result
    from supervisor.worker_health import recover_confirmed_dead_worker
    from supervisor.task_reaper import _stop_crashed_worker_pool, TerminalFileRecoveryPending

    W.WORKERS = {i: _make_worker(wid=i, busy_task_id=f'storm-done-{i}') for i in range(3)}
    W.RUNNING = q.RUNNING = {}
    for i, worker in W.WORKERS.items():
        task = _make_task(worker.busy_task_id)
        child = tmp_path / ('child-' + str(i))
        task['drive_root'] = str(child)
        W.RUNNING[task['id']] = {'task': task, 'attempt': 1, 'worker_id': i}
        write_task_result(child, task['id'], 'completed', result='saved-' + str(i))
    events = stdqueue.Queue()
    monkeypatch.setattr(W, 'get_event_q', lambda: events)
    monkeypatch.setattr(W, 'kill_workers', MagicMock(return_value=True))
    monkeypatch.setattr(W, 'respawn_worker', MagicMock())
    monkeypatch.setattr(W, 'load_state', lambda: {})
    monkeypatch.setattr(q, 'enqueue_task', MagicMock())
    monkeypatch.setattr('supervisor.terminal_delivery.deliver_miss_lane_outcome', lambda *_a, **_k: None)
    W.ensure_workers_healthy()
    jobs = [q._reap_queue.get_nowait() for _ in range(4)]
    with pytest.raises(TerminalFileRecoveryPending):
        _stop_crashed_worker_pool(jobs[-1])
    W.kill_workers.assert_not_called()
    for job in jobs[:-1]:
        recover_confirmed_dead_worker(job)
        done = events.get_nowait()
        assert done['_files_prepared_attempt'] == 1 and done['worker_id'] == job['worker_id']
        assert load_task_result(tmp_path, done['task_id'])['result'] == 'saved-' + str(job['worker_id'])
        # Model the normal event owner's release, after validating the real frame.
        W.RUNNING.pop(done['task_id'])
    _stop_crashed_worker_pool(jobs[-1])
    W.kill_workers.assert_called_once_with(disable_reason='worker_crash_storm')
    W.respawn_worker.assert_not_called()
    q.enqueue_task.assert_not_called()


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

        response = asyncio.run(api_project_from_task(_request(tmp_path, {
            "task_id": "crash", "id": "target", "name": "Target"})))
        assert response.status_code == 200
        assert admitted["_project_admission"] == registry.project_admission_basis(
            "target", registry.get_reserved_project(tmp_path, "target"), frozen=True)
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
        if basis_kind != "legacy":
            assert retry["_project_admission"] == admitted["_project_admission"]
        assert retry["workspace_root"] == "/synthetic/frozen-resource"
        assert not done


# ---------------------------------------------------------------------------
# Test: a confirmed worker death settles what its browser bridges left
# ---------------------------------------------------------------------------

def _orphaned_bridge_browser(root, task_id):
    """What a dead worker's bridge leaves: its server gone, its detached browser live."""
    import os
    import subprocess
    import sys
    from ouroboros import mcp_task_sessions
    from ouroboros.process_containment import CONTAINMENT_ENV_PREFIX

    marker = f"{CONTAINMENT_ENV_PREFIX}{mcp_task_sessions._scope(root, task_id)}deadbeef"
    spawn = ("import subprocess, sys; print(subprocess.Popen([sys.executable, '-c', 'import time; "
             "time.sleep(120)'], start_new_session=True, stdin=subprocess.DEVNULL, "
             "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).pid)")
    server = subprocess.run([sys.executable, "-c", spawn], env={**os.environ, marker: "1"},
                            capture_output=True, text=True, timeout=30, check=True)
    return marker, int(server.stdout)


def _bridge_retry_ctx(root, task_id):
    return SimpleNamespace(task_id=task_id, task_attempt=2, task_lifecycle_bound=True, drive_root=root,
                           task_metadata={}, task_contract={}, messages=[])


@pytest.mark.serial  # Real marked processes (tests/conftest lane policy).
@pytest.mark.skipif(sys.platform == "win32", reason="Detached environment-marker bridge custody is POSIX-only")
@pytest.mark.parametrize("closure", ["confirmed", "unconfirmed", "scan_raises"])
def test_confirmed_death_settles_bridge_scope_before_retry(tmp_path, monkeypatch, caplog, closure):
    import os
    import signal
    import supervisor.queue as q
    import supervisor.workers as W
    from ouroboros import mcp_client, mcp_task_sessions
    from ouroboros.process_containment import pid_is_zombie, pids_with_env_marker
    from supervisor.worker_health import recover_confirmed_dead_worker
    from tests.test_mcp_task_browser_bridge import _entry, _server

    for registry in ("_sessions", "_ended_attempts", "_failed_opens"):
        monkeypatch.setattr(mcp_task_sessions, registry, type(getattr(mcp_task_sessions, registry))())
    job, _events = _reserved_job(tmp_path, monkeypatch)
    task_id = job["task_id"]
    marker, orphan = _orphaned_bridge_browser(tmp_path, task_id)

    def live():
        return [pid for pid in pids_with_env_marker(marker) or [] if not pid_is_zombie(pid)]

    order = []
    actual_stop_scope = mcp_task_sessions.stop_scope

    def stop_scope(root, scoped_task, **kwargs):
        assert not q._queue_lock._is_owned() and task_id in W.RUNNING
        order.append(("settle", root, scoped_task))
        if closure == "scan_raises":
            raise OSError("process table unreadable")
        if closure == "unconfirmed":
            return {"closure": "unconfirmed: process table unreadable"}
        return actual_stop_scope(root, scoped_task, **kwargs)

    monkeypatch.setattr(mcp_task_sessions, "stop_scope", stop_scope)
    monkeypatch.setattr(q, "enqueue_task", lambda task, front=False: order.append(("retry", task["_attempt"])) or task)
    retry = _bridge_retry_ctx(tmp_path, task_id)
    cfg = mcp_client.normalize_server_config(_entry(_server(tmp_path)))
    try:
        assert live() == [orphan]
        with caplog.at_level("ERROR", logger="ouroboros.mcp_task_sessions"):
            recover_confirmed_dead_worker(job)
        # The death stands whatever the scan says: retried once, slot released.
        assert order == [("settle", tmp_path, task_id), ("retry", 2)]
        assert task_id not in W.RUNNING
        W.respawn_worker.assert_called_once_with(0)
        if closure == "confirmed":
            assert not live()
            pytest.importorskip("mcp")
            assert "browser_tabs" in {tool["name"] for tool in mcp_task_sessions.discover(cfg, retry, 15)}
        else:
            assert "Browser bridge closure unconfirmed" in caplog.text
            assert live() == [orphan]  # Never reported closed: the retry may not reopen.
            with pytest.raises(RuntimeError, match="earlier bridge processes are still live"):
                mcp_task_sessions.discover(cfg, retry, 15)
    finally:
        mcp_task_sessions.stop_task(retry)
        if live():
            os.kill(orphan, signal.SIGKILL)
