"""Regression tests for worker crash retry loop fixes.

Covers:
- Retry limit enforced (attempt > QUEUE_MAX_RETRIES → STATUS_FAILED, no requeue)
- Attempt counter incremented before requeue
- Already-completed task is not requeued after crash
- Crash storm detection works (no grace reset on respawn)
- Terminal event emitted when retry limit exhausted
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_worker_crash_state():
    """Crash history is process-global and must not leak between serial tests."""
    import supervisor.workers as workers

    workers.CRASH_TS.clear()
    workers._WORKER_POOL_DISABLED_REASON = ""
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
    w = MagicMock()
    w.wid = wid
    w.proc = proc
    w.busy_task_id = busy_task_id
    # Real Worker defaults reaping=False; without this the MagicMock auto-attr is truthy
    # and the new crash-detector reaping guard would skip the worker.
    w.reaping = False
    return w


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
        W.ensure_workers_healthy()

    assert len(enqueued) == 1, "Task should be requeued once"
    assert enqueued[0]["_attempt"] == 2, f"Expected _attempt=2, got {enqueued[0].get('_attempt')}"
    assert not service_dir.exists(), "Child-drive service logs should be archived after worker death"


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
        "supervisor.worker_health._emit_task_done_terminal",
        side_effect=lambda *args, **kwargs: terminal.append((args, kwargs)),
    ), patch(
        "ouroboros.task_results.load_task_result", return_value=None
    ), patch(
        "ouroboros.task_results.write_task_result", side_effect=fake_write
    ):
        W.ensure_workers_healthy()

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
        W.ensure_workers_healthy()

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
        W.ensure_workers_healthy()

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
        W.ensure_workers_healthy()

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
         patch("supervisor.workers.get_event_q", return_value=fake_queue):
        W.respawn_worker(0)

    assert W._LAST_SPAWN_TIME == original_time, (
        f"_LAST_SPAWN_TIME should NOT be reset by respawn_worker, "
        f"but changed from {original_time} to {W._LAST_SPAWN_TIME}"
    )


# ---------------------------------------------------------------------------
# Test: crash storm detection accumulates (grace not reset by respawn)
# ---------------------------------------------------------------------------

def test_crash_storm_detection_accumulates(tmp_path):
    """After multiple rapid crashes, CRASH_TS should accumulate >= 3 entries
    within 60s when _LAST_SPAWN_TIME is not reset by respawn_worker."""
    import supervisor.workers as W

    W.DRIVE_ROOT = tmp_path
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    W.QUEUE_MAX_RETRIES = 0  # Immediately fail, no retry
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
            try:
                W.ensure_workers_healthy()
            except Exception:
                pass

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
        existing_result = {"status": terminal_status, "result": "done"}

        with patch.object(sq, "enqueue_task", side_effect=lambda t, front=False: enqueued.append(dict(t))), \
             patch.object(sq, "persist_queue_snapshot", MagicMock()), \
             patch("supervisor.workers.respawn_worker"), \
             patch("supervisor.workers.send_with_budget"), \
             patch("supervisor.workers.load_state", return_value={}), \
             patch("ouroboros.task_results.load_task_result", return_value=existing_result), \
             patch("ouroboros.task_results.write_task_result"):
            W.ensure_workers_healthy()

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
        W.ensure_workers_healthy()

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
        W.ensure_workers_healthy()

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
        W.ensure_workers_healthy()

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    task_done_events = [e for e in events if e.get("type") == "task_done"]
    assert len(task_done_events) >= 1, (
        f"Expected task_done terminal event for deep_self_review crash, got: {events}"
    )
    assert task_done_events[0]["task_id"] == "dsr01"
    assert task_done_events[0]["status"] == "failed"
