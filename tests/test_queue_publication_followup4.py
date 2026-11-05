"""Queue publication, post-merge follow-up 4 (RV_MAIN3 R2/R3-A, R3-B, R5, R6).

A failed publication leaves the bundle on disk unknown (torn or old) with its certificate
revoked. From then on the publisher owes a fresh capture and a COMPLETE rewrite of every
document under a new generation, whatever the bytes say against the last cache, and the
failure clears only when that write lands; a quiet tick retries it, so one transient failure
whose change reverts never wedges the loop (R2/R3-A). An already-running quiet refresh never
resurrects a certificate a failed barrier revoked (R3-B). The Pause latch's barrier outside Q
keeps its typed refusal (R5). A kept saved queue is skipped before any row is copied (R6).
The reviewers' probes live here as the permanent pins.
"""

import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import queue_observation, task_status
from supervisor import queue, queue_publication, queue_snapshot


@pytest.fixture
def published(tmp_path, monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "INITIALIZED", True)
    for name, value in (("PENDING", []), ("RUNNING", {}), ("BUDGET_ROOT_FENCES", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    monkeypatch.setattr(queue_publication, "utc_now_iso", lambda: datetime.fromtimestamp(now[0], timezone.utc).isoformat())
    monkeypatch.setattr(queue_publication, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _s: None))
    for module in (queue_observation, task_status):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(queue_observation, "_sleep", lambda _sec: None)
    queue_publication.reset()
    yield tmp_path, now
    queue_publication.reset()


def _read(root, name):
    return json.loads((root / "state" / name).read_text(encoding="utf-8"))


def _task(task_id, **extra):
    return {"id": task_id, "type": "task", "chat_id": 1, "root_task_id": task_id, "text": f"task {task_id}", **extra}


def _seed(published):
    root, now = published
    queue.RUNNING["r"] = {"task": _task("r", model_waits={}), "attempt": 1, "worker_id": 3,
                          "started_at": now[0] - 100, "last_heartbeat_at": now[0] - 1,
                          "budget_paused_sec": 0.0, "soft_sent": False}
    assert queue.persist_queue_snapshot("seed") is True
    queue_publication.tick_publisher_started()
    return root, now


# ---- R3-B: an in-flight quiet refresh never resurrects a certificate a failed barrier revoked ------

def test_an_inflight_quiet_refresh_cannot_recertify_after_a_failed_barrier(published, monkeypatch):
    root, now = _seed(published)
    original = _read(root, "queue.alive")["generation"]
    now[0] += queue_publication.QUEUE_ALIVE_REFRESH_SEC * 2
    reached, proceed = threading.Event(), threading.Event()
    due = queue_publication._alive_due

    def paused_due(heartbeats):
        if threading.current_thread().name == "quiet-tick":
            reached.set()
            assert proceed.wait(5)
        return due(heartbeats)

    monkeypatch.setattr(queue_publication, "_alive_due", paused_due)
    answers, errors = [], []

    def refresh():
        try:
            answers.append(queue_publication.publish_queue_changes())
        except BaseException as exc:  # noqa: BLE001 - the thread's failure is the assertion's evidence
            errors.append(repr(exc))

    thread = threading.Thread(target=refresh, name="quiet-tick")
    thread.start()
    try:
        assert reached.wait(5)
        with queue._queue_lock:
            queue.RUNNING["r"]["owner_wait"] = {"kind": "review", "state": "waiting"}
        with monkeypatch.context() as fail:
            fail.setattr(queue, "atomic_write_text", lambda *_a, **_kw: False)
            assert queue.publish_queue_snapshot("barrier") is False
        assert not (root / "state/queue.alive").exists()
    finally:
        proceed.set()
        thread.join(5)
    assert not thread.is_alive() and errors == []
    assert not (root / "state/queue.alive").exists(), "the revoked certificate stays revoked"
    assert answers == [False], "the refresh answers what it did: nothing certified"
    # The next tick repairs the bundle and certifies a new generation.
    assert queue_publication.publish_queue_changes() is True
    assert _read(root, "queue.alive")["generation"] != original
    assert _read(root, "queue_snapshot.json")["running"][0]["owner_wait"] == {"kind": "review", "state": "waiting"}


# ---- R5: the Pause latch's barrier outside Q keeps its typed refusal ------------------------------

def test_a_raising_pause_barrier_answers_the_typed_refusal_and_runs_outside_q(tmp_path, monkeypatch):
    from supervisor import owner_pause_control as pause
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_pause import _live_tree

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    calls = []
    real_barrier, real_wake = q.publish_queue_snapshot, pause._wake_members

    def barrier(*a, **kw):
        calls.append(("barrier", q._queue_lock._is_owned()))
        return real_barrier(*a, **kw)

    def wake(*a, **kw):
        calls.append(("wake", q._queue_lock._is_owned()))
        return real_wake(*a, **kw)

    monkeypatch.setattr(q, "publish_queue_snapshot", barrier)
    monkeypatch.setattr(pause, "_wake_members", wake)
    answer = pause.request_owner_pause("root-1", request_id="order")
    assert answer["ok"] and answer["snapshot_persisted"]
    assert calls[0] == ("barrier", False) and calls[1] == ("wake", False), "published outside Q, before the wake"

    def raising(*_a, **_kw):
        raise OSError("fixture: the encoder failed")

    monkeypatch.setattr(q, "publish_queue_snapshot", raising)
    answer = pause.request_owner_pause("root-1", request_id="raise")
    assert answer == {"ok": False, "error": "pause_record_unwritable", "detail": "fixture: the encoder failed"}


# ---- R6: a kept saved queue is skipped before any row is copied -----------------------------------

def test_a_kept_saved_queue_costs_no_capture_per_tick(published, monkeypatch):
    root, _now = published
    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "state/queue_snapshot.json").write_text(
        json.dumps({"pending": [{"id": "saved", "task": {"id": "saved"}}], "running": []}), encoding="utf-8")
    queue_publication.tick_publisher_started()
    queue.persist_queue_snapshot("startup")  # a mark; the saved queue is not in this process's custody
    assert queue_publication.publish_queue_changes() is None, "kept: the first tick decides it"
    assert queue_publication._STATE["kept"] is not None
    captures, censuses = [], []
    capture, census = queue_publication._capture, queue_publication._census
    monkeypatch.setattr(queue_publication, "_capture", lambda *a, **k: captures.append(1) or capture(*a, **k))
    monkeypatch.setattr(queue_publication, "_census", lambda: censuses.append(1) or census())
    for _ in range(3):
        assert queue_publication.publish_queue_changes() is None
    assert captures == [] and censuses == [], "a kept saved queue is known from the memo, before any copy or census"
    # A change of custody (the restore read it, or work was admitted) decides again.
    queue.PENDING.append(_task("admitted"))
    queue.persist_queue_snapshot("admit")
    assert queue_publication.publish_queue_changes() is True and captures == [1]
