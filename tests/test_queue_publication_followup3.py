"""Queue publication, review follow-up 3 (RV_MAIN2 N1, N9, N10, N11 and the census NIT).

N1: the published pool's availability counts every started process (project slots and lane
carriers) and a disabled pool still refuses; pinned over a REAL published document at idle,
unstubbed. N9: a failed publication is retried by the next tick and ``queue.alive`` never
re-certifies the stale document meanwhile. N10: a newer capture always wins over an older one,
whatever their revisions. N11: the retry-custody publication repeats only when the claim's
facts change. NIT: the census carries only the model waits that are still waiting.
"""

import json
import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import queue_observation, task_status
from supervisor import direct_roots, queue, queue_publication, queue_snapshot


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
    monkeypatch.setattr(direct_roots, "collect_direct_roots", lambda: {"roots": [], "incomplete": False})
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


def _slot(wid, lane, busy=None):
    return types.SimpleNamespace(wid=wid, lane=lane, busy_task_id=busy, active_capacity=True, resume_hold="",
                                 reaping=False, readiness_exhausted=False, process_birth="b",
                                 ready_identity=(100 + wid, "b"), proc=types.SimpleNamespace(pid=100 + wid, is_alive=lambda: True))


def _idle_pool(monkeypatch, *, disabled=""):
    """The pool as boot leaves it and as every quiet period returns it: the lane's spare alone."""
    from supervisor import worker_pool_lifecycle, workers
    from supervisor.worker_pool_lifecycle import LANE_CONVERSATION

    monkeypatch.setattr(workers, "WORKERS", {0: _slot(0, LANE_CONVERSATION)})
    monkeypatch.setattr(workers, "_worker_pool_execution_state", lambda: {"disabled_reason": disabled})
    monkeypatch.setattr(worker_pool_lifecycle, "_slot_pid", lambda s: s.proc.pid)


# ---- N1: availability over the published document at idle ---------------------------------------

def test_an_idle_pool_with_the_lanes_spare_is_available_and_a_disabled_pool_refuses(published, monkeypatch):
    from ouroboros.context_runtime_facts import _queue_context_fact
    from ouroboros.tools.control_events import _promotion_pool_disabled_from_snapshot

    root, _now = published
    _idle_pool(monkeypatch)
    assert queue.persist_queue_snapshot("boot") is True
    document = _read(root, "queue_snapshot.json")
    assert document["worker_total"] == 0 and document["lane_carrier_total"] == 1, "the real idle shape"
    ctx = types.SimpleNamespace(drive_root=root, budget_drive_root="")
    assert _promotion_pool_disabled_from_snapshot(ctx) == "", "a started process answers; a quiet project pool is not absence"
    fact = _queue_context_fact({"budget_drive_root": str(root)})
    assert fact["worker_total"] == 0 and fact["lane_carrier_total"] == 1
    # A disabled or exhausted pool still refuses, whatever is started.
    _idle_pool(monkeypatch, disabled="crash_storm")
    assert queue.persist_queue_snapshot("storm") is True
    assert _promotion_pool_disabled_from_snapshot(ctx) == "crash_storm"
    # No process at all is absence.
    from supervisor import workers
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "_worker_pool_execution_state", lambda: {"disabled_reason": "no_workers"})
    assert queue.persist_queue_snapshot("empty") is True
    assert _promotion_pool_disabled_from_snapshot(ctx) == "no_workers"


def test_the_telegram_health_line_counts_every_started_process(published, monkeypatch):
    from skills.telegram.lib import telegram_health

    root, _now = published
    _idle_pool(monkeypatch)
    assert queue.persist_queue_snapshot("boot") is True
    snap = json.loads((root / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    assert telegram_health._started_processes(snap) == 1


# ---- N9: a failed publication is retried; the stale document is never re-certified ---------------

def test_a_failed_census_publication_is_retried_and_never_recertified_meanwhile(published, monkeypatch):
    root, now = published
    queue.RUNNING["r"] = {"task": _task("r"), "attempt": 1, "worker_id": 3, "started_at": now[0] - 100,
                          "last_heartbeat_at": now[0] - 1, "budget_paused_sec": 0.0, "soft_sent": False}
    assert queue.persist_queue_snapshot("start") is True
    generation = _read(root, "queue.alive")["generation"]
    queue_publication.tick_publisher_started()
    with queue._queue_lock:  # an unmarked change of a published fact (the census's business)
        queue.RUNNING["r"]["silence_attention"] = {"episode_id": "e1", "state": "pending_attention", "last_progress_at": now[0] - 50}
    with monkeypatch.context() as broken:
        broken.setattr(queue, "atomic_write_text", lambda *_a: False)  # the disk is full
        assert queue_publication.publish_queue_changes() is False
    assert not (root / "state/queue.alive").exists(), "the old certificate was revoked before the failed write"
    # Quiet ticks inside the refresh cadence and past it: the stale document is never certified again.
    now[0] += queue_publication.QUEUE_ALIVE_REFRESH_SEC * 2
    with monkeypatch.context() as broken:
        broken.setattr(queue, "atomic_write_text", lambda *_a: False)
        assert queue_publication.publish_queue_changes() is False, "the change is retried, not forgotten"
    assert not (root / "state/queue.alive").exists()
    assert "silence_attention" not in _read(root, "queue_snapshot.json")["running"][0]
    # The disk is back: the next tick publishes the change and certifies the new document.
    assert queue_publication.publish_queue_changes() is True
    assert _read(root, "queue_snapshot.json")["running"][0]["silence_attention"]["episode_id"] == "e1"
    alive = _read(root, "queue.alive")
    assert alive["generation"] != generation and alive["generation"] == _read(root, "queue_snapshot.json")["generation"]
    assert queue_publication.publish_queue_changes() is True, "and the quiet tick that follows refreshes as before"


# ---- N10: a newer capture always wins -------------------------------------------------------------

def test_a_newer_capture_is_never_overwritten_by_an_older_one_at_the_same_revision(published):
    root, now = published
    queue.PENDING.append(_task("a"))
    assert queue.persist_queue_snapshot("seed") is True
    queue_publication.tick_publisher_started()
    direct = {"roots": [], "incomplete": False}
    with queue._queue_lock:
        older = queue_publication._capture("barrier", direct)  # a barrier's capture, not yet written
    queue.PENDING.append(_task("b"))
    with queue._queue_lock:
        newer = queue_publication._capture("main_loop", direct, census=queue_publication._census())  # the tick's capture
    assert newer["revision"] == older["revision"], "the tick captured without a bump: the same revision"
    assert queue_publication._publish(newer) is True
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["a", "b"]
    assert queue_publication._publish(older) is True, "the older capture is subsumed, never written"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["a", "b"], "the newer facts stay on disk"


# ---- N11: the retry-custody publication repeats only when the claim's facts change ---------------

def test_the_retry_custody_publication_repeats_only_when_the_claims_facts_change(tmp_path, monkeypatch):
    from supervisor import workers

    pending = [{"id": "held", "type": "task", "_terminalization_retry": {"status": "cancelled", "trigger": "x", "reason": "pending"}}]
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir()
    publications = []
    monkeypatch.setattr(queue, "publish_queue_snapshot", lambda **kw: publications.append(kw["reason"]) or True)
    workers._PUBLISHED_RETRY_CUSTODY.clear()
    for _ in range(3):  # three assignment passes with the same unresolved claim
        workers._persist_pending_terminalization_retries(["held"])
    assert publications == ["pending_terminal_event_retry"], "one publication for one set of facts"
    pending[0]["_terminalization_retry"]["reason"] = "still pending, new detail"
    workers._persist_pending_terminalization_retries(["held"])
    assert len(publications) == 2, "a changed claim fact publishes again"
    workers._persist_pending_terminalization_retries([])
    workers._persist_pending_terminalization_retries(["held"])
    assert len(publications) == 3, "a claim that left and returned is a change"


# ---- NIT: the census carries only the waits that still wait ---------------------------------------

def test_the_census_prunes_resolved_model_waits(published):
    _root, now = published
    task = _task("r", model_waits={"w1": {"state": "resolved", "revision": 2, "task_attempt": 1}})
    queue.RUNNING["r"] = {"task": task, "attempt": 1, "started_at": now[0]}
    with queue._queue_lock:
        before = queue_publication._census()
        task["model_waits"]["w2"] = {"state": "resolved", "revision": 1, "task_attempt": 1}
        assert queue_publication._census() == before, "a resolved wait is history, not a published change"
        task["model_waits"]["w3"] = {"state": "waiting", "revision": 1, "task_attempt": 1}
        assert queue_publication._census() != before, "a wait that waits is published"
