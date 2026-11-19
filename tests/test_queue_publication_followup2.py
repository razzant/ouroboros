"""Queue publication, review follow-up 2 (RV_MAIN M5, M6, M7, W9 and the format stamp of the export).

M5: a mark is never read as "persisted" — it answers None while the tick publishes, every
runtime caller that acts on the publication reaches the barrier, and a guard finds a caller
that reads the mark's value. M6: a fact the publication carries that changed under Q without
a mark is published by the next tick (the census of volatile row facts). M7: the live root
cost figure is marked by a call-state frame, a wait lease and a server-side custody
transition, so the flush reads an in-flight call or an open run. W9: the Runtime queue fact
counts project slots; lane carriers are reported apart. The rollback export carries the
format an older body treats as current.
"""

import json
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


# ---- M6: a fact changed under Q without a mark is published by the next tick ----------------------

def _running(task_id, now):
    return {"task": _task(task_id, model_waits={}), "attempt": 1, "worker_id": 3, "started_at": now - 100,
            "last_heartbeat_at": now - 1, "last_progress_at": now - 50, "budget_paused_sec": 0.0, "soft_sent": False}


def test_an_unmarked_change_of_a_published_row_fact_is_published_by_the_tick(published):
    root, now = published
    queue.RUNNING["r"] = _running("r", now[0])
    assert queue.persist_queue_snapshot("start") is True
    queue_publication.tick_publisher_started()
    generation = _read(root, "queue.alive")["generation"]
    assert queue_publication.publish_queue_changes() is True
    assert _read(root, "queue.alive")["generation"] == generation, "an unchanged live set publishes nothing"
    # silence_attention.observe: a worker report mutates RUNNING in place under Q with no mark (#1376's source).
    with queue._queue_lock:
        queue.RUNNING["r"]["silence_attention"] = {"episode_id": "e1", "state": "pending_attention", "last_progress_at": now[0] - 50}
    assert queue_publication.publish_queue_changes() is True
    row = _read(root, "queue_snapshot.json")["running"][0]
    assert row["silence_attention"]["episode_id"] == "e1"
    second = _read(root, "queue.alive")["generation"]
    assert second != generation, "the census saw the unmarked change"
    # The quota clock (task_model_wait) and the model waits a worker reports are watched the same way.
    with queue._queue_lock:
        queue.RUNNING["r"]["model_wait_quota_clock"] = {"revision": 2, "active": True, "elapsed_sec": 3, "observed_at": now[0]}
        queue.RUNNING["r"]["task"]["model_waits"] = {"w1": {"state": "waiting", "revision": 1, "task_attempt": 1}}
    third = _read(root, "queue.alive")["generation"]
    assert queue_publication.publish_queue_changes() is True
    row = _read(root, "queue_snapshot.json")["running"][0]
    assert row["model_wait_quota_clock"]["revision"] == 2 and row["task"]["model_waits"]["w1"]["state"] == "waiting"
    assert _read(root, "queue.alive")["generation"] != third
    # Rest law: another tick over the unchanged rows writes no document.
    fourth = _read(root, "queue.alive")["generation"]
    assert queue_publication.publish_queue_changes() is True
    assert _read(root, "queue.alive")["generation"] == fourth


# ---- M7: the live root cost figure sees a call in flight and an open run -------------------------
