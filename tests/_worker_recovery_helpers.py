"""In-process worker recovery fixtures: pool events, a fake worker, and the real crash requeue."""

from types import SimpleNamespace
from unittest.mock import MagicMock


def _queued(task_id, **extra):
    """A pending queue row; a later attempt was admitted to dispatch before."""
    return {"admitted_dispatch": "possible" if extra.get("_attempt", 1) > 1 else "none", "id": task_id, "type": "task", "chat_id": 0, "_attempt": 1, "text": task_id, **extra}


def _pool_events(workers, monkeypatch):
    events: list = []
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False, raising=False)
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append), raising=False)
    return events


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


def _crash(queue, workers, monkeypatch, task_id):
    """The worker running ``task_id`` dies before any terminal: the live same-id crash requeue
    (``worker_health``) runs on the real queue. Doubled: the dead process, cost reconstruction and
    the slot's respawn."""
    from supervisor import worker_health

    meta = workers.RUNNING[task_id]
    wid = int(meta["worker_id"])
    dead = _make_worker(wid=wid, busy_task_id=task_id, exitcode=1)
    dead.reaping = True
    workers.WORKERS[wid] = dead
    monkeypatch.setattr(workers, "QUEUE_MAX_RETRIES", 2)  # the configurable limit, wide enough for two losses
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *_a, **_k: {})
    worker_health._recover_crashed_task_without_terminal(dict(
        worker=dead, task_id=task_id, task=meta["task"], meta=meta, worker_id=wid, exitcode=1,
        attempt=int(meta["attempt"]), drive_root=str(workers.DRIVE_ROOT)), queue)
    workers.WORKERS.pop(wid, None)
