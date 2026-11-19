"""A retained stop never parks the lifecycle serializer: only its job's first stop waits.

Reviewer counterexample: an idle retirement or unready teardown whose stop could not be
confirmed was re-driven on every health tick, each pass holding the serializer for another
bounded join, while assignment defers its whole turn on that serializer, so independent work
starved beside a ready idle slot. The failed slot itself stays charged until its death is seen.
Each re-drive repeats one bounded tree kill and joins without waiting; since owner 14 Oct
21:09 #4 no kill or join holds the serializer at all (it guards the membership step only).
"""

import json
import queue as stdqueue
import threading
import time
from types import SimpleNamespace

import pytest

from supervisor import queue, task_reaper, worker_pool_lifecycle as lifecycle, workers
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401
from tests.test_worker_demand import acknowledge, assigning as assigning, task  # noqa: F401
from tests.worker_pool_support import affirm_ready


_REAL_THREAD = threading.Thread  # the pool fixture stubs Thread; the reaper below is real


class _NativeChild(Process):
    """``join(timeout)`` really waits while the child lives; each join's bound is recorded."""

    def __init__(self, pid, *, wait=True):
        super().__init__(pid)
        self.joins, self.wait = [], wait

    def join(self, timeout=None):
        self.joins.append(timeout)
        if self.wait and self.alive:
            time.sleep(timeout or 0)


class _EndOfReaper(BaseException):
    """Ends the real reaper loop (it catches Exception only)."""


class _JobQueue(stdqueue.Queue):
    """The production reaper queue; a ``"stop"`` sentinel ends the real loop at teardown."""

    def get(self, *args, **kwargs):
        job = super().get(*args, **kwargs)
        if job == "stop":
            raise _EndOfReaper()
        return job


def _until_stopped(loop):
    try:
        loop()
    except _EndOfReaper:
        pass


def _wedge(monkeypatch):
    """The tree kill never lands (the child outlives SIGKILL); a queued error makes one kill raise."""
    kills, errors = [], []

    def kill(pid, **_kw):
        kills.append(pid)
        if errors:
            raise errors.pop()

    monkeypatch.setattr(lifecycle, "kill_worker_tree", kill)
    return kills, errors


def _watcher_handoffs(monkeypatch):
    """Each fresh slot's readiness handoff as (slot ids, readiness attempt)."""
    handed = []
    monkeypatch.setattr(lifecycle.threading, "Thread", lambda **kw: handed.append(
        (tuple(kw["args"][0]), kw["args"][2])) or SimpleNamespace(start=lambda: None))
    return handed


def _retiring_surplus(assigning, proc):
    """Slot 1 is an acknowledged idle surplus reserved for the reaper; slot 0 stays idle and ready."""
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    slot = affirm_ready(workers.Worker(1, proc, InputQueue()))
    workers.WORKERS[1] = slot
    acknowledge(slot)
    assert lifecycle.retire_quiescent_worker(slot)
    return slot


def _booting(assigning, proc):
    """Slot 1 is a booting child silent past its readiness window; slot 0 stays idle and ready."""
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    slot = workers.Worker(1, proc, InputQueue(), reaping=True, booting=True)
    workers.WORKERS[1] = slot
    return slot


def _reap_pass(monkeypatch):
    """One health tick's hand-back of retained jobs, then the real reaper loop over them."""
    task_reaper.retry_terminal_file_recoveries()
    handed = []
    while not queue._reap_queue.empty():
        handed.append(queue._reap_queue.get_nowait())

    def get():
        if not handed:
            raise _EndOfReaper()
        return handed.pop(0)

    monkeypatch.setattr(task_reaper, "reap_queue", SimpleNamespace(get=get, task_done=lambda: None))
    with pytest.raises(_EndOfReaper):
        task_reaper.reaper_loop()


def _retired_rows(root):
    path = root / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [(r["worker_id"], r["pid"], r["token"], r["replaced"]) for r in rows if r["type"] == "worker_retired"]


@pytest.mark.parametrize(("kind", "kill_raises"), [("idle", False), ("unready", True)])
def test_a_retained_stop_spends_one_bounded_join_and_its_re_drives_never_wait(assigning, monkeypatch, kind, kill_raises):
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    monkeypatch.setattr(task_reaper, "_deferred_reap_jobs", [])
    kills, errors = _wedge(monkeypatch)
    proc = _NativeChild(43, wait=False)
    if kind == "idle":
        slot = _retiring_surplus(assigning, proc)
    else:
        slot = _booting(assigning, proc)
        lifecycle._replace_unready_slot(1, slot, 0, time.time(), 1)  # the watcher's own bounded stop
        assert proc.joins == [2] and kills == [43] and not slot.booting
    job = queue._reap_queue.queue[0]
    watcher_joins, watcher_kills = len(proc.joins), len(kills)
    if kill_raises:
        errors.append(OSError("process tree unreadable"))  # the job's first reaper kill
    flags = []
    for _redrive in range(3):
        _reap_pass(monkeypatch)
        flags.append(job.get("first_stop"))
        assert [kept is job for kept in task_reaper._deferred_reap_jobs] == [True], "retained once"
        assert workers.WORKERS[1] is slot and slot.active_capacity and slot.reaping
        assert not lifecycle.worker_ready(slot)
        if kind == "idle":
            assert slot.retiring and slot.quiescence_token == job["token"]
        else:
            assert not slot.booting and not slot.readiness_exhausted and job["attempt"] == 1
        assert workers._ensure_workers_healthy_locked(queue) == ([], False) and queue._reap_queue.empty()
    assert proc.joins[watcher_joins:] == ([2, 0] if kill_raises else [2, 0, 0])
    assert flags == ([True, None, None] if kill_raises else [None] * 3), "only a completed kill spends the join"
    assert kills == [43] * (watcher_kills + 3), "every live pass still re-kills"

    handed = _watcher_handoffs(monkeypatch)
    proc.alive = False  # the stop lands later
    _reap_pass(monkeypatch)
    assert task_reaper._deferred_reap_jobs == [] and len(kills) == watcher_kills + 3, "a seen death is not re-killed"
    if kind == "idle":
        assert workers.WORKERS == {0: assigning.original}
        assert _retired_rows(assigning.root) == [(1, 43, job["token"], False)]
    else:
        assert handed == [((1,), 2)], "the same readiness count continues"
        assert workers.WORKERS[1] is not slot and workers.WORKERS[1].booting
        assert _retired_rows(assigning.root) == []


@pytest.mark.parametrize("kind", ["idle", "unready"])
def test_a_new_generation_drops_the_retained_job_and_owns_its_own_bounded_join(assigning, monkeypatch, kind):
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    monkeypatch.setattr(task_reaper, "_deferred_reap_jobs", [])
    kills, _errors = _wedge(monkeypatch)
    proc = _NativeChild(43, wait=False)
    if kind == "idle":
        slot = _retiring_surplus(assigning, proc)
    else:
        slot = _booting(assigning, proc)
        lifecycle._defer_unready_teardown(1, slot, 0, time.time(), 1)
    _reap_pass(monkeypatch)
    seen = (list(proc.joins), list(kills))
    assert seen == ([2], [43]) and task_reaper._deferred_reap_jobs
    successor = affirm_ready(workers.Worker(1, Process(44), InputQueue()))
    with queue._queue_lock:
        workers.WORKERS[1] = successor  # a pool restart installed its own slot under this id
    _reap_pass(monkeypatch)
    assert (proc.joins, kills) == seen and task_reaper._deferred_reap_jobs == []
    assert workers.WORKERS[1] is successor and not successor.reaping and successor.proc.is_alive()
    acknowledge(successor)
    assert lifecycle.retire_quiescent_worker(successor)
    assert queue._reap_queue.get_nowait()["first_stop"] is True


def _tick_until(predicate, ticks=12):
    """server.py's turn order: health (re-submits retained jobs), intake, assignment, idle sleep."""
    for tick in range(ticks):
        workers.ensure_workers_healthy()
        time.sleep(0.05)  # event drain, bridge intake, timeouts, maintenance
        workers.assign_tasks()
        if predicate():
            return tick
        time.sleep(0.5)
    return None


@pytest.mark.parametrize(("kind", "wedged"), [("idle", True), ("unready", False)])
def test_an_unconfirmable_stop_never_starves_independent_dispatch_or_growth(
        assigning, monkeypatch, record_property, kind, wedged):
    """Real reaper thread and assignment, cap 3, slot 0 idle and ready. Before the fix each re-drive
    held the serializer for another 2 s join and no turn dispatched in 12 ticks."""
    jobs = _JobQueue()
    monkeypatch.setattr(queue, "_reap_queue", jobs)
    monkeypatch.setattr(task_reaper, "reap_queue", jobs)
    monkeypatch.setattr(task_reaper, "_deferred_reap_jobs", [])
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    handed = _watcher_handoffs(monkeypatch)
    if wedged:
        _wedge(monkeypatch)
    proc = _NativeChild(43)
    if kind == "idle":
        slot = _retiring_surplus(assigning, proc)
    else:
        slot = _booting(assigning, proc)
        lifecycle._defer_unready_teardown(1, slot, 0, time.time(), 1)  # the watcher's unconfirmed stop
    token = slot.quiescence_token
    reaper = _REAL_THREAD(target=_until_stopped, args=(task_reaper.reaper_loop,), daemon=True)
    started = time.monotonic()
    reaper.start()
    try:
        workers.PENDING.append(task("fresh"))
        dispatched_at = _tick_until(lambda: "fresh" in workers.RUNNING)
        record_property("dispatched_at_tick_sec", json.dumps([dispatched_at, round(time.monotonic() - started, 2)]))
        assert dispatched_at is not None, "independent work starved beside a ready idle slot"
        assert workers.RUNNING["fresh"]["worker_id"] == 0
        if wedged:
            workers.PENDING.extend([task("second"), task("third")])
            assert _tick_until(lambda: len(proc.joins) >= 3 and len(workers.WORKERS) == 4) is not None
            assert proc.joins[0] == 2 and set(proc.joins[1:]) == {0}, f"a re-drive blocked again: {proc.joins}"
            assert workers.WORKERS[1] is slot and slot.active_capacity and slot.reaping
            # Growth ran beside it: the cap counts acting tasks and the machine measures the wedged process's
            # memory itself, so it is charged as retiring until its death is seen, never as a held number.
            assert len(assigning.created) == 2, "growth ran for both queued rows"
            proc.alive = False  # the stop lands
        else:
            assert dispatched_at <= 1 and proc.joins == [2]
        assert _tick_until(lambda: workers.WORKERS.get(1) is not slot) is not None
    finally:
        jobs.put("stop")
        reaper.join(10)
        record_property("join_bounds", json.dumps(proc.joins))
    assert not reaper.is_alive()
    if kind == "idle":
        assert _retired_rows(assigning.root) == [(1, 43, token, False)]
    else:
        assert ((1,), 2) in handed and _retired_rows(assigning.root) == [], "the same readiness count continues"
