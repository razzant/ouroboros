"""Cost law L9: the automatic schedule pass runs when it has something new to decide.

It used to start on every supervisor tick (31 a minute on 01.11), each pass taking Q and the
table's file lock twice and deep-copying the table, though nothing it decides on had changed.
The tick now asks ``schedule_time.pass_due``: a process's first pass, a write of the table by any
writer, the earliest FUTURE due instant the last completed pass left, or — only while rows wait
on facts the table does not revise (an occurrence in flight, an unobserved previous run) — the
existing 20 s observation cadence. The snapshot read takes no lock; the deciding transaction is
where Q is held. A due row still fires, and admits, through the real duty.
"""
from __future__ import annotations

import datetime as dt
import time
from types import SimpleNamespace


from ouroboros.runtime_limits import BODY_MAINTENANCE_INTERVAL_SEC
from ouroboros.task_results import write_task_result
from tests.test_schedule_occurrence import _row, _rows
from tests.test_schedule_occurrence import q as _q

q = _q


def _iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat()


def _edit(q, schedule_id, **fields):
    from supervisor import queue_schedules

    data = q.queue.load_schedule_store(q.root)
    next(row for row in data["tasks"] if row["id"] == schedule_id).update(fields)
    queue_schedules._write_scheduled_tasks(data, q.root)


def _pass(q):
    from supervisor.queue_schedules import automatic_acquisitions, check_scheduled_tasks

    with automatic_acquisitions():
        check_scheduled_tasks()


def _due(q, at=None):
    from supervisor.schedule_time import pass_due

    return pass_due(q.root, time.time() if at is None else at)


def test_a_quiet_table_wakes_only_at_its_next_due_instant_or_on_a_write(q):
    _row(q, "s1", cron=True)
    stored = _iso(time.time() + 3600)
    # The table keeps whole microseconds: the row's own instant is the stored one, not the float it was
    # rendered from (which the ISO text rounds, up about as often as down).
    due_at = dt.datetime.fromisoformat(stored).timestamp()
    _edit(q, "s1", next_run_at=stored)
    assert _due(q), "a process's first pass"
    _pass(q)
    now = time.time()
    assert not _due(q, now) and not _due(q, now + 10 * BODY_MAINTENANCE_INTERVAL_SEC), \
        "nothing waits on other facts: no observation cadence either"
    assert _due(q, due_at), "the row's own due instant"
    _edit(q, "s1", name="renamed")  # any writer: the owner door here, a worker's manage_schedules alike
    assert _due(q, now)
    _pass(q)
    assert not _due(q, now)


def test_a_row_waiting_on_another_task_is_re_observed_on_the_maintenance_cadence_never_per_tick(q):
    write_task_result(q.root, "run-1", "running", root_task_id="run-1")
    _row(q, "s1", cron=True)
    _edit(q, "s1", next_run_at=_iso(time.time() + 3600), last_task_id="run-1")
    _pass(q)
    end = time.time()
    assert not _due(q, end), "its run's outcome is unobserved, yet no pass per tick"
    assert _due(q, end + BODY_MAINTENANCE_INTERVAL_SEC)


def test_a_due_row_its_previous_run_still_blocks_waits_on_that_run_not_the_clock(q):
    q.pending.append({"id": "prev", "type": "task", "metadata": {"schedule_id": "s1"}})
    _row(q, "s1", cron=True)  # due since 2000, but its previous occurrence still queued
    _pass(q)
    end = time.time()
    assert "occurrence" not in _rows(q)["s1"], "a duplicate firing is never licensed"
    assert not _due(q, end), "a past due instant it could not start is no per-tick wake"
    assert _due(q, end + BODY_MAINTENANCE_INTERVAL_SEC)


def test_the_snapshot_takes_no_lock_and_only_the_deciding_read_holds_q(q, monkeypatch):
    from supervisor import queue as queue_module
    from supervisor import queue_schedules

    held = []
    real = queue_schedules.load_schedule_store

    def load(*args, **kwargs):
        held.append(queue_module._queue_lock._is_owned())
        return real(*args, **kwargs)

    _row(q, "s1", cron=True)
    _edit(q, "s1", next_run_at=_iso(time.time() + 3600))
    monkeypatch.setattr(queue_schedules, "load_schedule_store", load)
    _pass(q)
    assert held == [False, True]


def test_a_due_occurrence_still_admits_through_the_gate(q):
    _row(q, intent={"kind": "system_repo"})  # a one-shot due since 2000
    for _ in range(6):  # the tick: a pass only when due; admission's own write wakes the next
        if _due(q):
            _pass(q)
    [task] = q.pending
    assert task["metadata"]["schedule_id"] == "s1"
    end = time.time()
    assert not _due(q, end), "its admitted occurrence waits on the task, not on every tick"
    assert _due(q, end + BODY_MAINTENANCE_INTERVAL_SEC)


def test_the_tick_starts_the_schedule_duty_only_when_a_pass_is_due(q, monkeypatch):
    from ouroboros import server_maintenance as sm
    from supervisor import active_activity

    started = []

    def start(latch, name, target, args):
        started.append(name)
        latch.release()
        return True

    now = time.time()
    monkeypatch.setattr(sm, "_start_maintenance_thread", start)
    monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [now])
    monkeypatch.setattr(sm, "_LAST_COLD_CUSTODY_SWEEP", [now])
    monkeypatch.setattr(sm, "_tick_maintenance_child", lambda: None)
    monkeypatch.setattr(sm, "_stop_requested", lambda _stop=None: False)
    monkeypatch.setattr(q.queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: SimpleNamespace(actors=lambda: []))
    monkeypatch.setattr("ouroboros.maintenance.child.owed", lambda: False)
    _row(q, "s1", cron=True)
    _edit(q, "s1", next_run_at=_iso(now + 3600))

    def tick():
        sm._periodic_supervisor_maintenance([time.time()], [time.time()])
        return started.count("schedule-admission")

    assert tick() == 1
    _pass(q)  # the started duty's pass
    assert tick() == 1 and tick() == 1, "a quiet table starts no further pass"
    _edit(q, "s1", name="renamed")
    assert tick() == 2
