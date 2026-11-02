"""#1729 / #1730: the owner's Stop stays effective across a retained reap's health transfer.

The reaper retains a hard-bound reap (here the attempt's explicit deadline; neither bound retries)
whose terminal-file read stays unresolved (its slot reserved, its terminal owed) and the health
tick hands it back for a replay: retained -> queued -> in flight -> retained or settled. The real
owners run: the timeout decision (``enforce_task_timeouts``), ``reaper_loop`` and
``reap_timed_out_task``, the health tick (``retry_terminal_file_recoveries``), the HTTP Stop
(``api_task_cancel``), cancellation custody and its watchdog (``sweep_cancel_intents``). Doubled:
OS processes, model calls, the worker queue's ``put``, the reaper thread's blocking ``get`` and the
failing terminal-file read.

The contract: idle custody settles Cancelled under Q before any replay reads it, and that replay
publishes it once; a replay out with the reaper keeps the slot, so the recorded Stop answers
not confirmed (neither alive nor dead) and settles after it. Never a 404, a second ``task_done``,
a re-park or a successor.
"""
from __future__ import annotations

import queue as stdqueue
from types import SimpleNamespace

import pytest

from ouroboros import headless
from ouroboros.task_results import load_task_result
from supervisor import queue, workers
from tests.test_execution_clock_resume_handover import clock  # noqa: F401
from tests.test_retry_pause_identity import (  # noqa: F401
    _hung_first_attempt,
    _owner_stop,
    _paused_hung_attempt,
    _successors,
    reaper,
)
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


class _Drained(BaseException):
    """The pump's queue is empty: the real ``reaper_loop`` would block on ``get`` here."""


class _ReapPump(stdqueue.Queue):
    def get(self, *args, **kwargs):
        if self.empty():
            raise _Drained
        return super().get(*args, **kwargs)


def _retained_reap(host, monkeypatch, clock, reaper, hung=_paused_hung_attempt):  # noqa: F811
    """The real reaper loop retains a deadline reap whose terminal-file read fails until ``fault.on`` clears.

    The timeout decision, the health tick and ``reaper_loop`` share one queue, as in production
    (``supervisor.queue`` re-exports the reaper's). ``fault.during`` runs once inside the next read."""
    from supervisor import task_reaper

    pump = _ReapPump()
    monkeypatch.setattr(queue, "_reap_queue", pump)
    monkeypatch.setattr(task_reaper, "reap_queue", pump)
    monkeypatch.setattr(task_reaper, "_deferred_reap_jobs", [])
    hung(host, monkeypatch, clock)
    real, fault = headless.prepare_terminal_task_files, SimpleNamespace(on=True, during=None)

    def prepare(root, task):
        if str(task.get("id") or "") == "held":
            during, fault.during = fault.during, None
            if during:
                during()
            if fault.on:
                raise OSError("terminal files unreadable")
        return real(root, task)

    monkeypatch.setattr(headless, "prepare_terminal_task_files", prepare)
    events, respawned = [], []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    monkeypatch.setattr(workers, "respawn_worker", lambda wid, *_a, **_k: respawned.append(wid))
    reaper.arm()
    workers.RUNNING["held"]["task"]["deadline_at"] = "2000-01-01T00:00:00+00:00"  # the attempt's hard bound
    queue.enforce_task_timeouts()  # the bound opens its finalization grace
    clock[0] += float(queue.FINALIZATION_GRACE_SEC) + 1
    queue.enforce_task_timeouts()  # the grace ran out: the decision submits the reap
    [job] = list(pump.queue)
    assert job["deadline_reached"] and not {"will_retry", "retry_task_id"} & set(job)
    assert "held" not in workers.RUNNING and workers.WORKERS[0].reaping

    def pump_once():
        with pytest.raises(_Drained):
            task_reaper.reaper_loop()

    pump_once()
    assert [kept is job for kept in task_reaper._deferred_reap_jobs] == [True], "the first pass retains it"
    assert workers.WORKERS[0].reaping and not respawned and not _done(events)
    return SimpleNamespace(job=job, fault=fault, events=events, respawned=respawned, pump=pump,
                           pump_once=pump_once)


def _done(events):
    return [e["status"] for e in events if e.get("type") == "task_done" and e.get("task_id") == "held"]


def _supervisor_rounds(clock, retained, rounds=3):  # noqa: F811
    """The supervisor's own cadence, bounded: health tick, one reaper pass, the cancel watchdog."""
    from supervisor import task_reaper
    from supervisor.task_lifecycle import sweep_cancel_intents

    for _ in range(rounds):
        task_reaper.retry_terminal_file_recoveries()
        retained.pump_once()
        sweep_cancel_intents(now=clock[0] + 60)


def _cancelled_once(host, retained):  # noqa: F811
    from ouroboros.cancel_intents import active_intent
    from supervisor import task_reaper

    assert load_task_result(host.root, "held")["status"] == "cancelled"
    assert _done(retained.events) == ["cancelled"], "one terminal publication"
    assert retained.respawned == [0] and not task_reaper._deferred_reap_jobs and retained.pump.empty()
    assert not host.pending and "held" not in workers.RUNNING and not _successors(host)
    assert active_intent(host.root, "held") is None, "the owner's intent settled with its outcome"


def test_a_health_requeue_during_defer_keeps_custody_for_http_stop(host, monkeypatch, clock, reaper):  # noqa: F811
    """The next health tick runs after defer releases Q, before that pass's finally.

    Stop arrives before the queued replay starts. Cleanup of the OLD pass must not
    consume the NEW replay's flag or withdraw the retained job Stop addresses.
    """
    from supervisor import task_reaper

    retained = _retained_reap(host, monkeypatch, clock, reaper)
    real_warning, real_done = task_reaper.log.warning, retained.pump.task_done
    handbacks = []

    def health_before_finally(message, *args, **kwargs):
        real_warning(message, *args, **kwargs)
        if message == "Reaper custody pending for task %s, worker %s: %s":
            task_reaper.retry_terminal_file_recoveries()
            task_reaper.retry_terminal_file_recoveries()  # an already queued replay is not duplicated
            handbacks.append(retained.pump.qsize())

    def stop_before_next_get():
        real_done()
        raise _Drained  # the next replay remains queued, as while the reaper is unscheduled

    task_reaper.retry_terminal_file_recoveries()
    with monkeypatch.context() as interleave:
        interleave.setattr(task_reaper.log, "warning", health_before_finally)
        interleave.setattr(retained.pump, "task_done", stop_before_next_get)
        retained.pump_once()
    custody = [kept is retained.job for kept in task_reaper._deferred_reap_jobs]
    replaying = retained.job.get("_replaying")
    status, body = _owner_stop(host, "held")
    facts = {"queued_replays": retained.pump.qsize(), "retained": custody,
             "replaying": replaying, "http_status": status, "body": body}
    print("defer/requeue Stop witness:", facts)
    assert handbacks == [1] and facts["queued_replays"] == 1
    assert status == 202 and custody == [True] and replaying is True, facts
    assert body["cancel_state"] == "pending" and body["reason_code"] == "teardown_unconfirmed"
    _supervisor_rounds(clock, retained)
    _cancelled_once(host, retained)


def test_failed_replay_submission_keeps_retryable_stop_custody(host, monkeypatch, clock, reaper):  # noqa: F811
    from supervisor import task_reaper

    retained = _retained_reap(host, monkeypatch, clock, reaper)

    def unavailable(*_args, **_kwargs):
        raise OSError("reaper queue temporarily unavailable")

    with monkeypatch.context() as fault:
        fault.setattr(retained.pump, "put", unavailable)
        task_reaper.retry_terminal_file_recoveries()
    assert task_reaper._deferred_reap_jobs == [retained.job]
    assert not retained.job.get("_replaying") and retained.pump.empty()
    task_reaper.retry_terminal_file_recoveries()
    task_reaper.retry_terminal_file_recoveries()  # healthy handoff is not duplicated
    assert retained.pump.qsize() == 1 and retained.job.get("_replaying")
    status, body = _owner_stop(host, "held")
    assert status == 202 and body["reason_code"] == "teardown_unconfirmed"
    _supervisor_rounds(clock, retained)
    _cancelled_once(host, retained)


def test_stop_ends_a_reap_deferred_on_persistently_unreadable_terminal_files(host, monkeypatch, clock,  # noqa: F811
                                                                             reaper):  # noqa: F811
    """A disk fault that never clears keeps the reap deferred — never Failed, never retried — but the
    owner's Stop still ends it: the next health replay settles Cancelled once and frees the slot."""
    from supervisor import task_reaper

    retained = _retained_reap(host, monkeypatch, clock, reaper)
    task_reaper.retry_terminal_file_recoveries()
    retained.pump_once()  # a replay before any Stop stays deferred
    assert [kept is retained.job for kept in task_reaper._deferred_reap_jobs] == [True]
    assert workers.WORKERS[0].reaping and not retained.respawned and not host.pending
    assert load_task_result(host.root, "held")["status"] not in {"failed", "cancelled"}

    status, body = _owner_stop(host, "held")
    assert (status, body) == (200, {"ok": True, "task_id": "held"}), "Stop reaches the reaper's retained custody"
    task_reaper.retry_terminal_file_recoveries()
    retained.pump_once()  # the next health replay, the fault unchanged
    _cancelled_once(host, retained)


@pytest.mark.parametrize("stop_at", ["queued", "in_flight"])
def test_stop_reaches_a_retained_reap_in_every_state_of_its_health_transfer(host, monkeypatch, clock,  # noqa: F811
                                                                            reaper, stop_at):  # noqa: F811
    """The fault never clears. A job the reaper has out for its replay keeps the slot, so the recorded
    Stop answers 'not confirmed' (never 404) and the watchdog finishes it once the replay defers again."""
    from ouroboros.cancel_intents import active_intent
    from supervisor import task_reaper

    retained, stops = _retained_reap(host, monkeypatch, clock, reaper), []
    task_reaper.retry_terminal_file_recoveries()  # the health tick hands the job back to the reaper
    if stop_at == "queued":
        stops.append(_owner_stop(host, "held"))
    else:
        retained.fault.during = lambda: stops.append(_owner_stop(host, "held"))
        retained.pump_once()
    [(status, body)] = stops
    recorded = bool(active_intent(host.root, "held"))
    _supervisor_rounds(clock, retained)
    _cancelled_once(host, retained)
    assert status == 202 and body["cancel_state"] == "pending" and body["reason_code"] == "teardown_unconfirmed", body
    assert recorded, "the Stop is durable while the replay owns the slot"


def test_a_health_transfer_inside_the_stop_publishes_one_terminal(host, monkeypatch, clock, reaper):  # noqa: F811
    """Stop sees the idle retained job, then the health tick hands it back before the cancel write."""
    from supervisor import cancel_publication, task_reaper

    retained = _retained_reap(host, monkeypatch, clock, reaper)
    real_audit = cancel_publication._audit_delegated_runs_on_kill

    def audit(*args, **kwargs):  # the cancel's own pre-write custody audit
        monkeypatch.setattr(cancel_publication, "_audit_delegated_runs_on_kill", real_audit)
        task_reaper.retry_terminal_file_recoveries()
        return real_audit(*args, **kwargs)

    monkeypatch.setattr(cancel_publication, "_audit_delegated_runs_on_kill", audit)
    status, _body = _owner_stop(host, "held")
    _supervisor_rounds(clock, retained)
    _cancelled_once(host, retained)
    assert status == 202, "the replay it handed back owns the write; no second publisher"


@pytest.mark.parametrize("hung", ["paused", "unpaused"])
def test_stop_inside_the_replay_that_clears_the_fault_keeps_one_honest_terminal(host, monkeypatch, clock,  # noqa: F811
                                                                                reaper, hung):  # noqa: F811
    """The fault clears in the very replay the Stop lands in, which owns the slot and the write: the
    dead attempt keeps its honest deadline failure, paused or not (a hard bound holds no pause and
    admits no successor), and the recorded Stop settles over that terminal."""
    from ouroboros.cancel_intents import active_intent
    from supervisor import task_reaper

    retained = _retained_reap(host, monkeypatch, clock, reaper,
                              hung=_paused_hung_attempt if hung == "paused" else _hung_first_attempt)
    stops = []

    def stop_then_clear():
        stops.append((*_owner_stop(host, "held"), bool(active_intent(host.root, "held"))))
        retained.fault.on = False

    task_reaper.retry_terminal_file_recoveries()
    retained.fault.during = stop_then_clear
    retained.pump_once()
    [(status, _body, recorded)] = stops
    _supervisor_rounds(clock, retained)
    stored = load_task_result(host.root, "held")
    assert not host.pending and "held" not in workers.RUNNING and not _successors(host)
    assert (stored["status"], stored["reason_code"]) == ("failed", "deadline"), stored
    assert _done(retained.events) == [stored["status"]], "one terminal publication, of the durable truth"
    assert retained.respawned == [0] and not task_reaper._deferred_reap_jobs and retained.pump.empty()
    assert active_intent(host.root, "held") is None
    assert status == 202 and recorded, "the owner's Stop reached the reaper's custody"


def test_a_failed_cancel_write_keeps_the_retained_reap_and_the_watchdog_finishes(host, monkeypatch, clock,  # noqa: F811
                                                                                 reaper):  # noqa: F811
    """An unwritable Cancelled row is not a cancellation: the reap stays retained, nothing publishes."""
    from ouroboros import task_results
    from ouroboros.cancel_intents import active_intent
    from supervisor import task_reaper

    retained = _retained_reap(host, monkeypatch, clock, reaper)
    real_write = task_results.write_task_result

    def refused(root, task_id, status, *args, **kwargs):
        if task_id == "held" and status == "cancelled":
            raise OSError("result store unwritable")
        return real_write(root, task_id, status, *args, **kwargs)

    monkeypatch.setattr(task_results, "write_task_result", refused)
    status, _body = _owner_stop(host, "held")
    assert status == 202 and active_intent(host.root, "held")
    task_reaper.retry_terminal_file_recoveries()
    retained.pump_once()
    assert load_task_result(host.root, "held")["status"] not in {"failed", "cancelled"}
    assert [kept is retained.job for kept in task_reaper._deferred_reap_jobs] == [True] and not _done(retained.events)
    monkeypatch.setattr(task_results, "write_task_result", real_write)
    _supervisor_rounds(clock, retained)
    _cancelled_once(host, retained)


def test_a_contended_cancel_write_releases_q_and_the_watchdog_finishes(host, monkeypatch, clock, reaper):  # noqa: F811
    """Real store contention (another writer holds the store) meets the miss-lane Cancelled write
    under Q: Q is free again while the store stays contended, nothing is cancelled or published, the
    reap stays retained, and the watchdog settles Cancelled once after the contention ends."""
    import threading
    import time

    from ouroboros import task_results
    from ouroboros.cancel_intents import active_intent
    from ouroboros.store import open_store
    from supervisor import task_reaper

    retained = _retained_reap(host, monkeypatch, clock, reaper)
    real_write, witness, threads = task_results.write_task_result, {}, []

    def contended(root, task_id, status, *args, **kwargs):
        if task_id != "held" or status != "cancelled" or witness:
            return real_write(root, task_id, status, *args, **kwargs)
        held, ended = threading.Event(), threading.Event()

        def writer():  # the store's own write transaction, held until the probe has answered
            with open_store(root, write=True, barrier=True):
                held.set()
                ended.wait(30)

        def probe():  # another supervisor thread wanting Q while the store is contended
            started = time.perf_counter()  # the fixture's clock fakes time.monotonic
            got = queue._queue_lock.acquire(timeout=3.0)
            witness.update(q_free=got, contended=not ended.is_set(), waited=time.perf_counter() - started)
            if got:
                queue._queue_lock.release()
            ended.set()

        threads.append(threading.Thread(target=writer, daemon=True))
        threads[0].start()
        assert held.wait(5), "the contending writer holds the store"
        threads.append(threading.Thread(target=probe, daemon=True))
        threads[1].start()
        return real_write(root, task_id, status, *args, **kwargs)

    monkeypatch.setattr(task_results, "write_task_result", contended)
    status, body = _owner_stop(host, "held")
    for thread in threads:
        thread.join(10)
    facts = {"http": status, "witness": witness, "stored": load_task_result(host.root, "held")["status"],
             "recorded": bool(active_intent(host.root, "held")),
             "retained": [kept is retained.job for kept in task_reaper._deferred_reap_jobs],
             "replaying": retained.job.get("_replaying"), "done": _done(retained.events)}
    print("contended cancel write witness:", facts)
    assert threads and not any(thread.is_alive() for thread in threads), facts
    assert witness["q_free"] and witness["contended"] and witness["waited"] < 2.0, facts
    assert status == 202 and facts["stored"] not in {"failed", "cancelled"} and facts["recorded"], facts
    assert body["cancel_state"] == "pending" and body["reason_code"] == "teardown_unconfirmed", body
    assert facts["retained"] == [True] and not facts["replaying"] and not facts["done"], facts
    _supervisor_rounds(clock, retained)
    _cancelled_once(host, retained)
