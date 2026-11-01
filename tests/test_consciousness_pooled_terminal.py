"""Every recorded terminal of a pooled own wake settles the clock once, and honestly (#2489).

The real dispatcher, result store, orphan sweep and clock run; only the worker slot and the
lane the clock would launch next are inert. ``interrupted``, cancellation custody and unknown
storage settle nothing: they still have an owner. An early ``completed`` beside the fault's
``infra_failed`` axis is a failure whatever the lifecycle string says.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import consciousness as clock_module, headless
from ouroboros.consciousness import (
    BackgroundConsciousness,
    INTERVAL_STATE_KEY,
    LAST_WAKE_STATE_KEY,
    OWN_WAKE_STATE_KEY,
)
from ouroboros.consciousness_wake import wake_task_metadata
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import execution_owner_record
from tests import test_consciousness as clocks, test_terminal_file_boundary as terminals
from tests.test_consciousness import DEFAULT, T0, _launched

clock = clocks.clock
terminal_context = terminals.terminal_context
file_recovery = terminals.file_recovery

OWN = wake_task_metadata("act", "heartbeat")
WAKE = "wake0001"
FINISH = T0 + 5000


@pytest.fixture
def pooled(clock, terminal_context, monkeypatch):
    """The admitted own wake on a worker slot: real queue maps, a real dispatcher, its RUNNING row."""
    from supervisor import queue

    ctx, worker, pushed = terminal_context
    handed_off = _launched(clock)
    owner = execution_owner_record(clock.root, {"id": WAKE, "_attempt": 1}, "pooled")
    task = {"id": WAKE, "type": "task", "_attempt": 1, "chat_id": 7, "metadata": dict(OWN), "_execution_owner": owner}
    ctx.RUNNING.clear()
    ctx.RUNNING[WAKE] = {"task": task, "attempt": 1, "worker_id": 7}
    worker.busy_task_id = WAKE
    ctx.consciousness = clock.clock
    monkeypatch.setattr(queue, "RUNNING", ctx.RUNNING)
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(clock_module.time, "time", lambda: FINISH)
    deadline = clock.clock.next_wake_at
    handed_off(WAKE, True)  # the direct stack returned over its handoff: this ended no logical wake
    assert clock.clock.live_wake() == WAKE and clock.clock.next_wake_at == deadline
    assert LAST_WAKE_STATE_KEY not in clock.store
    return SimpleNamespace(clock=clock.clock, store=clock.store, launches=clock.launches, receipt=clock.receipt,
                           root=clock.root, ctx=ctx, worker=worker, pushed=pushed, owner=owner, task=task,
                           deadline=deadline)


def _running_row(pooled, **fields):
    write_task_result(pooled.root, WAKE, "running", execution_owner=pooled.owner, metadata=dict(OWN),
                      task_attempt=1, **fields)


def _dispatch(pooled, status, **fields):
    from supervisor.events import dispatch_event

    dispatch_event({"type": "task_done", "task_id": WAKE, "worker_id": 7, "status": status,
                    "_files_prepared_attempt": 1, **fields}, pooled.ctx)


def _settled(pooled, *, backoff, interval=DEFAULT):
    assert pooled.clock.next_wake_at == FINISH + interval * backoff
    assert pooled.clock._backoff == backoff and pooled.store[LAST_WAKE_STATE_KEY] == FINISH
    assert pooled.store[OWN_WAKE_STATE_KEY] == {**pooled.store[OWN_WAKE_STATE_KEY], "task_id": WAKE, "finished": True}
    assert pooled.clock.live_wake() == "" and pooled.clock.status_snapshot()["last_wake_outcome"] == (
        "failed" if backoff > 1 else "done")


def _unsettled(pooled):
    assert pooled.clock.next_wake_at == pooled.deadline and LAST_WAKE_STATE_KEY not in pooled.store
    assert pooled.clock._backoff == 1 and pooled.store[OWN_WAKE_STATE_KEY]["finished"] is False


# --- the lifecycle-fault terminal ----------------------------------------------------------


def test_lifecycle_fault_settles_the_known_wake_once_with_the_chosen_interval(pooled):
    _running_row(pooled)
    pooled.store[INTERVAL_STATE_KEY] = 2000
    _dispatch(pooled, "running")
    assert load_task_result(pooled.root, WAKE)["reason_code"] == "task_done_lifecycle_fault"
    assert WAKE not in pooled.ctx.RUNNING and pooled.worker.busy_task_id is None
    _settled(pooled, backoff=2, interval=2000)
    # A repeated terminal and a restarted clock neither move it again nor start over it.
    settled = pooled.clock.next_wake_at
    _dispatch(pooled, "completed")
    assert pooled.clock.next_wake_at == settled and pooled.clock._backoff == 2
    again = BackgroundConsciousness(pooled.root, pooled.root / "repo", lambda: 7, now=FINISH)
    again._wake_finished(WAKE, True, metadata=OWN, pooled=True)
    assert again.next_wake_at == settled and pooled.store[LAST_WAKE_STATE_KEY] == FINISH
    assert pooled.clock.tick(FINISH + 1) == "not_due" and len(pooled.launches) == 1


def test_a_missing_result_fault_keeps_the_known_identity_and_settles(pooled, file_recovery):
    """The real path: a terminal without a result returns from file recovery to the fault lane."""
    from supervisor import task_reaper
    from supervisor.events import dispatch_event

    _ctx, _worker, _pushed, jobs, returned = file_recovery
    assert load_task_result(pooled.root, WAKE) is None
    dispatch_event({"type": "task_done", "task_id": WAKE, "worker_id": 7, "status": "completed"}, pooled.ctx)
    task_reaper._recover_terminal_files(jobs.get_nowait())
    _unsettled(pooled)  # the fault lane has not spoken yet
    dispatch_event(returned.get_nowait(), pooled.ctx)
    stored = load_task_result(pooled.root, WAKE)
    assert stored["status"] == "failed" and stored["reason_code"] == "task_done_lifecycle_fault"
    assert stored["metadata"]["wake_cause"] == "own_clock"  # the row says whose it was, from its RUNNING row
    assert stored["execution_owner"] == pooled.owner and stored["task_attempt"] == 1
    from ouroboros.task_status import task_execution_observation
    assert task_execution_observation(pooled.root, stored)["kind"] == "pooled"
    assert pooled.pushed[-1]["execution_owner"] == pooled.owner
    _settled(pooled, backoff=2)


def test_an_early_completed_checkpoint_beside_a_fault_counts_failed(pooled, file_recovery):
    from ouroboros.outcomes import normalize_outcome_axes
    from supervisor import task_reaper
    from supervisor.events import dispatch_event

    _ctx, _worker, _pushed, jobs, returned = file_recovery
    child = headless.prepare_task_drive(pooled.root, WAKE, "empty")
    pooled.task["drive_root"] = str(child)
    write_task_result(pooled.root, WAKE, "completed", result="Authored note", metadata=dict(OWN),
                      execution_owner=pooled.owner, task_attempt=1,
                      root_phase_checkpoint={"post_task_synthesis": "completed"})
    dispatch_event({"type": "task_done", "task_id": WAKE, "worker_id": 7, "status": "completed"}, pooled.ctx)
    task_reaper._recover_terminal_files(jobs.get_nowait())
    dispatch_event(returned.get_nowait(), pooled.ctx)
    stored = load_task_result(pooled.root, WAKE)
    assert stored["status"] == "completed" and normalize_outcome_axes(stored)["execution"]["status"] == "infra_failed"
    assert stored["result"] == "Authored note"
    _settled(pooled, backoff=2)


@pytest.mark.parametrize("status,axes,backoff", [
    ("completed", None, 1),
    ("failed", None, 2),
    ("completed", {"execution": {"status": "infra_failed", "reason_code": "provider_lost"}}, 2),
])
def test_the_ordinary_terminal_reads_the_outcome_not_only_the_lifecycle_word(pooled, status, axes, backoff):
    write_task_result(pooled.root, WAKE, status, result="note", metadata=dict(OWN), execution_owner=pooled.owner,
                      task_attempt=1, artifact_status="ready", **({"outcome_axes": axes} if axes else {}))
    _dispatch(pooled, status)
    assert WAKE not in pooled.ctx.RUNNING
    _settled(pooled, backoff=backoff)


def test_a_reaped_terminal_without_identity_on_its_row_keeps_the_identity_of_its_event(pooled):
    """The reaper pops RUNNING before its terminal dispatches; an unstarted row knows only its status."""
    write_task_result(pooled.root, WAKE, "failed", result="reaped", artifact_status="ready")
    pooled.ctx.RUNNING.clear()
    pooled.worker.busy_task_id = None
    _dispatch(pooled, "failed", metadata=dict(OWN))
    _settled(pooled, backoff=2)


# --- what has an owner settles nothing ----------------------------------------------------


def test_interrupted_recoverable_work_keeps_the_clock_until_its_real_terminal(pooled):
    _running_row(pooled)
    write_task_result(pooled.root, WAKE, "interrupted", result="interrupted")
    _dispatch(pooled, "interrupted")
    _unsettled(pooled)
    # Restored by the same generation's owner: the same id runs again and really ends.
    pooled.ctx.RUNNING[WAKE] = {"task": pooled.task, "attempt": 1, "worker_id": 7}
    write_task_result(pooled.root, WAKE, "completed", result="note", task_attempt=1, artifact_status="ready")
    _dispatch(pooled, "completed")
    _settled(pooled, backoff=1)


def test_cancellation_custody_owns_the_fault_until_it_settles(pooled):
    from ouroboros.cancel_intents import request_cancel

    _running_row(pooled)
    request_cancel(pooled.root, WAKE, reason="test stop", source="owner", requested_by="owner")
    _dispatch(pooled, "running")
    assert WAKE in pooled.ctx.RUNNING and load_task_result(pooled.root, WAKE)["status"] == "running"
    _unsettled(pooled)
    assert pooled.clock.live_wake() == WAKE


def test_unknown_fault_storage_retains_recovery_and_settles_nothing(pooled, file_recovery, monkeypatch):
    import ouroboros.task_results as results

    _ctx, _worker, _pushed, jobs, _returned = file_recovery
    _running_row(pooled)
    with monkeypatch.context() as patched:
        patched.setattr(results, "write_task_result", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        _dispatch(pooled, "running")
    assert WAKE in pooled.ctx.RUNNING and pooled.worker.busy_task_id == WAKE
    assert not jobs.empty()  # the existing file-recovery owner kept it
    _unsettled(pooled)
    assert pooled.clock.live_wake() == WAKE


# --- a terminal no dispatcher reported: only the tick that would start over it reads it ----


def _carrier_gone(pooled):
    pooled.ctx.RUNNING.clear()
    pooled.worker.busy_task_id = None


def test_the_orphan_sweep_terminal_settles_at_the_due_tick_instead_of_a_new_wake(pooled, monkeypatch):
    from ouroboros.task_status import reconcile_orphaned_running_tasks
    from ouroboros.utils import append_jsonl
    from tests._shared import write_queue_document

    _carrier_gone(pooled)
    write_task_result(pooled.root, WAKE, "running", execution_owner=pooled.owner, metadata=dict(OWN), task_attempt=1,
                      result="Task is running.", ts="2026-05-28T00:00:00+00:00")
    (pooled.root / "state").mkdir(exist_ok=True)
    write_queue_document(pooled.root / "state" / "queue_snapshot.json",
                         {"ts": clock_module._iso(FINISH - 5), "pending": [], "running": []})
    events = pooled.root / "logs" / "events.jsonl"
    append_jsonl(events, {"ts": "2026-05-28T00:00:01+00:00", "type": "llm_round", "task_id": WAKE})
    append_jsonl(events, {"ts": clock_module._iso(FINISH - 10), "type": "worker_boot"})
    assert reconcile_orphaned_running_tasks(pooled.root) == 1
    assert load_task_result(pooled.root, WAKE)["reason_code"] == "orphaned_running_after_worker_restart"
    launches = len(pooled.launches)
    assert pooled.clock.tick(pooled.deadline - 1) == "not_due"
    _unsettled(pooled)  # no due attempt, no read: the sweep's row waits for the tick that would launch over it
    assert pooled.clock.tick(FINISH) == "not_due"
    assert len(pooled.launches) == launches
    _settled(pooled, backoff=2)
    assert pooled.clock.tick(FINISH + 2 * DEFAULT) == "launched" and len(pooled.launches) == launches + 1


@pytest.mark.parametrize("liveness", [True, None])
def test_a_recorded_terminal_with_live_or_unknown_effects_is_not_adopted(pooled, monkeypatch, liveness):
    """A settled row is no proof its carrier ended: reap in progress, retained review, billing synthesis."""
    from supervisor import queue

    _carrier_gone(pooled)
    write_task_result(pooled.root, WAKE, "failed", result="note", metadata=dict(OWN), execution_owner=pooled.owner,
                      task_attempt=1)
    probes = []
    monkeypatch.setattr(queue, "task_settlement_liveness", lambda task_id: probes.append(task_id) or liveness)
    launches = len(pooled.launches)
    assert pooled.clock.tick(FINISH) == "wake_live"
    assert probes == [WAKE] and len(pooled.launches) == launches
    _unsettled(pooled)
    monkeypatch.setattr(queue, "task_settlement_liveness", lambda task_id: False)
    assert pooled.clock.tick(FINISH) == "not_due"
    _settled(pooled, backoff=2)


@pytest.mark.parametrize("shape", ["unreadable", "absent", "nonterminal", "interrupted", "direct", "cancelled"])
def test_what_the_tick_cannot_prove_settles_nothing_and_keeps_todays_start(pooled, monkeypatch, shape):
    """Unknown never counts completed, and no unprovable row wedges the alarm."""
    import ouroboros.task_results as results

    _carrier_gone(pooled)
    if shape == "unreadable":
        real = results.load_task_result

        def unreadable(root, task_id, **kwargs):
            if task_id == WAKE and kwargs.get("strict"):
                raise OSError("task store unreadable")
            return real(root, task_id, **kwargs)

        monkeypatch.setattr(results, "load_task_result", unreadable)
    elif shape == "nonterminal":
        _running_row(pooled)
    elif shape == "interrupted":
        write_task_result(pooled.root, WAKE, "interrupted", execution_owner=pooled.owner, metadata=dict(OWN), task_attempt=1)
    elif shape == "direct":
        direct = execution_owner_record(pooled.root, {"id": WAKE, "_attempt": 1}, "direct")
        write_task_result(pooled.root, WAKE, "completed", result="note", execution_owner=direct, metadata=dict(OWN),
                          task_attempt=1, _is_direct_chat=True)
    elif shape == "cancelled":  # a Stop whose event never arrived: its door's own next start stands
        write_task_result(pooled.root, WAKE, "cancelled", result="stopped", execution_owner=pooled.owner,
                          metadata=dict(OWN), task_attempt=1)
    launches = len(pooled.launches)
    pooled.receipt["task_id"] = "wake0002"
    assert pooled.clock.tick(FINISH) == "launched"
    assert len(pooled.launches) == launches + 1
    assert pooled.clock._backoff == 1 and LAST_WAKE_STATE_KEY not in pooled.store
    assert pooled.clock.status_snapshot()["last_wake_outcome"] == "running"


def test_a_due_clock_reads_the_row_once_and_a_quiet_clock_never_reads_it(pooled, monkeypatch):
    """The narrow check is no reconciler: one addressed read, only at the launch it would replace."""
    import ouroboros.task_results as results

    _carrier_gone(pooled)
    reads = []
    real = results.load_task_result

    def counted(root, task_id, **kwargs):
        reads.append(task_id)
        return real(root, task_id, **kwargs)

    monkeypatch.setattr(results, "load_task_result", counted)
    assert pooled.clock.tick(pooled.deadline - 1) == "not_due"
    assert reads == []
    pooled.store["bg_consciousness_enabled"] = False
    assert pooled.clock.tick(pooled.deadline - 1) == "disabled"
    assert reads == []


def test_a_focus_or_child_row_under_another_id_never_touches_the_clock(pooled):
    """Own-clock custody follows one id; the pooled hook still filters by cause and carrier."""
    from supervisor.consciousness_carriers import finish_pooled_wake

    focus = {**OWN, "wake_cause": "focus", "consciousness_cause": "place:mail"}
    finish_pooled_wake(pooled.ctx, WAKE, {"status": "failed", "metadata": focus, "execution_owner": pooled.owner})
    finish_pooled_wake(pooled.ctx, "other", {"status": "failed", "metadata": OWN, "execution_owner": pooled.owner})
    finish_pooled_wake(pooled.ctx, WAKE, {"status": "failed", "metadata": OWN, "execution_owner": {"kind": "direct"}})
    _unsettled(pooled)
