"""A launch's dispatch evidence lands whole before anything says ``possible`` (Audit05 B).

The real launcher (``worker_assignment._claim_worker_launch`` / ``assign_tasks``) over
real schedule admission and a real SQLite holder. A refused or rolled-back mark leaves
the same first dispatch for every consumer (budget stop, owner Restart, Pause/Resume,
deadline, restore); a commit with an unknown outcome is held visibly; a committed mark
whose later step failed may finish and put once in this generation only; an entered put
binds its worker. Restore splits positive ``possible`` from contention, and a legacy
row whose own receipt proves no first handoff recovers under its stop's own rule.
"""
from __future__ import annotations

import contextlib
import copy
from types import SimpleNamespace

import pytest

from ouroboros import store
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, queue_schedules as schedules, task_admission, worker_assignment
from tests._store_journal_shared import exclusive_hold
from tests.test_g1_followup_policy import _assignment_world, register, row, world as _world

pytestmark = pytest.mark.serial
followup_world = _world  # fixture alias: injection does not shadow the import (F811)


@pytest.fixture
def world(followup_world, monkeypatch):
    from supervisor import workers

    # The real launch writes drive obligations through the pool, too.
    monkeypatch.setattr(workers, "DRIVE_ROOT", followup_world[1])
    return followup_world


@contextlib.contextmanager
def held_store(root):
    assert store.store_path(root).exists()
    holder = exclusive_hold(store.store_path(root))  # its reads busy too, under WAL as well
    try:
        yield
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def admitted(world, relation="independent"):
    _, root, pending = world
    registered = register(world, relation)
    queue.check_scheduled_tasks()
    [task] = pending
    return registered, task


def claim(task, sent, *, zero_budget=False):
    worker = SimpleNamespace(wid=0, in_q=SimpleNamespace(put=lambda launched: sent.append(copy.deepcopy(launched))))
    with (store.acquisition_budget(0) if zero_budget else contextlib.nullcontext()), queue._queue_lock:
        return worker_assignment._claim_worker_launch(queue, task, worker)


def snapshot_row(root, task_id):
    import json

    data = json.loads((root / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    return next(row["task"] if "task" in row else row for row in data["pending"]
                if (row.get("task") or row).get("id") == task_id)


def restore(rows, **kwargs):
    from supervisor.queue_snapshot import _retain_snapshot_pending

    retained, _, consumed, _ = _retain_snapshot_pending([copy.deepcopy(r) for r in rows], [], stale=False, **kwargs)
    assert not consumed
    return retained


def hold_reason(task):
    from ouroboros.work_stop import continuation_refusal, hold_fact
    return ((hold_fact(task) or continuation_refusal(task)) or {}).get("reason")


@pytest.mark.parametrize("zero_budget", [False, True])
@pytest.mark.parametrize("locked", [False, True])
def test_marks_commit_together_or_change_nothing(world, monkeypatch, zero_budget, locked):
    """Probe04 reversed: a held store at the canonical mark refuses everything, memory included."""
    _, root, pending = world
    _, task = admitted(world)
    sent, holder, armed = [], contextlib.ExitStack(), [locked]
    real_commit = task_admission.commit_dispatch_marks

    def commit(candidate, projections):
        if armed[0]:
            armed[0] = False
            holder.enter_context(held_store(root))
        return real_commit(candidate, projections)

    monkeypatch.setattr(task_admission, "commit_dispatch_marks", commit)
    with holder:
        outcome = claim(task, sent, zero_budget=zero_budget)
    stored = load_task_result(root, task["id"])
    if locked:
        assert outcome is False and sent == [] and pending == [task]
        assert task["admitted_dispatch"] == "none"
        assert stored["status"] == "scheduled" and stored["schedule_admission"]["dispatch"] == "none"
        assert stored.get("admitted_dispatch", "none") == "none"
        assert snapshot_row(root, task["id"])["admitted_dispatch"] == "none"
        [restored] = restore([task])
        assert not restored.get("_project_admission_restore_hold") and hold_reason(restored) != "dispatch_outcome_unknown"
        assert claim(task, sent) is True  # released: the same work is handed off exactly once
    else:
        assert outcome is True
    assert len(sent) == 1 and sent[0]["id"] == task["id"]
    stored = load_task_result(root, task["id"])
    assert stored["schedule_admission"]["dispatch"] == "possible" and stored["admitted_dispatch"] == "possible"
    assert stored["admitted_dispatch_attempt"] == 1 and stored["status"] == "running"


@pytest.mark.parametrize("refuser", ["schedule", "project"])
def test_one_write_and_either_refusal_writes_neither_mark(world, monkeypatch, refuser):
    _, root, _ = world
    _, task = admitted(world)
    stored = load_task_result(root, task["id"])
    if refuser == "schedule":
        write_task_result(root, task["id"], "scheduled",
                          schedule_admission={**stored["schedule_admission"], "token": "foreign"})
    else:
        write_task_result(root, task["id"], "scheduled", admitted_dispatch_attempt=2)
    writes = []
    real_write = task_admission.write_task_result
    monkeypatch.setattr(task_admission, "write_task_result",
                        lambda *a, **k: writes.append(k.get("_field_projector")) or real_write(*a, **k))
    sent = []
    assert claim(task, sent) is False and sent == [] and task["admitted_dispatch"] == "none"
    assert len(writes) == 1, "one task-result transaction carries both owners' marks"
    after = load_task_result(root, task["id"])
    assert after["schedule_admission"]["dispatch"] == "none" and after["status"] == "scheduled"
    assert after.get("admitted_dispatch", "none") == "none"


@pytest.mark.parametrize("unknown", ["readback_raised", "commit_unconfirmed"])
def test_an_unknown_commit_puts_at_most_once_here_and_never_after_a_crash(world, monkeypatch, unknown):
    """A commit whose outcome is unknown is possible: this process re-reads it and puts once;
    after a crash only the durable possible remains, held on restore."""
    world, workers, sent = _assignment_world(world, monkeypatch)
    _, root, pending = world
    _, task = admitted(world)
    real_load, real_write = task_admission.load_task_result, task_admission.write_task_result
    armed = [True]

    def load(*args, **kwargs):
        if unknown == "readback_raised" and armed[0]:
            armed[0] = False
            error = store.StoreBusy("readback busy", reason="contention")
            error.rollback_confirmed = True
            raise error
        return real_load(*args, **kwargs)

    def write(*args, **kwargs):
        written = real_write(*args, **kwargs)
        if unknown == "commit_unconfirmed" and armed[0] and kwargs.get("_field_projector"):
            armed[0] = False
            error = store.StoreBusy("commit outcome unknown", reason="contention")
            error.rollback_confirmed = False
            raise error
        return written

    monkeypatch.setattr(task_admission, "load_task_result", load)
    monkeypatch.setattr(task_admission, "write_task_result", write)
    workers.assign_tasks()
    assert sent == [] and pending == [task] and task["admitted_dispatch"] == "possible"
    assert load_task_result(root, task["id"])["schedule_admission"]["dispatch"] == "possible"
    assert snapshot_row(root, task["id"])["admitted_dispatch"] == "possible"
    stale = copy.deepcopy(task)
    crashed = dict(worker_assignment._COMMITTED_UNSENT)
    worker_assignment._COMMITTED_UNSENT.clear()  # a crash: the next process has only durable facts
    [held] = restore([stale])
    assert hold_reason(held) == "dispatch_outcome_unknown"
    worker_assignment._COMMITTED_UNSENT.update(crashed)  # back in the same process
    workers.assign_tasks()
    workers.assign_tasks()
    assert [entry[1]["id"] for entry in sent] == [task["id"]], "one put, by the process that owns the launch"


@pytest.mark.parametrize("step", ["row_write", "snapshot", "drive_started"])
def test_a_committed_mark_finishes_and_puts_once_in_the_same_generation(world, monkeypatch, step):
    """Postcommit/preput: the process-local witness lets the same row, attempt and marks
    finish once; reconcile never retires or reclaims the live pending launch meanwhile."""
    world, workers, sent = _assignment_world(world, monkeypatch)
    _, root, pending = world
    registered, task = admitted(world)
    failures = []
    real_rows, real_persist = schedules._write_scheduled_tasks, queue.persist_queue_snapshot

    def rows(data, *args, **kwargs):
        if step == "row_write" and not failures and any(
                (r.get("occurrence") or {}).get("dispatch") == "possible" for r in data["tasks"]):
            failures.append(step)
            raise OSError("row write failed")
        return real_rows(data, *args, **kwargs)

    def persist(**kwargs):
        if step == "snapshot" and not failures and kwargs.get("reason") == "worker_launch_claimed":
            failures.append(step)
            return False
        return real_persist(**kwargs)

    from ouroboros import obligations

    real_started = obligations.drive_started

    def started(root_, candidate):
        if step == "drive_started" and not failures:
            failures.append(step)
            raise store.StoreBusy("pending-drives row busy", reason="contention")
        return real_started(root_, candidate)

    monkeypatch.setattr(schedules, "_write_scheduled_tasks", rows)
    monkeypatch.setattr(queue, "persist_queue_snapshot", persist)
    monkeypatch.setattr(obligations, "drive_started", started)
    workers.assign_tasks()
    assert failures == [step] and sent == [] and pending == [task]
    assert row(root, registered["id"])["occurrence"].get("dispatch") != "possible", "the row is marked last"
    assert load_task_result(root, task["id"])["schedule_admission"]["dispatch"] == "possible"
    assert task["id"] in worker_assignment._COMMITTED_UNSENT
    before = row(root, registered["id"])
    queue.check_scheduled_tasks()  # reconcile between commit and first put: the live launch stays
    assert row(root, registered["id"]) == before and pending == [task]
    workers.assign_tasks()
    assert [entry[1]["id"] for entry in sent] == [task["id"]] and not pending
    assert row(root, registered["id"])["occurrence"]["dispatch"] == "possible"
    assert task["id"] not in worker_assignment._COMMITTED_UNSENT
    workers.assign_tasks()
    assert len(sent) == 1


def test_a_committed_mark_never_replays_after_a_crash(world, monkeypatch):
    """The witness is process memory: a crash leaves only the durable possible, held on restore."""
    _, root, pending = world
    _, task = admitted(world)
    real_persist = queue.persist_queue_snapshot
    monkeypatch.setattr(queue, "persist_queue_snapshot",
                        lambda **kw: False if kw.get("reason") == "worker_launch_claimed" else real_persist(**kw))
    assert claim(task, []) is False
    stale = snapshot_row(root, task["id"])
    assert stale["admitted_dispatch"] == "none"  # the locator snapshot, written before the commit
    worker_assignment._COMMITTED_UNSENT.clear()  # the process ended
    [restored] = restore([stale])
    assert hold_reason(restored) == "dispatch_outcome_unknown", "a committed mark is never silently dropped"


def test_project_rows_keep_their_project_custody_after_a_stale_commit(tmp_path, monkeypatch):
    from tests.test_schedule_occurrence import _project, _row
    from supervisor import queue as queue_module, state, workers
    from supervisor.task_admission import revalidate_project_holds

    root = tmp_path / "data"
    root.mkdir()
    state.init(root)
    state.save_state({"owner_chat_id": 1})
    queue_module.init(root)
    pending: list = []
    queue_module.init_queue_refs(pending, {}, {"value": 0})
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    q = SimpleNamespace(queue=queue_module, pending=pending, root=root)
    _project(q, "proj")
    _row(q, "s1", intent={"kind": "room_default"}, project_id="proj", chat_id=None)
    queue_module.check_scheduled_tasks()
    [task] = pending
    assert task.get("project_id") == "proj"
    real_persist = queue_module.persist_queue_snapshot
    monkeypatch.setattr(queue_module, "persist_queue_snapshot",
                        lambda **kw: False if kw.get("reason") == "worker_launch_claimed" else real_persist(**kw))
    assert claim(task, []) is False
    stale = snapshot_row(root, task["id"])
    worker_assignment._COMMITTED_UNSENT.clear()
    [restored] = restore([stale])
    assert restored["_project_admission_restore_hold"]
    pending[:] = [restored]
    monkeypatch.setattr(queue_module, "persist_queue_snapshot", real_persist)
    revalidate_project_holds()
    assert pending[0].get("_project_admission_restore_hold"), "a committed Project mark keeps its custody"


def test_a_put_that_raises_after_entry_binds_its_worker(world, monkeypatch):
    world, workers, sent = _assignment_world(world, monkeypatch)
    _, root, pending = world
    _, task = admitted(world)
    second = {"id": "second", "type": "task", "chat_id": 0, "text": "another", "_project_scope_none": True}
    queue.enqueue_task(second, restoring_snapshot=True)

    def put(launched):
        sent.append((0, copy.deepcopy(launched)))
        raise BrokenPipeError("the feeder raised after the item was queued")

    workers.WORKERS[0].in_q.put = put
    workers.assign_tasks()
    assert [entry[1]["id"] for entry in sent] == [task["id"]]
    assert workers.WORKERS[0].busy_task_id == task["id"] and queue.RUNNING[task["id"]]["worker_id"] == 0
    assert [t["id"] for t in pending] == ["second"], "an uncertain slot runs no second task"
    assert task["id"] not in worker_assignment._COMMITTED_UNSENT
    workers.assign_tasks()
    assert len(sent) == 1


def test_in_process_consumers_see_a_definite_refusal_as_never_dispatched(world, monkeypatch):
    """Budget stop, owner Restart, Pause/Resume and deadline expiry after a rolled-back mark."""
    from supervisor.events_budget import HOLD_OWNER_RESTART
    from supervisor.queue_timeouts import pending_deadline_expirable
    from supervisor.queue_transitions import queued_admitted_dispatch
    from supervisor.restart_retention import hold_never_started

    _, root, pending = world
    _, task = admitted(world)
    real_commit = task_admission.commit_dispatch_marks

    def commit(candidate, projections):
        with held_store(root):
            return real_commit(candidate, projections)

    monkeypatch.setattr(task_admission, "commit_dispatch_marks", commit)
    assert claim(task, []) is False
    assert task["admitted_dispatch"] == "none"
    assert queued_admitted_dispatch(task) is False
    probe = copy.deepcopy(task)
    assert hold_never_started(probe, [], []) is True and hold_reason(probe) == HOLD_OWNER_RESTART
    expired = {**copy.deepcopy(task), "deadline_at": "2000-01-01T00:00:00+00:00"}
    assert pending_deadline_expirable(expired) is True


def test_budget_exhaustion_after_a_definite_refusal_pauses_before_dispatch(world, monkeypatch):
    world, workers, sent = _assignment_world(world, monkeypatch)
    _, root, pending = world
    _, task = admitted(world)
    real_commit = task_admission.commit_dispatch_marks

    def commit(candidate, projections):
        with held_store(root):
            return real_commit(candidate, projections)

    monkeypatch.setattr(task_admission, "commit_dispatch_marks", commit)
    workers.assign_tasks()
    assert sent == [] and task["admitted_dispatch"] == "none"
    monkeypatch.setattr("supervisor.state.budget_remaining", lambda *_a, **_k: 0.0)
    workers.assign_tasks()
    assert task["_budget_pause"]["status"] == "paused_before_dispatch" and task["_budget_pause"]["replay_safe"]
    assert load_task_result(root, task["id"])["status"] == "scheduled"


def test_restore_drops_contention_for_reconcile_and_holds_positive_possible(world, monkeypatch):
    from supervisor import schedule_occurrence as occurrences

    _, root, pending = world
    registered, task = admitted(world)
    with held_store(root):
        assert occurrences.restore_verdict(copy.deepcopy(task)) == "busy"
        assert restore([task]) == [], "contention at boot is not a manual hold"
    pending.clear()
    queue.check_scheduled_tasks()  # readable again: the occurrence protocol republishes the same task
    assert [t["id"] for t in pending] == [task["id"]]
    assert occurrences.record_dispatch_possible(task)
    stale = {**copy.deepcopy(task), "admitted_dispatch": "none"}
    [held] = restore([stale])
    assert hold_reason(held) == "dispatch_outcome_unknown"


@pytest.mark.parametrize("prior", ["fresh", "held_unknown", "project_held"])
def test_legacy_no_first_dispatch_rows_recover_under_their_stop_rule(world, monkeypatch, prior):
    """A row an old launcher left at ``possible`` whose own receipt proves no first handoff."""
    from supervisor import restart_retention
    from supervisor.events_budget import HOLD_SAVED_WORK, hold_budget_row
    from supervisor.task_admission import revalidate_project_holds

    _, root, pending = world
    _, task = admitted(world)
    assert task_admission.record_project_dispatch_possible(task)  # the old launcher's first mark landed
    legacy = {**copy.deepcopy(task), "admitted_dispatch": "possible"}  # ...then its schedule mark failed
    if prior == "held_unknown":
        hold_budget_row(legacy, reason="dispatch_outcome_unknown", detail="old restore", result_root=root)
    if prior == "project_held":
        legacy["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed",
                                                     "detail": "The task may already have reached a worker"}
    [stopped] = restore([legacy])
    assert stopped["admitted_dispatch"] == "none"
    assert hold_reason(stopped) == HOLD_SAVED_WORK, "an application stop holds it like any never-started row"
    monkeypatch.setattr(restart_retention, "fresh_return_transaction", lambda _root: {"queue_ids": [task["id"]]})
    [returned] = restore([legacy])
    assert returned["admitted_dispatch"] == "none" and hold_reason(returned) is None
    if prior == "project_held":
        assert returned["_project_admission_restore_hold"]
        pending[:] = [returned]
        revalidate_project_holds()
        assert not pending[0].get("_project_admission_restore_hold"), "release reads the same receipt proof"


def test_rows_without_an_occurrence_never_borrow_the_receipt_proof(world, monkeypatch):
    """A non-schedule first attempt the snapshot calls possible stays held: no receipt proves it unsent."""
    from supervisor import schedule_occurrence as occurrences
    from supervisor.task_admission import revalidate_project_holds

    _, root, pending = world
    plain = queue.enqueue_task({"id": "plain-root", "type": "task", "chat_id": 0, "text": "plain"})
    assert plain.get("_project_scope_none") is True and plain["admitted_dispatch"] == "none"
    write_task_result(root, plain["id"], "scheduled", result="accepted")
    assert task_admission.record_project_dispatch_possible(plain)  # an earlier launcher's mark
    started = {**copy.deepcopy(plain), "admitted_dispatch": "possible"}
    assert occurrences.first_dispatch_unsent(copy.deepcopy(started)) is False
    [held] = restore([started])
    assert held["admitted_dispatch"] == "possible" and held["_project_admission_restore_hold"]
    pending[:] = [held]
    revalidate_project_holds()
    assert pending[0].get("_project_admission_restore_hold"), "possibly dispatched work is never released"


@pytest.mark.parametrize("case", ["retry_attempt", "receipt_possible"])
def test_legacy_discriminator_keeps_every_other_possible_unknown(world, monkeypatch, case):
    from supervisor import schedule_occurrence as occurrences

    _, root, pending = world
    _, task = admitted(world)
    legacy = {**copy.deepcopy(task), "admitted_dispatch": "possible"}
    if case == "retry_attempt":
        legacy["_attempt"] = 2
        [held] = restore([legacy])
        assert held["admitted_dispatch"] == "possible" and held["_project_admission_restore_hold"]
    else:
        assert occurrences.record_dispatch_possible(task)
        [held] = restore([legacy])
        assert held["admitted_dispatch"] == "possible" and hold_reason(held) == "dispatch_outcome_unknown"


@pytest.mark.parametrize("stop,truth", [("returning", "unsent"), ("app_stop", "unsent"),
                                        ("owner_restart", "unsent"), ("returning", "possible")])
def test_a_receipt_busy_at_boot_is_rechecked_without_another_boot(world, monkeypatch, stop, truth):
    """The snapshot says possible and the receipt is busy: this stop's own rule applies under the
    Project custody that re-reads the receipt each tick. Proven unsent recovers; possible stays held."""
    from supervisor import restart_retention, schedule_occurrence as occurrences
    from supervisor.events_budget import HOLD_OWNER_RESTART, HOLD_SAVED_WORK
    from supervisor.task_admission import revalidate_project_holds
    from ouroboros import delegate_recovery
    from ouroboros.work_stop import current_stop, stop_presentation

    _, root, pending = world
    _, task = admitted(world)
    assert task_admission.record_project_dispatch_possible(task)  # an old launcher's first mark
    if truth == "possible":
        assert occurrences.record_dispatch_possible(task)
    legacy = {**copy.deepcopy(task), "admitted_dispatch": "possible"}
    if stop == "returning":
        monkeypatch.setattr(restart_retention, "fresh_return_transaction", lambda _root: {"queue_ids": [task["id"]]})
    if stop == "owner_restart":
        transaction = delegate_recovery.prepare_owner_restart(root)
        monkeypatch.setattr(delegate_recovery, "_PROCESS_GENERATION", transaction["generation"] + "-successor")
        assert delegate_recovery.boot_door(root) == "owner_restart"
    stop_hold = {"returning": None, "app_stop": HOLD_SAVED_WORK, "owner_restart": HOLD_OWNER_RESTART}[stop]
    with held_store(root), store.acquisition_budget(0.2):  # contention reads the same, without a long wait
        assert occurrences.restore_verdict(copy.deepcopy(legacy)) == "busy"
        [restored] = restore([legacy])
    assert restored["_project_admission_restore_hold"] and hold_reason(restored) == stop_hold
    pending[:] = [restored]
    revalidate_project_holds()
    queue.check_scheduled_tasks()  # the live row is the occurrence: never republished beside it
    [row_] = pending
    assert hold_reason(row_) == stop_hold, "the stop's own hold stays independent of the receipt check"
    if stop == "owner_restart":
        for stopped in (restored, row_):
            assert current_stop(stopped)["door"] == "owner_restart"
            assert stop_presentation(stopped)["reason"]["text"] == "Held after a restart"
    if truth == "unsent":
        assert row_["admitted_dispatch"] == "none" and not row_.get("_project_admission_restore_hold")
    else:
        assert row_["admitted_dispatch"] == "possible" and row_["_project_admission_restore_hold"]


def unsent_launch(world, monkeypatch):
    """A committed launch whose later step failed: the marks landed, the put was never entered."""
    world, workers, sent = _assignment_world(world, monkeypatch)
    _, root, pending = world
    _, task = admitted(world)
    real_persist, failed = queue.persist_queue_snapshot, []

    def persist(**kwargs):
        if kwargs.get("reason") == "worker_launch_claimed" and not failed:
            failed.append(True)
            return False
        return real_persist(**kwargs)

    monkeypatch.setattr(queue, "persist_queue_snapshot", persist)
    workers.assign_tasks()
    assert failed and sent == [] and pending == [task] and task["admitted_dispatch"] == "possible"
    assert worker_assignment._unsent(task) is not None
    assert load_task_result(root, task["id"])["schedule_admission"]["dispatch"] == "possible"
    return root, pending, workers, sent, task


def test_a_budget_stop_before_the_put_claims_no_prior_dispatch(world, monkeypatch):
    root, pending, workers, sent, task = unsent_launch(world, monkeypatch)
    monkeypatch.setattr("supervisor.state.budget_remaining", lambda *_a, **_k: 0.0)
    workers.assign_tasks()
    assert sent == [] and pending == [task]
    assert task["_budget_pause"]["status"] == "paused_before_dispatch" and task["_budget_pause"]["replay_safe"]
    assert load_task_result(root, task["id"])["status"] == "scheduled", "never terminal 'after prior dispatch'"


def test_a_deadline_before_the_put_terminalizes_without_dispatch_and_retires_the_witness(world, monkeypatch):
    import queue as stdqueue
    from supervisor import queue_timeouts

    root, pending, workers, sent, task = unsent_launch(world, monkeypatch)
    jobs = stdqueue.Queue()
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "_reap_queue", jobs)
    monkeypatch.setattr(queue_timeouts, "_pending_deadline_jobs", {})
    task["deadline_at"] = "2000-01-01T00:00:00+00:00"  # the calendar deadline passed before any put
    workers.assign_tasks()
    while not jobs.empty():
        queue_timeouts._reap_pending_deadline(jobs.get_nowait())
    stored = load_task_result(root, task["id"])
    assert sent == [] and pending == [] and stored["status"] == "failed"
    assert stored["reason_code"] == queue_timeouts.REASON_DEADLINE
    workers.assign_tasks()
    assert task["id"] not in worker_assignment._COMMITTED_UNSENT, "its row left the queue: the witness goes too"


def test_the_witness_carries_its_pre_commit_deadline_claim(world, monkeypatch):
    """A saved successor's exact non-put claim survives a failed later step, under launch admission."""
    _, root, pending = world
    _, task = admitted(world)
    claim_ = {"task_id": task["id"], "claim": {"marker": "pre-commit"}}
    real_persist = queue.persist_queue_snapshot
    monkeypatch.setattr(queue, "persist_queue_snapshot",
                        lambda **kw: False if kw.get("reason") == "worker_launch_claimed" else real_persist(**kw))
    with queue._queue_lock:
        assert worker_assignment._commit_launch(queue, task, claim_) is None
    seen = []
    monkeypatch.setattr(worker_assignment, "_deadline_refuses_launch",
                        lambda _q, candidate, unsubmitted=None: seen.append(unsubmitted) or unsubmitted is not None)
    assert claim(task, []) is False
    assert seen == [claim_], "only the in-admission check runs, with the claim taken before the commit"


def test_an_unknown_project_commit_never_resolves_on_an_earlier_mark(world, monkeypatch):
    """A same-attempt mark another launch published is not this commit: it re-runs its owner's check."""
    _, root, pending = world
    row_ = queue.enqueue_task({"id": "waiting-root", "type": "task", "chat_id": 0, "text": "resume after the answer"})
    write_task_result(root, row_["id"], "running", owner_wait={"state": "waiting", "wait_id": "w1"})
    assert task_admission.record_project_dispatch_possible(row_)  # the original attempt's own handoff
    row_["_owner_wait_resume"] = {"wait_id": "w1"}
    real_write, armed = task_admission.write_task_result, [True]

    def write(*args, **kwargs):
        if armed[0] and kwargs.get("_field_projector"):
            armed[0] = False
            error = store.StoreBusy("commit outcome unknown", reason="contention")
            error.rollback_confirmed = False
            raise error  # and in fact nothing landed
        return real_write(*args, **kwargs)

    monkeypatch.setattr(task_admission, "write_task_result", write)
    with queue._queue_lock:
        assert worker_assignment._commit_launch(queue, row_) is None
        write_task_result(root, row_["id"], "running", owner_wait={"state": "answered", "wait_id": "w1"})
        assert worker_assignment._commit_launch(queue, row_) is False, "the stale handoff is refused, never put"
    assert worker_assignment._unsent(row_) is not None


@pytest.mark.parametrize("completed_first", [False, True])
def test_stop_after_a_raised_put_captures_the_bound_worker(world, monkeypatch, completed_first):
    """The real Stop custody reaches the possibly executing worker; natural completion still wins."""
    from ouroboros import cancel_intents
    from ouroboros.task_results import STATUS_COMPLETED
    from supervisor import task_lifecycle
    from tests._cancel_intents_shared import _LiveProc, _reap_spawned_live_procs  # noqa: F401

    world, workers, sent = _assignment_world(world, monkeypatch)
    _, root, pending = world
    _, task = admitted(world)

    def put(launched):
        sent.append((0, copy.deepcopy(launched)))
        raise BrokenPipeError("raised after the item was queued")

    workers.WORKERS[0].in_q.put = put
    workers.assign_tasks()
    assert queue.RUNNING[task["id"]]["worker_id"] == 0
    proc = _LiveProc()
    workers.WORKERS[0].proc, workers.WORKERS[0].reaping = proc, False
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(queue, "_emit_cancel_task_done", lambda *a, **k: None)
    if completed_first:
        write_task_result(root, task["id"], STATUS_COMPLETED, result="finished before the Stop",
                          final_answer="finished before the Stop", cost_usd=0.0, cost_final=True,
                          cost_accounting_status="available")
    # As every in-process ingress does: live ownership (the bound worker) lets a Stop reach a
    # task whose result already settled; custody without an open intent touches nothing.
    cancel_intents.request_cancel(root, task["id"], reason="Explicit Stop", source="http_single",
                                  requested_by="owner",
                                  allow_settled_target=queue.task_has_live_ownership(task["id"]))
    outcome = task_lifecycle.cancel_task_custody(task["id"])
    assert not proc.is_alive(), "Stop reaches the worker the raised put may have started"
    if completed_first:
        assert outcome == task_lifecycle.CANCEL_ALREADY_SETTLED
        assert load_task_result(root, task["id"])["status"] == STATUS_COMPLETED
    else:
        assert outcome == task_lifecycle.CANCEL_CANCELLED
    assert len(sent) == 1
