"""A worker launch persists on the admission lane, outside Q: reserve, persist, revalidate, put.

The real assignment pass over real queue maps, a real preparation generation and real SQLite
holds. Both directions: a contended launch never waits under Q and dispatches once when the store
frees; a Stop or Pause arriving between the reservation and the put wins and nothing is sent; a
refused or failed persist releases the slot and keeps the row as residual demand; no worker ever
holds two reservations; and with nothing contending the same pass launches at once.
"""
from __future__ import annotations

import contextlib
import threading
import time
import traceback
from types import SimpleNamespace

import pytest

from ouroboros import store
from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
from supervisor import queue, task_admission, task_lifecycle, worker_assignment, workers
from supervisor import workspace_preparation as prep
from tests._store_journal_shared import use_journal
from tests.test_admission_contention import database_reader
from tests.test_scheduled_receipt_contention import install_worker
from tests.test_workspace_preparation import _WAIT_SEC
from tests.test_workspace_preparation import env as env
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial

# Storage opened while Q is owned by paths this change does not move, each its own owner's: the
# pre-assignment cancel consult and the Project hold revalidation read task rows under Q at zero
# wait, and the pass's money reading is the usage store's bounded display read.
_UNTOUCHED_UNDER_Q = {"_drop_cancelled_pending", "revalidate_project_holds",
                      "_retry_terminalization_pending_for_assignment", "budget_remaining"}


def wait_for(predicate, what):
    deadline = time.monotonic() + _WAIT_SEC
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def store_opens_under_q(monkeypatch):
    """Every store connection opened while Q is owned, as the innermost supervisor function; an
    entry outside the untouched allow-list is the old defect come back."""
    seen = []
    real = store.connection

    @contextlib.contextmanager
    def connection(*args, **kwargs):
        if queue._queue_lock._is_owned():
            names = [frame.name for frame in traceback.extract_stack()]
            seen.append(next((name for name in reversed(names) if name in _UNTOUCHED_UNDER_Q), tuple(names[-10:])))
        with real(*args, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(store, "connection", connection)
    return seen


def scheduled_child(env, tid="launch-child", root="root"):
    """A queued child with its SCHEDULED receipt, as admission leaves it (project-scope-none)."""
    for task_id in ("root", "parent"):
        write_task_result(env.data, task_id, "running", root_task_id="root")
    task = {"id": tid, "type": "task", "chat_id": 7, "text": "work", "description": "work",
            "root_task_id": root, "parent_task_id": "parent", "delegation_role": "subagent",
            "memory_mode": "empty", "budget_drive_root": str(env.data), "_project_scope_none": True}
    write_task_result(env.data, tid, STATUS_SCHEDULED, parent_task_id="parent", root_task_id=root,
                      delegation_role="subagent", result="Subagent accepted and scheduled.")
    with queue._queue_lock:
        admitted = queue.enqueue_task(task)
    assert not admitted.get("_admission_blocked"), admitted
    return admitted


def hold_at_marks(env, monkeypatch, stack, *, write):
    """Begin the hold at the first dispatch-marks write, as a reader (rollback journal) or a writer (WAL)."""
    original = task_admission.write_task_result
    held, attempts, failures = [], [], []

    def marks(root, task_id, status, **fields):
        if task_id != "launch-child" or fields.get("_field_projector") is None:
            return original(root, task_id, status, **fields)
        attempts.append(threading.current_thread().name)
        if not held:
            held.append(stack.enter_context(database_reader(env.data, write=write)))
        try:
            return original(root, task_id, status, **fields)
        except store.StoreBusy as exc:
            failures.append((exc.reason, getattr(exc, "rollback_confirmed", None)))
            raise

    monkeypatch.setattr(task_admission, "write_task_result", marks)
    return held, attempts, failures


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
def test_contended_launch_never_waits_under_q_and_dispatches_once_when_the_store_frees(env, monkeypatch, journal):
    use_journal(monkeypatch, journal)
    worker, delivered = install_worker(env, monkeypatch)
    row = scheduled_child(env)
    under_q = store_opens_under_q(monkeypatch)
    with contextlib.ExitStack() as stack:
        held, attempts, failures = hold_at_marks(env, monkeypatch, stack, write=journal == "WAL")
        workers.assign_tasks()  # reserves under Q and hands the launch to the admission lane
        assert worker.busy_task_id == "launch-child" and row.get("_launch_claim") and env.pending_ids() == ["launch-child"]
        wait_for(lambda: len(failures) >= 2, "the marks retried at the lane")
        assert set(attempts) == {prep.ADMISSION_LANE} and all(reason == "contention" for reason, _rb in failures)
        assert delivered == [] and "launch-child" not in queue.RUNNING
        workers.assign_tasks()  # a second pass finds the slot taken and the row claimed: no second launch
        assert len([t for t in delivered]) == 0 and sum(1 for w in workers.WORKERS.values() if w.busy_task_id) == 1
        held[0]()
        assert env.gen.wait_idle(_WAIT_SEC)
    assert [task["id"] for task in delivered] == ["launch-child"]
    assert "launch-child" in queue.RUNNING and env.pending_ids() == [] and "_launch_claim" not in row
    assert worker.busy_task_id == "launch-child"
    stored = env.row("launch-child")
    # The marks landed before the put; without a fork drive the worker's own start publishes RUNNING.
    assert stored["admitted_dispatch"] == "possible" and stored["status"] == STATUS_SCHEDULED
    assert all(isinstance(entry, str) for entry in under_q), [entry for entry in under_q if not isinstance(entry, str)]
    # The other direction: nothing contending, the same pass launches a second child at once.
    second = affirm_ready(SimpleNamespace(wid=1, busy_task_id=None,
                                          in_q=SimpleNamespace(put=lambda task: delivered.append(dict(task)))))
    workers.WORKERS[1] = second
    scheduled_child(env, "launch-second")
    workers.assign_tasks()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert [task["id"] for task in delivered] == ["launch-child", "launch-second"] and second.busy_task_id == "launch-second"
    assert all(isinstance(entry, str) for entry in under_q), [entry for entry in under_q if not isinstance(entry, str)]


@pytest.mark.parametrize("control", ["stop", "pause"])
def test_control_between_reservation_and_put_wins_and_frees_the_slot(env, monkeypatch, control):
    from supervisor.events_budget import _set_root_budget_pause_locked

    worker, delivered = install_worker(env, monkeypatch)
    row = scheduled_child(env)
    committed, proceed = threading.Event(), threading.Event()
    original = worker_assignment._commit_launch

    def after_marks(queue_module, candidate, unsubmitted=None):
        projections = original(queue_module, candidate, unsubmitted)
        if candidate.get("id") == "launch-child" and projections:
            committed.set()  # the marks are durable; the put has not been entered
            assert proceed.wait(_WAIT_SEC)
        return projections

    monkeypatch.setattr(worker_assignment, "_commit_launch", after_marks)
    workers.assign_tasks()
    assert committed.wait(_WAIT_SEC) and worker.busy_task_id == "launch-child"
    with queue._queue_lock:
        if control == "stop":
            task_lifecycle.CANCELLED_ROOT_FENCES["root"] = "2026-10-20T00:00:00Z"
        else:
            _set_root_budget_pause_locked("root", {"cause": "owner_pause", "fence_id": "between-marks-and-put"})
    proceed.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert delivered == [] and "launch-child" not in queue.RUNNING
    assert worker.busy_task_id is None and "_launch_claim" not in row  # the slot and the row are free again
    assert env.pending_ids() == ["launch-child"] and env.row("launch-child")["admitted_dispatch"] == "possible"
    assert worker_assignment._unsent(row) is not None  # committed, never put: inspected by this process, never re-sent
    workers.assign_tasks()  # the fence still stands: no launch, no second reservation
    assert env.gen.wait_idle(_WAIT_SEC) and delivered == [] and worker.busy_task_id is None


@pytest.mark.parametrize("failure", ["refused", "raised"])
def test_persist_refused_or_failed_releases_the_slot_and_keeps_residual_demand(env, monkeypatch, failure):
    worker, delivered = install_worker(env, monkeypatch)
    row = scheduled_child(env)
    calls = []

    def commit(queue_module, candidate, unsubmitted=None):
        calls.append(candidate.get("id"))
        if failure == "raised":
            raise OSError("store unavailable")
        return False  # refused: an owner's predicate wrote nothing

    monkeypatch.setattr(worker_assignment, "_commit_launch", commit)
    workers.assign_tasks()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert calls == ["launch-child"] and delivered == []
    assert worker.busy_task_id is None and "_launch_claim" not in row and env.pending_ids() == ["launch-child"]
    assert "launch-child" not in queue.RUNNING and env.row("launch-child")["status"] == STATUS_SCHEDULED
    workers.assign_tasks()  # still residual demand: the next pass claims it again, once
    assert env.gen.wait_idle(_WAIT_SEC) and calls == ["launch-child", "launch-child"]
    assert sum(1 for w in workers.WORKERS.values() if w.busy_task_id) == 0


def test_uncertain_marks_commit_is_inspected_by_token_and_never_resent(env, monkeypatch):
    worker, delivered = install_worker(env, monkeypatch)
    row = scheduled_child(env)
    original = task_admission.commit_dispatch_marks
    outcomes = []

    def unknown_once(task, projections):
        result = original(task, projections)
        outcomes.append(result)
        return None if len(outcomes) == 1 else result  # the first commit's readback is unknown

    monkeypatch.setattr(task_admission, "commit_dispatch_marks", unknown_once)
    workers.assign_tasks()  # one pass: the same lane job re-reads its own marks and puts once
    assert env.gen.wait_idle(_WAIT_SEC)
    assert [task["id"] for task in delivered] == ["launch-child"] and "launch-child" in queue.RUNNING
    assert worker.busy_task_id == "launch-child" and worker_assignment._unsent(row) is None
    assert outcomes == [True]  # the marks were never rewritten


def occupy_lane(generation):
    entered, release = threading.Event(), threading.Event()

    def occupy():
        entered.set()
        assert release.wait(_WAIT_SEC)

    assert generation.submit(occupy, prep.ADMISSION_LANE) and entered.wait(_WAIT_SEC)
    return release


def test_a_generation_closed_before_the_marks_leaves_the_row_never_started(env, monkeypatch):
    """The stop arrives while the launch waits on the lane: nothing durable is written, so after the
    restart the row is plainly never-started (not "may already have reached a worker")."""
    import copy
    import json

    from supervisor.queue_snapshot import _retain_snapshot_pending
    from supervisor.restart_retention import never_started

    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", env.data / "state" / "queue_snapshot.json")
    worker, delivered = install_worker(env, monkeypatch)
    row = scheduled_child(env)
    release = occupy_lane(env.gen)
    workers.assign_tasks()  # the claim is taken; the job waits behind the occupying one
    assert row.get("_launch_claim") and worker.busy_task_id == "launch-child"
    prep.close_generation(env.gen)  # Restart, Update or Quit while the job waits
    release.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert delivered == [] and "launch-child" not in queue.RUNNING and worker.busy_task_id is None
    assert env.row("launch-child").get("admitted_dispatch") != "possible" and worker_assignment._unsent(row) is None
    snapshot = queue.QUEUE_SNAPSHOT_PATH
    saved = (next(r["task"] for r in json.loads(snapshot.read_text(encoding="utf-8"))["pending"]
                  if r["task"]["id"] == "launch-child") if snapshot.exists() else dict(row))
    worker_assignment._COMMITTED_UNSENT.clear()  # the next process has no memory witness
    retained, _unseen, consumed, _roots = _retain_snapshot_pending([copy.deepcopy(saved)], [], stale=False)
    assert consumed == [] and len(retained) == 1 and never_started(retained[0])


def test_marks_committed_while_the_generation_was_open_finish_their_handoff(env, monkeypatch):
    """A launch whose marks landed before the stop is handed to its worker, as at base: refusing it
    then would leave the proof of a non-put only in this process's memory."""
    worker, delivered = install_worker(env, monkeypatch)
    scheduled_child(env)
    committed, proceed = threading.Event(), threading.Event()
    original = worker_assignment._commit_launch

    def after_marks(queue_module, candidate, unsubmitted=None):
        projections = original(queue_module, candidate, unsubmitted)
        if candidate.get("id") == "launch-child" and projections:
            committed.set()
            assert proceed.wait(_WAIT_SEC)
        return projections

    monkeypatch.setattr(worker_assignment, "_commit_launch", after_marks)
    workers.assign_tasks()
    assert committed.wait(_WAIT_SEC)
    prep.close_generation(env.gen)  # the server stops between the durable marks and the put
    proceed.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert [task["id"] for task in delivered if "id" in task] == ["launch-child"]
    assert "launch-child" in queue.RUNNING and worker.busy_task_id == "launch-child"


def accepted_root(env, tid="accepted-root"):
    """An ordinary accepted root: every fresh non-Project admission carries the host-unscoped mark."""
    write_task_result(env.data, tid, "scheduled", chat_id=7, description="accepted owner work")
    with queue._queue_lock:
        row = queue.enqueue_task({"id": tid, "type": "task", "chat_id": 7, "text": "accepted owner work",
                                  "description": "accepted owner work", "budget_drive_root": str(env.data),
                                  "_project_scope_none": True})
    assert not row.get("_admission_blocked"), row
    return row


def hold_after(monkeypatch, owner, name, tid):
    """Hold the lane right after ``owner.name`` returned for ``tid``: its durable step has landed."""
    reached, proceed = threading.Event(), threading.Event()
    original = getattr(owner, name)

    def held(*args, **kwargs):
        result = original(*args, **kwargs)
        task = next(arg for arg in args if isinstance(arg, dict) and "id" in arg)
        if task["id"] == tid and result is not None and result is not False:
            reached.set()
            assert proceed.wait(_WAIT_SEC)
        return result

    monkeypatch.setattr(owner, name, held)
    return reached, proceed


def stop_door():
    """A process-ending stop's first step, run as the door runs it: off the lane, outside Q."""
    outcome = []
    door = threading.Thread(target=lambda: outcome.append(prep.pause_admission(_WAIT_SEC)), name="stop-door")
    door.start()
    return door, outcome


def delivered_ids(delivered):
    return [task["id"] for task in delivered if "id" in task]


@pytest.mark.parametrize("held_step", ["marks_committed", "commit_returning"])
def test_a_stop_waits_for_a_committed_launch_to_reach_its_worker(env, monkeypatch, held_step):
    """Restart, Update or Quit lands between a launch's committed marks and its put (or while that
    commit is still returning): the stop pauses admission and waits for the lane, so the launch
    reaches its worker before the pool retires and nothing is left between its marks and a worker."""
    worker, delivered = install_worker(env, monkeypatch)
    row = accepted_root(env)
    owner, name = ((worker_assignment, "_commit_launch") if held_step == "marks_committed"
                   else (task_admission, "commit_dispatch_marks"))
    reached, proceed = hold_after(monkeypatch, owner, name, row["id"])
    workers.assign_tasks()
    assert reached.wait(_WAIT_SEC) and delivered == []
    door, paused = stop_door()
    time.sleep(0.2)
    assert paused == [] and env.gen.launches_closed()  # the stop waits; no new launch starts meanwhile
    assert not env.gen.closed()  # a child's admission is queued as ever: only launches pause
    proceed.set()
    door.join(_WAIT_SEC)
    assert paused == [True]
    assert delivered_ids(delivered) == [row["id"]] and row["id"] in queue.RUNNING
    assert worker_assignment._unsent(row) is None and worker.busy_task_id == row["id"]


def test_a_stop_waits_for_a_scheduled_launch_to_reach_its_worker(env, monkeypatch):
    """The same for an occurrence a schedule fired: its receipt says ``possible`` only once the put
    is certain to follow, so restore never holds a never-sent occurrence (and its schedule fires on)."""
    from supervisor import schedule_occurrence
    from tests.test_schedule_occurrence import _row

    worker, delivered = install_worker(env, monkeypatch)
    monkeypatch.setattr("ouroboros.config.get_bg_wakeup_min_sec", lambda: 0)
    _row(SimpleNamespace(queue=queue, pending=queue.PENDING, root=env.data), intent={"kind": "system_repo"})
    queue.check_scheduled_tasks()
    [row] = queue.PENDING
    assert schedule_occurrence.restore_verdict(row) == "replay"
    reached, proceed = hold_after(monkeypatch, worker_assignment, "_commit_launch", row["id"])
    workers.assign_tasks()
    assert reached.wait(_WAIT_SEC)
    door, paused = stop_door()
    time.sleep(0.2)
    assert paused == []
    proceed.set()
    door.join(_WAIT_SEC)
    assert paused == [True] and delivered_ids(delivered) == [row["id"]] and row["id"] in queue.RUNNING


def test_a_paused_lane_writes_no_marks_and_resume_reopens_it(env, monkeypatch):
    """A launch still waiting on the lane when the stop pauses it refuses before any durable step
    (the row is plainly never-started); a stop that does not end the process reopens admission."""
    worker, delivered = install_worker(env, monkeypatch)
    row = scheduled_child(env)
    release = occupy_lane(env.gen)
    workers.assign_tasks()  # the claim is taken; the job waits behind the occupying one
    assert row.get("_launch_claim") and worker.busy_task_id == "launch-child"
    door, paused = stop_door()
    time.sleep(0.2)
    assert paused == []
    release.set()
    door.join(_WAIT_SEC)
    assert paused == [True] and delivered == [] and worker.busy_task_id is None and "_launch_claim" not in row
    assert env.row("launch-child").get("admitted_dispatch") != "possible" and worker_assignment._unsent(row) is None
    workers.assign_tasks()  # still paused: nothing starts
    assert env.gen.wait_idle(_WAIT_SEC) and delivered == []
    prep.resume_admission()
    assert not env.gen.launches_closed()
    workers.assign_tasks()
    assert env.gen.wait_idle(_WAIT_SEC) and delivered_ids(delivered) == ["launch-child"]


def test_an_unresolved_committed_launch_is_finished_while_a_stop_waits(env, monkeypatch):
    """No later assignment pass retries a launch once its stop began: the lane itself finishes a
    committed launch whose last step stayed unresolved (here the occurrence-row mark, once)."""
    worker, delivered = install_worker(env, monkeypatch)
    row = accepted_root(env)
    marks = []
    original_mark = worker_assignment._mark_launch_row

    def unresolved_once(candidate, projections):
        marks.append(candidate["id"])
        return None if len(marks) == 1 else original_mark(candidate, projections)

    monkeypatch.setattr(worker_assignment, "_mark_launch_row", unresolved_once)
    reached, proceed = hold_after(monkeypatch, worker_assignment, "_commit_launch", row["id"])
    workers.assign_tasks()
    assert reached.wait(_WAIT_SEC)
    door, paused = stop_door()
    time.sleep(0.2)
    proceed.set()
    door.join(_WAIT_SEC)
    assert paused == [True] and marks == [row["id"], row["id"]]
    assert delivered_ids(delivered) == [row["id"]] and row["id"] in queue.RUNNING


def test_contention_after_the_marks_retries_while_a_stop_waits(env, monkeypatch):
    """A closing generation withdraws only a launch that wrote nothing: contention met after the marks
    committed (here the occurrence-row mark's table) is retried at the lane and the launch is put."""
    worker, delivered = install_worker(env, monkeypatch)
    row = accepted_root(env)
    marks = []
    original_mark = worker_assignment._mark_launch_row

    def contended_once(candidate, projections):
        marks.append(candidate["id"])
        if len(marks) == 1:
            raise store.StoreBusy("held occurrence table", reason="contention")
        return original_mark(candidate, projections)

    monkeypatch.setattr(worker_assignment, "_mark_launch_row", contended_once)
    reached, proceed = hold_after(monkeypatch, worker_assignment, "_commit_launch", row["id"])
    workers.assign_tasks()
    assert reached.wait(_WAIT_SEC)
    door, paused = stop_door()
    time.sleep(0.2)
    proceed.set()
    door.join(_WAIT_SEC)
    assert paused == [True] and marks == [row["id"], row["id"]]
    assert delivered_ids(delivered) == [row["id"]] and row["id"] in queue.RUNNING


def unresolved_once(monkeypatch, kind, tid):
    """The first attempt after the marks stays unresolved: its commit reads back unknown, or the
    occurrence-row mark meets a busy table; ``reached`` is set once it did."""
    attempts, reached, proceed = [], threading.Event(), threading.Event()
    proceed.set()
    owner, name = ((task_admission, "commit_dispatch_marks") if kind == "unknown_commit"
                   else (worker_assignment, "_mark_launch_row"))
    original = getattr(owner, name)

    def unresolved(candidate, projections):
        outcome = original(candidate, projections) if kind == "unknown_commit" or attempts else None
        attempts.append(outcome)
        if candidate["id"] == tid and len(attempts) == 1:
            reached.set()
            assert proceed.wait(_WAIT_SEC)
            return None
        return outcome

    monkeypatch.setattr(owner, name, unresolved)
    return attempts, reached, proceed


@pytest.mark.parametrize("kind", ["unknown_commit", "unwritten_row_mark"])
def test_a_committed_launch_stays_on_the_lane_until_it_is_put(env, monkeypatch, kind):
    """An unresolved committed launch is never handed to a later assignment pass, where a stop that comes
    first (a Restart is read before the pass) would find the lane idle and leave its marks without a
    worker: the same job retries and puts once, the marks never rewritten."""
    worker, delivered = install_worker(env, monkeypatch)
    row = accepted_root(env)
    attempts, _reached, _proceed = unresolved_once(monkeypatch, kind, row["id"])
    workers.assign_tasks()  # one pass
    assert env.gen.wait_idle(_WAIT_SEC)
    assert delivered_ids(delivered) == [row["id"]] and row["id"] in queue.RUNNING
    assert worker_assignment._unsent(row) is None and len(attempts) == (1 if kind == "unknown_commit" else 2)


def test_a_stop_after_an_unresolved_attempt_still_waits_for_the_launch(env, monkeypatch):
    worker, delivered = install_worker(env, monkeypatch)
    row = accepted_root(env)
    attempts, reached, proceed = unresolved_once(monkeypatch, "unwritten_row_mark", row["id"])
    proceed.clear()
    workers.assign_tasks()
    assert reached.wait(_WAIT_SEC)
    door, paused = stop_door()
    time.sleep(0.2)
    assert paused == []  # the committed launch is still the lane's: the stop waits for it
    proceed.set()
    door.join(_WAIT_SEC)
    assert paused == [True] and delivered_ids(delivered) == [row["id"]] and len(attempts) == 2


def test_a_stop_proceeds_once_its_bound_passes(env, monkeypatch):
    """The wait is a stop's hard bound, never an indefinite hold: a lane that does not empty in time
    answers False and the stop goes on (such a launch then restores as possibly dispatched)."""
    release = occupy_lane(env.gen)
    try:
        assert prep.pause_admission(0.2) is False and env.gen.launches_closed()
    finally:
        release.set()
    assert env.gen.lane_idle(prep.ADMISSION_LANE, _WAIT_SEC)
    prep.resume_admission()
    assert not env.gen.launches_closed()


def test_a_launch_on_the_lane_counts_against_its_roots_cap_in_the_same_pass(env, monkeypatch):
    worker, delivered = install_worker(env, monkeypatch)
    second = affirm_ready(SimpleNamespace(wid=1, busy_task_id=None,
                                          in_q=SimpleNamespace(put=lambda task: delivered.append(dict(task)))))
    workers.WORKERS[1] = second
    monkeypatch.setattr("ouroboros.config.get_max_active_subagents_per_root", lambda: 1)
    first, other = scheduled_child(env), scheduled_child(env, "launch-second")
    with contextlib.ExitStack() as stack:
        # A hold that contends the marks write in either journal mode: a read holds a rollback journal's COMMIT,
        # but blocks nothing under WAL, which shipped SQLite (3.53) runs on a local disk.
        held, _attempts, failures = hold_at_marks(env, monkeypatch, stack, write=True)
        workers.assign_tasks()  # the first child's launch waits on the lane; the second sees it as started
        wait_for(lambda: failures, "the first launch reached the held store")
        assert first.get("_launch_claim") and not other.get("_launch_claim") and second.busy_task_id is None
        held[0]()
        assert env.gen.wait_idle(_WAIT_SEC)
    launched = [task["id"] for task in delivered if "id" in task]  # worker commands carry no task id
    assert launched == ["launch-child"] and env.pending_ids() == ["launch-second"]
    assert env.row("launch-second").get("admitted_dispatch") != "possible"  # no durable marks for the excess
