"""Who continues saved work after a stop depends on the stop (#1563, owner 2026-10-08).

A UI Restart / planned restart / managed update RETURNS active work and the
formerly runnable queue — but only under its fresh, acknowledged restart
transaction. Quit, a crash, Panic and an unacknowledged restart HOLD the same
saved work for an explicit Resume. Accepted work never expires; earlier holds
and a Stop are never lifted. Driven through the real kill, snapshot and boot
restore, and the real Resume selection.
"""
from __future__ import annotations

import json
import time

import pytest

from ouroboros import working_checkpoint as wc
from ouroboros.task_results import STATUS_RUNNING, STATUS_SCHEDULED, load_task_result, write_task_result
from ouroboros.work_stop import INTERRUPTED_SELECTION, interrupted_selection
from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_restart_retention import _pool_events, _queued
from tests.worker_pool_support import affirm_ready


def _working_run(root, workers, task_id, *, attempt=1, parent=""):
    write_task_result(root, task_id, STATUS_RUNNING, chat_id=0, root_task_id=parent or task_id)
    _ctx, limit = _loop_ctx(root, task_id, attempt=attempt)
    wc.save_round(limit, "pre_effect")
    workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 0, "_attempt": attempt,
                                         "root_task_id": parent or task_id,
                                         **({"parent_task_id": parent} if parent else {})},
                                "worker_id": len(workers.RUNNING), "attempt": attempt}


def _plain_run(root, workers, task_id):
    write_task_result(root, task_id, STATUS_RUNNING, chat_id=0)
    workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 0, "root_task_id": task_id},
                                "worker_id": 9, "attempt": 1}


def _stop_and_boot(queue, workers, *, age_sec=0):
    """The application stops (its teardown kill), then the next generation boots."""
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True)
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    if age_sec:
        snap["ts"] = "2000-01-01T00:00:00+00:00"
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap))
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    return queue.restore_pending_from_snapshot()


def _crash(queue, workers, monkeypatch, task_id):
    """The worker running ``task_id`` dies before any terminal: the live same-id crash requeue
    (``worker_health``) runs on the real queue. Doubled: the dead process, cost reconstruction and
    the slot's respawn."""
    from supervisor import worker_health
    from tests.test_worker_crash_retry import _make_worker

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


def _ack(root, transaction_id):
    from ouroboros import delegate_recovery as dr

    row = dr._read_restart_transaction(root, transaction_id)
    dr._write_restart_transaction(root, {**row, "status": "normal_exit_acknowledged", "exit_code": 42})


@pytest.mark.parametrize("promised", ["task_ids", "return_ids", "none"])
def test_refused_return_reports_planned_and_saved_promises(tmp_path, monkeypatch, promised):
    from supervisor.restart_retention import recover_saved_running

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _plain_run(tmp_path, workers, "unsaved")
    assert recover_saved_running(workers.RUNNING.values(), tmp_path,
        transaction={promised: ["unsaved"]}, cause="restart") == []
    path = tmp_path / "logs/supervisor.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    refused = [row for row in rows if row.get("type") == "restart_return_refused"]
    assert bool(refused) is (promised != "none")
    if refused:
        assert refused[-1]["refused"] == {"unsaved": "no_usable_recovery_source"}


@pytest.mark.parametrize("age_sec", [0, 7 * 86400], ids=["fresh", "week-old"])
def test_quit_keeps_running_work_and_all_accepted_queue_held_for_resume(tmp_path, monkeypatch, age_sec):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    _working_run(tmp_path, workers, "saved-child", parent="saved-root")
    _plain_run(tmp_path, workers, "unsaved")
    write_task_result(tmp_path, "queued", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append(_queued("queued", root_task_id="queued"))

    _stop_and_boot(queue, workers, age_sec=age_sec)

    rows = {row["id"]: row for row in workers.PENDING}
    assert set(rows) == {"saved-root", "saved-child", "queued"}
    for task_id in ("saved-root", "saved-child", "queued"):
        hold = budget_hold_fact(rows[task_id])
        assert hold and hold["reason"] == HOLD_SAVED_WORK and hold["door"] == "app_stop"
        assert load_task_result(tmp_path, task_id)["status"] == STATUS_SCHEDULED
    handoff = rows["saved-root"]["_working_recovery"]
    assert handoff["from_attempt"] == 1 and rows["saved-root"]["_attempt"] == 2 and handoff["cause"] == "app_stop"
    assert "_working_recovery" not in rows["queued"], "a never-started row starts, it does not continue"
    # Work with nothing saved keeps the old path (cancelled, Continue offered), never a silent resume.
    from ouroboros.cancel_intents import has_active_intent

    assert load_task_result(tmp_path, "unsaved")["status"] == "cancelled"
    assert not has_active_intent(tmp_path, "saved-root", strict=True)

    # Only the owner's Resume releases it, under the same id.
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    assert queue.resume_budget_paused_task("saved-root")["ok"]
    assert budget_hold_fact(rows["saved-root"]) is None
    assert rows["saved-root"]["_working_recovery"]["source_ref"], "it continues its frozen state"
    assert queue.resume_budget_paused_task("queued")["ok"]
    assert budget_hold_fact(rows["queued"]) is None


RESUMED = "resumed-child"


def _ended_work(root):
    """The work the helper served is over: its root completed on its own."""
    write_task_result(root, "ended-parent", "running", chat_id=0, root_task_id="ended-parent", delegation_role="root")
    write_task_result(root, "ended-parent", "completed", result="done")


def _resume_and_run(root, queue, state, workers, monkeypatch):
    """The owner's Resume of the held helper, the real assignment pass to one inert pooled worker, and the dispatched
    attempt's loop: it takes what it was handed (its exact grant, else its saved working state) and saves its own work.
    Only that worker's inert handle can be killed; no operating-system PID is used."""
    import queue as stdqueue

    from ouroboros import budget_pause
    from supervisor import worker_pool_lifecycle
    from tests.test_budget_pause_lineage import _consume
    from tests.test_owner_wait_restart import InertProcess

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    assert queue.resume_budget_paused_task(RESUMED)["ok"]
    commands, proc = stdqueue.Queue(), InertProcess()
    workers.WORKERS[0] = affirm_ready(workers.Worker(0, proc, commands))
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda _task: True)

    def kill_inert(pid, **_kwargs):
        assert pid == proc.pid
        proc.terminate()

    monkeypatch.setattr(workers, "kill_worker_tree", kill_inert)
    monkeypatch.setattr(worker_pool_lifecycle, "kill_worker_tree", kill_inert)
    workers.assign_tasks()
    launched = commands.get_nowait()
    assert launched["id"] == RESUMED and RESUMED in workers.RUNNING
    ctx, limit = _loop_ctx(root, RESUMED, attempt=launched["_attempt"])
    ctx.root_task_id = "ended-parent"
    if isinstance(launched.get("_budget_pause_resume"), dict):
        _consume(monkeypatch, ctx, limit, launched["_budget_pause_resume"])
        assert budget_pause.budget_pause_row(root, RESUMED)["grant"]["consumed_at"]
    else:
        assert wc.load_recovery(ctx, launched["_working_recovery"])
        wc.consume_recovery(ctx, launched["_working_recovery"])
    wc.save_round(limit, "pre_effect")
    return launched, limit


def _resumed_helper_of_ended_work(root, queue, state, workers, monkeypatch, *, door):
    """A helper of work that ended, running on the owner's explicit Resume when the application stops: of its exact
    budget Pause (``grant``), or of the saved-work hold an earlier stop left it in (``saved``). Nothing is stamped."""
    from tests._budget_pause_exact_helpers import _parked

    _ended_work(root)
    lineage = {"parent_task_id": "ended-parent", "delegation_role": "subagent"}
    if door == "grant":
        _parked(root, monkeypatch, task_id=RESUMED, root_task_id="ended-parent", extra=lineage)
    else:
        _working_run(root, workers, RESUMED, parent="ended-parent")
        workers.RUNNING[RESUMED]["task"].update(lineage)
        _stop_and_boot(queue, workers)  # its work saved, held for Resume; the owner's Resume comes before any sweep
    return _resume_and_run(root, queue, state, workers, monkeypatch)


def _kept_resume(root, row, door, attempt):
    """The accepted Resume the boot hold over ``attempt``'s saved work still answers to, read where each door keeps it."""
    from ouroboros import budget_pause

    assert row["_working_recovery"]["source_task_id"] == RESUMED and row["_working_recovery"]["from_attempt"] == attempt
    if door == "saved":
        assert interrupted_selection(row)["selected_by"] == "owner"
    else:  # the consumed grant on its durable pause, the selection carrier gone with the exact handoff
        assert interrupted_selection(row) == {}
        assert budget_pause.budget_pause_row(root, RESUMED)["state"] == budget_pause.STATE_RESUMED


@pytest.mark.parametrize("door", ["grant", "saved"])
@pytest.mark.parametrize("intent", ["none", "stale_closure", "owner_stop_joined"])
def test_a_helper_whose_resume_won_survives_stop_boot_and_the_next_sweep(tmp_path, monkeypatch, intent, door):
    """#3047: the app stopped while a resumed helper of ended work ran; its boot hold is no abandoned park."""
    from ouroboros import cancel_intents as ci
    from supervisor.ended_work_helpers import WORK_FINISHED, close_abandoned_helpers
    from supervisor.task_lifecycle import sweep_cancel_intents

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    launched, _limit = _resumed_helper_of_ended_work(tmp_path, queue, state, workers, monkeypatch, door=door)
    if intent != "none":  # minted while the row still read parked, before custody saw the Resume's worker
        ci.request_cancel(tmp_path, RESUMED, source=WORK_FINISHED, reason="the work it helped has ended")
    if intent == "owner_stop_joined":
        ci.request_cancel(tmp_path, RESUMED, source="http_single", reason="Stop", requested_by="owner")

    _stop_and_boot(queue, workers)
    assert close_abandoned_helpers() == []
    outcomes = sweep_cancel_intents(now=time.time() + 3600)

    row = load_task_result(tmp_path, RESUMED)
    returned = next((task for task in workers.PENDING if task["id"] == RESUMED), None)
    assert not ci.active_intent(tmp_path, RESUMED)
    if intent == "owner_stop_joined":
        assert row["status"] == "cancelled" and returned is None and row["followup_stop"]  # the joined Stop, carried out
        return
    assert outcomes == ({} if intent == "none" else {RESUMED: "not_found"})
    assert row["status"] == STATUS_SCHEDULED
    assert returned["_working_recovery"]["source_ref"] and budget_hold_fact(returned)["reason"] == HOLD_SAVED_WORK
    _kept_resume(tmp_path, returned, door, launched["_attempt"])
    assert close_abandoned_helpers() == [] and sweep_cancel_intents(now=time.time() + 7200) == {}  # not just one tick
    _stop_and_boot(queue, workers)  # held, it waits through a further stop and boot as well
    returned = next(task for task in workers.PENDING if task["id"] == RESUMED)
    assert close_abandoned_helpers() == [] and sweep_cancel_intents(now=time.time() + 7200) == {}
    assert queue.resume_budget_paused_task(RESUMED)["ok"] and budget_hold_fact(returned) is None


def test_a_saved_work_resume_stopped_before_its_dispatch_is_not_abandoned(tmp_path, monkeypatch):
    """The owner resumed the held helper and the application stopped again before any worker took it: restore holds
    that accepted row for Resume again, and the selection its hold replaced stays beside the same saved source."""
    from supervisor.ended_work_helpers import close_abandoned_helpers
    from supervisor.task_lifecycle import sweep_cancel_intents

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _ended_work(tmp_path)
    _working_run(tmp_path, workers, RESUMED, parent="ended-parent")
    workers.RUNNING[RESUMED]["task"]["delegation_role"] = "subagent"
    _stop_and_boot(queue, workers)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    assert queue.resume_budget_paused_task(RESUMED)["ok"]
    _stop_and_boot(queue, workers)
    returned = next(task for task in workers.PENDING if task["id"] == RESUMED)
    assert budget_hold_fact(returned)["reason"] == HOLD_SAVED_WORK
    assert close_abandoned_helpers() == [] and sweep_cancel_intents(now=time.time() + 3600) == {}
    assert load_task_result(tmp_path, RESUMED)["status"] == STATUS_SCHEDULED
    assert returned["_working_recovery"]["from_attempt"] == 1 and interrupted_selection(returned)["selected_by"] == "owner"


def test_a_stop_hold_without_its_attempts_resume_still_closes_with_the_ended_work(tmp_path, monkeypatch):
    """A helper that was only running when its work ended, which nobody resumed, supplies no accepted work."""
    from supervisor.ended_work_helpers import WORK_FINISHED, close_abandoned_helpers
    from supervisor.task_lifecycle import sweep_cancel_intents

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _ended_work(tmp_path)
    _working_run(tmp_path, workers, RESUMED, parent="ended-parent")
    workers.RUNNING[RESUMED]["task"]["delegation_role"] = "subagent"
    _stop_and_boot(queue, workers)
    held = next(t for t in workers.PENDING if t["id"] == RESUMED)
    assert budget_hold_fact(held)["reason"] == HOLD_SAVED_WORK and interrupted_selection(held) == {}

    assert close_abandoned_helpers() == [RESUMED]
    assert sweep_cancel_intents(now=time.time() + 3600) == {RESUMED: "cancelled"}
    row = load_task_result(tmp_path, RESUMED)
    assert row["status"] == "cancelled" and row["cancel_origin"]["source"] == WORK_FINISHED
    assert row["pause_at_close"]["_budget_pause_hold"]["reason"] == HOLD_SAVED_WORK


def test_a_grant_consumed_by_a_later_attempt_than_the_saved_work_is_no_accepted_work(tmp_path, monkeypatch):
    """The consumed grant answers only for the attempt whose saved work the stop interrupted. The real grant-door
    Resume, dispatch and consumption, then the one stamped field changed (the grant now read as a later attempt's):
    that supplies no accepted work, so the helper closes with its ended work (the clause's negative control)."""
    from ouroboros import budget_pause
    from supervisor.ended_work_helpers import WORK_FINISHED, close_abandoned_helpers
    from supervisor.task_lifecycle import sweep_cancel_intents

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    launched, _limit = _resumed_helper_of_ended_work(tmp_path, queue, state, workers, monkeypatch, door="grant")
    consumed = budget_pause.budget_pause_row(tmp_path, RESUMED)
    assert 0 < consumed["task_attempt"] <= launched["_attempt"] and consumed["grant"]["consumed_at"]
    budget_pause.set_budget_pause(tmp_path, RESUMED, {**consumed, "task_attempt": launched["_attempt"] + 6})
    _stop_and_boot(queue, workers)
    held = next(task for task in workers.PENDING if task["id"] == RESUMED)
    assert budget_hold_fact(held)["reason"] == HOLD_SAVED_WORK and interrupted_selection(held) == {}

    assert close_abandoned_helpers() == [RESUMED]
    assert sweep_cancel_intents(now=time.time() + 3600) == {RESUMED: "cancelled"}
    row = load_task_result(tmp_path, RESUMED)
    assert row["status"] == "cancelled" and row["cancel_origin"]["source"] == WORK_FINISHED


@pytest.mark.parametrize("door", ["grant", "saved"])
def test_a_newer_pause_of_the_resumed_attempt_ends_its_consumed_resume(tmp_path, monkeypatch, door):
    """The owner's Resume was consumed (its exact grant, or its saved-work selection, which the park copies along),
    then the same attempt paused again for money (the loop's real pause and the supervisor's park): that newer Pause
    waits for its own Resume, so after the next stop the helper closes with its ended work, its newer pause kept
    beside the closed result."""
    from ouroboros import budget_pause
    from supervisor.ended_work_helpers import WORK_FINISHED, close_abandoned_helpers
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.task_lifecycle import sweep_cancel_intents
    from tests._budget_pause_exact_helpers import _fast_hold, _quiet_external, _supervisor_ctx

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _launched, limit = _resumed_helper_of_ended_work(tmp_path, queue, state, workers, monkeypatch, door=door)
    resumed = budget_pause.budget_pause_row(tmp_path, RESUMED)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    with pytest.raises(budget_pause.BudgetPauseRequested):
        budget_pause.request_pause(limit, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED, scope="global",
                                   reason_text="money gone again", root_task_id="ended-parent")
    budget_pause.end_dispatch_fence(RESUMED)
    newer = budget_pause.budget_pause_row(tmp_path, RESUMED)
    assert newer["pause_id"] != resumed.get("pause_id")
    install_exact_budget_pause(_supervisor_ctx(tmp_path, workers, queue, [], []), RESUMED,
                               budget_pause.exact_pause_marker(newer)["checkpoint"])
    _stop_and_boot(queue, workers)

    assert close_abandoned_helpers() == [RESUMED]
    assert sweep_cancel_intents(now=time.time() + 3600) == {RESUMED: "cancelled"}
    row = load_task_result(tmp_path, RESUMED)
    assert row["status"] == "cancelled" and row["cancel_origin"]["source"] == WORK_FINISHED
    assert row["pause_at_close"]["_budget_pause"]["checkpoint"]["pause_id"] == newer["pause_id"]


@pytest.mark.parametrize("door", ["grant", "saved"])
@pytest.mark.parametrize("joined_stop", [False, True])
@pytest.mark.parametrize("stale_closure", [False, True])
def test_a_resumed_helpers_later_stop_is_not_a_new_abandonment(tmp_path, monkeypatch, joined_stop, stale_closure, door):
    """The selected work continues through its boot hold; a second application stop does not cancel that decision.
    The owner's second Resume selects the saved-work hold; the real assignment and loop take it up before that stop."""
    from ouroboros.cancel_intents import request_cancel
    from supervisor.ended_work_helpers import WORK_FINISHED, close_abandoned_helpers
    from supervisor.task_lifecycle import sweep_cancel_intents

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _resumed_helper_of_ended_work(tmp_path, queue, state, workers, monkeypatch, door=door)
    _stop_and_boot(queue, workers)
    assert close_abandoned_helpers() == []
    launched, _limit = _resume_and_run(tmp_path, queue, state, workers, monkeypatch)
    if stale_closure:
        request_cancel(tmp_path, RESUMED, source=WORK_FINISHED)
    if joined_stop:
        request_cancel(tmp_path, RESUMED, source="http_single", requested_by="owner")
    _stop_and_boot(queue, workers)
    assert close_abandoned_helpers() == []
    sweep_cancel_intents(now=time.time() + 3600)
    row = load_task_result(tmp_path, RESUMED)
    assert row["status"] == ("cancelled" if joined_stop else STATUS_SCHEDULED)
    assert close_abandoned_helpers() == []
    if not joined_stop:
        returned = next(task for task in workers.PENDING if task["id"] == RESUMED)
        _kept_resume(tmp_path, returned, "saved", launched["_attempt"])  # the newest Resume is the selection


def test_a_saved_child_of_an_unsaved_interrupted_parent_is_never_held_alone(tmp_path, monkeypatch):
    """Invariant 19 stands: saved work never revives a child whose parent was interrupted."""
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _plain_run(tmp_path, workers, "unsaved-root")
    _working_run(tmp_path, workers, "saved-child", parent="unsaved-root")
    _stop_and_boot(queue, workers)
    assert "saved-child" not in {row["id"] for row in workers.PENDING if not row.get("_terminalization_retry")}
    assert load_task_result(tmp_path, "unsaved-root")["status"] == "cancelled"


def test_a_fresh_acknowledged_restart_returns_work_and_queue_once(tmp_path, monkeypatch):
    from supervisor.restart_retention import fresh_return_transaction, prepare_restart_returns

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    write_task_result(tmp_path, "queued", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append(_queued("queued", root_task_id="queued"))
    returning = prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING, transaction_id="tx-restart")
    assert returning == {"saved-root"}
    _ack(tmp_path, "tx-restart")

    _stop_and_boot(queue, workers, age_sec=3600)

    rows = {row["id"]: row for row in workers.PENDING}
    assert set(rows) == {"saved-root", "queued"}
    assert budget_hold_fact(rows["saved-root"]) is None and budget_hold_fact(rows["queued"]) is None
    assert rows["saved-root"]["_working_recovery"]["cause"] == "restart"
    assert fresh_return_transaction(tmp_path) == {}, "a later stop can never reuse this Restart"

    # A later Quit holds the same kind of work again.
    for task_id in list(rows):
        workers.PENDING.remove(rows[task_id])
    _working_run(tmp_path, workers, "later-run")
    _stop_and_boot(queue, workers)
    assert budget_hold_fact(next(r for r in workers.PENDING if r["id"] == "later-run"))["reason"] == HOLD_SAVED_WORK


def test_an_unacknowledged_restart_holds_like_a_crash(tmp_path, monkeypatch):
    from supervisor.restart_retention import prepare_restart_returns

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING, transaction_id="tx-aborted")
    _stop_and_boot(queue, workers)  # the exit was never acknowledged
    assert budget_hold_fact(workers.PENDING[0])["reason"] == HOLD_SAVED_WORK


@pytest.mark.parametrize("restart", [False, True], ids=["quit", "acknowledged-restart"])
def test_saved_successor_keeps_frozen_source_across_another_crash_before_dispatch(tmp_path, monkeypatch, restart):
    from supervisor.restart_retention import prepare_restart_returns

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    if restart:
        prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING, transaction_id="first-restart")
        _ack(tmp_path, "first-restart")
    _stop_and_boot(queue, workers)
    [first] = workers.PENDING
    source = dict(first["_working_recovery"])
    assert first["_attempt"] == 2

    # Restore itself writes the pending snapshot; another application crash can
    # happen before that successor ever reaches a worker or consumes its source.
    workers.PENDING.clear()
    assert queue.restore_pending_from_snapshot() == 1
    [again] = workers.PENDING
    assert again["_working_recovery"] == source
    assert again["_attempt"] == 2, "an unstarted successor consumes no new attempt"
    assert budget_hold_fact(again)["reason"] == HOLD_SAVED_WORK
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    assert queue.resume_budget_paused_task("saved-root")["ok"]
    assert again["_working_recovery"] == source
    ctx, _limit = _loop_ctx(tmp_path, "saved-root", attempt=2)
    assert wc.load_recovery(ctx, source)["working"]["boundary"] == "pre_effect"


@pytest.mark.parametrize(("door", "origin"), [
    ("restart", "restarted"),
    ("quit", "resumed"),
    ("crash", "crash_retry"),
    ("panic", "restarted"),
])
def test_runnable_saved_successor_waiting_for_capacity_follows_the_next_stop(tmp_path, monkeypatch, origin, door):
    """A pending successor still owns saved work; another stop does not buy an attempt."""
    import queue as stdqueue

    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_owner_wait_restart import InertProcess

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    if origin == "crash_retry":  # the one automatic retry left: same id, continuing saved work
        _crash(queue, workers, monkeypatch, "saved-root")
    else:
        if origin == "restarted":
            prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING, transaction_id="first-restart")
            _ack(tmp_path, "first-restart")
        _stop_and_boot(queue, workers)
    [successor] = workers.PENDING
    if origin == "resumed":
        monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
        assert queue.resume_budget_paused_task("saved-root")["ok"]
    assert budget_hold_fact(successor) is None
    assert successor["_attempt"] == 2
    handoff = dict(successor["_working_recovery"])

    if door in {"restart", "panic"}:
        from ouroboros import delegate_recovery

        prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING, transaction_id="second-restart")
        transaction = delegate_recovery._read_restart_transaction(tmp_path, "second-restart")
        assert transaction["queue_ids"] == ["saved-root"]
        _ack(tmp_path, "second-restart")
    if door == "panic":
        (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    if door == "crash":
        queue.persist_queue_snapshot(reason="main_loop")
        workers.PENDING.clear()
        queue.restore_pending_from_snapshot()
    else:
        _stop_and_boot(queue, workers)
    [restored] = workers.PENDING
    kept = {key: value for key, value in restored["_working_recovery"].items() if key != INTERRUPTED_SELECTION}
    assert restored["_attempt"] == 2 and kept == handoff  # the same frozen source
    # a Resume this stop interrupted before its dispatch stays beside that source (#3047)
    assert bool(interrupted_selection(restored)) == (origin == "resumed" and door == "quit")
    ctx, _limit = _loop_ctx(tmp_path, "saved-root", attempt=2)
    assert wc.load_recovery(ctx, handoff)["working"]["boundary"] == "pre_effect"
    commands = stdqueue.Queue()
    workers.WORKERS[0] = affirm_ready(workers.Worker(0, InertProcess(), commands))
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda _task: True)
    workers.assign_tasks()
    if door == "restart":
        sent = commands.get_nowait()
        assert sent["id"] == "saved-root" and sent["_working_recovery"] == handoff
        assert commands.empty() and set(workers.RUNNING) == {"saved-root"}
        assert not workers.PENDING
    else:
        assert commands.empty() and not workers.RUNNING
        assert budget_hold_fact(restored)["reason"] == HOLD_SAVED_WORK


@pytest.mark.parametrize("control", ["prior_hold", "stop", "bad_source"])
def test_restart_does_not_return_a_saved_successor_without_current_admission(tmp_path, monkeypatch, control):
    import queue as stdqueue

    from ouroboros import delegate_recovery
    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.cancel_intents import request_cancel
    from supervisor.events_budget import HOLD_OWNER_RESTART, hold_budget_row
    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_owner_wait_restart import InertProcess

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING, transaction_id="initial")
    _ack(tmp_path, "initial")
    _stop_and_boot(queue, workers)
    [successor] = workers.PENDING
    if control == "prior_hold":
        hold_budget_row(successor, reason=HOLD_OWNER_RESTART, result_root=tmp_path)
    elif control == "owner_hold":
        successor["_owner_hold"] = {"reason": "owner_selection"}
    elif control == "stop":
        request_cancel(tmp_path, "saved-root", reason="owner stop", source="owner", requested_by="owner")
    elif control == "bad_source":
        source = successor["_working_recovery"]["source_ref"]
        (task_artifact_dir_path(tmp_path, "saved-root") / source["path"]).write_bytes(b"broken")
    else:
        write_task_result(tmp_path, "saved-root", "running", admitted_dispatch="possible",
                          admitted_dispatch_attempt=2, task_attempt=2)
    prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING, transaction_id="controlled")
    assert delegate_recovery._read_restart_transaction(tmp_path, "controlled")["queue_ids"] == []
    _ack(tmp_path, "controlled")
    _stop_and_boot(queue, workers)
    commands = stdqueue.Queue()
    workers.WORKERS[0] = affirm_ready(workers.Worker(0, InertProcess(), commands))
    workers.assign_tasks()
    assert commands.empty() and not workers.RUNNING
    if control == "prior_hold":
        assert budget_hold_fact(workers.PENDING[0])["reason"] == HOLD_OWNER_RESTART
    if control == "stop":
        assert not any(row["id"] == "saved-root" for row in workers.PENDING)


def test_a_panic_after_an_acknowledged_restart_returns_nothing_by_itself(tmp_path, monkeypatch):
    """Panic wins over a fresh return transaction and over an owner Restart's flag."""
    from supervisor.restart_retention import prepare_restart_returns

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    write_task_result(tmp_path, "queued", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append(_queued("queued", root_task_id="queued"))
    assert prepare_restart_returns(tmp_path, workers.RUNNING, workers.PENDING, transaction_id="tx-r") == {"saved-root"}
    _ack(tmp_path, "tx-r")
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "owner_restart_no_resume.flag").write_text("owner_restart", encoding="utf-8")
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    _stop_and_boot(queue, workers)
    rows = {row["id"]: row for row in workers.PENDING}
    assert set(rows) == {"saved-root", "queued"}
    assert budget_hold_fact(rows["saved-root"])["door"] == "panic"
    assert budget_hold_fact(rows["queued"]) is not None, "the queue waits for Resume too"


def test_a_crash_with_no_teardown_still_recovers_from_the_last_snapshot(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    write_task_result(tmp_path, "queued", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append(_queued("queued", root_task_id="queued"))
    queue.persist_queue_snapshot(reason="main_loop")  # the last tick before the crash; no kill ran
    workers.RUNNING.clear()
    workers.PENDING.clear()
    queue.restore_pending_from_snapshot()
    rows = {row["id"]: row for row in workers.PENDING}
    assert set(rows) == {"saved-root", "queued"}
    assert all(budget_hold_fact(row)["reason"] == HOLD_SAVED_WORK for row in rows.values())


def test_panic_holds_and_a_stop_or_an_earlier_hold_is_never_lifted(tmp_path, monkeypatch):
    from ouroboros.cancel_intents import request_cancel
    from supervisor.events_budget import HOLD_OWNER_RESTART, hold_budget_row

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    _working_run(tmp_path, workers, "stopped-root")
    request_cancel(tmp_path, "stopped-root", reason="owner stop", source="owner", requested_by="owner")
    write_task_result(tmp_path, "held-before", STATUS_SCHEDULED, chat_id=0)
    held = _queued("held-before", root_task_id="held-before")
    hold_budget_row(held, reason=HOLD_OWNER_RESTART, result_root=tmp_path)
    workers.PENDING.append(held)
    queue.persist_queue_snapshot(reason="main_loop")
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    workers.RUNNING.clear()
    workers.PENDING[:] = []
    queue.restore_pending_from_snapshot()
    rows = {row["id"]: row for row in workers.PENDING}
    assert budget_hold_fact(rows["saved-root"])["door"] == "panic"
    assert "stopped-root" not in rows, "Stop wins over saved work"
    assert budget_hold_fact(rows["held-before"])["reason"] == HOLD_OWNER_RESTART


def test_an_owner_restart_returns_its_named_work_instead_of_cancelling_it(tmp_path, monkeypatch):
    from ouroboros import server_restart
    from ouroboros.cancel_intents import has_active_intent

    # This fixture owns one restart generation; do not leak its notice facts
    # into later tests that replace the stop owner with an inert callback.
    monkeypatch.setattr(server_restart, "_LAST_RETURNING", set())
    monkeypatch.setattr(server_restart, "_LAST_RETURN_TX", {"id": ""})
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    _plain_run(tmp_path, workers, "unsaved")
    from tests._work_stop_helpers import live_restart
    transaction = live_restart(tmp_path, monkeypatch)
    monkeypatch.setattr(server_restart, "_LAST_RETURN_TX", {"id": transaction["transaction_id"]})
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_restart, "_stop_owned_daemon", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.delegate_custody.reconcile_orphaned_runs", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.owned_shutdown.stop_owned_work", lambda *_a, **_k: None)
    ctx = type("Ctx", (), {"RUNNING": workers.RUNNING, "PENDING": workers.PENDING,
                           "kill_workers": staticmethod(workers.kill_workers)})()
    stopped = server_restart._stop_owned_work(ctx)
    assert stopped == ["unsaved"] and server_restart._LAST_RETURNING == {"saved-root"}
    assert has_active_intent(tmp_path, "unsaved", strict=True)
    assert not has_active_intent(tmp_path, "saved-root", strict=True)
    assert "saved-root" in workers.RUNNING, "kept for the next boot, never terminalized"
    assert load_task_result(tmp_path, "saved-root")["status"] == STATUS_RUNNING


def test_a_warm_owner_pause_park_becomes_the_exact_cold_pause_its_tree_resumes(tmp_path, monkeypatch):
    from ouroboros import budget_pause, owner_pause
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    write_task_result(tmp_path, "root", STATUS_RUNNING, chat_id=0)
    fence, _ = owner_pause.install_fence(tmp_path, "root", request_id="pause")
    ctx, limit = _loop_ctx(tmp_path, "root")
    wait = checkpoint_owner_wait(ctx, limit.messages, {}, {}, 4, [], set(),
                                 pause={"fence_id": fence["fence_id"], "generation": 1, "root_task_id": "root",
                                        "detached_reviews": [], "external_runs": {"runs": [], "custody_read": "ok"}})
    set_owner_wait(tmp_path, "root", {**wait, "state": "waiting"})
    workers.RUNNING["root"] = {"task": {"id": "root", "type": "task", "chat_id": 0, "root_task_id": "root"},
                               "worker_id": 0, "attempt": 1}
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True)
    row = budget_pause.budget_pause_row(tmp_path, "root")
    assert row["reason"] == "owner" and row["rail"] == "owner_pause" and row["state"] == "paused"
    assert row["owner_fence_id"] == fence["fence_id"] and row["settlement"] == "settled"
    assert load_task_result(tmp_path, "root")["owner_wait"]["state"] == "retained"
    parked = next(task for task in workers.PENDING if task["id"] == "root")
    assert parked["_budget_pause"]["reason"] == "owner"


class _Proc:
    def __init__(self, pid):
        self.pid, self.alive = pid, True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass


def test_a_managed_update_window_never_times_out_the_saved_work_it_returns(tmp_path, monkeypatch):
    """``kill_workers_for_update`` returns mid-process: apply, smoke or the assisted
    resolver keep the tick running until the restart. A kept row has no live attempt,
    on a vacant slot or one the resolver reuses, so the timeout rail never asks it to
    finalize, reaps or retries it; live work keeps idle and deadline, and a Stop is
    never kept. The acknowledged update then returns both under their own ids."""
    import datetime
    import queue as stdqueue
    from types import SimpleNamespace

    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_mailbox import _mailbox_path, mailbox_lines
    from supervisor import queue_timeouts, worker_pool_lifecycle
    from supervisor.restart_retention import RETAINED_FOR_BOOT, prepare_restart_returns

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    procs, reaps, clock = {}, stdqueue.Queue(), [time.time()]
    t0 = clock[0]
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    for module in (workers, worker_pool_lifecycle):
        monkeypatch.setattr(module, "kill_worker_tree", lambda pid, **_k: setattr(procs[pid], "alive", False))
    monkeypatch.setattr(workers, "respawn_worker", lambda *_a, **_k: None)
    monkeypatch.setattr(queue, "_reap_queue", reaps)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 120)
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 1920)  # the window this timeline was written for
    monkeypatch.setattr(queue_timeouts, "time", SimpleNamespace(time=lambda: clock[0]))

    def run(task_id, wid, *, progress_at, **task):
        if task_id not in workers.RUNNING:
            write_task_result(tmp_path, task_id, STATUS_RUNNING, chat_id=0)
            workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 0,
                                                 "root_task_id": task_id, **task}, "worker_id": wid, "attempt": 1}
        procs[60000 + wid] = _Proc(60000 + wid)
        workers.WORKERS[wid] = workers.Worker(wid, procs[60000 + wid], stdqueue.Queue(), busy_task_id=task_id)
        workers.RUNNING[task_id].update(started_at=t0 - 1500, last_progress_at=progress_at, last_heartbeat_at=t0)

    def controls(task_id):
        path = _mailbox_path(tmp_path, task_id)
        return [json.loads(line).get("kind") for line in mailbox_lines(path.read_text())] if path.exists() else []

    for task_id in ("saved-a", "saved-b", "stopped"):  # slots 0, 1, 2; 25 min into a long tool call
        _working_run(tmp_path, workers, task_id)
        run(task_id, workers.RUNNING[task_id]["worker_id"], progress_at=t0 - 1500)
    request_cancel(tmp_path, "stopped", reason="owner stop", source="owner", requested_by="owner")
    prepare_restart_returns(tmp_path, workers.RUNNING, list(workers.PENDING), transaction_id="tx-update")
    assert workers.kill_workers_for_update(result_reason="Managed update.", terminal_status="interrupted") == []
    assert {task_id: meta.get(RETAINED_FOR_BOOT) for task_id, meta in workers.RUNNING.items()} == {
        "saved-a": True, "saved-b": True}, "a Stop is never kept"
    assert not workers.WORKERS

    # The assisted resolver reuses slot 0; another live run has an explicit deadline.
    run("resolver", 0, progress_at=t0 - 1500)
    deadline = datetime.datetime.fromtimestamp(t0 + 600, datetime.timezone.utc).isoformat()
    run("deadline-run", 3, progress_at=t0, deadline_at=deadline)
    asked, reaped = {}, []
    for step in range(9):  # apply, smoke and resolver review well past idle window + grace
        clock[0] = t0 + 300 * step
        queue.enforce_task_timeouts()
        for task_id, meta in workers.RUNNING.items():
            if meta.get("finalization_reason") and task_id not in asked:
                asked[task_id] = (meta["finalization_reason"], controls(task_id))
        while not reaps.empty():
            job = reaps.get_nowait()
            reaped.append(job["task_id"])
            queue._reap_timed_out_task(job)

    assert asked == {"deadline-run": ("deadline", ["finalize_now"])}
    assert reaped == ["deadline-run"]
    assert workers.RUNNING["resolver"]["silence_attention"]["state"] == "pending_attention"
    assert load_task_result(tmp_path, "deadline-run")["reason_code"] == "deadline"
    for task_id in ("saved-a", "saved-b"):
        assert workers.RUNNING[task_id].get("finalization_requested_at") is None
        assert load_task_result(tmp_path, task_id)["status"] == STATUS_RUNNING
        assert "finalize_now" not in controls(task_id), "nothing tells the returned attempt to finish"

    _ack(tmp_path, "tx-update")
    queue.persist_queue_snapshot(reason="main_loop")  # the last tick before the update's restart
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    workers.WORKERS.clear()
    queue.restore_pending_from_snapshot()
    rows = {row["id"]: row for row in workers.PENDING}
    for task_id in ("saved-a", "saved-b"):
        assert budget_hold_fact(rows[task_id]) is None and rows[task_id]["_attempt"] == 2
        assert rows[task_id]["_working_recovery"]["cause"] == "restart"


_SAVED_QUEUE_TIMES = {"ordinary": "2026-09-29T08:00:00+00:00", "held": "2026-10-05T09:30:00+00:00",
                      "saved-run": "2026-10-11T11:15:28.500000+00:00", "nested": "2026-10-11T02:44:00+00:00"}


def test_a_stop_keeps_each_saved_queue_time_through_restore_and_its_readers(tmp_path, monkeypatch):
    """#1805: the snapshot keeps ``queued_at`` on its outer row and restore carries it back
    on every return path (ordinary, Project-held, saved running work), across a second
    restore too; a row that kept none stays unknown, and only a new task is stamped now."""
    import datetime

    from ouroboros import server_routing_context as routing
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from tests.test_project_routing_v664 import _ctx

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-run")
    workers.RUNNING["saved-run"]["task"]["queued_at"] = _SAVED_QUEUE_TIMES["saved-run"]
    for task_id in ("ordinary", "held", "nested", "legacy"):
        write_task_result(tmp_path, task_id, STATUS_SCHEDULED, chat_id=0)
    workers.PENDING += [
        _queued("ordinary", root_task_id="ordinary", queued_at=_SAVED_QUEUE_TIMES["ordinary"]),
        _queued("held", root_task_id="held", queued_at=_SAVED_QUEUE_TIMES["held"], _project_admission_restore_hold={
            "reason": "project_routing_fence_lookup_failed", "detail": "verify the Project first"}),
        _queued("nested", root_task_id="nested", queued_at=_SAVED_QUEUE_TIMES["nested"]),
        _queued("legacy", root_task_id="legacy", queued_at=_SAVED_QUEUE_TIMES["ordinary"]),
    ]
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True)
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text())
    saved = {row["id"]: row for row in snap["pending"]}
    assert saved["ordinary"]["queued_at"] == _SAVED_QUEUE_TIMES["ordinary"], "serialized on the outer row"
    # An older writer kept the time only inside its task; another row kept it nowhere.
    saved["nested"].pop("queued_at")
    saved["nested"]["task"]["queued_at"] = _SAVED_QUEUE_TIMES["nested"]
    saved["legacy"].pop("queued_at")
    saved["legacy"]["task"].pop("queued_at", None)
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap))
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    queue.restore_pending_from_snapshot()

    expected = {**{key: value for key, value in _SAVED_QUEUE_TIMES.items()}, "legacy": None}
    rows = {row["id"]: row for row in workers.PENDING}
    assert {task_id: rows[task_id]["queued_at"] for task_id in expected} == expected
    assert rows["held"]["_project_admission_restore_hold"] and rows["saved-run"]["_working_recovery"]

    def epoch(value):
        return datetime.datetime.fromisoformat(value).timestamp() if value else 0.0

    census = {row["activity_id"]: row["started_at"] for row in _chat_activities_snapshot_safe(tmp_path, direct_turns=[])}
    assert {task_id: census[task_id] for task_id in expected} == {
        task_id: epoch(value) for task_id, value in expected.items()}, "an unknown age is never this boot's"
    roots = {row["task_id"]: row["started_at"] for row in routing.addressable_roots(
        _ctx(tmp_path, pending=list(workers.PENDING))).rows()}
    assert {task_id: roots[task_id] for task_id in expected} == expected

    workers.PENDING.clear()  # the restore published its own snapshot; a second crash reads it back
    queue.restore_pending_from_snapshot()
    assert {row["id"]: row["queued_at"] for row in workers.PENDING} == expected

    before = datetime.datetime.now(datetime.timezone.utc)
    write_task_result(tmp_path, "new", STATUS_SCHEDULED, chat_id=0)
    fresh = queue.enqueue_task(_queued("new", root_task_id="new", queued_at=_SAVED_QUEUE_TIMES["ordinary"]))
    assert datetime.datetime.fromisoformat(fresh["queued_at"]) >= before, "a new admission is stamped now"


@pytest.mark.parametrize(("outer", "nested", "expected"), [
    ({"queued_at": "2026-10-01T00:00:00+00:00"}, "2026-01-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00"),
    ({"queued_at": None}, "2026-01-01T00:00:00+00:00", None),
    ({}, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
    ({}, None, None),
    ({"queued_at": True}, None, None),
    ({"queued_at": float("nan")}, None, None),
    ({"queued_at": float("inf")}, None, None),
    ({"queued_at": 0}, None, None),
    ({"queued_at": ""}, "2026-01-01T00:00:00+00:00", None),
    ({"queued_at": ["2026-10-01"]}, None, None),
], ids=["outer", "explicit-null", "nested-fallback", "absent", "bool", "nan", "inf",
        "zero", "blank", "not-a-time"])
def test_the_saved_row_decoder_reads_the_outer_queue_time_first(outer, nested, expected):
    """The outer value decides; only its absence falls back to a nested copy; nothing is guessed."""
    from supervisor.queue_snapshot import _saved_pending_task
    from supervisor.task_admission import restore_terminalization_retry

    task = {"id": "t", **({"queued_at": nested} if nested else {})}
    decoded = _saved_pending_task({"id": "t", **outer, "task": task})
    assert decoded["queued_at"] == expected and task.get("queued_at") == nested
    pending: list = []
    retry = restore_terminalization_retry({**decoded, "_terminalization_retry": {"status": "cancelled"}},
                                          pending=pending, running={}, queue_seq_counter_ref={"value": 0},
                                          sort_pending=lambda: None)
    assert retry["queued_at"] == expected, "a shutdown-custody row keeps the saved time or its unknown"


# R1 #1805 (``direct_pause.retain_direct_hold`` stamping a numeric ``queued_at`` on a failed inline park's
# hold) is retired with the in-process park: a direct turn is a conversation-lane RUNNING row whose pause
# the supervisor parks like any carried task's, so no hold of that kind is minted here any more.
