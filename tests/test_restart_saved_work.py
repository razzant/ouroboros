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
from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_restart_retention import _pool_events, _queued

pytestmark = pytest.mark.serial


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


def _ack(root, transaction_id):
    from ouroboros import delegate_recovery as dr

    row = dr._read_restart_transaction(root, transaction_id)
    dr._write_restart_transaction(root, {**row, "status": "normal_exit_acknowledged", "exit_code": 42})


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
        assert hold and hold["reason"] == HOLD_SAVED_WORK and hold["stop_cause"] == "app_stop"
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


@pytest.mark.parametrize("origin", ["restarted", "resumed", "idle_retry"])
@pytest.mark.parametrize("door", ["restart", "quit", "crash", "panic"])
def test_runnable_saved_successor_waiting_for_capacity_follows_the_next_stop(tmp_path, monkeypatch, origin, door):
    """A pending successor still owns saved work; another stop does not buy an attempt."""
    import queue as stdqueue

    from supervisor.restart_retention import prepare_restart_returns
    from tests.test_owner_wait_restart import InertProcess

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _working_run(tmp_path, workers, "saved-root")
    if origin == "idle_retry":
        from supervisor import task_reaper

        task = workers.RUNNING.pop("saved-root")["task"]
        accepted, attempt, _reason, _ = task_reaper._enqueue_retry(
            queue, task, task_id="saved-root", retry_task_id="", attempt=1,
            terminal_reason="idle_timeout", recon_fields={})
        assert accepted and attempt == 2
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
    assert restored["_attempt"] == 2 and restored["_working_recovery"] == handoff
    ctx, _limit = _loop_ctx(tmp_path, "saved-root", attempt=2)
    assert wc.load_recovery(ctx, handoff)["working"]["boundary"] == "pre_effect"
    commands = stdqueue.Queue()
    workers.WORKERS[0] = workers.Worker(0, InertProcess(), commands)
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


@pytest.mark.parametrize("control", ["prior_hold", "owner_hold", "stop", "bad_source", "attempt_dispatched"])
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
    workers.WORKERS[0] = workers.Worker(0, InertProcess(), commands)
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
    assert budget_hold_fact(rows["saved-root"])["stop_cause"] == "panic"
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
    assert budget_hold_fact(rows["saved-root"])["stop_cause"] == "panic"
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
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 900)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 1800)  # idle window 1920 s
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

    assert asked == {"resolver": ("idle_timeout", ["finalize_now"]),
                     "deadline-run": ("deadline", ["finalize_now"])}
    assert sorted(reaped) == ["deadline-run", "resolver"]
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
