"""#939 x Batch4 Q1A: the owner's Continue keeps the work's explicit execution budget, once.

The budget is the same work's (Batch4: "a new task id does not reset the expenses and limits of
the whole assignment"; #939: execution excluding queue and confirmed pauses, on the existing
clock). 900 seconds with 700 executed leaves 200: the successor keeps the total and starts from
the closed execution, which ``settle_execution_carry`` turns into its clock's offset once.

Driven through the real seams: queue admission, ``assign_tasks`` and its RUNNING row, the
worker's start record, the terminal writers that release that row (the settled ``task_done``,
the crash terminal, the owner Restart's kill, the hard-bound teardown), ``admit_continuation``
with its claim, the successor's assignment and the worker's own clock. No current producer
opens an owner-addressed root with an API/CLI budget (an API root records no opening that
Continue accepts), so each predecessor is admitted with both through the real queue; every
later fact is produced by the runtime. A total that was never closed for the row's last
attempt is unknown: refused, never read as zero, a fresh budget or no limit.
"""

from __future__ import annotations

import copy
import queue as stdqueue
import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._cancel_intents_shared import _reap_spawned_live_procs  # noqa: F401 -- autouse reaper of _LiveProc
from tests.test_owner_continue import NONCE
from tests.worker_pool_support import affirm_ready

BUDGET = 900


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path_factory.mktemp("projects_root")))


@pytest.fixture
def pool(tmp_path, monkeypatch):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda task: True)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_k: True)
    return SimpleNamespace(root=tmp_path, queue=queue, state=state, workers=workers, mp=monkeypatch)


def _admit_root(pool, task_id="budget-1", budget=BUDGET):
    """A root admitted with an owner opening and an explicit execution budget."""
    from ouroboros.task_results import write_task_result

    origin = {"origin_message_text": "Find the crash; fifteen minutes of work at most",
              "origin_message_ref": {"chat_id": 7, "client_message_id": f"m-{task_id}"}}
    write_task_result(pool.root, task_id, "scheduled", chat_id=7, root_task_id=task_id, title="Find the crash",
                      execution_timeout_sec=budget, billing_group={
                          "billing_group_id": task_id, "billing_group_limit_usd": 20.0,
                          "billing_group_limit_source": "initial_task_admission",
                          "billing_group_limit_revision": "admission-1"}, **origin)
    admitted = pool.queue.enqueue_task({"id": task_id, "type": "task", "chat_id": 7, "root_task_id": task_id,
                                        "text": "Find the crash", "execution_timeout_sec": budget, **origin})
    assert not admitted.get("_admission_blocked"), admitted


def _start(pool, task_id, *, executed):
    """The real assignment and the worker's start record; then ``executed`` seconds of work."""
    from ouroboros.agent import OuroborosAgent

    sent = []
    pool.workers.WORKERS[7] = affirm_ready(SimpleNamespace(
        wid=7, busy_task_id=None, reaping=False, in_q=SimpleNamespace(put=lambda row: sent.append(copy.deepcopy(row)))))
    pool.workers.assign_tasks()
    assert [row["id"] for row in sent] == [task_id]
    OuroborosAgent._persist_running_record(SimpleNamespace(env=SimpleNamespace(drive_root=pool.root)), dict(sent[0]))
    meta = pool.workers.RUNNING[task_id]
    meta["started_at"] -= executed
    return sent[0], meta


def _settle_by_task_done(pool, task_id, *, reason_code="provider_unavailable"):
    """The worker's own failed terminal, then the supervisor's real ``task_done`` handling."""
    from ouroboros.task_results import write_task_result
    from supervisor.events import _handle_task_done
    from tests.test_task_done_durable_vs_transport import _mk_ctx

    write_task_result(pool.root, task_id, "failed", reason_code=reason_code, result="The provider was unavailable.")
    ctx = _mk_ctx(pool.root, [], [])
    ctx.RUNNING, ctx.WORKERS, ctx.PENDING = pool.workers.RUNNING, pool.workers.WORKERS, pool.workers.PENDING
    _handle_task_done({"type": "task_done", "task_id": task_id, "worker_id": 7, "task_type": "task",
                       "status": "failed", "_files_prepared_attempt": 1}, ctx)
    assert task_id not in pool.workers.RUNNING


def _close(pool, task_id):
    from ouroboros.task_results import load_task_result

    return load_task_result(pool.root, task_id).get("execution_consumed")


def _continue(pool, task_id):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    ack = admit_continuation(task_id, action_nonce=NONCE)
    assert ack["ok"] is True, ack
    row = next(task for task in pool.workers.PENDING if task["id"] == ack["successor_task_id"])
    binding = load_task_result(pool.root, task_id)["continued_by"]["binding"]
    return row, binding


def test_continue_keeps_the_total_and_the_closed_execution_once(pool):
    """900 total, 700 executed: the successor's window is 200 before it runs, its own
    execution then adds once, and a second Continue carries that sum, never a fresh 900."""
    from ouroboros.task_results import load_task_result

    _admit_root(pool)
    _start(pool, "budget-1", executed=700)
    _settle_by_task_done(pool, "budget-1")
    closed = _close(pool, "budget-1")
    assert closed["attempt"] == 1 and closed["sec"] == pytest.approx(700, abs=30)

    row, binding = _continue(pool, "budget-1")

    assert binding["execution_timeout_sec"] == BUDGET and binding["execution_consumed_sec"] == closed["sec"]
    assert row["execution_timeout_sec"] == BUDGET and row["_execution_handoff_sec"] == closed["sec"]
    assert row.get("_execution_consumed_sec", 0) == 0, "settled once, at assignment"
    stored = load_task_result(pool.root, row["id"])
    assert stored["execution_timeout_sec"] == BUDGET
    assert stored["execution_consumed"] == {"sec": closed["sec"], "attempt": 0}
    # The successor's real assignment settles the carry once; the worker's own clock reads it.
    sent, meta = _start(pool, row["id"], executed=0)
    assert "_execution_handoff_sec" not in meta["task"] and "_execution_handoff_sec" not in sent
    assert meta["task"]["_execution_consumed_sec"] == pytest.approx(closed["sec"])
    from tests._execution_clock_seams import agent_start

    waiter = agent_start(sent, pool.root, pool.mp)
    assert waiter.execution_window_remaining() == pytest.approx(BUDGET - closed["sec"], abs=30)

    # Fifty more seconds, then another technical end: the next Continue carries 750, not 1450.
    meta["started_at"] -= 50
    _settle_by_task_done(pool, row["id"])
    second = _close(pool, row["id"])
    assert second["attempt"] == 1 and second["sec"] == pytest.approx(closed["sec"] + 50, abs=30)
    row2, binding2 = _continue(pool, row["id"])
    assert binding2["execution_timeout_sec"] == BUDGET and binding2["execution_consumed_sec"] == second["sec"]
    assert row2["_execution_handoff_sec"] == second["sec"]


def test_a_successor_that_never_started_hands_the_same_total_on(pool):
    from ouroboros.task_results import load_task_result
    from tests.test_restart_retention import _pool_events

    _admit_root(pool)
    _start(pool, "budget-1", executed=700)
    _settle_by_task_done(pool, "budget-1")
    first, binding = _continue(pool, "budget-1")
    _pool_events(pool.workers, pool.mp)
    pool.mp.setattr(pool.workers, "WORKERS", {})
    assert pool.workers.kill_workers(terminal_status="cancelled", result_reason="server shutdown",
                                     stop_source="server_shutdown")
    assert load_task_result(pool.root, first["id"])["cancel_origin"]["source"] == "server_shutdown"

    second, binding2 = _continue(pool, first["id"])

    assert binding2["execution_timeout_sec"] == BUDGET
    assert binding2["execution_consumed_sec"] == binding["execution_consumed_sec"]
    assert second["_execution_handoff_sec"] == binding["execution_consumed_sec"]


def test_the_owner_restart_kill_closes_the_attempt_it_interrupts(pool):
    from ouroboros.task_results import load_task_result
    from tests.test_restart_retention import _pool_events, _restart_door

    _admit_root(pool)
    _start(pool, "budget-1", executed=400)
    pool.mp.setattr(pool.workers, "WORKERS", {})  # the worker slot is gone; the door settles the row
    _pool_events(pool.workers, pool.mp)
    _restart_door(pool.root, pool.mp, pool.workers)
    stored = load_task_result(pool.root, "budget-1")
    assert stored["cancel_origin"]["source"] == "owner_restart"
    assert stored["execution_consumed"]["attempt"] == 1
    assert stored["execution_consumed"]["sec"] == pytest.approx(400, abs=30)

    _row, binding = _continue(pool, "budget-1")

    assert binding["execution_consumed_sec"] == stored["execution_consumed"]["sec"]


@pytest.mark.serial  # a real worker process is killed (tests/_cancel_intents_shared._LiveProc)
def test_an_owner_restart_cancel_that_wins_the_race_closes_the_attempt(pool):
    """The Restart door records its ``owner_restart`` intent before ``kill_workers``; cancellation
    custody can win that race, kill the worker and release the RUNNING row itself
    (``cancel_publication._publish_cancelled_task``). That release closes the attempt too."""
    from ouroboros import cancel_intents as ci
    from ouroboros.task_results import load_task_result
    from supervisor import task_lifecycle
    from tests._cancel_intents_shared import _LiveProc

    _admit_root(pool)
    _start(pool, "budget-1", executed=400)
    proc = pool.workers.WORKERS[7].proc = _LiveProc()
    pool.mp.setattr(pool.workers, "respawn_worker", lambda wid: None, raising=False)
    pool.mp.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    pool.mp.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    pool.mp.setattr(pool.queue, "_emit_cancel_task_done", lambda *_a, **_k: None)
    ci.request_cancel(pool.root, "budget-1", reason="Owner restart", source="owner_restart",
                      requested_by="owner", requested_stop_policy="immediate", allow_settled_target=True)

    assert task_lifecycle.cancel_task_custody("budget-1") == task_lifecycle.CANCEL_CANCELLED

    assert not proc.is_alive() and "budget-1" not in pool.workers.RUNNING
    stored = load_task_result(pool.root, "budget-1")
    assert stored["status"] == "cancelled" and stored["cancel_origin"]["source"] == "owner_restart"
    assert stored["execution_consumed"]["attempt"] == 1
    assert stored["execution_consumed"]["sec"] == pytest.approx(400, abs=30)
    _row, binding = _continue(pool, "budget-1")
    assert binding["execution_consumed_sec"] == stored["execution_consumed"]["sec"]


def test_a_crash_terminal_closes_the_dead_attempt(pool):
    from ouroboros.task_results import load_task_result
    from supervisor.worker_health import recover_confirmed_dead_worker
    from tests.test_restart_retention import _pool_events

    _admit_root(pool)
    _sent, meta = _start(pool, "budget-1", executed=300)
    worker = pool.workers.WORKERS[7]
    worker.proc = SimpleNamespace(pid=worker.proc.pid, is_alive=lambda: False, exitcode=-9)
    worker.reaping = True
    _pool_events(pool.workers, pool.mp)
    pool.mp.setattr(pool.workers, "reconstruct_task_cost", lambda *_a, **_k: {"cost_accounting_status": "available"})
    pool.mp.setattr(pool.workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)

    recover_confirmed_dead_worker({"kind": "confirmed_dead_worker", "worker_id": 7, "worker": worker,
                                   "task_id": "budget-1", "meta": meta, "task": dict(meta["task"]), "attempt": 1,
                                   "exitcode": -9, "drive_root": str(pool.root), "skip_respawn": True})

    stored = load_task_result(pool.root, "budget-1")
    assert stored["status"] == "failed" and stored["reason_code"] == "worker_crash_signal"
    assert stored["execution_consumed"]["attempt"] == 1
    assert stored["execution_consumed"]["sec"] == pytest.approx(300, abs=30)
    _row, binding = _continue(pool, "budget-1")
    assert binding["execution_consumed_sec"] == stored["execution_consumed"]["sec"]


def test_a_worker_that_saved_its_own_terminal_then_died_closes_the_attempt(pool):
    """The crash recovery publishes a dead worker's saved terminal as a ``task_done`` while the
    RUNNING row stays; the ordinary handler then closes the attempt before releasing it."""
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.events import _handle_task_done
    from supervisor.worker_health import recover_confirmed_dead_worker
    from tests.test_restart_retention import _pool_events
    from tests.test_task_done_durable_vs_transport import _mk_ctx

    _admit_root(pool)
    _sent, meta = _start(pool, "budget-1", executed=250)
    write_task_result(pool.root, "budget-1", "failed", reason_code="provider_unavailable",
                      result="The provider was unavailable.")
    worker = pool.workers.WORKERS[7]
    worker.proc = SimpleNamespace(pid=worker.proc.pid, is_alive=lambda: False, exitcode=1)
    worker.reaping = True
    events = _pool_events(pool.workers, pool.mp)
    pool.mp.setattr(pool.workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)

    recover_confirmed_dead_worker({"kind": "confirmed_dead_worker", "worker_id": 7, "worker": worker,
                                   "task_id": "budget-1", "meta": meta, "task": dict(meta["task"]), "attempt": 1,
                                   "exitcode": 1, "drive_root": str(pool.root), "skip_respawn": True})
    [done] = [event for event in events if event.get("type") == "task_done"]
    assert pool.workers.RUNNING["budget-1"] is meta and "execution_consumed" not in load_task_result(
        pool.root, "budget-1")
    ctx = _mk_ctx(pool.root, [], [])
    ctx.RUNNING, ctx.WORKERS, ctx.PENDING = pool.workers.RUNNING, pool.workers.WORKERS, pool.workers.PENDING
    _handle_task_done(done, ctx)

    stored = load_task_result(pool.root, "budget-1")
    assert "budget-1" not in pool.workers.RUNNING
    assert stored["status"] == "failed" and stored["reason_code"] == "provider_unavailable"
    assert stored["execution_consumed"]["attempt"] == 1
    assert stored["execution_consumed"]["sec"] == pytest.approx(250, abs=30)
    _row, binding = _continue(pool, "budget-1")
    assert binding["execution_consumed_sec"] == stored["execution_consumed"]["sec"]


def test_a_spent_budget_is_a_reached_limit_not_a_continue(pool):
    """The hard-bound teardown closes the attempt before its RUNNING row goes to the reaper,
    whose later terminal merges over it; a spent budget refuses like a reached deadline."""
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import queue_timeouts
    from supervisor.continuation_admission import admit_continuation

    _admit_root(pool)
    _sent, meta = _start(pool, "budget-1", executed=BUDGET + 5)
    meta.update(finalization_requested_at=time.time() - pool.queue.FINALIZATION_GRACE_SEC - 1,
                finalization_reason="absolute_ceiling")
    reaped = stdqueue.Queue()
    pool.mp.setattr(pool.queue, "_reap_queue", reaped)
    pool.mp.setattr(pool.queue, "_ensure_reaper_started", lambda: None)

    queue_timeouts._enforce_task_timeouts_locked(pool.workers, time.time(), 0, {})

    assert "budget-1" not in pool.workers.RUNNING and reaped.get_nowait()["terminal_reason"] == "absolute_ceiling"
    closed = _close(pool, "budget-1")
    assert closed["attempt"] == 1 and closed["sec"] >= BUDGET
    write_task_result(pool.root, "budget-1", "failed", reason_code="absolute_ceiling")  # the reaper's terminal
    stored = load_task_result(pool.root, "budget-1")
    assert stored["execution_consumed"] == closed
    assert continuation_offer(stored, "budget-1") == {
        "eligible": False, "cause": "execution_deadline", "refusal": "hard_limit_reached"}
    assert admit_continuation("budget-1", action_nonce=NONCE) == {
        "ok": False, "error": "hard_limit_reached", "cause": "execution_deadline"}
    assert "continued_by" not in load_task_result(pool.root, "budget-1") and pool.workers.PENDING == []


@pytest.mark.parametrize("close", [
    None,  # a record settled before its terminal writers kept the total
    {"sec": 120.0, "attempt": 1},  # a later attempt started and never closed
    {"sec": "120", "attempt": 2},
    {"sec": -1.0, "attempt": 2},
    {"sec": 120.0, "attempt": True},
], ids=["never_recorded", "stale_attempt", "text", "negative", "bool_attempt"])
def test_an_unknown_consumed_total_is_refused_never_regranted(pool, close):
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation

    _admit_root(pool)
    pool.workers.PENDING.clear()
    write_task_result(pool.root, "budget-1", "failed", reason_code="provider_unavailable", task_attempt=2,
                      **({"execution_consumed": close} if close else {}))
    stored = load_task_result(pool.root, "budget-1")

    assert continuation_offer(stored, "budget-1") == {
        "eligible": False, "cause": "provider_unavailable", "refusal": "execution_budget_unknown"}
    assert admit_continuation("budget-1", action_nonce=NONCE) == {
        "ok": False, "error": "execution_budget_unknown", "cause": "provider_unavailable"}
    assert "continued_by" not in load_task_result(pool.root, "budget-1") and pool.workers.PENDING == []


def test_work_without_an_explicit_budget_keeps_todays_continue(pool):
    from ouroboros.task_results import load_task_result

    _admit_root(pool, budget=0)  # 0 adds no limit (the API's meaning)
    _start(pool, "budget-1", executed=700)
    _settle_by_task_done(pool, "budget-1")
    assert "execution_consumed" not in load_task_result(pool.root, "budget-1")

    row, binding = _continue(pool, "budget-1")

    assert not {"execution_timeout_sec", "execution_consumed_sec"} & set(binding)
    assert "execution_timeout_sec" not in row and "_execution_handoff_sec" not in row
    assert "execution_consumed" not in load_task_result(pool.root, row["id"])


def test_a_close_written_after_the_binding_was_read_refuses_the_stale_claim(pool):
    """The claim re-reads the budget under the predecessor's lock: a binding built from an
    earlier read (a late terminal writer closed more execution since) writes nothing."""
    from ouroboros.owner_continue import claim_on_predecessor, execution_carry
    from ouroboros.task_results import load_task_result, write_task_result

    _admit_root(pool)
    _start(pool, "budget-1", executed=700)
    _settle_by_task_done(pool, "budget-1")
    stale = execution_carry(load_task_result(pool.root, "budget-1"))
    write_task_result(pool.root, "budget-1", "failed", execution_consumed={"sec": 760.0, "attempt": 1})
    binding = {"successor_task_id": "budget-1-cX", **stale}

    with pytest.raises(ValueError, match="execution_budget_changed"):
        claim_on_predecessor(pool.root, "budget-1", NONCE, binding)

    assert "continued_by" not in load_task_result(pool.root, "budget-1")
