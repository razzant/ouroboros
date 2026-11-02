"""The conversation lane's slots: outside the project pool's count, assignment and fences (A1).

Equality laws (owner decision 1A/4A): a lane slot is never chosen for PENDING work; one
ready lane spare is kept and replenished outside ``MAX_WORKERS``; a project crash storm
and readiness exhaustion capture and terminalise project slots only; the carrier facts
on ``execution_owner`` are additive and old rows decode as they did.
"""

import queue as stdqueue
import time

import pytest

from ouroboros.task_carrier import (
    EXECUTOR_INLINE_DEGRADED, EXECUTOR_WORKER, LANE_CONVERSATION, LANE_PROJECT,
    carrier_executor, conversation_origin, execution_kind, inline_execution,
)
from supervisor import conversation_lane, queue, worker_health, worker_pool_lifecycle as lifecycle, workers
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial

# The shared pool fixture stubs the lane's maintenance; these tests read the real one.
_REAL_LANE_RECONCILE = conversation_lane.reconcile_conversation_lane


@pytest.fixture(autouse=True)
def real_lane(pool, monkeypatch):
    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", _REAL_LANE_RECONCILE)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))  # a started generation
    return pool


def task(tid, **fields):
    return {"id": tid, "type": "task", "chat_id": 0, "text": "work", **fields}


def add_slot(wid, *, lane, busy=None, pid=None):
    slot = workers.Worker(wid, Process(pid or 5000 + wid), InputQueue(), busy_task_id=busy, lane=lane)
    affirm_ready(slot)
    workers.WORKERS[wid] = slot
    return slot


@pytest.fixture
def assigning(pool, monkeypatch):
    from supervisor import evolution_lifecycle, state
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    return pool


def test_lane_slot_is_never_chosen_for_pending_work(assigning, monkeypatch):
    monkeypatch.setattr(workers, "MAX_WORKERS", 3)
    spare = add_slot(1, lane=LANE_CONVERSATION)
    workers.PENDING.append(task("queued"))
    workers.assign_tasks()  # the sole project worker is busy: demand spawns a project slot, never takes the spare
    assert spare.busy_task_id is None and spare.in_q.empty()
    assert "queued" in {row["id"] for row in workers.PENDING}
    created = [slot for slot in workers.WORKERS.values() if slot.booting]
    assert len(created) == 1 and created[0].lane == LANE_PROJECT


def test_one_lane_spare_is_kept_and_replenished_outside_max_workers(pool, monkeypatch):
    monkeypatch.setattr(workers, "MAX_WORKERS", 1)  # the project pool is full (its one worker is busy)
    lifecycle.reconcile_worker_capacity(lambda: 0)
    lane = [slot for slot in workers.WORKERS.values() if slot.lane == LANE_CONVERSATION]
    assert len(lane) == 1 and lane[0].booting and len(pool.created) == 1
    lifecycle.reconcile_worker_capacity(lambda: 0)
    assert len(pool.created) == 1  # a booting spare is already charged
    affirm_ready(lane[0])
    lane[0].reaping = False
    lane[0].busy_task_id = "turn"  # a turn claimed the spare
    lifecycle.reconcile_worker_capacity(lambda: 0)
    assert len(pool.created) == 2  # replenished, while MAX_WORKERS stays reached for project work
    assert sum(1 for slot in workers.WORKERS.values() if slot.lane == LANE_CONVERSATION) == 2


def test_lane_spare_never_spawns_in_a_marker_window(pool, monkeypatch):
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "managed_update_tx:assisted_resolution")
    lifecycle.reconcile_worker_capacity(lambda: 0)
    assert not pool.created


def test_lane_spare_is_not_doubled_beside_a_parked_unready_carrier(pool):
    parked = add_slot(1, lane=LANE_CONVERSATION)
    parked.readiness_exhausted, parked.reaping, parked.booting = True, True, False
    parked.proc.alive = False
    lifecycle.reconcile_worker_capacity(lambda: 0)
    assert not pool.created


def test_project_pool_keeps_no_spare_once_the_lane_holds_one(pool, monkeypatch):
    """The pool's one quiescent spare is the lane's: an idle project worker is probed for retirement."""
    monkeypatch.setattr(workers, "MAX_WORKERS", 4)
    workers.RUNNING.clear()
    pool.original.busy_task_id = None  # the only project worker is now idle
    add_slot(1, lane=LANE_CONVERSATION)
    lifecycle.reconcile_worker_capacity(lambda: 0)
    probe = pool.original.in_q.get_nowait()
    assert probe["type"] == "worker_quiescence"
    assert not pool.created


def test_project_pool_without_a_lane_spare_keeps_its_last_idle_worker(pool, monkeypatch):
    monkeypatch.setattr(workers, "MAX_WORKERS", 4)
    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", lambda: None)
    workers.RUNNING.clear()
    pool.original.busy_task_id = None
    lifecycle.reconcile_worker_capacity(lambda: 0)
    assert pool.original.in_q.empty()  # the sole slot of the table is never probed away


def test_storm_capture_and_project_fence_spare_the_lane(pool, monkeypatch):
    """Three dead project workers fence the project pool; the live lane carrier and its row survive."""
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0.0)
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 1})
    monkeypatch.setattr(workers, "append_jsonl", lambda *a, **k: True)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda _pid: None)
    monkeypatch.setattr(workers, "_audit_delegate_terminal_custody", lambda *a, **k: None)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *a, **k: {
        "cost_accounting_status": "unavailable", "cost_final": False})
    workers.RUNNING.clear()
    dead = []
    for wid in (0, 1, 2):
        slot = workers.WORKERS.get(wid) or add_slot(wid, lane=LANE_PROJECT)
        slot.busy_task_id = f"p{wid}"
        slot.proc.alive, slot.proc.exitcode = False, -9
        workers.RUNNING[f"p{wid}"] = {"task": task(f"p{wid}"), "worker_id": wid, "attempt": 1,
                                      "started_at": time.time(), "last_heartbeat_at": time.time()}
        dead.append(slot)
    carrier = add_slot(7, lane=LANE_CONVERSATION, busy="turn")
    workers.RUNNING["turn"] = {"task": task("turn", _is_direct_chat=True), "worker_id": 7, "attempt": 1,
                               "started_at": time.time(), "last_heartbeat_at": time.time(),
                               "lane": LANE_CONVERSATION}
    workers.CRASH_TS.extend([time.time(), time.time()])
    _ids, disabled = workers._ensure_workers_healthy_locked(queue)
    assert disabled and workers._WORKER_POOL_DISABLED_REASON == "worker_crash_storm"
    jobs = []
    while not queue._reap_queue.empty():
        jobs.append(queue._reap_queue.get_nowait())
    storm = next(job for job in jobs if job["kind"] == "worker_crash_storm")
    assert {wid for wid, _slot in storm["workers"]} == {0, 1, 2}  # the lane carrier is not the storm's
    assert carrier.busy_task_id == "turn" and not carrier.reaping
    # The fence's stop owns project slots only: the lane carrier and its row stay.
    assert workers.kill_workers(disable_reason="worker_crash_storm", lanes=frozenset({LANE_PROJECT}))
    assert set(workers.WORKERS) == {7} and workers.WORKERS[7] is carrier
    assert set(workers.RUNNING) == {"turn"}
    # The project fence never blocks the lane: its spare is replenished under the latch.
    lifecycle.reconcile_worker_capacity(lambda: 0)
    assert len(pool.created) == 1 and pool.created and workers.WORKERS[8].lane == LANE_CONVERSATION
    # ... and a project respawn stays refused by the same latch.
    assert lifecycle.respawn_worker(0) is False


def test_exhaustion_disablement_ignores_lane_rows_and_slots(pool, monkeypatch):
    monkeypatch.setattr(workers, "kill_workers", lambda **kw: pytest.fail(f"project stop with {kw}"))
    workers.RUNNING.clear()
    exhausted = pool.original
    exhausted.busy_task_id, exhausted.readiness_exhausted, exhausted.reaping = None, True, True
    exhausted.proc.alive = False
    add_slot(1, lane=LANE_CONVERSATION, busy="turn")
    workers.RUNNING["turn"] = {"task": task("turn"), "worker_id": 1, "attempt": 1, "lane": LANE_CONVERSATION,
                               "started_at": time.time(), "last_heartbeat_at": time.time()}
    stopped = []
    monkeypatch.setattr(workers, "kill_workers", lambda **kw: stopped.append(kw) or True)
    assert workers.disable_exhausted_worker_pool() is True
    assert stopped and stopped[0]["lanes"] == frozenset({LANE_PROJECT})


def test_execution_state_counts_project_slots_only(pool):
    add_slot(1, lane=LANE_CONVERSATION)
    state = lifecycle._worker_pool_execution_state()
    assert state["worker_count"] == 1 and state["available"]
    workers.WORKERS.pop(0)  # the project pool retired to zero at quiescence: still a started pool
    state = lifecycle._worker_pool_execution_state()
    assert state["worker_count"] == 0 and state["available"] and state["disabled_reason"] == ""


def test_spawn_workers_starts_the_lane_spare_first(pool, monkeypatch):
    workers.WORKERS.clear()
    monkeypatch.setattr(workers, "_record_worker_processes", lambda *a, **k: None)
    monkeypatch.setattr(workers.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: None})())
    monkeypatch.setattr(workers, "events_log_cursor", lambda: (0, 0, 0))
    monkeypatch.setattr(workers, "event_queue_generation", lambda: "gen")
    monkeypatch.setattr(workers, "append_jsonl", lambda *a, **k: True)
    import supervisor.worker_process as wp
    monkeypatch.setattr(wp, "spawn_worker_process", lambda ctx, wid, in_q, out_q, repo, drive: Process(900 + wid))
    workers.spawn_workers()
    assert [slot.lane for slot in workers.WORKERS.values()] == [LANE_CONVERSATION]
    workers.WORKERS.clear()
    workers.spawn_workers(3)
    assert sorted(slot.lane for slot in workers.WORKERS.values()) == [LANE_CONVERSATION, LANE_PROJECT, LANE_PROJECT]


def test_carrier_facts_are_additive_and_old_rows_decode(tmp_path):
    from ouroboros.task_status import execution_owner_record

    legacy_direct = {"id": "a", "_is_direct_chat": True}
    assert execution_kind(legacy_direct) == "direct" and inline_execution(legacy_direct)
    assert conversation_origin(legacy_direct) and carrier_executor(legacy_direct) == ""
    legacy_pooled = {"id": "b", "_execution_owner": {"kind": "pooled"}}
    assert not inline_execution(legacy_pooled) and not conversation_origin(legacy_pooled)
    lane = {"id": "c", "_is_direct_chat": True,
            "_execution_owner": {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}}
    assert execution_kind(lane) == "direct" and conversation_origin(lane) and not inline_execution(lane)
    degraded = {"id": "d", "_is_direct_chat": True,
                "_execution_owner": {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_INLINE_DEGRADED}}
    assert inline_execution(degraded)
    pooled = execution_owner_record(tmp_path, {"id": "e"}, "pooled")
    assert pooled["lane"] == LANE_PROJECT and pooled["executor"] == EXECUTOR_WORKER
    rebound = execution_owner_record(tmp_path, {**lane, "_attempt": 1}, "direct")  # the agent re-binds the record
    assert rebound["kind"] == "direct" and rebound["lane"] == LANE_CONVERSATION and rebound["executor"] == EXECUTOR_WORKER
    plain = execution_owner_record(tmp_path, {"id": "f", "_is_direct_chat": True}, "direct")
    assert "lane" not in plain and "executor" not in plain  # a legacy admission records nothing it does not know
