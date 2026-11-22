"""The conversation lane's slots: outside the project pool's count, assignment and fences (A1).

Equality laws (owner decision 1A/4A): a lane slot is never chosen for PENDING work; one
ready lane spare is kept and replenished outside ``MAX_WORKERS``; a project crash storm
and readiness exhaustion capture and terminalise project slots only; the carrier facts
on ``execution_owner`` are additive and old rows decode as they did.
"""

import threading
import time

import pytest

from ouroboros.task_carrier import (
    EXECUTOR_INLINE_DEGRADED, EXECUTOR_WORKER, LANE_CONVERSATION, LANE_PROJECT,
    carrier_executor, conversation_origin, execution_kind, inline_execution,
)
from supervisor import conversation_lane, queue, worker_chat_lane, worker_pool_lifecycle as lifecycle, workers
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready


_THREAD = threading.Thread  # the real class: the pool fixture stubs ``threading.Thread`` for the pool's own threads

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
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(3))
    spare = add_slot(1, lane=LANE_CONVERSATION)
    workers.PENDING.append(task("queued"))
    workers.assign_tasks()  # the sole project worker is busy: demand spawns a project slot, never takes the spare
    assert spare.busy_task_id is None and spare.in_q.empty()
    assert "queued" in {row["id"] for row in workers.PENDING}
    created = [slot for slot in workers.WORKERS.values() if slot.booting]
    assert len(created) == 1 and created[0].lane == LANE_PROJECT


def test_one_lane_spare_is_kept_and_replenished_outside_max_workers(pool, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(1))  # the project pool is full (its one worker is busy)
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
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(4))
    workers.RUNNING.clear()
    pool.original.busy_task_id = None  # the only project worker is now idle
    add_slot(1, lane=LANE_CONVERSATION)
    lifecycle.reconcile_worker_capacity(lambda: 0)
    probe = pool.original.in_q.get_nowait()
    assert probe["type"] == "worker_quiescence"
    assert not pool.created


def test_project_pool_without_a_lane_spare_keeps_its_last_idle_worker(pool, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", str(4))
    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", lambda: None)
    workers.RUNNING.clear()
    pool.original.busy_task_id = None
    lifecycle.reconcile_worker_capacity(lambda: 0)
    assert pool.original.in_q.empty()  # the sole slot of the table is never probed away


def test_three_project_deaths_fence_nothing_and_the_exhaustion_fence_spares_the_lane(pool, monkeypatch):
    """Three dead project workers are three recoveries, no fence (ARCHITECTURE §5: the crash-storm fence is
    gone); the live lane carrier and its row survive them. The one project fence left, readiness exhaustion,
    still spares the lane: its stop owns project slots only and the lane's spare is replenished under the latch."""
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0.0)
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 1})
    monkeypatch.setattr(workers, "append_jsonl", lambda *a, **k: True)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda _pid: None)
    monkeypatch.setattr(workers, "_audit_delegate_terminal_custody", lambda *a, **k: None)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *a, **k: {
        "cost_accounting_status": "unavailable", "cost_final": False})
    workers.RUNNING.clear()
    for wid in (0, 1, 2):
        slot = workers.WORKERS.get(wid) or add_slot(wid, lane=LANE_PROJECT)
        slot.busy_task_id = f"p{wid}"
        slot.proc.alive, slot.proc.exitcode = False, -9
        workers.RUNNING[f"p{wid}"] = {"task": task(f"p{wid}"), "worker_id": wid, "attempt": 1,
                                      "started_at": time.time(), "last_heartbeat_at": time.time()}
    carrier = add_slot(7, lane=LANE_CONVERSATION, busy="turn")
    workers.RUNNING["turn"] = {"task": task("turn", _is_direct_chat=True), "worker_id": 7, "attempt": 1,
                               "started_at": time.time(), "last_heartbeat_at": time.time(),
                               "lane": LANE_CONVERSATION}
    _ids, disabled = workers._ensure_workers_healthy_locked(queue)
    assert not disabled and workers._WORKER_POOL_DISABLED_REASON == ""
    jobs = []
    while not queue._reap_queue.empty():
        jobs.append(queue._reap_queue.get_nowait())
    assert [job["kind"] for job in jobs] == ["confirmed_dead_worker"] * 3  # three recoveries, no stop job
    assert {job["worker_id"] for job in jobs} == {0, 1, 2} and not any(job.get("skip_respawn") for job in jobs)
    assert carrier.busy_task_id == "turn" and not carrier.reaping
    # The exhaustion fence's stop owns project slots only: the lane carrier and its row stay.
    assert workers.kill_workers(disable_reason="worker_readiness_exhausted", lanes=frozenset({LANE_PROJECT}))
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


def test_a_failed_boot_spawn_is_recorded_and_the_next_generation_clears_it(pool, monkeypatch):
    """S9 keeps a server's Presence turns waiting for its pool, but a boot whose carrier spawn failed has no
    supervisor loop to retry it: that observed physical fact is recorded so Presence runs on the disclosed
    degraded executor instead of a typed retry forever; the next started generation clears it."""
    workers.WORKERS.clear()
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", None)  # the boot: no generation has armed yet
    monkeypatch.setattr(workers, "_record_worker_processes", lambda *a, **k: None)
    monkeypatch.setattr(workers.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: None})())
    monkeypatch.setattr(workers, "events_log_cursor", lambda: (0, 0, 0))
    monkeypatch.setattr(workers, "event_queue_generation", lambda: "gen")
    monkeypatch.setattr(workers, "append_jsonl", lambda *a, **k: True)
    import supervisor.worker_process as wp

    def refuse(*_args):
        raise OSError("fork refused")
    monkeypatch.setattr(wp, "spawn_worker_process", refuse)
    assert conversation_lane.spawn_refusal() == ""
    with pytest.raises(OSError):
        workers.spawn_workers()
    assert not conversation_lane.armed() and conversation_lane.spawn_refusal() == "carrier_spawn_refused:OSError"
    monkeypatch.setattr(wp, "spawn_worker_process", lambda ctx, wid, in_q, out_q, repo, drive: Process(900 + wid))
    workers.spawn_workers()
    assert conversation_lane.armed() and conversation_lane.spawn_refusal() == ""


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


def _stop_doubles(monkeypatch):
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0.0)
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 1})
    monkeypatch.setattr(workers, "append_jsonl", lambda *a, **k: True)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda _pid: None)
    monkeypatch.setattr(workers, "_audit_delegate_terminal_custody", lambda *a, **k: None)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *a, **k: {
        "cost_accounting_status": "unavailable", "cost_final": False})


def test_a_lane_turn_admitted_during_a_project_stops_teardown_survives_the_stop(pool, monkeypatch):
    """S43 defect 2: the rows a project stop spares are chosen under the same Q hold as its terminalization.
    A hello admitted while the doomed slots were torn down outside Q is the lane's too: never failed by
    that stop with "Worker process crashed (crash storm)"."""
    from ouroboros.task_results import load_task_result

    _stop_doubles(monkeypatch)
    workers.RUNNING.clear()
    dead = workers.WORKERS.get(0) or add_slot(0, lane=LANE_PROJECT)
    dead.busy_task_id, dead.proc.alive, dead.proc.exitcode = "p0", False, -9
    workers.RUNNING["p0"] = {"task": task("p0"), "worker_id": 0, "attempt": 1, "started_at": time.time(),
                             "last_heartbeat_at": time.time()}
    real_teardown = workers._tear_down_slots

    def teardown_then_hello(doomed):
        real_teardown(doomed)  # Q is dropped here; the owner's hello is admitted in that gap
        workers.RUNNING["hello"] = {"task": task("hello", _is_direct_chat=True), "worker_id": None, "attempt": 1,
                                    "started_at": time.time(), "lane": LANE_CONVERSATION}

    monkeypatch.setattr(workers, "_tear_down_slots", teardown_then_hello)
    assert workers.kill_workers(disable_reason="worker_readiness_exhausted", lanes=frozenset({LANE_PROJECT}))
    assert "p0" not in workers.RUNNING
    assert "hello" in workers.RUNNING and workers.RUNNING["hello"]["worker_id"] is None  # still awaiting its carrier
    assert (load_task_result(pool.root, "hello") or {}).get("status") != "failed"


def test_a_turn_admitted_while_the_spare_boots_is_dispatched_through_a_marker_window(pool, monkeypatch):
    """S43 defect 3: a marker window forbids a spawn, never a waiting turn's carrier. The spare that was
    booting at admission takes the turn on the next reconcile, inside the window; nothing spawns."""
    spare = add_slot(7, lane=LANE_CONVERSATION)
    spare.booting = spare.reaping = True  # spawned, not yet opened by the readiness owner
    meta = conversation_lane.admit_turn(task("t", _is_direct_chat=True), chat_id=1, client_message_id="cm")
    worker_chat_lane._degrade_if_no_carrier_can_start(meta)
    assert meta["worker_id"] is None and all(w.executor == EXECUTOR_WORKER for w in workers.WORKERS.values())
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "managed_update")  # the window closes
    spare.booting = spare.reaping = False  # the readiness owner opened the spare
    conversation_lane.reconcile_conversation_lane()
    assert meta["worker_id"] == 7 and spare.busy_task_id == meta["task"]["id"]
    assert len(pool.created) == 0  # the window forbade a spawn, as it should


def test_the_reaper_respawns_no_carrier_in_a_marker_window(pool, monkeypatch):
    """``worker_pool_lifecycle.respawn_worker`` (LG's stand finding): a dead carrier's replacement is not started
    inside an assisted update's conflict-marker window — a child would import the markers — although
    ``reconcile_conversation_lane`` already promised nothing spawns there. The refusal is typed and raised before
    any process starts, so the reaper's refusal episode keeps the slot and retries it each tick; the same respawn
    starts the replacement once the window ends. The degraded executor covers the lane's turns meanwhile."""
    from supervisor import queue as q, task_reaper

    # The lane's spawn-refusal fact is module state: this test's refusals stay its own (the real lane never
    # reaches its spawn inside a window: the gate is closed there and it returns before spawning).
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", [""])
    dead = add_slot(1, lane=LANE_CONVERSATION)
    dead.proc.alive, dead.reaping = False, True
    monkeypatch.setattr(worker_chat_lane, "marker_window_closed", lambda: True)
    with pytest.raises(lifecycle.SpawnRefusedInMarkerWindow):
        lifecycle.respawn_worker(1)
    assert not pool.created and workers.WORKERS[1] is dead, "no child in the window; the dead slot keeps its place"
    # The reaper's door: the refusal opens its episode and hands the slot back to the next tick.
    monkeypatch.setattr(task_reaper, "_RESPAWN_REFUSALS", {})
    task_reaper._respawn_after_reap(q, workers, 1, expected_worker=dead)
    assert not pool.created and dead.reaping is False
    assert "SpawnRefusedInMarkerWindow" in task_reaper._RESPAWN_REFUSALS[1]["cause"]
    # Growth under demand and the pool's last member's replacement respect the same window: no child, the
    # dead member keeps its place (a non-last member is retired, never replaced, window or not).
    lifecycle.reconcile_worker_capacity(lambda: 3)
    assert not pool.created
    workers.WORKERS.pop(0)  # the dead carrier is the pool's last member now
    assert lifecycle.retire_or_replace_worker(1, dead) == "" and workers.WORKERS[1] is dead
    # The window ends: the same respawn starts the replacement and the episode closes.
    monkeypatch.setattr(worker_chat_lane, "marker_window_closed", lambda: False)
    task_reaper._respawn_after_reap(q, workers, 1, expected_worker=dead)
    assert len(pool.created) == 1 and workers.WORKERS[1] is not dead and 1 not in task_reaper._RESPAWN_REFUSALS


class _ObservedLifecycle:
    """The real lifecycle serializer, observed: every thread that begins a blocking acquire is named."""

    def __init__(self, inner):
        self._inner, self.waiting = inner, []

    def acquire(self, blocking=True, timeout=-1):
        if blocking:
            self.waiting.append(threading.current_thread().name)
        return self._inner.acquire(blocking, timeout)

    def release(self):
        self._inner.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


class _BoundedQ:
    """Q itself, taken with a bound: a reversed lock order fails this test instead of hanging the run."""

    def __init__(self, inner, bound):
        self._inner, self._bound = inner, bound

    def __enter__(self):
        if not self._inner.acquire(timeout=self._bound):
            raise TimeoutError("Q is held by a thread that waits for the lifecycle serializer")
        return self

    def __exit__(self, *exc):
        self._inner.release()


def test_a_stop_never_holds_q_while_a_carrier_spawn_holds_the_lifecycle_serializer(pool, monkeypatch):
    """One global lock order, lifecycle before Q (review C1). A carrier spawn (``replenish_after_claim`` from a
    Host thread, the tick's reconcile) holds the lifecycle serializer through its process start and then takes Q
    to install the slot; a stop on another thread (Restart, shutdown) must wait for the serializer BEFORE it takes
    Q. The spawn is the real ``_spawn_worker_slot``: its process start returns once the stop is blocked on the
    serializer, and its Q is bounded, so the reversed order (Q, then the serializer) fails here, not in a hang."""
    _stop_doubles(monkeypatch)
    add_slot(0, lane=LANE_PROJECT)
    serializer = _ObservedLifecycle(lifecycle._WORKER_LIFECYCLE_LOCK)
    monkeypatch.setattr(lifecycle, "_WORKER_LIFECYCLE_LOCK", serializer)
    monkeypatch.setattr(lifecycle, "_queue_lock", _BoundedQ(queue._queue_lock, 5.0))

    def start_once_the_stop_waits(ctx, wid, *args):
        deadline = time.monotonic() + 10
        while "stopper" not in serializer.waiting and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "stopper" in serializer.waiting, "the stop never reached the lifecycle serializer"
        return Process(7000 + wid)

    monkeypatch.setattr("supervisor.worker_process.spawn_worker_process", start_once_the_stop_waits)
    holding, spawned, stopped, failures = threading.Event(), [], [], []

    def spawner():
        try:
            with serializer:
                holding.set()
                spawned.append(lifecycle._spawn_worker_slot(5, lane=LANE_CONVERSATION))
        except BaseException as exc:
            failures.append(exc)
            holding.set()

    def stopper():
        try:
            stopped.append(workers.kill_workers(lanes=frozenset({LANE_PROJECT})))
        except BaseException as exc:
            failures.append(exc)

    spawn_thread = _THREAD(target=spawner, name="spawner", daemon=True)
    stop_thread = _THREAD(target=stopper, name="stopper", daemon=True)
    spawn_thread.start()
    assert holding.wait(10)
    stop_thread.start()
    spawn_thread.join(30)
    stop_thread.join(30)
    assert not spawn_thread.is_alive() and not stop_thread.is_alive()
    assert failures == [] and spawned == [True] and stopped == [True]
    assert workers.WORKERS[5].lane == LANE_CONVERSATION and 0 not in workers.WORKERS  # the project stop spared the carrier
