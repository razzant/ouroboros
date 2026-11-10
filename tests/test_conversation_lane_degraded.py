"""The degraded executor is chosen only on a physical fact, runs the same worker body on a server
thread, and is disclosed (owner decision 2A; E7).

A refused carrier spawn or a repo-writer marker window runs the admitted turn inline with the
owner's notice, a journal row and a vitals count; a merely busy lane leaves the row awaiting its
carrier. A live call is the other inline executor (``executor=inline_call``), named in its row.
"""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.task_carrier import EXECUTOR_INLINE_CALL, EXECUTOR_INLINE_DEGRADED, LANE_CONVERSATION
from supervisor import conversation_lane, inline_carrier, worker_chat_lane, workers
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready

_REAL_THREAD = threading.Thread  # captured at collection: the pool fixture stubs the lifecycle's threads
_REAL_LANE_RECONCILE = conversation_lane.reconcile_conversation_lane  # the shared pool fixture stubs it


class Actor:
    def __init__(self):
        self.ran, self.owner_wait_callback, self._event_queue = [], None, None
        self.env = SimpleNamespace(drive_root=None, drive_path=lambda rel: None)

    def handle_task(self, task):
        self.ran.append(task)
        return [{"type": "task_done", "task_id": task["id"], "chat_id": task["chat_id"], "status": "completed"}]


@pytest.fixture
def degraded(pool, monkeypatch):
    from supervisor import message_bus

    sent = []
    monkeypatch.setattr(message_bus, "_BRIDGE", SimpleNamespace(send_chat_action=lambda *a, **k: None,
                                                                 broadcast=lambda m: None, push_log=lambda *a, **k: None))
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: sent.append((a, k)))
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", [""])  # a fresh generation records no refusal
    actor = Actor()
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: actor)
    monkeypatch.setattr(inline_carrier, "_INLINE", type(inline_carrier._INLINE)(maxlen=50))
    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=_REAL_THREAD))  # a real carrier thread
    monkeypatch.setattr(inline_carrier, "_DEGRADED_TOTAL", [0])
    (pool.root / "logs").mkdir(parents=True, exist_ok=True)
    pool.actor, pool.sent = actor, sent
    return pool


def _inline_slot():
    return next((w for w in workers.WORKERS.values() if w.executor == EXECUTOR_INLINE_DEGRADED), None)


def _await_inline(slot):
    slot.proc.join(10)
    assert not slot.proc.is_alive()


@pytest.mark.parametrize("fact", ["spawn_refused", "marker_window"])
def test_a_physical_fact_runs_the_turn_inline_and_discloses_it(degraded, monkeypatch, fact):
    if fact == "spawn_refused":
        monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", ["carrier_spawn_refused:OSError"])
    else:
        monkeypatch.setattr(worker_chat_lane, "marker_window_closed", lambda: True)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-d"})
    slot = _inline_slot()
    assert slot is not None and slot.lane == LANE_CONVERSATION and slot.proc.pid is None
    _await_inline(slot)
    assert degraded.actor.ran and degraded.actor.ran[0]["text"] == "hello"
    assert "task_contract" in degraded.actor.ran[0]  # the same first step a carrier runs
    tid = degraded.actor.ran[0]["id"]
    meta = workers.RUNNING[tid]
    assert meta["executor"] == EXECUTOR_INLINE_DEGRADED and meta["task"]["_execution_owner"]["executor"] == EXECUTOR_INLINE_DEGRADED
    assert slot.wid not in workers.WORKERS  # the pseudo-slot ends with its thread
    vitals = inline_carrier.inline_vitals()
    assert vitals["degraded_total"] == 1 and vitals["inline_executions"][-1]["task_id"] == tid
    reason = vitals["inline_executions"][-1]["reason"]
    assert reason == ("carrier_spawn_refused:OSError" if fact == "spawn_refused" else "repo_writer_gate_closed")
    notice = [k for a, k in degraded.sent if k.get("progress_meta", {}).get("task_incident") == "conversation_degraded_executor"]
    assert len(notice) == 1
    rows = [json.loads(line) for line in (degraded.root / "logs" / "supervisor.jsonl").read_text().splitlines()]
    assert [row["executor"] for row in rows if row["type"] == "conversation_inline_executor"] == [EXECUTOR_INLINE_DEGRADED]


def test_a_busy_lane_never_selects_the_degraded_executor(degraded):
    busy = workers.Worker(5, Process(5005), InputQueue(), lane=LANE_CONVERSATION, busy_task_id="other")
    affirm_ready(busy)
    workers.WORKERS[5] = busy
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-b"})
    assert _inline_slot() is None and not degraded.actor.ran
    meta = next(m for m in workers.RUNNING.values() if m.get("lane") == LANE_CONVERSATION and m["task"]["text"] == "hello")
    assert meta["worker_id"] is None  # awaiting its carrier: the next ready one, or the spare being replenished
    assert inline_carrier.inline_vitals()["degraded_total"] == 0


def test_a_booting_spare_is_a_carrier_that_will_come(degraded, monkeypatch):
    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", ["carrier_spawn_refused:OSError"])
    booting = workers.Worker(6, Process(5006), InputQueue(), lane=LANE_CONVERSATION, reaping=True, booting=True)
    workers.WORKERS[6] = booting
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-c"})
    assert _inline_slot() is None and not degraded.actor.ran


def test_under_the_project_fence_the_tick_still_starts_a_waiting_turns_executor(degraded, monkeypatch):
    """S43 defect 1: the lane is independent of the project pool's exhaustion fence. Under the fence the tick
    reconciles the lane all the same: the lane's refused spawn is noticed and the waiting turn's degraded
    executor starts, instead of the owner's hello waiting forever beside a fenced pool."""
    from supervisor import worker_pool_lifecycle

    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "worker_readiness_exhausted")
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 1})
    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", _REAL_LANE_RECONCILE)
    busy = workers.Worker(5, Process(5005), InputQueue(), lane=LANE_CONVERSATION, busy_task_id="other")
    affirm_ready(busy)
    workers.WORKERS[5] = busy

    def refused(*_a, **_k):
        raise OSError("fork refused")

    monkeypatch.setattr(worker_pool_lifecycle, "_spawn_worker_slot", refused)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-fence"})
    meta = next(m for m in workers.RUNNING.values() if m.get("lane") == LANE_CONVERSATION and m["task"]["text"] == "hello")
    assert meta["worker_id"] is None and not degraded.actor.ran  # awaiting: a busy lane is no physical fact
    workers.assign_tasks()  # the tick under the fence: the lane is reconciled, the refusal is now the fact
    deadline = time.time() + 10
    while not degraded.actor.ran and time.time() < deadline:
        time.sleep(0.02)
    assert degraded.actor.ran and degraded.actor.ran[0]["text"] == "hello"
    assert meta["executor"] == EXECUTOR_INLINE_DEGRADED
    assert inline_carrier.inline_vitals()["inline_executions"][-1]["reason"] == "carrier_spawn_refused:OSError"
    for slot in list(workers.WORKERS.values()):  # the executor's thread ends with its pseudo-slot
        if slot.executor == EXECUTOR_INLINE_DEGRADED:
            slot.proc.join(10)


def test_the_degraded_executor_is_a_health_and_runtime_vitals_fact(degraded, monkeypatch):
    """Owner decision 2A: the degraded executor is a fact of /api/health (in memory) and of the Runtime
    vitals the mind reads (from the journal row the disclosure wrote), not only a chat notice."""
    import asyncio
    import json as _json

    from ouroboros.context_runtime_facts import vitals_fact
    from ouroboros.gateway import state as gateway_state

    monkeypatch.setattr(conversation_lane, "_SPAWN_REFUSED", ["carrier_spawn_refused:OSError"])
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-v"})
    slot = _inline_slot()
    _await_inline(slot)
    tid = degraded.actor.ran[0]["id"]
    fact = inline_carrier.inline_executors_fact()
    assert fact["degraded_total"] == 1 and fact["recent"][-1]["task_id"] == tid
    assert fact["recent"][-1]["executor"] == EXECUTOR_INLINE_DEGRADED
    monkeypatch.setattr(gateway_state, "_live_boot", lambda request: {"phase": "serving"})
    health = _json.loads(asyncio.run(gateway_state.api_health(SimpleNamespace())).body)
    assert health["inline_executors"]["degraded_total"] == 1 and health["inline_executors"]["recent"][-1]["task_id"] == tid
    vitals = vitals_fact(degraded.root, observed_at="2026-11-02T12:00:00+00:00")
    assert vitals is not None and any(EXECUTOR_INLINE_DEGRADED in line and tid in line for line in vitals["inline_executors"])


def test_a_degraded_executor_whose_thread_cannot_start_leaves_the_row_admitted_and_no_pseudo_slot(degraded, monkeypatch):
    """A process that can start no thread ran nothing: the pseudo-slot and the activity entry leave, the row
    is admitted and awaiting again (its owner record names the worker executor), and the caller learns it by
    type (``InlineExecutorUnavailable``) instead of a bound row whose executor never ran."""
    from ouroboros.task_carrier import EXECUTOR_WORKER
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.conversation_lane import admit_turn

    class Refusing:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name", "")

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(inline_carrier, "threading", SimpleNamespace(Thread=Refusing))
    from supervisor.conversation_lane import publish_awaiting_rows

    task = {"id": "t-unstarted", "chat_id": 1, "text": "hello", "metadata": {}}
    with workers._queue_lock:
        meta = admit_turn(task, chat_id=1, client_message_id="cm-u")
    assert publish_awaiting_rows()  # the admission's barrier, outside Q, before any executor may hold the row
    with workers._queue_lock:
        with pytest.raises(inline_carrier.InlineExecutorUnavailable):
            inline_carrier.start_inline_turn(meta, executor=EXECUTOR_INLINE_DEGRADED, reason="pool_not_started")
    assert workers.RUNNING["t-unstarted"] is meta and meta["worker_id"] is None and "executor" not in meta
    assert meta["task"]["_execution_owner"]["executor"] == EXECUTOR_WORKER
    assert _inline_slot() is None and get_direct_activity_registry().get("t-unstarted") is None



# ---- a live call's row is the inline executor's from admission to bind (review F1) ------------------------------

def _ready_spare(wid=5):
    spare = workers.Worker(wid, Process(5000 + wid), InputQueue(), lane=LANE_CONVERSATION, busy_task_id=None)
    affirm_ready(spare)
    workers.WORKERS[wid] = spare
    return spare


def test_a_live_call_turn_is_reserved_for_its_inline_executor_from_admission_to_bind(degraded, monkeypatch):
    """The row's barrier runs outside Q, and the supervisor tick reconciles the lane in that window with a READY idle
    spare at hand. The row is the admitting executor's: the tick hands it to no carrier, spawns no carrier for it,
    and the owner's words run once, on the inline executor."""
    from supervisor.inline_carrier import EXECUTOR_INLINE_CALL

    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", _REAL_LANE_RECONCILE)
    spare = _ready_spare()
    real_publish, ticked = conversation_lane.publish_awaiting_rows, []

    def publish_then_tick():
        ok = real_publish()
        if not ticked:  # the tick's own pass runs the barrier again: once is the window
            ticked.append(True)
            conversation_lane.reconcile_conversation_lane()
        return ok

    monkeypatch.setattr(conversation_lane, "publish_awaiting_rows", publish_then_tick)
    meta = inline_carrier.run_inline_turn(1, "hello from the call", None, None, {"client_message_id": "cm-live"},
                                          executor=EXECUTOR_INLINE_CALL)
    assert ticked and meta is not None
    assert spare.busy_task_id is None and spare.in_q.empty(), "the tick's dispatch never handed the row to the spare"
    assert degraded.created == [], "a reserved row awaits no carrier: the tick started none for it"
    assert meta["executor"] == EXECUTOR_INLINE_CALL and meta["worker_id"] is not None and "_inline_bind" not in meta
    deadline = time.time() + 10
    while not degraded.actor.ran and time.time() < deadline:
        time.sleep(0.02)
    assert [t["text"] for t in degraded.actor.ran] == ["hello from the call"]
    assert [row["executor"] for row in inline_carrier.inline_vitals()["inline_executions"]] == [EXECUTOR_INLINE_CALL]
    for slot in list(workers.WORKERS.values()):
        if slot.executor == EXECUTOR_INLINE_CALL:
            slot.proc.join(10)


def test_a_reserved_row_gets_its_barrier_and_no_carrier_until_its_executor_binds_it(degraded):
    """The mechanism under the law above: the admission barrier lands for a reserved row like any other, and the lane's
    two dispatch doors pass it by while a READY spare sits idle; ``start_inline_turn`` clears the mark as it binds."""
    from supervisor.conversation_lane import admit_turn

    spare = _ready_spare()
    task = {"id": "t-reserved", "chat_id": 1, "text": "hello", "metadata": {}}
    with workers._queue_lock:
        meta = admit_turn(task, chat_id=1, client_message_id="cm-r", reserved=True)
    assert meta["_unpublished"] and meta["_inline_bind"]
    assert conversation_lane.publish_awaiting_rows() and "_unpublished" not in meta
    with workers._queue_lock:
        assert conversation_lane.dispatch_awaiting_turns() == 0
    conversation_lane.dispatch_admitted_turn(meta)
    assert meta["worker_id"] is None and spare.busy_task_id is None and spare.in_q.empty()
    with workers._queue_lock:
        slot = inline_carrier.start_inline_turn(meta, executor=EXECUTOR_INLINE_DEGRADED, reason="test")
    assert meta["worker_id"] == slot.wid and "_inline_bind" not in meta
    slot.proc.join(10)


def test_start_inline_turn_refuses_a_row_a_carrier_already_holds(degraded):
    """No second executor: a row bound to a carrier is refused by type, with no pseudo-slot, actor or disclosure."""
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.conversation_lane import admit_turn

    task = {"id": "t-held", "chat_id": 1, "text": "hello", "metadata": {}}
    with workers._queue_lock:
        meta = admit_turn(task, chat_id=1, client_message_id="cm-h")
    assert conversation_lane.publish_awaiting_rows()
    meta["worker_id"] = 5  # a carrier holds the turn
    with workers._queue_lock, pytest.raises(RuntimeError, match="held"):
        inline_carrier.start_inline_turn(meta, executor=EXECUTOR_INLINE_DEGRADED, reason="pool_not_started")
    assert meta["worker_id"] == 5 and "executor" not in meta and _inline_slot() is None
    assert get_direct_activity_registry().get("t-held") is None
    assert inline_carrier.inline_vitals()["inline_executions"] == []


@pytest.mark.parametrize("executor", [EXECUTOR_INLINE_CALL, EXECUTOR_INLINE_DEGRADED])
def test_inline_start_phases_reach_the_admitted_row(degraded, monkeypatch, executor):
    """The real inline thread/worker endpoint keeps admission's generation, like the pool handoff."""
    out = InputQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *a, **k: out)

    def handle(task):
        degraded.actor.ran.append(task)
        for kind in ("context_building_started", "context_building_finished"):
            degraded.actor._event_queue.put({"type": "log_event", "data": {"type": kind, "task_id": task["id"]}})
        return []

    monkeypatch.setattr(degraded.actor, "handle_task", handle)
    task = worker_chat_lane._seed_task(1, "hello", None, None, {"client_message_id": "cm-phase"})
    task["_conversation_prepare"] = {"quiet": False}
    with workers._queue_lock:
        meta = conversation_lane.admit_turn(task, chat_id=1, client_message_id="cm-phase", reserved=True)
    assert conversation_lane.publish_awaiting_rows()
    with workers._queue_lock:
        slot = inline_carrier.start_inline_turn(meta, executor=executor, reason="test")
    _await_inline(slot)
    frames = []
    while not out.empty():
        frame = out.get_nowait()
        if frame.get("type") == "direct_turn_phase":
            frames.append(frame)
    assert [frame["phase"] for frame in frames] == ["context_building", "context_built"]
    ctx = SimpleNamespace(RUNNING=workers.RUNNING)
    conversation_lane.handle_direct_turn_phase({**frames[0], "phase_generation": "previous", "phase": "dispatched"}, ctx)
    assert not meta.get("turn_phase")
    for frame in frames:
        conversation_lane.handle_direct_turn_phase(frame, ctx)
        assert meta.get("turn_phase") == frame["phase"]
    assert all(frame.get("phase_generation") == meta["phase_generation"] for frame in frames)
