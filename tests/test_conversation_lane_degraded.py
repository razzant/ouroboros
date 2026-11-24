"""The degraded executor is chosen only on a physical fact, runs the same worker body on a server
thread, and is disclosed (owner decision 2A; E7).

A refused carrier spawn or a repo-writer marker window runs the admitted turn inline with the
owner's notice, a journal row and a vitals count; a merely busy lane leaves the row awaiting its
carrier. A live call is the other inline executor (``executor=inline_call``), named in its row.
A Presence author's pseudo-slot is negative: the boot's old-workers stop leaves a live one its turn
and row while the pool starts beside it; every other stop takes it.
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


# ---- a live Presence author's negative pseudo-slot: only the boot's old-workers stop spares it (#3149) ----------

@pytest.fixture
def author(degraded, monkeypatch):
    """The pre-queue Host's Presence author: no queue and no armed pool in this process yet, where
    ``dispatch_presence_turn`` admits (a server owning its queue refuses typed before its pool arms:
    ``tests/test_conversation_lane_presence.py``), so the lane's own dispatch runs the body on the degraded
    executor over ``_UnarmedFrames``. The body is held (the seam under test is the stop); released, it sends
    the one ``presence_turn_settled`` frame ``carry_presence_turn`` sends, which the real handler settles.

    Nothing here signals a process or attaches a controller: ``_tear_down_slots`` calls workers' own
    ``kill_worker_tree``, which the pool fixture's lifecycle double does not replace, so it is an inert
    record; the platform's process-tree kills fail the test if anything else reaches them. A spawn forks
    no child: context, process factory and ledger stay stubs."""
    import os

    from ouroboros import platform_layer, presence_carrier
    from ouroboros.presence_runner import PresenceTurnExecutions, PresenceTurnResult
    from supervisor import queue, worker_process

    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", None)  # no pool generation has armed yet
    monkeypatch.setattr(queue, "INITIALIZED", False)  # nor does this process own a queue
    monkeypatch.setattr(queue, "OWNER_PID", os.getpid())
    torn = []
    monkeypatch.setattr(workers, "kill_worker_tree", lambda pid, **kwargs: torn.append(pid))
    for name in ("kill_pid_tree", "request_process_tree_kill"):
        monkeypatch.setattr(platform_layer, name, lambda *a, _name=name, **k: pytest.fail(f"a real {_name}"))
    monkeypatch.setattr(workers, "_audit_delegate_terminal_custody", lambda *a, **k: None)
    monkeypatch.setattr(workers, "mp", SimpleNamespace(get_context=lambda method: SimpleNamespace(Queue=InputQueue)))
    monkeypatch.setattr(worker_process, "spawn_worker_process",
                        lambda ctx, wid, in_q, event_q, repo_dir, drive_root: Process(3000 + wid))
    monkeypatch.setattr(workers, "_record_worker_processes", lambda: None)
    monkeypatch.setattr(workers, "_verify_worker_sha_after_spawn", lambda *a, **k: None)

    entered, release, ran = threading.Event(), threading.Event(), []

    def held_body(agent, task, *, wid, out_q, **kwargs):
        turn_id = str(task["id"])
        ran.append(turn_id)
        entered.set()
        release.wait(10)
        result = PresenceTurnResult(outcome="message", text="Hi", task_id=turn_id)
        out_q.put({"type": "presence_turn_settled", "turn_id": turn_id, "task_id": turn_id, "physical_id": turn_id,
                   "result": presence_carrier._result_payload(result), "error": None,
                   **presence_carrier._frame_identity(task, wid, None)})
        return []

    monkeypatch.setattr(worker_process, "run_worker_task", held_body)
    settled, released = [], []
    real_settle = presence_carrier.handle_presence_turn_settled

    def settle(evt, ctx):  # the real handler, counted: ``_UnarmedFrames`` reads it at each frame
        settled.append(evt["turn_id"])
        real_settle(evt, ctx)

    monkeypatch.setattr(presence_carrier, "handle_presence_turn_settled", settle)
    registry = PresenceTurnExecutions()  # the Host's waiters

    def start(turn_id):
        assert not conversation_lane.armed() and not queue.owns_live_queue(degraded.root)
        task = {"id": turn_id, "chat_id": 1, "text": "hello", "metadata": {}, "_presence_turn": True,
                presence_carrier.PRESENCE_RUN_KEY: {"skill_name": "telegram-bot"}}

        def dispatch(execution):  # admitted as ``dispatch_presence_turn`` admits a Host turn: never reserved
            with workers._queue_lock, workers._repo_writer_gate_lock:
                meta = conversation_lane.admit_turn(task, chat_id=1, client_message_id=f"src-{turn_id}",
                                                    quiet=True, reply_expected=False)
            conversation_lane.dispatch_admitted_turn(meta)

        execution, started = registry.start_or_dispatch(turn_id, identity=f"id-{turn_id}", reserve=lambda: True,
                                                          release=lambda: released.append(turn_id), dispatch=dispatch)
        assert started and entered.wait(10)  # the body holds its turn now
        meta = workers.RUNNING[turn_id]
        assert meta["executor"] == EXECUTOR_INLINE_DEGRADED and "_inline_bind" not in meta
        assert inline_carrier.inline_vitals()["inline_executions"][-1]["reason"] == "pool_not_started"
        return execution, meta, workers.WORKERS[meta["worker_id"]]

    degraded.start, degraded.release, degraded.ran, degraded.registry = start, release, ran, registry
    degraded.torn, degraded.settled, degraded.released = torn, settled, released
    return degraded


def test_boot_sweep_keeps_a_live_presence_author_its_row_and_the_new_pool(author):
    """The boot's old-workers stop (server.py ``kill_workers(preserve_pending=True, startup=True)``) took a
    live Host author's slot and RUNNING row (``lanes=None`` owns every slot), and the spawn's empty-pool
    invariant then refused or lost the new pool. The live author keeps its own turn instead, every other
    slot stays the stop's, the pool starts beside it, and the author's one settlement ends its row."""
    from supervisor import worker_pool_lifecycle

    execution, meta, slot = author.start("t-live-author")
    assert slot.wid < 0 and slot.proc.is_alive()
    author.original.proc.alive = False  # a stopped worker slot; its row "owner" is still RUNNING on it

    assert workers.kill_workers(preserve_pending=True, startup=True)

    assert 0 not in workers.WORKERS and "owner" not in workers.RUNNING  # retired and settled, as at every boot
    assert author.torn == [42]  # its process tree met only the inert record; the author has none
    assert workers.WORKERS[slot.wid] is slot and slot.proc.is_alive()
    assert not slot.reaping and not slot.proc.stop_requested  # neither asked to stop nor reaped
    assert workers.RUNNING["t-live-author"] is meta and meta["worker_id"] == slot.wid
    assert not execution.result.done() and author.settled == [] and author.released == []  # the Host still waits
    state = worker_pool_lifecycle._worker_pool_execution_state()  # a lone Host thread: no pool, carrier or capacity
    assert state["available"] is False and state["disabled_reason"] == "no_workers"
    assert worker_pool_lifecycle.pool_slot_counts()["lane_carrier_total"] == 0

    workers.spawn_workers()  # the ordinary new pool, beside the live thread
    fresh = workers.WORKERS[0]
    assert fresh is not author.original and fresh.lane == LANE_CONVERSATION and fresh.booting
    assert workers.WORKERS[slot.wid] is slot and slot.proc.is_alive()
    assert workers.RUNNING["t-live-author"] is meta
    assert worker_pool_lifecycle._worker_pool_execution_state()["available"] is True

    # One Host result: released, the author sends its one settlement through its own frames; the real handler
    # answers the Host's waiter and releases the row, and the body's exit owns its pseudo-slot. (The waiter is
    # answered before the row is released: join the carrier before reading the rest.)
    author.release.set()
    assert execution.result.result(timeout=10).text == "Hi"
    slot.proc.join(10)
    assert not slot.proc.is_alive() and slot.wid not in workers.WORKERS and author.ran == ["t-live-author"]
    assert author.settled == ["t-live-author"] and author.released == ["t-live-author"]
    assert "t-live-author" not in workers.RUNNING and author.registry.find("t-live-author") is None
    assert workers.WORKERS[0] is fresh and fresh.busy_task_id is None and fresh.booting  # the new pool, untouched
    assert worker_pool_lifecycle._worker_pool_execution_state()["available"] is True


def test_an_ordinary_stop_still_takes_a_live_presence_author(author):
    """Every other stop keeps whole-table ownership (Panic, Restart, a managed update): without ``startup``
    the live author is asked to stop cooperatively, its slot is taken and its row settles as before."""
    from ouroboros.task_results import load_task_result

    _, _, slot = author.start("t-ordinary-stop")

    assert workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                                archive_service_logs=False, reconcile_delegate_custody=False)

    assert slot.proc.stop_requested and slot.wid not in workers.WORKERS  # the cooperative request reached it
    assert author.torn == [42]  # only the worker slot with a process met the (inert) process-tree seam
    assert "t-ordinary-stop" not in workers.RUNNING  # its row settled, not spared
    assert load_task_result(author.root, "t-ordinary-stop")["status"] == "cancelled"
    author.release.set()
    slot.proc.join(10)  # the unconfirmed thread ends at its own boundary; its late frame finds no row to release
    assert "t-ordinary-stop" not in workers.RUNNING and author.ran == ["t-ordinary-stop"]


def test_boot_sweep_spares_a_published_unbound_presence_admission_for_the_new_pool_carrier(author, monkeypatch):
    """#3149: ``admit_turn`` writes the RUNNING row under Q and ``publish_awaiting_rows`` clears ``_unpublished``
    outside Q before ``dispatch_admitted_turn`` re-takes Q to bind a carrier. If ``kill_workers(startup=True)`` and
    ``spawn_workers()`` run in that unlocked window, the unbound Presence row survives (while dead worker rows and
    non-Presence lane rows are settled), waits for the booting carrier instead of starting a second inline thread,
    dispatches once to the opened carrier, and settles once."""
    from ouroboros import presence_carrier
    from ouroboros.presence_runner import PresenceTurnResult
    from supervisor import queue

    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", _REAL_LANE_RECONCILE)
    author.original.proc.alive = False
    author.original.busy_task_id = "stale-worker-presence"
    workers.RUNNING["stale-worker-presence"] = {
        "task": {"id": "stale-worker-presence", "chat_id": 1, "text": "old", "_presence_turn": True,
                 "_execution_owner": {"kind": "presence", "lane": LANE_CONVERSATION}},
        "worker_id": 0, "attempt": 1, "started_at": time.time(), "last_heartbeat_at": time.time(),
        "lane": LANE_CONVERSATION,
    }
    with workers._queue_lock, workers._repo_writer_gate_lock:
        direct_meta = conversation_lane.admit_turn(
            {"id": "stale-direct", "chat_id": 1, "text": "hi", "_is_direct_chat": True, "metadata": {}},
            chat_id=1, client_message_id="cm-stale-direct",
        )
    assert conversation_lane.publish_awaiting_rows() and "_unpublished" not in direct_meta

    turn_id = "t-unbound-pool"
    task = {"id": turn_id, "chat_id": 1, "text": "hello", "metadata": {}, "_presence_turn": True,
            "drive_root": str(author.root), presence_carrier.PRESENCE_RUN_KEY: {"skill_name": "telegram-bot"}}
    real_publish, booted = conversation_lane.publish_awaiting_rows, []

    def publish_then_boot():
        ok = real_publish()
        if not booted:
            booted.append(True)
            meta = workers.RUNNING[turn_id]
            assert "_unpublished" not in meta and meta["worker_id"] is None
            queue.INITIALIZED = True
            assert workers.kill_workers(preserve_pending=True, startup=True)
            assert 0 not in workers.WORKERS and author.torn == [42]
            assert "stale-worker-presence" not in workers.RUNNING and "stale-direct" not in workers.RUNNING
            assert workers.RUNNING[turn_id] is meta and meta["worker_id"] is None
            workers.spawn_workers()
        return ok

    monkeypatch.setattr(conversation_lane, "publish_awaiting_rows", publish_then_boot)

    def dispatch(_execution):
        with workers._queue_lock, workers._repo_writer_gate_lock:
            meta = conversation_lane.admit_turn(task, chat_id=1, client_message_id=f"src-{turn_id}",
                                                quiet=True, reply_expected=False)
        conversation_lane.dispatch_admitted_turn(meta)

    execution, started = author.registry.start_or_dispatch(
        turn_id, identity=f"id-{turn_id}", reserve=lambda: True,
        release=lambda: author.released.append(turn_id), dispatch=dispatch,
    )
    assert started and booted
    meta = workers.RUNNING[turn_id]
    fresh = workers.WORKERS[0]
    assert fresh.lane == LANE_CONVERSATION and fresh.booting
    assert _inline_slot() is None and author.ran == [] and meta["worker_id"] is None  # awaiting the booting carrier

    affirm_ready(fresh)
    fresh.reaping = False
    conversation_lane.reconcile_conversation_lane()
    assert meta["worker_id"] == 0 and fresh.busy_task_id == turn_id
    dispatched = fresh.in_q.get_nowait()
    assert dispatched["id"] == turn_id and fresh.in_q.empty()
    assert workers.WORKERS[1].lane == LANE_CONVERSATION  # the claimed spare was replenished

    result = PresenceTurnResult(outcome="message", text="Hi", task_id=turn_id)
    presence_carrier.handle_presence_turn_settled({
        "type": "presence_turn_settled", "turn_id": turn_id, "task_id": turn_id, "physical_id": turn_id,
        "result": presence_carrier._result_payload(result), "error": None,
        **presence_carrier._frame_identity(dispatched, 0, None), "pid": fresh.proc.pid,
    }, workers)
    assert execution.result.result(timeout=1).text == "Hi"
    assert author.settled == [turn_id] and author.released == [turn_id]
    assert turn_id not in workers.RUNNING and fresh.busy_task_id is None


def test_boot_sweep_before_spawn_lets_published_unbound_presence_bind_inline_and_settle_once(author, monkeypatch):
    """#3149: when ``kill_workers(startup=True)`` runs in the unlocked post-publication window and
    ``dispatch_admitted_turn`` re-takes Q BEFORE ``spawn_workers()``, the spared row binds one negative Host
    pseudo-slot, the new pool starts beside it, and the turn settles once."""
    from ouroboros import presence_carrier

    author.original.proc.alive = False
    turn_id = "t-unbound-inline"
    task = {"id": turn_id, "chat_id": 1, "text": "hello", "metadata": {}, "_presence_turn": True,
            "drive_root": str(author.root), presence_carrier.PRESENCE_RUN_KEY: {"skill_name": "telegram-bot"}}
    real_publish, swept = conversation_lane.publish_awaiting_rows, []

    def publish_then_sweep():
        ok = real_publish()
        if not swept:
            swept.append(True)
            assert workers.kill_workers(preserve_pending=True, startup=True)
            assert 0 not in workers.WORKERS and turn_id in workers.RUNNING
        return ok

    monkeypatch.setattr(conversation_lane, "publish_awaiting_rows", publish_then_sweep)

    def dispatch(_execution):
        with workers._queue_lock, workers._repo_writer_gate_lock:
            meta = conversation_lane.admit_turn(task, chat_id=1, client_message_id=f"src-{turn_id}",
                                                quiet=True, reply_expected=False)
        conversation_lane.dispatch_admitted_turn(meta)

    execution, started = author.registry.start_or_dispatch(
        turn_id, identity=f"id-{turn_id}", reserve=lambda: True,
        release=lambda: author.released.append(turn_id), dispatch=dispatch,
    )
    assert started and swept
    slot = _inline_slot()
    assert slot is not None and slot.wid < 0 and workers.RUNNING[turn_id]["worker_id"] == slot.wid
    workers.spawn_workers()
    assert 0 in workers.WORKERS and workers.WORKERS[slot.wid] is slot
    author.release.set()
    assert execution.result.result(timeout=10).text == "Hi"
    slot.proc.join(10)
    assert slot.wid not in workers.WORKERS and turn_id not in workers.RUNNING
    assert author.ran == [turn_id] and author.settled == [turn_id] and author.released == [turn_id]


def test_ordinary_stop_and_cancel_custody_still_settle_an_unbound_presence_admission(author, monkeypatch):
    """Without ``startup=True``, a whole-table stop and per-task ``cancel_task_custody`` both settle an unbound
    published Presence row cleanly without starting a carrier."""
    from ouroboros import presence_carrier
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import load_task_result
    from supervisor import task_lifecycle

    with workers._queue_lock, workers._repo_writer_gate_lock:
        meta = conversation_lane.admit_turn(
            {"id": "t-unbound-shutdown", "chat_id": 1, "text": "hello", "metadata": {}, "_presence_turn": True,
             "drive_root": str(author.root), presence_carrier.PRESENCE_RUN_KEY: {"skill_name": "telegram-bot"}},
            chat_id=1, client_message_id="src-shutdown", quiet=True, reply_expected=False,
        )
    assert conversation_lane.publish_awaiting_rows() and "_unpublished" not in meta and meta["worker_id"] is None
    assert workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                                archive_service_logs=False, reconcile_delegate_custody=False)
    assert "t-unbound-shutdown" not in workers.RUNNING
    assert load_task_result(author.root, "t-unbound-shutdown")["status"] == "cancelled"

    real_publish, cancelled = conversation_lane.publish_awaiting_rows, []

    def publish_then_cancel():
        ok = real_publish()
        if not cancelled:
            cancelled.append(True)
            request_cancel(author.root, "t-unbound-cancel", reason="owner stop", source="http")
            assert task_lifecycle.cancel_task_custody("t-unbound-cancel", deliver=False) == "cancelled"
        return ok

    monkeypatch.setattr(conversation_lane, "publish_awaiting_rows", publish_then_cancel)
    with workers._queue_lock, workers._repo_writer_gate_lock:
        meta2 = conversation_lane.admit_turn(
            {"id": "t-unbound-cancel", "chat_id": 1, "text": "hello", "metadata": {}, "_presence_turn": True,
             "drive_root": str(author.root), presence_carrier.PRESENCE_RUN_KEY: {"skill_name": "telegram-bot"}},
            chat_id=1, client_message_id="src-cancel", quiet=True, reply_expected=False,
        )
    conversation_lane.dispatch_admitted_turn(meta2)
    assert cancelled and _inline_slot() is None and author.ran == []
    assert "t-unbound-cancel" not in workers.RUNNING
    assert load_task_result(author.root, "t-unbound-cancel")["status"] == "cancelled"
