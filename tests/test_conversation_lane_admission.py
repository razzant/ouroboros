"""Owner turns are admitted to the conversation lane and executed by a carrier, never in the server (A2).

E1: the admission constructs no agent and runs no ``handle_task``; the RUNNING row carries the
origin (``kind="direct"``) and the carrier facts (``lane``, ``executor``). E7: a "hello" during a
project crash storm is admitted to the lane, never into PENDING. The carrier's first step lays the
owner's input onto the admitted seed, and the census lists the turn from its RUNNING row.
"""

import os
import queue as stdqueue
import time
from types import SimpleNamespace

import pytest

from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION, LANE_PROJECT
from supervisor import conversation_lane, worker_chat_lane, worker_pool_lifecycle as lifecycle, workers
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready


_REAL_LANE_RECONCILE = conversation_lane.reconcile_conversation_lane


@pytest.fixture
def lane(pool, monkeypatch):
    """The pool fixture plus a bridge, a lane spare and a server that must never run cognition."""
    from supervisor import message_bus

    sent, actions = [], []
    bridge = SimpleNamespace(send_chat_action=lambda *a, **k: actions.append((a, k)),
                             broadcast=lambda msg: sent.append(msg), push_log=lambda *a, **k: None)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: sent.append({"text": a[1] if len(a) > 1 else "", **k}))
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: pytest.fail("the server constructed an agent"))
    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", _REAL_LANE_RECONCILE)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    spare = workers.Worker(5, Process(5005), InputQueue(), lane=LANE_CONVERSATION)
    affirm_ready(spare)
    workers.WORKERS[5] = spare
    pool.spare, pool.sent, pool.actions = spare, sent, actions
    return pool


def test_owner_turn_is_admitted_to_the_lane_and_runs_in_the_carrier(lane):
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-1"})
    rows = [(tid, meta) for tid, meta in workers.RUNNING.items() if meta.get("lane") == LANE_CONVERSATION]
    assert len(rows) == 1
    tid, meta = rows[0]
    task = meta["task"]
    assert task["_is_direct_chat"] and task["_execution_owner"]["kind"] == "direct"
    assert task["_execution_owner"]["lane"] == LANE_CONVERSATION and task["_execution_owner"]["executor"] == EXECUTOR_WORKER
    assert meta["worker_id"] == 5 and lane.spare.busy_task_id == tid
    dispatched = lane.spare.in_q.get_nowait()
    assert dispatched["id"] == tid and dispatched["_conversation_prepare"]["client_message_id"] == "cm-1"
    assert dispatched["_fresh_identity"] and "_fresh_identity" not in task  # host-only, by value to the carrier
    assert not workers.PENDING
    assert lane.actions and lane.actions[0][0][1] == "typing" and lane.actions[0][1]["activity_id"] == tid
    # Census: the turn is listed from its row, not from a process-local actor.
    assert [row["activity_id"] for row in conversation_lane.lane_turn_rows()] == [tid]
    assert conversation_lane.lane_turn_rows()[0]["phase"] == "working"


def test_hello_under_an_exhausted_project_pool_is_admitted_to_the_lane_never_pending(lane, monkeypatch):
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "worker_readiness_exhausted")
    workers.WORKERS[0].proc.alive = False  # every project worker is dead
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-2"})
    assert not workers.PENDING
    assert [m for m in workers.RUNNING.values() if m.get("lane") == LANE_CONVERSATION and m["worker_id"] == 5]


def test_without_a_ready_carrier_the_turn_awaits_and_the_next_tick_dispatches_it(lane):
    lane.spare.busy_task_id = "other"
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-3"})
    meta = next(m for m in workers.RUNNING.values() if m.get("lane") == LANE_CONVERSATION and m["task"]["text"] == "hello")
    assert meta["worker_id"] is None
    assert conversation_lane.lane_turn_rows()[-1]["phase"] == "preparing"
    lifecycle.reconcile_worker_capacity(lambda: 0)  # one carrier for the waiting turn, plus the spare
    assert len(lane.created) == 2
    fresh = [w for w in workers.WORKERS.values() if w.lane == LANE_CONVERSATION and w.booting]
    assert len(fresh) == 2
    ready = fresh[0]
    affirm_ready(ready)
    ready.reaping = False
    lifecycle.reconcile_worker_capacity(lambda: 0)
    assert meta["worker_id"] == ready.wid and ready.in_q.get_nowait()["id"] == meta["task"]["id"]


def test_wake_is_admitted_quietly_with_its_input_bound_before_dispatch(lane, monkeypatch):
    from supervisor import state
    monkeypatch.setattr(worker_chat_lane, "wake_gate_refusal", lambda: "")
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True, "owner_chat_id": 1})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    monkeypatch.setattr("ouroboros.consciousness.panic_blocks_wake", lambda root: False)
    bound = []
    second = workers.Worker(6, Process(5006), InputQueue(), lane=LANE_CONVERSATION)  # a wake leaves the owner's spare
    affirm_ready(second)
    workers.WORKERS[6] = second
    receipt = worker_chat_lane.handle_wake_direct(
        1, "[Wake-up]", {"initiator": "consciousness", "wake_cause": "own_clock", "work_kind": "consciousness"},
        bind_input=lambda task: bound.append(task["id"]))
    assert receipt["admitted"] and bound == [receipt["task_id"]]
    meta = workers.RUNNING[receipt["task_id"]]
    assert meta["lane"] == LANE_CONVERSATION and meta["worker_id"] == 5 and second.busy_task_id is None
    assert lane.spare.in_q.get_nowait()["_conversation_prepare"]["quiet"] is True


def test_carrier_first_step_lays_the_input_onto_the_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    task = worker_chat_lane._seed_task(1, "", None, None, {"client_message_id": "cm-5", "chat_attachment_uploads": [],
                                                           "chat_attachments": [{"kind": "file"}]})
    task["_conversation_prepare"] = {"quiet": False}
    agent = SimpleNamespace()
    prepared = worker_chat_lane.prepare_conversation_task(agent, task, tmp_path)
    assert prepared is task and "_conversation_prepare" not in task
    assert task["text"] == "(file attached)"  # the placeholder for an empty caption
    assert isinstance(task.get("task_contract"), dict) and task.get("tool_profile")


def test_lane_tasks_get_the_full_endpoint_and_pooled_tasks_the_telemetry_stamp():
    from supervisor.log_addressing import task_event_queue

    inner = stdqueue.Queue()
    fired = []
    lane_task = {"id": "t1", "chat_id": 7, "_is_direct_chat": True,
                 "_execution_owner": {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}}
    endpoint = task_event_queue(inner, lane_task, initiator="consciousness", on_first_work=lambda: fired.append(1))
    endpoint.put({"type": "tool_call_started", "task_id": "t1", "tool": "read_file"})
    frame = inner.get_nowait()
    assert frame["cancelable"] and frame["initiator"] == "consciousness" and frame["_is_direct_chat"] and fired == [1]
    pooled = task_event_queue(inner, {"id": "t2", "chat_id": 7, "_execution_owner": {"kind": "pooled", "lane": LANE_PROJECT}},
                              on_first_work=lambda: fired.append(2))
    pooled.put({"type": "tool_call_started", "task_id": "t2", "tool": "read_file"})
    frame = inner.get_nowait()
    assert "cancelable" not in frame and fired == [1]


def test_carrier_phase_frames_land_on_the_row_and_titles_broadcast(pool):
    ctx = SimpleNamespace(RUNNING=workers.RUNNING, bridge=SimpleNamespace(broadcast=lambda m: broadcasts.append(m)))
    broadcasts = []
    conversation_lane.handle_direct_turn_phase({"task_id": "owner", "task_attempt": 3, "phase": "context_building"}, ctx)
    assert workers.RUNNING["owner"]["turn_phase"] == "context_building"
    conversation_lane.handle_direct_turn_phase({"task_id": "owner", "task_attempt": 3, "phase": "started"}, ctx)
    assert workers.RUNNING["owner"]["turn_phase"] == "context_building"  # facts only move forward
    conversation_lane.handle_direct_turn_phase({"task_id": "owner", "task_attempt": 2, "phase": "dispatched"}, ctx)
    assert workers.RUNNING["owner"]["turn_phase"] == "context_building"  # another attempt's frame
    conversation_lane.handle_task_named({"task_id": "owner", "suggested_name": "Title"}, ctx)
    assert broadcasts == [{"type": "task_named", "task_id": "owner", "suggested_name": "Title"}]


def test_conversation_rows_have_no_project_execution_ceiling():
    from ouroboros.runtime_limits import resolve_task_execution_timeout_sec

    lane_task = {"id": "t", "_execution_owner": {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}}
    assert resolve_task_execution_timeout_sec(lane_task, global_ceiling=3600) is None
    assert resolve_task_execution_timeout_sec({**lane_task, "execution_timeout_sec": 60}, global_ceiling=3600) == 60
    pooled = {"id": "p", "_execution_owner": {"kind": "pooled", "lane": LANE_PROJECT, "executor": EXECUTOR_WORKER}}
    assert resolve_task_execution_timeout_sec(pooled, global_ceiling=3600) == 3600


def test_a_rows_start_phase_belongs_to_the_first_execution_of_its_attempt(pool):
    """#2862: a pooled root's RUNNING row is fresh for each dispatch; a Resume may keep its attempt, so a late frame
    of the execution before it (or of any other) moves nothing, and a frame naming none keeps the attempt rule."""
    ctx = SimpleNamespace(RUNNING=workers.RUNNING)
    workers.RUNNING["pooled"] = {"task": {"id": "pooled"}, "attempt": 1, "worker_id": 4,
                                  "phase_generation": "gen-new"}
    frame = {"task_id": "pooled", "task_attempt": 1, "phase_generation": "gen-new"}
    conversation_lane.handle_direct_turn_phase({**frame, "execution_id": "new", "phase": "context_building"}, ctx)
    conversation_lane.handle_direct_turn_phase({**frame, "phase_generation": "gen-old", "execution_id": "old",
                                                "phase": "dispatched"}, ctx)
    assert workers.RUNNING["pooled"]["turn_phase"] == "context_building"
    conversation_lane.handle_direct_turn_phase({**frame, "execution_id": "new", "phase": "requested"}, ctx)
    conversation_lane.handle_direct_turn_phase({**frame, "execution_id": "new", "phase": "context_built"}, ctx)
    assert workers.RUNNING["pooled"]["turn_phase"] == "requested"  # never back after its own call started
    # The next dispatch's row starts clean: the earlier execution's step does not carry into it.
    workers.RUNNING["pooled"] = {"task": {"id": "pooled"}, "attempt": 1, "worker_id": 5,
                                  "phase_generation": "gen-newer"}
    conversation_lane.handle_direct_turn_phase({**frame, "phase_generation": "gen-newer", "execution_id": "newer",
                                                "phase": "context_building"}, ctx)
    assert workers.RUNNING["pooled"]["turn_phase"] == "context_building"
    workers.RUNNING.pop("pooled")


def test_a_stale_execution_reporting_first_never_owns_the_next_dispatchs_row(pool):
    """#2862: after a Resume that kept the attempt, the earlier execution's late frame may reach the fresh row
    before the new execution reports. Its slot no longer holds the row, so it binds nothing; on the same slot the
    pooled producer's own first frame binds the row at once, before the execution's later frames."""
    import queue as stdqueue

    ctx = SimpleNamespace(RUNNING=workers.RUNNING)
    workers.RUNNING["pooled"] = {"task": {"id": "pooled"}, "attempt": 1, "worker_id": 5,
                                  "phase_generation": "gen-new"}
    frame = {"task_id": "pooled", "task_attempt": 1, "phase_generation": "gen-new"}
    conversation_lane.handle_direct_turn_phase({**frame, "worker_id": 4, "execution_id": "old",
                                                "phase": "dispatched"}, ctx)
    assert "turn_phase" not in workers.RUNNING["pooled"] and "turn_execution" not in workers.RUNNING["pooled"]
    out_q = stdqueue.Queue()
    frames = worker_chat_lane.pooled_turn_phases({"id": "pooled", "chat_id": 7, "_attempt": 1,
                                                  "_phase_generation": "gen-new"}, out_q, worker_id=5)
    first = out_q.get_nowait()
    assert first == {"type": "direct_turn_phase", "task_id": "pooled", "task_attempt": 1, "worker_id": 5,
                     "phase_generation": "gen-new",
                     "execution_id": frames.execution.execution_id, "phase": "started"}
    conversation_lane.handle_direct_turn_phase(first, ctx)
    # A same-slot late frame of another execution, after the new one bound the row, moves nothing.
    conversation_lane.handle_direct_turn_phase({**frame, "phase_generation": "gen-old", "worker_id": 5,
                                                "execution_id": "old",
                                                "phase": "requested"}, ctx)
    assert workers.RUNNING["pooled"]["turn_phase"] == "started"
    frames.observe("log_event", {"type": "context_building_started", "task_id": "pooled"})
    conversation_lane.handle_direct_turn_phase(out_q.get_nowait(), ctx)
    assert workers.RUNNING["pooled"]["turn_phase"] == "context_building"
    assert workers.RUNNING["pooled"]["turn_execution"] == frames.execution.execution_id
    workers.RUNNING.pop("pooled")


def test_direct_resume_dispatch_stamps_a_fresh_phase_generation(pool):
    """A direct-lane Resume fences a late frame even when its worker is reused."""
    task = {"id": "direct-resume", "chat_id": 7, "_attempt": 1,
            "_conversation_prepare": {"quiet": False}}
    meta = {"task": task, "worker_id": None, "attempt": 1,
            "lane": LANE_CONVERSATION, "last_heartbeat_at": time.time(),
            "last_progress_at": time.time(), "phase_generation": "awaiting-generation"}
    slot = workers.Worker(5, Process(5005), InputQueue(), lane=LANE_CONVERSATION)
    affirm_ready(slot)
    workers.WORKERS[5] = slot
    workers.RUNNING[task["id"]] = meta
    ctx = SimpleNamespace(RUNNING=workers.RUNNING)
    try:
        conversation_lane.handle_direct_turn_phase(
            {"task_id": task["id"], "task_attempt": 1, "worker_id": 5,
             "phase_generation": "old-generation", "execution_id": "old",
             "phase": "dispatched"}, ctx)
        assert "turn_phase" not in meta and "turn_execution" not in meta

        assert conversation_lane.hand_to_carrier(meta, slot)
        dispatched = slot.in_q.get_nowait()
        generation = dispatched["_phase_generation"]
        assert generation and meta["phase_generation"] == generation

        # The old execution used the same attempt and slot. It must not claim
        # the fresh row after the new dispatch boundary.
        conversation_lane.handle_direct_turn_phase(
            {"task_id": task["id"], "task_attempt": 1, "worker_id": 5,
             "phase_generation": "old-generation", "execution_id": "old",
             "phase": "dispatched"}, ctx)
        assert "turn_phase" not in meta and "turn_execution" not in meta

        conversation_lane.handle_direct_turn_phase(
            {"task_id": task["id"], "task_attempt": 1, "worker_id": 5,
             "phase_generation": generation, "execution_id": "new",
             "phase": "started"}, ctx)
        assert meta["turn_phase"] == "started"
        assert meta["turn_execution"] == "new"
    finally:
        workers.RUNNING.pop(task["id"], None)
        workers.WORKERS.pop(5, None)


def test_carrier_turn_receipt_measures_the_owner_wait_from_the_admission(tmp_path):
    out_q = stdqueue.Queue()
    admitted_at = time.time() - 5.0
    task = {"id": "t9", "chat_id": 1, "text": "hello", "_is_direct_chat": True,
            "origin_message_ref": {"ts": "2026-11-02T00:00:00+00:00", "client_message_id": "cm"},
            "metadata": {"client_message_id": "cm"},
            "_conversation_prepare": {"admitted_at": admitted_at, "reply_expected": True, "client_message_id": "cm",
                                      "census": {"direct_turns_at_entry": 2}}}
    turn = worker_chat_lane.CarrierTurn(task, tmp_path, out_q, 7)
    receipt = turn.execution.receipt()
    assert receipt["handler_wait_sec"] >= 4.9 and receipt["census"] == {"direct_turns_at_entry": 2}
    assert (receipt["worker_id"], receipt["pid"]) == (7, os.getpid())  # the carrier slot and process it ran in (S11)
    assert receipt["phases"]["started"] >= 0.0 and turn.execution.phase == "started"
    turn.observe("log_event", {"type": "context_building_started", "task_id": "t9"})
    assert out_q.get_nowait() == {"type": "direct_turn_phase", "task_id": "t9", "task_attempt": 1, "worker_id": 7,
                                  "execution_id": turn.execution.execution_id, "phase": "context_building"}


def _wake_doubles(monkeypatch):
    from supervisor import state
    monkeypatch.setattr(worker_chat_lane, "wake_gate_refusal", lambda: "")
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True, "owner_chat_id": 1})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **k: 5.0)
    monkeypatch.setattr("ouroboros.consciousness.panic_blocks_wake", lambda root: False)


WAKE = {"initiator": "consciousness", "wake_cause": "own_clock", "work_kind": "consciousness"}


def test_the_reserved_spare_is_the_owners_and_a_wake_waits_for_its_own_carrier(lane, monkeypatch):
    """Owner decision 4A: a wake never takes the last idle carrier. Its own carrier is started at its
    admission; the owner's next turn reaches the spare without a boot wait, and the claim replenishes
    the spare at once, not at the next tick."""
    _wake_doubles(monkeypatch)
    receipt = worker_chat_lane.handle_wake_direct(1, "[Wake-up]", dict(WAKE), bind_input=lambda task: None)
    wake = workers.RUNNING[receipt["task_id"]]
    assert wake["worker_id"] is None, "the last idle carrier is the owner's spare"
    assert len(lane.created) == 1 and workers.WORKERS[6].lane == LANE_CONVERSATION  # the wake's carrier, now
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-own"})
    owner = next(m for m in workers.RUNNING.values() if m.get("lane") == LANE_CONVERSATION and m["task"]["text"] == "hello")
    assert owner["worker_id"] == 5, "the owner's turn took the spare with no boot behind the wake"
    assert len(lane.created) == 2, "the claim replenished the spare without a tick"
    for wid in (6, 7):  # the readiness owner opens the two booting carriers
        slot = workers.WORKERS[wid]
        slot.booting = slot.reaping = False
        affirm_ready(slot)
    conversation_lane.reconcile_conversation_lane()
    assert wake["worker_id"] == 6 and workers.WORKERS[7].busy_task_id is None  # the wake runs; one spare stays


def test_consecutive_owner_turns_reuse_the_returned_carrier_without_a_boot(lane):
    def row(text):
        return next(m for m in workers.RUNNING.values() if m.get("lane") == LANE_CONVERSATION and m["task"]["text"] == text)

    worker_chat_lane.handle_chat_direct(1, "one", None, task_metadata={"client_message_id": "cm-1"})
    first = row("one")
    assert first["worker_id"] == 5 and len(lane.created) == 1  # the claim replenished at once
    lane.spare.busy_task_id = None  # the task_done handler released the carrier
    workers.RUNNING.pop(first["task"]["id"])
    worker_chat_lane.handle_chat_direct(1, "two", None, task_metadata={"client_message_id": "cm-2"})
    assert row("two")["worker_id"] == 5, "the returned carrier takes the next turn; no spawn wait"
    assert len(lane.created) == 1  # the booting replacement already stands as the spare



def _awaiting_turn(lane, text, client_message_id):
    lane.spare.busy_task_id = "other"  # the spare is busy: the owner's new turn awaits a carrier
    worker_chat_lane.handle_chat_direct(1, text, None, task_metadata={"client_message_id": client_message_id})
    meta = next(m for m in workers.RUNNING.values() if m.get("lane") == LANE_CONVERSATION and m["task"]["text"] == text)
    assert meta["worker_id"] is None
    return meta["task"]["id"]


def _dispatched_once_carriers_are_ready():
    """The lane starts carriers, the readiness watcher marks them READY, the tick reconciles: the ids handed out."""
    lifecycle.reconcile_worker_capacity(lambda: 0)
    for slot in [w for w in workers.WORKERS.values() if w.lane == LANE_CONVERSATION and w.booting]:
        affirm_ready(slot)
        slot.reaping = False
    lifecycle.reconcile_worker_capacity(lambda: 0)
    handed = set()
    for slot in [w for w in workers.WORKERS.values() if w.lane == LANE_CONVERSATION]:
        while True:
            try:
                handed.add(slot.in_q.get_nowait().get("id"))  # quiescence probes carry none
            except stdqueue.Empty:
                break
    return handed - {None}


def test_stop_of_a_turn_awaiting_its_carrier_cancels_it_and_no_carrier_ever_runs_it(lane, monkeypatch):
    """Review K1: a lane turn no carrier holds yet (RUNNING, ``worker_id`` None, no stored row) has no process to
    kill. Custody takes it like a PENDING row: typed ``cancelled`` with its pre-start cost, its ``task_done``
    published, the intent settled; a carrier that becomes ready afterwards is never handed it."""
    from ouroboros.cancel_intents import active_intent, request_cancel
    from ouroboros.task_results import load_task_result
    from supervisor import queue, task_lifecycle

    events = stdqueue.Queue()
    monkeypatch.setattr(workers, "get_event_q", lambda: events)
    task_id = _awaiting_turn(lane, "hello", "cm-k1")
    request_cancel(queue.DRIVE_ROOT, task_id, reason="owner stop", source="owner", requested_by="owner")
    assert task_lifecycle.cancel_task_custody(task_id) == task_lifecycle.CANCEL_CANCELLED
    assert task_id not in workers.RUNNING and active_intent(queue.DRIVE_ROOT, task_id) is None
    assert (load_task_result(queue.DRIVE_ROOT, task_id) or {}).get("status") == "cancelled"
    done = [frame for frame in list(events.queue) if frame.get("type") == "task_done"]
    assert [(frame["task_id"], frame["status"]) for frame in done] == [(task_id, "cancelled")]
    assert task_id not in _dispatched_once_carriers_are_ready()


def test_a_turn_awaiting_its_carrier_with_a_stop_on_record_is_never_dispatched(lane):
    """The dispatch side of K1: until custody settles it, a Stop's intent on record holds the awaiting row back
    from every carrier (and from the degraded executor); the turn beside it without one is dispatched."""
    from ouroboros.cancel_intents import request_cancel
    from supervisor import queue

    held = _awaiting_turn(lane, "stopped hello", "cm-k1-held")
    free = _awaiting_turn(lane, "free hello", "cm-k1-free")
    request_cancel(queue.DRIVE_ROOT, held, reason="owner stop", source="owner", requested_by="owner")
    handed = _dispatched_once_carriers_are_ready()
    assert free in handed and held not in handed
    assert workers.RUNNING[held]["worker_id"] is None and workers.RUNNING[free]["worker_id"] is not None


def test_a_failed_admission_barrier_keeps_the_turn_unstarted_until_its_row_lands(lane, monkeypatch):
    """Astra M1: the row is on disk before any carrier can hold the turn. A failed barrier (False: the documents
    on disk are the previous generation's) keeps the row visibly unstarted and the READY spare untouched, pass
    after pass; once a publication lands (the lane's pass runs the barrier outside Q, then dispatches under it),
    the turn goes to the carrier."""
    from supervisor import queue

    real = queue.publish_queue_snapshot
    monkeypatch.setattr(queue, "publish_queue_snapshot", lambda reason="": False)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-m1"})
    meta = next(m for m in workers.RUNNING.values() if m.get("lane") == LANE_CONVERSATION)
    assert meta["worker_id"] is None and meta["_unpublished"] and lane.spare.busy_task_id is None and lane.spare.in_q.empty()
    conversation_lane.publish_awaiting_rows()  # the barrier still fails: the turn stays held
    with workers._queue_lock:
        conversation_lane.dispatch_awaiting_turns()
    assert meta["worker_id"] is None and meta["_unpublished"] and lane.spare.in_q.empty()
    monkeypatch.setattr(queue, "publish_queue_snapshot", real)
    conversation_lane.reconcile_conversation_lane()  # the tick's pass: the barrier outside Q, the dispatch under it
    assert "_unpublished" not in meta
    assert meta["worker_id"] == lane.spare.wid and lane.spare.in_q.get_nowait()["id"] == meta["task"]["id"]


def test_an_owner_turns_row_is_written_under_q_and_published_outside_it(lane, monkeypatch):
    """Q spans memory only: the admission transaction (Q and the repo-writer gate lock) writes the row and marks
    the queue; the barrier that puts the row on disk runs after it released both, and the carrier takes the
    turn under Q again. Without the two-hold shape the barrier ran, and could wait on the disk, under Q."""
    from supervisor import queue

    seen = []
    real_publish, real_mark = queue.publish_queue_snapshot, queue.persist_queue_snapshot

    def publish(reason=""):
        seen.append(("barrier", reason, queue._queue_lock._is_owned(), workers._repo_writer_gate_lock._is_owned()))
        return real_publish(reason=reason)

    def mark(reason=""):
        seen.append(("mark", reason, queue._queue_lock._is_owned(), workers._repo_writer_gate_lock._is_owned()))
        return real_mark(reason=reason)

    monkeypatch.setattr(queue, "publish_queue_snapshot", publish)
    monkeypatch.setattr(queue, "persist_queue_snapshot", mark)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-q"})
    meta = next(m for m in workers.RUNNING.values() if m.get("lane") == LANE_CONVERSATION)
    assert meta["worker_id"] == lane.spare.wid and "_unpublished" not in meta
    kinds = [(kind, reason, under_q, under_gate) for kind, reason, under_q, under_gate in seen
             if reason in ("conversation_turn_admitted", "conversation_turn_dispatched")]
    assert kinds[0] == ("mark", "conversation_turn_admitted", True, True), "the row is marked inside the transaction"
    assert ("barrier", "conversation_turn_admitted", False, False) in kinds, "the barrier runs outside Q and the gate"
    assert kinds.index(("barrier", "conversation_turn_admitted", False, False)) < kinds.index(("mark", "conversation_turn_dispatched", True, False)), \
        "the carrier takes the turn only after its row is on disk"


def test_a_dead_carriers_retry_whose_barrier_failed_waits_unstarted_for_its_row(lane, monkeypatch):
    """Astra M1's law on the crash path: ``readmit_after_crash`` (crash recovery's lane re-admission, under Q)
    keeps a retry whose barrier failed awaiting with no carrier and the READY spare free; the lane's next pass
    publishes once, then hands it out (the crash path itself: tests/test_worker_crash_retry.py)."""
    from supervisor import queue

    task = {"id": "turn-r", "type": "task", "chat_id": 1, "text": "hello", "_is_direct_chat": True, "_attempt": 2,
            "_execution_owner": {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER},
            "_conversation_prepare": {"admitted_at": time.time(), "quiet": False, "reply_expected": True,
                                      "client_message_id": "cm-r", "census": {}}}
    real = queue.publish_queue_snapshot
    monkeypatch.setattr(queue, "publish_queue_snapshot", lambda reason="": False)
    with queue._queue_lock:
        meta = conversation_lane.readmit_after_crash(task, {"client_message_id": "cm-r", "reply_expected": True})
    assert meta["attempt"] == 2 and meta["worker_id"] is None and meta.get("_unpublished") is True
    conversation_lane.dispatch_admitted_turn(meta)  # the caller's step outside Q: the barrier fails, the row stays held
    assert meta.get("_unpublished") is True and lane.spare.busy_task_id is None and lane.spare.in_q.empty()
    monkeypatch.setattr(queue, "publish_queue_snapshot", real)
    conversation_lane.dispatch_admitted_turn(meta)
    assert meta["worker_id"] == lane.spare.wid and lane.spare.in_q.get_nowait()["id"] == "turn-r"


def test_a_damaged_cancel_intent_projection_holds_only_what_it_cannot_read(lane):
    """Review N5: the lane's Stop hold reads the projection as every behaviour read does (fail-soft with
    disclosure). A quarantined intent row holds its own turn and the turn beside it starts; a projection that
    cannot be read at all, which no Stop can be minted into, holds no turn: it is disclosed, never a silent hold
    of every awaiting turn while READY carriers sit idle."""
    import json

    from supervisor import queue

    path = queue.DRIVE_ROOT / "state" / "cancel_intents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    quarantined = _awaiting_turn(lane, "damaged hello", "cm-n5-row")
    free = _awaiting_turn(lane, "free hello", "cm-n5-free")
    path.write_text(json.dumps({"intents": {quarantined: "not a row"}}), encoding="utf-8")
    handed = _dispatched_once_carriers_are_ready()
    assert free in handed and quarantined not in handed
    path.write_text("{not json", encoding="utf-8")
    assert quarantined in _dispatched_once_carriers_are_ready()  # an unreadable projection holds nothing
    trail = (queue.DRIVE_ROOT / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    assert '"op": "active_intents"' in trail and '"op": "active_intents_row"' in trail  # both disclosed


def test_custody_publishes_only_its_own_row_never_a_resend_admitted_under_the_same_id(lane, monkeypatch):
    """Review N7: custody keeps the running row through kill and persist and pops it at its publication. A resend
    admitted under the same id meanwhile (another carrier's row) is another attempt's, as worker health and the
    reaper read a row: custody settles its own attempt, leaves that row in RUNNING and emits no terminal for it."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import write_task_result
    from supervisor import queue, task_lifecycle, worker_pool_lifecycle
    from tests.conversation_carrier_support import lane_row

    task = {"id": "turn-n7", "type": "task", "chat_id": 1, "text": "hello", "_is_direct_chat": True}
    old = lane_row(workers, dict(task), wid=97)
    write_task_result(queue.DRIVE_ROOT, "turn-n7", "running", chat_id=1, _is_direct_chat=True)
    emitted, resend = [], {}

    def kill(pid, **_kw):  # the old carrier dies; a resend is admitted under the same id before the publication
        old.proc.is_alive = lambda: False
        lane_row(workers, dict(task), wid=98)
        resend["row"] = workers.RUNNING["turn-n7"]

    monkeypatch.setattr(worker_pool_lifecycle, "kill_worker_tree", kill)
    monkeypatch.setattr(queue, "_emit_cancel_task_done", lambda _t, tid, **_k: emitted.append(tid))
    request_cancel(queue.DRIVE_ROOT, "turn-n7", reason="owner stop", source="owner", requested_by="owner")
    task_lifecycle.cancel_task_custody("turn-n7")
    assert workers.RUNNING.get("turn-n7") is resend["row"] and resend["row"]["worker_id"] == 98
    assert emitted == []  # no terminal of the old attempt for the row the resend owns


@pytest.mark.parametrize(("places_state", "live", "resumes"), [
    ("paused", "presence", True),
    ("empty", "direct", False),
    ("unreadable", "presence", True),
])
def test_a_correspondents_presence_turn_never_holds_the_owners_auto_resume(lane, monkeypatch, live, resumes, places_state):
    """Review N8: the Host serves at boot before auto-resume, so a correspondent's Presence turn can be a live lane
    row when the planned restart's resume runs. It is its place's conversation, never the owner's: only an owner
    turn (base's registered direct actors) holds the resume."""
    from tests.conversation_carrier_support import lane_row

    started = []
    from tests.test_places import paused_inventory
    suffix = paused_inventory(workers.DRIVE_ROOT, places_state)
    (workers.DRIVE_ROOT / "state").mkdir(parents=True, exist_ok=True)
    (workers.DRIVE_ROOT / "state" / "pending_restart_verify.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr(worker_chat_lane, "handle_chat_direct", lambda chat_id, text, *_a, **_k: started.append(text))
    monkeypatch.setattr("ouroboros.delegate_recovery.import_legacy_restart", lambda _root: None)
    monkeypatch.setattr("ouroboros.delegate_recovery.pending_owner_restart", lambda _root: None)
    monkeypatch.setattr("ouroboros.installation_stop.panic_input", lambda _root: "")
    turn = {"id": f"{live}-turn", "type": "task", "chat_id": 5 if live == "presence" else 7, "text": "hi",
            "_is_direct_chat": True, "_execution_owner": {"kind": live}}
    lane_row(workers, {**turn, **({"_presence_turn": True} if live == "presence" else {})}, wid=97)
    worker_chat_lane.auto_resume_after_restart()
    assert bool(started) is resumes and all(text.startswith("[auto-resume after restart]") for text in started)
    if resumes:
        original = "[auto-resume after restart] Continue your work. Read scratchpad and identity — they contain context of what you were doing."
        assert started == [original + (f"\n{suffix}" if suffix else "")]
