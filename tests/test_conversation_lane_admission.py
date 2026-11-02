"""Owner turns are admitted to the conversation lane and executed by a carrier, never in the server (A2).

E1: the admission constructs no agent and runs no ``handle_task``; the RUNNING row carries the
origin (``kind="direct"``) and the carrier facts (``lane``, ``executor``). E7: a "hello" during a
project crash storm is admitted to the lane, never into PENDING. The carrier's first step lays the
owner's input onto the admitted seed, and the census lists the turn from its RUNNING row.
"""

import queue as stdqueue
import time
from types import SimpleNamespace

import pytest

from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION, LANE_PROJECT
from supervisor import conversation_lane, queue, worker_chat_lane, worker_pool_lifecycle as lifecycle, workers
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial

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
    from supervisor.direct_roots import collect_direct_roots
    assert collect_direct_roots() == {"roots": [], "incomplete": False}


def test_hello_during_a_project_crash_storm_is_admitted_to_the_lane_never_pending(lane, monkeypatch):
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "worker_crash_storm")
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
    receipt = worker_chat_lane.handle_wake_direct(
        1, "[Wake-up]", {"initiator": "consciousness", "wake_cause": "own_clock", "work_kind": "consciousness"},
        bind_input=lambda task: bound.append(task["id"]))
    assert receipt["admitted"] and bound == [receipt["task_id"]]
    meta = workers.RUNNING[receipt["task_id"]]
    assert meta["lane"] == LANE_CONVERSATION and meta["worker_id"] == 5
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


def test_carrier_turn_receipt_measures_the_owner_wait_from_the_admission(tmp_path):
    out_q = stdqueue.Queue()
    admitted_at = time.time() - 5.0
    task = {"id": "t9", "chat_id": 1, "text": "hello", "_is_direct_chat": True,
            "origin_message_ref": {"ts": "2026-11-02T00:00:00+00:00", "client_message_id": "cm"},
            "metadata": {"client_message_id": "cm"},
            "_conversation_prepare": {"admitted_at": admitted_at, "reply_expected": True, "client_message_id": "cm",
                                      "census": {"direct_turns_at_entry": 2}}}
    turn = worker_chat_lane.CarrierTurn(task, tmp_path, out_q)
    receipt = turn.execution.receipt()
    assert receipt["handler_wait_sec"] >= 4.9 and receipt["census"] == {"direct_turns_at_entry": 2}
    assert receipt["phases"]["started"] >= 0.0 and turn.execution.phase == "started"
    turn.observe("log_event", {"type": "context_building_started", "task_id": "t9"})
    assert out_q.get_nowait() == {"type": "direct_turn_phase", "task_id": "t9", "task_attempt": 1, "phase": "context_building"}
