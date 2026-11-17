"""A consciousness wake-up is an ordinary Main turn (Background Consciousness redesign, P1).

The lane's door admits a wake — its RUNNING row under the conversation lane is in the census
before the receipt returns — and a lane carrier executes it: the carrier's first step observes,
renders and binds the wake's input from the facts the clock admitted with (``_wake_prepare``),
and the wake's end reaches the clock as a pooled terminal; the door calls no finish callback
(owner decision 1A). A wake carries its facts as plain task metadata — the origin label, the
ledger category, the model role, the withheld tools — and ``set_next_wakeup`` clamps into the
configured bounds and persists the choice on the runtime state. The label's journey through
frames, rows and replay is pinned in ``test_consciousness_initiator_label.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
from supervisor import conversation_lane, workers
from supervisor.active_activity import get_direct_activity_registry
from supervisor.consciousness_carriers import finish_pooled_wake, pooled_terminal_ok, queued_wakes
from tests.conversation_carrier_support import carry

TS = "2026-09-16T12:00:00Z"
WAKE_META = {
    "initiator": "consciousness", "usage_category": "consciousness", "wake_reason": "heartbeat",
    "consciousness_autonomy": "act", "model_role": "consciousness",
    "wake_cause": "own_clock", "consciousness_cause": "own", "work_kind": "consciousness",
}
PLACEHOLDER = "[Wake-up · heartbeat]"  # what the clock admits; the carrier renders the wake from its facts


def _lane(monkeypatch, tmp_path, *, event_q=None, sent=None):
    """The lane's doors over a fresh pool table with two READY idle conversation carriers: the first (returned)
    carries the wake, the second is the owner's reserved spare a wake never takes (owner decision 4A).

    The server runs no cognition here: an actor exists only when a test hands one to ``carry``,
    which runs the dispatched task through the real worker body on this thread, as the carrier would.
    """
    from supervisor import message_bus, queue as supervisor_queue, state
    from tests.conversation_carrier_support import install_lane_spare

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    shared = event_q if event_q is not None else queue.Queue()
    running, pending = {}, []
    for module in (workers, supervisor_queue):
        monkeypatch.setattr(module, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(module, "RUNNING", running)
        monkeypatch.setattr(module, "PENDING", pending)
    monkeypatch.setattr(supervisor_queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state" / "queue_snapshot.json")
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "get_event_q", lambda: shared)
    monkeypatch.setattr(workers, "send_with_budget",
                        lambda *a, **kw: (sent if sent is not None else []).append((a, kw)))
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: pytest.fail("the server constructed an agent"))
    # A wake is admissible only when the installed owner's consciousness toggle
    # is positively known to be on. An absent state is deliberately unknown.
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    bridge = SimpleNamespace(send_chat_action=lambda *a, **kw: None, broadcast=lambda msg: None)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    workers.open_repo_writer_admission()
    carrier = install_lane_spare(workers, 90)  # the idle carrier the wake takes
    install_lane_spare(workers, 91)  # the owner's reserved spare: a wake never takes the last idle carrier (4A)
    return carrier


def _wake_facts(tmp_path, **facts):
    """The clock's ``_wake_prepare``: the facts only its process holds, as ``consciousness._launch`` binds them."""
    return {"boundary": None, "since": 0.0, "now": 1_900_000_000.0, "reason": "heartbeat", "level": "act",
            "repo_dir": str(tmp_path / "repo"), "last_wake_at": 0.0, "last_wake_task_id": "", "disabled_tools": [],
            "window": {"settled_usd": 0.0, "unknown_unmetered": 0, "limit_usd": 5.0, "spend_by_cause": None},
            "counted_roots": [], "max_tasks": 2, "interval": 600, "appointment_at": 0.0, **facts}


def _bind_facts(tmp_path, **facts):
    def bind(task):
        task["_wake_prepare"] = _wake_facts(tmp_path, **facts)
    return bind


def _drain(bus):
    frames = []
    while True:
        try:
            frames.append(bus.get_nowait())
        except queue.Empty:
            return frames


def _terminal(root, task, status="completed", **fields):
    """What the real agent persists for the turn's end: the row with its metadata and execution owner."""
    from ouroboros.task_results import write_task_result

    return write_task_result(root, task["id"], status, chat_id=task["chat_id"], _is_direct_chat=True,
                             metadata=task["metadata"], execution_owner=task["_execution_owner"], **fields)


def _settle(task_id, clock):
    """The task_done handler's own-clock delivery: ``finish_pooled_wake`` over the durable terminal, then the
    RUNNING row's release (``events_task_done._finish_task_done_dispatch`` runs them in that order)."""
    from ouroboros.task_results import load_task_result

    meta = workers.RUNNING.get(task_id) or {}
    row = load_task_result(workers.DRIVE_ROOT, task_id) or {}
    finish_pooled_wake(SimpleNamespace(consciousness=clock), task_id, row, dict(meta.get("task") or {}))
    workers.RUNNING.pop(task_id, None)
    return row


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


# --- the lane: synchronous admission, carried execution ---------------------------


def test_wake_owns_its_running_row_before_the_receipt_returns_and_its_end_is_a_pooled_terminal(monkeypatch, tmp_path):
    events = queue.Queue()
    spare = _lane(monkeypatch, tmp_path, event_q=events)
    finished: list = []
    seen: list = []

    class Actor:
        def handle_task(self, task):
            seen.append(task)
            _terminal(tmp_path, task, result="a thought")
            return [{"type": "send_message", "task_id": task["id"], "chat_id": task["chat_id"], "text": "a thought"}]

    receipt = workers.handle_wake_direct(1, "wake text", dict(WAKE_META),
                                         on_finished=lambda tid, ok: finished.append((tid, ok)))
    assert receipt["admitted"] is True and receipt["reason"] == ""
    task_id = receipt["task_id"]
    # Admitted synchronously: the census lists the wake from its RUNNING row before any body runs,
    # the lane's spare holds it, and the server registers no actor for it.
    meta = workers.RUNNING[task_id]
    assert meta["lane"] == LANE_CONVERSATION and meta["worker_id"] == spare.wid and spare.busy_task_id == task_id
    owner = meta["task"]["_execution_owner"]
    assert owner["kind"] == "direct" and owner["lane"] == LANE_CONVERSATION and owner["executor"] == EXECUTOR_WORKER
    assert [row["activity_id"] for row in conversation_lane.lane_turn_rows()] == [task_id]
    assert queued_wakes() == [task_id] and get_direct_activity_registry().get(task_id) is None
    assert seen == [] and _drain(events) == []  # nothing of the body ran in this process
    carry(workers, spare, Actor(), out_q=events)
    task, = seen
    assert task["id"] == task_id and task["type"] == "task" and task["_is_direct_chat"] is True
    assert task["text"] == "wake text" and task["chat_id"] == 1
    assert task["root_task_id"] == task_id  # its own root; nothing above it
    for absent in ("_presence_turn", "delegation_role", "project_id", "parent_task_id"):
        assert absent not in task
    for key, value in WAKE_META.items():
        assert task["metadata"][key] == value
    assert isinstance(task.get("task_contract"), dict)
    assert finished == [] and spare.busy_task_id is None  # the door calls nothing back: the end is the pooled terminal
    # The final rides the turn's origin label and lane fact by value.
    final = next(frame for frame in _drain(events) if frame["type"] == "send_message")
    assert final["initiator"] == "consciousness" and final["_is_direct_chat"] is True
    assert final["chat_id"] == 1 and final["worker_id"] == spare.wid
    # The end reaches the clock as the pooled terminal the task_done handler delivers.
    settled: list = []
    clock = SimpleNamespace(_wake_finished=lambda tid, ok, **kw: settled.append((tid, ok, kw.get("pooled"))))
    row = _settle(task_id, clock)
    assert pooled_terminal_ok(row) is True and settled == [(task_id, True, True)]
    assert task_id not in workers.RUNNING and queued_wakes() == []


def test_the_carrier_binds_the_observation_before_the_turn_runs_and_keeps_the_complete_text(monkeypatch, tmp_path):
    """The carrier's first step: the clock's facts become the observation; the complete rendering is the
    task's text, the exact source is stored under the wake and the compact projection names it; the
    accepted boundary rides the task metadata to the clock. The server observed nothing."""
    from ouroboros import consciousness_wake as wake
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import load_task_result

    spare = _lane(monkeypatch, tmp_path)
    seen: dict = {}

    class Actor:
        def handle_task(self, task):
            seen["metadata"] = dict(task["metadata"])
            seen["text"] = task["text"]
            _terminal(tmp_path, task)
            return []

    (tmp_path / "logs" / "chat.jsonl").write_text(json.dumps({
        "ts": TS, "direction": "in", "chat_id": 1, "source": "web", "text": "owner words"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(wake, "render_wake_message", lambda *a, **kw: f"RENDERED:{kw['events']}")
    bound_at: list = []

    def bind(task):  # the clock's binder: once the wake has its id, before its row and its dispatch
        bound_at.append((task["id"], task["id"] in workers.RUNNING, list(spare.in_q)))
        task["_wake_prepare"] = _wake_facts(tmp_path)

    receipt = workers.handle_wake_direct(1, PLACEHOLDER, dict(WAKE_META), bind_input=bind)
    task_id = receipt["task_id"]
    assert receipt["admitted"] is True and bound_at == [(task_id, False, [])]
    dispatched, = list(spare.in_q)
    assert dispatched["text"] == PLACEHOLDER and dispatched["_wake_prepare"]["since"] == 0.0
    assert "wake_observation" not in dispatched["metadata"]
    carry(workers, spare, Actor())
    bound = seen["metadata"][wake.WAKE_OBSERVATION_KEY]
    assert seen["text"].startswith("RENDERED:") and "1 owner message (counted, not listed: my view holds the words)" in seen["text"]  # the complete observation, never the projection
    assert bound["composition"] == {"owner_message": 1} and bound["window"]["basis"] == "time_bootstrap"
    assert bound["projection_text"].startswith("RENDERED:") and bound["source"]["sha256"] in bound["projection_text"]
    assert bound["accepted_boundary"]["task_id"] == task_id  # the clock reads it from the terminal (consciousness._settle_wake)
    # The source outlives the wake and is readable by the published digest; a later wake
    # uses the runtime_data handle (``consolidator.retain_memory_source``), no new root.
    raw = read_actor_source_bytes(tmp_path, task_id, {**bound["source"], "root": "artifact_store"})
    assert bound["source"]["read"]["arguments"]["root"] == "runtime_data"
    assert (tmp_path / bound["source"]["read"]["arguments"]["path"]).read_bytes() == raw
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert rows[0]["kind"] == "wake_observation" and rows[1]["kind"] == "owner_message"
    assert rows[1]["chat_offset"] == 0 and "owner words" in rows[1]["line"]
    durable = load_task_result(tmp_path, task_id)["metadata"][wake.WAKE_OBSERVATION_KEY]
    assert durable["accepted_boundary"] == json.loads(json.dumps(bound["accepted_boundary"]))


def test_the_bound_source_survives_handoff_and_cleanup_for_a_read_file_consumer(monkeypatch, tmp_path):
    """The metadata the carrier bound rides the parkable record (a budget-pause handoff) and the
    durable running record verbatim; the wake's own read_file reads its pointer during the
    turn, and after the turn ended and its row was released a later turn reads the same exact
    bytes through the durable handle. The task text stays the whole observation."""
    import pathlib

    from ouroboros import consciousness_wake as wake
    from ouroboros.budget_pause import parkable_direct_task
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.tool_context import ToolContext

    repo = pathlib.Path(__file__).resolve().parents[1]
    spare = _lane(monkeypatch, tmp_path)
    seen: dict = {}

    class Actor:
        def handle_task(self, task):
            seen["task"] = task
            # What the real agent persists at start (``_persist_running_record``: metadata verbatim).
            write_task_result(tmp_path, task["id"], "running", _is_direct_chat=True, chat_id=task["chat_id"],
                              metadata=task["metadata"], execution_owner=task["_execution_owner"])
            reader = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id=task["id"], task_metadata=task["metadata"])
            seen["during"] = _read_file(reader, **task["metadata"][wake.WAKE_OBSERVATION_KEY]["source"]["read"]["arguments"])
            seen["parked"] = parkable_direct_task(task)
            write_task_result(tmp_path, task["id"], "completed", result="done")
            return []

    (tmp_path / "logs" / "chat.jsonl").write_text(json.dumps({
        "ts": TS, "direction": "in", "chat_id": 1, "source": "web", "text": "owner words"}) + "\n", encoding="utf-8")
    receipt = workers.handle_wake_direct(1, PLACEHOLDER, dict(WAKE_META), bind_input=_bind_facts(tmp_path))
    assert receipt["admitted"] is True
    task_id = receipt["task_id"]
    carry(workers, spare, Actor())
    settled: list = []
    _settle(task_id, SimpleNamespace(_wake_finished=lambda tid, ok, **kw: settled.append(ok)))
    assert settled == [True] and task_id not in workers.RUNNING and spare.busy_task_id is None  # the turn's row is gone
    bound = seen["task"]["metadata"][wake.WAKE_OBSERVATION_KEY]
    event_line = (tmp_path / bound["source"]["read"]["arguments"]["path"]).read_text(encoding="utf-8").splitlines()[1]
    assert event_line in seen["during"]
    assert seen["parked"]["metadata"][wake.WAKE_OBSERVATION_KEY] == bound  # a parked wake resumes with it
    durable = load_task_result(tmp_path, task_id)["metadata"][wake.WAKE_OBSERVATION_KEY]
    assert durable == json.loads(json.dumps(bound))
    later = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="nextwake", task_metadata={})
    after = _read_file(later, **durable["source"]["read"]["arguments"])
    assert event_line in after and f"lines 1–{durable['source']['lines']} of {durable['source']['lines']}" in after
    assert "1 owner message (counted, not listed: my view holds the words)" in seen["task"]["text"]  # the complete observation stayed the input
    later_observation = wake.observe_wake(tmp_path, boundary=bound["accepted_boundary"], since=1_900_000_000.0,
                                          now=1_900_000_600.0)
    assert later_observation.window['transitions_basis'] == 'accepted_inventory'
    assert not later_observation.gaps
    assert not any(kind == 'owner_message' for kind, _offset, _line in later_observation.events)
    assert "owner_message" not in later_observation.counts()  # accepted input is not replayed


@pytest.mark.parametrize("fault", ["binder_raises", "preparation_fails"])
def test_a_failed_binding_or_preparation_keeps_the_admitted_text_as_the_only_input(monkeypatch, tmp_path, fault):
    from ouroboros import consciousness_wake as wake

    spare = _lane(monkeypatch, tmp_path)
    seen: dict = {}

    class Actor:
        def handle_task(self, task):
            seen.update(task)
            _terminal(tmp_path, task)
            return []

    if fault == "binder_raises":  # the clock's binder fails: the wake travels without its facts
        def bind(_task):
            raise OSError("source store unavailable")
    else:  # the carrier's first step fails: the admitted text stays the wake's input
        bind = _bind_facts(tmp_path)
        monkeypatch.setattr(wake, "observe_wake",
                            lambda *a, **kw: (_ for _ in ()).throw(OSError("chat chain unavailable")))
    receipt = workers.handle_wake_direct(1, "complete text", dict(WAKE_META), bind_input=bind)
    assert receipt["admitted"] is True
    carry(workers, spare, Actor())
    assert seen["text"] == "complete text" and "wake_observation" not in seen["metadata"]
    assert "_wake_prepare" not in seen


def test_wake_refusals_are_typed_and_start_nothing(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module
    from supervisor import state

    sent: list = []
    spare = _lane(monkeypatch, tmp_path, sent=sent)
    monkeypatch.setattr(agent_module, "make_agent", lambda **kw: (_ for _ in ()).throw(AssertionError("no actor")))
    monkeypatch.setattr(state, "load_state", lambda: {})
    assert workers.handle_wake_direct(1, "wake", dict(WAKE_META))["reason"] == "consciousness_disabled_or_unknown"
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 0)
    assert workers.handle_wake_direct(1, "wake", dict(WAKE_META)) == {
        "admitted": False, "task_id": "", "reason": "budget_exhausted"}

    def unavailable(*a, **kw):
        raise RuntimeError("ledger down")

    monkeypatch.setattr(state, "budget_remaining", unavailable)
    assert workers.handle_wake_direct(1, "wake", dict(WAKE_META))["reason"] == "cost_accounting_unavailable"
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    workers.close_repo_writer_admission("test-update")
    try:
        assert workers.handle_wake_direct(1, "wake", dict(WAKE_META))["reason"] == "repo_writer_gate_closed"
    finally:
        workers.open_repo_writer_admission()
    assert get_direct_activity_registry().snapshot() == []
    assert workers.RUNNING == {} and list(spare.in_q) == [] and spare.busy_task_id is None  # no row, no dispatch
    # A closed gate refuses the wake QUIETLY: the owner's "🔒" lock notice is for the owner's turn.
    assert not [call for call in sent if "🔒" in str(call[0][1])]
    # A refused wake writes no budget/cost notice of its own into the chat.
    assert not [call for call in sent if "Budget" in str(call[0][1]) or "accounting" in str(call[0][1])]


def test_a_wake_on_a_subscription_route_is_admitted_at_the_reached_global_limit(monkeypatch, tmp_path):
    """A reached global limit refuses paid admission only; the refusal above is a paid Main's."""
    from supervisor import state

    _lane(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_MODEL_CONSCIOUSNESS", "claudexor::codex=gpt-6-astra")
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 0)
    receipt = workers.handle_wake_direct(1, "wake", dict(WAKE_META))
    assert receipt["admitted"] is True and receipt["task_id"]


def test_wake_runner_failure_is_the_turns_own_terminal_and_reaches_the_clock_as_ok_false(monkeypatch, tmp_path):
    """An exception from the body is the task's own terminal (``worker_process._record_task_exception``),
    written through the agent's env; the clock reads it as a runner failure through the pooled terminal."""
    from ouroboros.task_results import load_task_result

    sent: list = []
    spare = _lane(monkeypatch, tmp_path, sent=sent)
    finished: list = []
    env = SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path,
                          drive_path=lambda *parts: tmp_path.joinpath(*parts))

    class Actor:
        def __init__(self):
            self.env = env

        def handle_task(self, task):
            raise RuntimeError("model unreachable")

    receipt = workers.handle_wake_direct(1, "wake", dict(WAKE_META),
                                         on_finished=lambda tid, ok: finished.append((tid, ok)))
    assert receipt["admitted"] is True
    task_id = receipt["task_id"]
    with pytest.raises(RuntimeError, match="model unreachable"):
        carry(workers, spare, Actor())
    assert finished == [] and spare.busy_task_id is None
    row = load_task_result(tmp_path, task_id)
    assert row["status"] == "failed" and row["reason_code"] == "task_exception"
    assert "model unreachable" in row["result"]
    assert pooled_terminal_ok(row) is False  # the clock reads the carrier's end as a runner failure
    settled: list = []
    _settle(task_id, SimpleNamespace(_wake_finished=lambda tid, ok, **kw: settled.append((tid, ok))))
    assert settled == [(task_id, False)] and task_id not in workers.RUNNING
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["type"] == "task_error" and row["task_id"] == task_id for row in rows)


def test_owner_turn_is_admitted_by_the_door_and_executed_by_the_carrier(monkeypatch, tmp_path):
    from ouroboros import project_naming

    events = queue.Queue()
    spare = _lane(monkeypatch, tmp_path, event_q=events)
    monkeypatch.setattr(project_naming, "spawn_turn_namer", lambda *a, **kw: None)

    class Actor:
        def handle_task(self, task):
            self.task = task
            return [{"type": "send_message", "task_id": task["id"], "chat_id": task["chat_id"], "text": "hi"}]

    actor = Actor()
    workers.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "c-1"})
    assert not hasattr(actor, "task")  # the door ran no body
    task_id = spare.busy_task_id
    meta = workers.RUNNING[task_id]
    assert meta["task"]["text"] == "hello" and meta["client_message_id"] == "c-1" and meta["lane"] == LANE_CONVERSATION
    assert get_direct_activity_registry().snapshot() == []
    carry(workers, spare, actor, out_q=events)
    assert actor.task["text"] == "hello" and "initiator" not in actor.task.get("metadata", {})
    final = next(frame for frame in _drain(events) if frame["type"] == "send_message")
    assert "initiator" not in final and final["_is_direct_chat"] is True


def test_turn_event_queue_stamps_the_initiator_on_the_turn_events_only():
    from supervisor.log_addressing import TurnEventQueue

    inner = queue.Queue()
    turn = TurnEventQueue(inner, "t1", 7, initiator="consciousness")
    own = {"type": "tool_call_started", "task_id": "t1"}
    turn.put(own)
    assert inner.get_nowait() is own
    assert own["initiator"] == "consciousness" and own["_is_direct_chat"] is True and own["chat_id"] == 7
    other = {"type": "tool_call_started", "task_id": "t2"}
    turn.put_nowait(other)
    assert "initiator" not in other
    wrapped = {"type": "log_event", "data": {"type": "task_heartbeat", "task_id": "t1", "initiator": "kept"}}
    turn.put(wrapped)
    assert wrapped["data"]["initiator"] == "kept"
    plain = TurnEventQueue(inner, "t1", 7)
    silent = {"type": "send_message", "task_id": "t1"}
    plain.put(silent)
    assert "initiator" not in silent and silent["_is_direct_chat"] is True


# --- the wake's metadata: withheld tools, model role, ledger category ---------


def test_metadata_disabled_tools_reach_the_registry_guard_through_the_contract(tmp_path):
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.tools.registry_guards import _disabled_tools
    from ouroboros.tools.tool_context import ToolContext

    task = {"id": "w", "type": "task", "chat_id": 1, "text": "wake", "_is_direct_chat": True,
            "metadata": {**WAKE_META, "disabled_tools": ["toggle_evolution", "commit_reviewed"]}}
    attach_task_contract(task)
    assert {"toggle_evolution", "commit_reviewed"} <= set(task["task_contract"]["disabled_tools"])
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path,
                      task_metadata=task["metadata"], task_contract=task["task_contract"])
    assert {"toggle_evolution", "commit_reviewed"} <= set(_disabled_tools(ctx))


def test_model_role_resolves_the_consciousness_slots_and_falls_back_to_main():
    from ouroboros.agent_dispatch import _initial_effort_for, model_role_slot_override
    from ouroboros.model_slots import task_model_binding
    from ouroboros.route_choice import following_route

    meta = {"model_role": "consciousness"}
    assert task_model_binding({"metadata": meta})[0] == "consciousness"
    assert task_model_binding({"metadata": {}})[0] == "main"
    with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_CONSCIOUSNESS": "", "OUROBOROS_EFFORT_MAX": "xhigh",
                                      "OUROBOROS_EFFORT_TASK": "medium", "USE_LOCAL_CONSCIOUSNESS": ""}):
        assert model_role_slot_override(meta) is None  # an empty slot is Main
        # A wake starts at the top of the owner's effort range; an ordinary turn at its recommended level.
        assert _initial_effort_for({"metadata": meta}, "task") == "xhigh"
        assert _initial_effort_for({"metadata": {}}, "task") == "medium"
    # The range's top is never below the recommended level (the tolerant read).
    with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_CONSCIOUSNESS": "", "OUROBOROS_EFFORT_MAX": "",
                                      "OUROBOROS_EFFORT_TASK": "xhigh"}):
        assert _initial_effort_for({"metadata": meta}, "task") == "xhigh"
    # An empty model slot's own local box, checked under a cloud Main, runs Main's model on the local runtime
    # (В25=B); unchecked, the box is the row's resting state ("Uses Main.") and never takes the role off a
    # local Main (owner, 12.11, #3272), for Light as for consciousness.
    with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_CONSCIOUSNESS": "", "USE_LOCAL_CONSCIOUSNESS": "true",
                                      "USE_LOCAL_MAIN": "false", "OUROBOROS_MODEL": "openai/gpt-5.6-sol"}):
        assert model_role_slot_override(meta) == ("openai/gpt-5.6-sol", True)
    with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_CONSCIOUSNESS": "", "USE_LOCAL_CONSCIOUSNESS": "false",
                                      "OUROBOROS_MODEL_LIGHT": "", "USE_LOCAL_LIGHT": "false",
                                      "USE_LOCAL_MAIN": "true", "OUROBOROS_MODEL": "local/model"}):
        assert model_role_slot_override(meta) is None
        assert following_route("consciousness") == ("local/model", True)
        assert model_role_slot_override({"model_role": "light"}) is None
    with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_CONSCIOUSNESS": "", "USE_LOCAL_CONSCIOUSNESS": "true",
                                      "USE_LOCAL_MAIN": "true", "OUROBOROS_MODEL": "local/model"}):
        assert model_role_slot_override(meta) is None  # equal flags: Main, the same prefix
    with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_CONSCIOUSNESS": "openai/gpt-5.6-sol", "USE_LOCAL_CONSCIOUSNESS": "true"}):
        assert model_role_slot_override(meta) == ("openai/gpt-5.6-sol", True)
    with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_CONSCIOUSNESS": "openai/gpt-5.6-sol", "USE_LOCAL_CONSCIOUSNESS": ""}):
        assert model_role_slot_override(meta) == ("openai/gpt-5.6-sol", False)
    assert model_role_slot_override({}) is None
    assert model_role_slot_override({"model_role": "main"}) is None
    assert model_role_slot_override({"model_role": "fallback"}) is None
    assert _initial_effort_for({"reasoning_effort": "xhigh", "metadata": meta}, "task") == "xhigh"


def test_prepare_task_context_binds_the_route_through_route_choice_and_pins_no_wake_slot():
    """The agent binds a task's route at the one ``route_choice`` seam; a wake carries no frozen model
    (``task_model_override`` stays empty) and follows its slot per call instead (owner 1A, route_choice)."""
    import inspect
    from types import SimpleNamespace

    from ouroboros import agent as agent_module
    from ouroboros.route_choice import bind_task_route, open_route

    source = inspect.getsource(agent_module.OuroborosAgent._prepare_task_context)
    assert "route_choice.bind_task_route(ctx, task_metadata" in source
    assert "model_role_slot_override(task_metadata)" not in source
    ctx = SimpleNamespace(task_metadata={"model_role": "consciousness"})
    bind_task_route(ctx, ctx.task_metadata, presence=False)
    assert ctx.task_model_override is None and ctx.task_use_local_override is None
    with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_CONSCIOUSNESS": "openai::consc", "USE_LOCAL_CONSCIOUSNESS": ""}):
        assert open_route(ctx, SimpleNamespace(default_model=lambda: "main/model")) == (
            "openai::consc", False, {"kind": "following", "role": "consciousness"})


def test_handle_task_ledger_category_comes_from_usage_category(monkeypatch, tmp_path):
    from ouroboros import config, model_wait, subagent_runtime, usage_accounting

    captured: list = []

    @contextlib.contextmanager
    def fake_scope(scope):
        captured.append(scope)
        yield

    monkeypatch.setattr(usage_accounting, "usage_scope", fake_scope)
    monkeypatch.setattr(model_wait, "task_model_wait_scope", lambda **kw: contextlib.nullcontext())
    from ouroboros.settings_integrity import task_settings_snapshot

    monkeypatch.setattr(subagent_runtime, "apply_task_start_settings_or_disclose",
                        lambda *a, **k: task_settings_snapshot({}, os.environ))
    monkeypatch.setattr(config, "task_settings_scope", lambda snapshot: contextlib.nullcontext())
    from tests._agent_entry_helpers import task_entry_agent

    agent = task_entry_agent(tmp_path, lambda task: [])
    agent.handle_task({"id": "w1", "type": "task", "chat_id": 1, "text": "wake", "metadata": dict(WAKE_META)})
    agent.handle_task({"id": "o1", "type": "task", "chat_id": 1, "text": "hello"})
    agent.handle_task({"id": "e1", "type": "evolution", "chat_id": 1, "text": "evolve", "metadata": {"initiator": "consciousness"}})
    assert [(scope.task_id, scope.category) for scope in captured] == [
        ("w1", "consciousness"), ("o1", "task"), ("e1", "evolution")]


# --- set_next_wakeup: clamp, persist, honest reply --------------------------------


def test_set_next_wakeup_clamps_persists_and_speaks_honestly(tmp_path, monkeypatch):
    from ouroboros.tools import control, control_runtime
    from supervisor import state

    (tmp_path / "state").mkdir(parents=True)
    (tmp_path / "locks").mkdir(parents=True)
    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    assert control._set_next_wakeup is control_runtime._set_next_wakeup
    ctx = SimpleNamespace(task_id="w1")
    with mock.patch.dict(os.environ, {"OUROBOROS_BG_WAKEUP_MIN": "120", "OUROBOROS_BG_WAKEUP_MAX": "600"}):
        low = control._set_next_wakeup(ctx, 30)
        assert low.startswith("OK: consciousness is off") and "120 s" in low and "clamped into 120-600 s" in low
        assert state.load_state()["consciousness_next_interval_sec"] == 120
        high = control._set_next_wakeup(ctx, 99999)
        assert "600 s" in high and "clamped" in high
        assert state.load_state()["consciousness_next_interval_sec"] == 600
        state.update_state(lambda st: st.__setitem__("bg_consciousness_enabled", True))
        spoken = control._set_next_wakeup(ctx, 300)
        assert spoken.startswith("OK: the wake-up interval is now 300 s;") and "pending keeps its time" in spoken
        assert state.load_state()["consciousness_next_interval_sec"] == 300
        assert "TOOL_ARG_ERROR" in control._set_next_wakeup(ctx, "soon")
        assert state.load_state()["consciousness_next_interval_sec"] == 300


def test_consciousness_status_answers_the_caller_only_with_sourced_persisted_facts(tmp_path, monkeypatch):
    """#1324: status is a read for the caller, never a line in the owner's chat;
    in-memory clock facts are named as not read, not guessed. The newest skip is a
    persisted fact with the allowance window it met (A46). Start and stop keep
    their supervisor path and their owner notice."""
    from ouroboros.tools import control
    from supervisor import events_runtime_controls, state

    (tmp_path / "state").mkdir(parents=True)
    (tmp_path / "locks").mkdir(parents=True)
    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    skip = {"at": "2026-09-26T09:30:00+00:00", "reason": "allowance_exhausted", "next_wake_at": "2026-09-26T10:00:00+00:00",
            "error": "", "allowance": {"settled_usd": 20.4, "unknown_unmetered": 0, "limit_usd": 20.0,
                                       "spend_by_cause": [], "resets_at": "2026-09-26T10:00:00+00:00"}}
    state.update_state(lambda st: st.update({
        "bg_consciousness_enabled": True, "consciousness_next_wake_at": 1790416800.0,
        "consciousness_last_wake_at": 1790413200.0, "consciousness_next_interval_sec": 900,
        "consciousness_last_skip": skip}))
    ctx = SimpleNamespace(task_id="w1", pending_events=[], drive_root=tmp_path, task_metadata={})
    with mock.patch.dict(os.environ, {"OUROBOROS_BG_WAKEUP_MIN": "120", "OUROBOROS_BG_WAKEUP_MAX": "600"}):
        facts = json.loads(control._toggle_consciousness(ctx, "status"))
    assert ctx.pending_events == []  # nothing for the supervisor to publish
    assert facts["source"] == str(tmp_path / "state" / "state.json") and facts["observed_at"]
    assert "source_modified_at" not in facts and "read_gap" not in facts and facts["not_recorded"] == ["stopped_by"]
    assert facts["enabled"] is True and facts["chosen_interval_sec"] == 900
    assert facts["stored_next_wake_at"] == "2026-09-26T10:00:00+00:00"
    assert facts["last_wake_ended_at"] == "2026-09-26T09:00:00+00:00"
    assert facts["configured_bounds_sec"] == {"min": 120, "max": 600, "source": "owner settings, not the state file"}
    assert facts["last_skip"] == skip and facts["notes"][1].startswith("last_skip is the newest skipped wake")
    assert "last_wake_outcome" not in facts and "early-wake" not in facts["notes"][1]
    assert "no event moves it" in facts["notes"][0]

    def unpublished():
        raise AssertionError("status must not be published")

    sent, clock = [], SimpleNamespace(start=lambda source: "enabled", stop=lambda source: "disabled", status_snapshot=unpublished)
    supervisor = SimpleNamespace(consciousness=clock, load_state=lambda: {"owner_chat_id": 5},
                                 send_with_budget=lambda chat, text, **kw: sent.append((chat, text, kw)))
    events_runtime_controls._handle_toggle_consciousness({"action": "status"}, supervisor)
    assert sent == []
    assert control._toggle_consciousness(ctx, "stop") == "OK: consciousness 'stop' requested."
    assert ctx.pending_events[-1]["action"] == "stop"
    events_runtime_controls._handle_toggle_consciousness({"action": "stop"}, supervisor)
    assert sent == [(5, "🧠 disabled", {"role": "system", "system_type": "consciousness_notice"})]


def _tree(root):
    """Every path under ``root`` with its bytes (None for a directory)."""
    return {str(path.relative_to(root)): (None if path.is_dir() else path.read_bytes())
            for path in sorted(root.rglob("*"))}


def _initialized_state(root, stored, *, witness="init-1"):
    """A primary copy of a completed initialization (#1307); ``witness=None`` leaves none."""
    (root / "state").mkdir(parents=True, exist_ok=True)
    if witness:
        (root / "state" / "state.initialized.json").write_text(
            json.dumps({"initialization_id": witness, "phase": "complete"}), encoding="utf-8")
    (root / "state" / "state.json").write_text(json.dumps({"initialization_id": "init-1", **stored}), encoding="utf-8")


def test_consciousness_status_reads_the_callers_canonical_root_and_writes_nothing(tmp_path, monkeypatch):
    """The read goes to the caller's canonical data root (``budget_drive_root``
    over a child's own execution drive), never to the process-global state path,
    and leaves every byte of both roots as it found them: no lock file, no
    defaults, no repair. A field the file lacks is named, not defaulted."""
    from ouroboros.tools import control
    from supervisor import state

    canonical, forked, wrong = (tmp_path / name for name in ("canonical", "forked", "wrong"))
    for root, stored in ((canonical, {"bg_consciousness_enabled": False, "consciousness_next_interval_sec": 1800,
                                      "consciousness_next_wake_at": 0}),
                         (forked, {"bg_consciousness_enabled": True}),
                         (wrong, {"bg_consciousness_enabled": True, "consciousness_next_interval_sec": 60})):
        _initialized_state(root, stored)
    for name, path in (("STATE_PATH", wrong / "state" / "state.json"),
                       ("STATE_LAST_GOOD_PATH", wrong / "state" / "state.last_good.json"),
                       ("STATE_LOCK_PATH", wrong / "locks" / "state.lock")):
        monkeypatch.setattr(state, name, path)
    before = _tree(tmp_path)
    ctx = SimpleNamespace(task_id="child", pending_events=[], drive_root=forked,
                          task_metadata={"budget_drive_root": str(canonical)})
    facts = json.loads(control._toggle_consciousness(ctx, "status"))
    assert _tree(tmp_path) == before and ctx.pending_events == []
    assert facts["source"] == str(canonical / "state" / "state.json")
    assert facts["enabled"] is False and facts["chosen_interval_sec"] == 1800
    assert facts["stored_next_wake_at"] == 0  # recorded zero stays zero, not a guessed time
    assert facts["not_recorded"] == ["stopped_by", "last_wake_ended_at", "last_skip"]
    assert "last_wake_ended_at" not in facts
    # A root task with no budget root reads its own drive, which IS its canonical root.
    facts = json.loads(control._toggle_consciousness(SimpleNamespace(pending_events=[], drive_root=forked), "status"))
    assert facts["source"] == str(forked / "state" / "state.json") and facts["enabled"] is True
    assert _tree(tmp_path) == before


def test_consciousness_status_names_a_read_gap_instead_of_defaults(tmp_path):
    """Missing, unreadable and corrupt state each come back as that gap: no field
    is fabricated (not ``enabled: false``, not zeros), the backup the repairing
    loader would restore from is not read, and nothing is created or rewritten."""
    from ouroboros.tools import control

    def status(root):
        before = _tree(root)
        facts = json.loads(control._toggle_consciousness(SimpleNamespace(pending_events=[], drive_root=root), "status"))
        assert _tree(root) == before, "a status read must not write"
        assert facts["not_read"] == ["enabled", "stopped_by", "stored_next_wake_at", "last_wake_ended_at",
                                     "chosen_interval_sec", "last_skip"]
        assert not {"enabled", "stopped_by", "stored_next_wake_at", "last_wake_ended_at", "chosen_interval_sec",
                    "last_skip"} & set(facts)
        assert facts["source"] == str(root / "state" / "state.json") and facts["observed_at"]
        return facts["read_gap"]

    missing = tmp_path / "missing"
    missing.mkdir()
    assert status(missing).startswith("missing:") and not (missing / "state").exists()

    backup = json.dumps({"bg_consciousness_enabled": True, "consciousness_next_interval_sec": 60})
    for name, body in (("corrupt", b'{"bg_consciousness_enabled": tr'), ("undecodable", b"\xff\xfe{"),
                       ("list", b"[true]"), ("null", b"null")):
        root = tmp_path / name
        (root / "state").mkdir(parents=True)
        (root / "state" / "state.json").write_bytes(body)
        (root / "state" / "state.last_good.json").write_text(backup, encoding="utf-8")
        assert status(root).startswith("corrupt:"), name

    unreadable = tmp_path / "unreadable"
    (unreadable / "state" / "state.json").mkdir(parents=True)  # present, but no file can be read there
    assert status(unreadable).startswith("unreadable:")


def test_consciousness_status_keeps_an_unproven_toggle_unknown_and_names_a_kept_panic_flag(tmp_path):
    """The toggle is a #1307 control: the status read reports a stored value only when
    that copy proves it (a completed initialization witness of its identity, no
    recovery-unconfirmed mark), and names a kept Panic flag, which bars every wake."""
    from ouroboros.tools import control

    def status(name, stored, **kw):
        root = tmp_path / name
        _initialized_state(root, {"bg_consciousness_enabled": True, "consciousness_next_interval_sec": 900,
                                  **stored}, **kw)
        if name == "panic":
            (root / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
        before = _tree(root)
        facts = json.loads(control._toggle_consciousness(SimpleNamespace(pending_events=[], drive_root=root), "status"))
        assert _tree(root) == before and facts["chosen_interval_sec"] == 900
        return facts

    proven = status("proven", {})
    assert proven["enabled"] is True and "panic_flag_kept" not in proven
    recovered = status("recovered", {"_recovery": {"source": "backup", "unconfirmed": ["bg_consciousness_enabled"]}})
    assert recovered["enabled"] == {"status": "unknown", "reason": "unconfirmed after a state recovery"}
    assert status("no_witness", {}, witness=None)["enabled"] == {
        "status": "unknown", "reason": "initialization_witness_missing"}
    assert status("foreign", {}, witness="init-2")["enabled"]["reason"] == "initialization_identity_mismatch"
    panic = status("panic", {})
    assert panic["enabled"] is True and "no wake starts" in panic["panic_flag_kept"]


def test_consciousness_status_names_who_stopped_it_and_an_unrecorded_stop_stays_unknown(tmp_path):
    """The mind sees the hold (``bg_consciousness_stop_source``): who stopped it, as stored; an off with none recorded
    (a never-started install, a stop from before the key) is unknown until the owner starts consciousness once; a
    source the copy cannot prove is unknown, like the toggle beside it."""
    from ouroboros.tools import control

    def status(name, stored):
        root = tmp_path / name
        _initialized_state(root, {"bg_consciousness_enabled": False, **stored})
        return json.loads(control._toggle_consciousness(SimpleNamespace(pending_events=[], drive_root=root), "status"))

    for source in ("owner", "panic", "agent_tool"):
        assert status(source, {"bg_consciousness_stop_source": source})["stopped_by"] == source
    never = status("never", {})
    assert never["enabled"] is False and "stopped_by" not in never and "stopped_by" in never["not_recorded"]
    recovered = status("recovered", {"bg_consciousness_stop_source": "agent_tool", "_recovery": {
        "source": "backup", "unconfirmed": ["bg_consciousness_stop_source"]}})
    assert recovered["stopped_by"] == {"status": "unknown", "reason": "unconfirmed after a state recovery"}


def test_wake_affordances_describe_the_real_alarm_and_the_status_audience():
    from ouroboros.tools.control import get_tools

    schemas = {entry.name: entry.schema for entry in get_tools()}
    wake = schemas["set_next_wakeup"]["description"]
    for phrase in ("after a wake-up ends", "OUROBOROS_BG_WAKEUP_MIN/MAX", "already pending keeps its time",
                   "stored for later", "failed wake-up doubles the interval (up to MAX)", "retries after MIN",
                   "exhausted allowance waits for its reset or a changed daily limit",
                   "sooner than MIN after the last wake-up, boot or skip",
                   "Events never bring a wake-up forward"):
        assert phrase in wake
    # #1603 (owner 12 Oct): only the scheduled time; no event pulls the alarm earlier
    assert "brings the next wake-up forward" not in wake and "pending event" not in wake
    assert len(wake) < 600, "an operational paragraph, not the alarm's whole algorithm"
    assert "answered to you only" in schemas["toggle_consciousness"]["description"]
    assert "a start does not lift a stop the owner or Panic placed" in schemas["toggle_consciousness"]["description"]


def test_an_admission_that_raises_leaves_no_running_row_or_registered_turn(monkeypatch, tmp_path):
    """A row nobody carries would read as a live wake forever (opus round 3): a raised admission writes none."""
    spare = _lane(monkeypatch, tmp_path)
    monkeypatch.setattr(conversation_lane, "admit_turn",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("carrier table unavailable")))
    receipt = workers.handle_wake_direct(1, "wake", dict(WAKE_META))
    assert receipt == {"admitted": False, "task_id": "", "reason": "admission_failed"}
    assert workers.RUNNING == {} and list(spare.in_q) == [] and spare.busy_task_id is None
    assert get_direct_activity_registry().snapshot() == [] and queued_wakes() == []
