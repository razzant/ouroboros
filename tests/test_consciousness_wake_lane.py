"""A consciousness wake-up is an ordinary Main turn (Background Consciousness redesign, P1).

The direct lane admits a turn (registered in the census before the receipt
returns) and then executes it; a wake carries its facts as plain task
metadata — the origin label, the ledger category, the model role, the
withheld tools — and ``set_next_wakeup`` clamps into the configured bounds and
persists the choice on the runtime state. The label's journey through frames,
rows and replay is pinned in ``test_consciousness_initiator_label.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
from types import SimpleNamespace
from unittest import mock

from supervisor import workers
from supervisor.active_activity import get_direct_activity_registry

TS = "2026-09-16T12:00:00Z"
WAKE_META = {
    "initiator": "consciousness", "usage_category": "consciousness", "wake_reason": "heartbeat",
    "consciousness_autonomy": "act", "model_role": "consciousness",
}


def _lane(monkeypatch, tmp_path, *, event_q=None, sent=None):
    from supervisor import message_bus, state

    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    shared = event_q if event_q is not None else queue.Queue()
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "get_event_q", lambda: shared)
    monkeypatch.setattr(workers, "send_with_budget",
                        lambda *a, **kw: (sent if sent is not None else []).append((a, kw)))
    # A wake is admissible only when the installed owner's consciousness toggle
    # is positively known to be on. An absent state is deliberately unknown.
    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(send_chat_action=lambda *a, **kw: None))
    workers.open_repo_writer_admission()


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


# --- the lane: synchronous admission, threaded execution -----------------------


def test_wake_is_registered_before_the_receipt_returns_and_reports_its_end(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module

    events = queue.Queue()
    _lane(monkeypatch, tmp_path, event_q=events)
    entered, release = threading.Event(), threading.Event()
    finished: list = []
    actors: list = []

    class Actor:
        def handle_task(self, task):
            self.task = task
            entered.set()
            assert release.wait(10)
            return [{"type": "send_message", "task_id": task["id"], "chat_id": task["chat_id"], "text": "a thought"}]

    def make_agent(**kwargs):
        actors.append(Actor())
        return actors[-1]

    monkeypatch.setattr(agent_module, "make_agent", make_agent)
    receipt = workers.handle_wake_direct(1, "wake text", dict(WAKE_META),
                                         on_finished=lambda tid, ok: finished.append((tid, ok)))
    assert receipt["admitted"] is True and receipt["reason"] == ""
    task_id = receipt["task_id"]
    # Registered synchronously: the census lists the wake before its body runs.
    entry = get_direct_activity_registry().get(task_id)
    assert entry is not None and entry.kind == "direct_chat" and entry.chat_id == 1
    assert entry.actor is actors[0]
    assert entered.wait(10)
    task = actors[0].task
    assert task["id"] == task_id and task["type"] == "task" and task["_is_direct_chat"] is True
    assert task["text"] == "wake text" and task["chat_id"] == 1
    for absent in ("_presence_turn", "delegation_role", "project_id", "parent_task_id", "root_task_id"):
        assert absent not in task
    for key, value in WAKE_META.items():
        assert task["metadata"][key] == value
    assert isinstance(task.get("task_contract"), dict)
    assert finished == []
    release.set()
    assert _wait_for(lambda: bool(finished))
    assert finished == [(task_id, True)]
    assert get_direct_activity_registry().get(task_id) is None
    # The drained final rides the turn's origin label and lane fact by value.
    drained = events.get(timeout=5)
    assert drained["initiator"] == "consciousness" and drained["_is_direct_chat"] is True
    assert drained["chat_id"] == 1


def test_wake_observation_is_bound_after_registration_and_before_the_turn_runs(monkeypatch, tmp_path):
    """The immutable observation source exists, named on the task, before the body starts."""
    from ouroboros import agent as agent_module
    from ouroboros import consciousness_wake as wake
    from ouroboros.artifacts import read_actor_source_bytes

    _lane(monkeypatch, tmp_path)
    seen: dict = {}
    finished: list = []

    class Actor:
        def handle_task(self, task):
            seen["metadata"] = dict(task["metadata"])
            seen["text"] = task["text"]
            return []

    monkeypatch.setattr(agent_module, "make_agent", lambda **kw: Actor())
    (tmp_path / "logs" / "chat.jsonl").write_text(json.dumps({
        "ts": TS, "direction": "in", "chat_id": 1, "source": "web", "text": "owner words"}) + "\n", encoding="utf-8")
    observation = wake.observe_wake(tmp_path, boundary=None, since=0.0, now=1_900_000_000.0)
    order: list = []

    def bind(task):
        order.append(get_direct_activity_registry().get(task["id"]) is not None)  # already registered
        wake.bind_wake_observation(tmp_path, task, observation, lambda events: f"PROJECTED:{events}")

    receipt = workers.handle_wake_direct(1, "the complete wake text", dict(WAKE_META), bind_input=bind,
                                         on_finished=lambda tid, ok: finished.append(ok))
    assert receipt["admitted"] is True and _wait_for(lambda: bool(finished)) and order == [True]
    bound = seen["metadata"][wake.WAKE_OBSERVATION_KEY]
    assert seen["text"] == "the complete wake text"  # the original host input is never replaced
    assert bound["composition"] == {"owner_message": 1} and bound["window"]["basis"] == "time_bootstrap"
    assert bound["projection_text"].startswith("PROJECTED:") and bound["source"]["sha256"] in bound["projection_text"]
    # The source outlives the wake and is readable by the published digest; a later wake
    # uses the runtime_data handle (``consolidator.retain_memory_source``), no new root.
    raw = read_actor_source_bytes(tmp_path, receipt["task_id"], {**bound["source"], "root": "artifact_store"})
    assert raw == observation.source_bytes()
    assert bound["source"]["read"]["arguments"]["root"] == "runtime_data"
    assert (tmp_path / bound["source"]["read"]["arguments"]["path"]).read_bytes() == raw
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert rows[0]["kind"] == "wake_observation" and rows[1]["kind"] == "owner_message"
    assert rows[1]["chat_offset"] == 0 and "owner words" in rows[1]["line"]


def test_the_bound_source_survives_handoff_and_cleanup_for_a_read_file_consumer(monkeypatch, tmp_path):
    """The metadata the lane bound rides the parkable record (a budget-pause handoff) and the
    durable running record verbatim; the wake's own read_file reads its pointer during the
    turn, and after the turn ended and its registry entry was released a later turn reads the
    same exact bytes through the durable handle. The task text stays the whole original input."""
    import pathlib

    from ouroboros import agent as agent_module
    from ouroboros import consciousness_wake as wake
    from ouroboros.budget_pause import parkable_direct_task
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.tool_context import ToolContext

    repo = pathlib.Path(__file__).resolve().parents[1]
    _lane(monkeypatch, tmp_path)
    seen: dict = {}
    finished: list = []

    class Actor:
        def handle_task(self, task):
            seen["task"] = task
            # What the real agent persists at start (``_persist_running_record``: metadata verbatim).
            write_task_result(tmp_path, task["id"], "running", _is_direct_chat=True, chat_id=task["chat_id"],
                              metadata=task["metadata"])
            reader = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id=task["id"], task_metadata=task["metadata"])
            seen["during"] = _read_file(reader, **task["metadata"][wake.WAKE_OBSERVATION_KEY]["source"]["read"]["arguments"])
            seen["parked"] = parkable_direct_task(task)
            write_task_result(tmp_path, task["id"], "completed", result="done")
            return []

    monkeypatch.setattr(agent_module, "make_agent", lambda **kw: Actor())
    (tmp_path / "logs" / "chat.jsonl").write_text(json.dumps({
        "ts": TS, "direction": "in", "chat_id": 1, "source": "web", "text": "owner words"}) + "\n", encoding="utf-8")
    observation = wake.observe_wake(tmp_path, boundary=None, since=0.0, now=1_900_000_000.0)
    receipt = workers.handle_wake_direct(
        1, "the complete wake text", dict(WAKE_META), on_finished=lambda tid, ok: finished.append(ok),
        bind_input=lambda task: seen.update(boundary=wake.bind_wake_observation(tmp_path, task, observation, lambda events: events)))
    assert receipt["admitted"] is True and _wait_for(lambda: bool(finished)) and finished == [True]
    task_id = receipt["task_id"]
    assert get_direct_activity_registry().get(task_id) is None  # the turn's registration is gone
    bound = seen["task"]["metadata"][wake.WAKE_OBSERVATION_KEY]
    event_line = observation.source_bytes().decode("utf-8").splitlines()[1]
    assert event_line in seen["during"]
    assert seen["parked"]["metadata"][wake.WAKE_OBSERVATION_KEY] == bound  # a parked wake resumes with it
    durable = load_task_result(tmp_path, task_id)["metadata"][wake.WAKE_OBSERVATION_KEY]
    assert durable == json.loads(json.dumps(bound))
    later = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id="nextwake", task_metadata={})
    after = _read_file(later, **durable["source"]["read"]["arguments"])
    assert event_line in after and f"lines 1–{durable['source']['lines']} of {durable['source']['lines']}" in after
    assert seen["task"]["text"] == "the complete wake text"
    later_observation = wake.observe_wake(tmp_path, boundary=seen['boundary'], since=1_900_000_000.0,
                                          now=1_900_000_600.0)
    assert later_observation.window['transitions_basis'] == 'accepted_inventory'
    assert not later_observation.gaps
    assert not any(kind == 'owner_message' for kind, _offset, _line in later_observation.events)
    assert 'owner words' not in later_observation.full_text()  # accepted input is not replayed


def test_a_failed_observation_binding_keeps_the_complete_text_as_the_only_input(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module

    _lane(monkeypatch, tmp_path)
    seen: dict = {}
    finished: list = []

    class Actor:
        def handle_task(self, task):
            seen.update(task)
            return []

    monkeypatch.setattr(agent_module, "make_agent", lambda **kw: Actor())

    def broken(_task):
        raise OSError("source store unavailable")

    receipt = workers.handle_wake_direct(1, "complete text", dict(WAKE_META), bind_input=broken,
                                         on_finished=lambda tid, ok: finished.append(ok))
    assert receipt["admitted"] is True and _wait_for(lambda: bool(finished)) and finished == [True]
    assert seen["text"] == "complete text" and "wake_observation" not in seen["metadata"]


def test_wake_refusals_are_typed_and_start_nothing(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module
    from supervisor import state

    sent: list = []
    _lane(monkeypatch, tmp_path, sent=sent)
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
    # A closed gate refuses the wake QUIETLY: the owner's "🔒" lock notice is for the owner's turn.
    assert not [call for call in sent if "🔒" in str(call[0][1])]
    # A refused wake writes no budget/cost notice of its own into the chat.
    assert not [call for call in sent if "Budget" in str(call[0][1]) or "accounting" in str(call[0][1])]


def test_wake_runner_failure_reports_ok_false_and_concludes_the_turn(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module

    sent: list = []
    _lane(monkeypatch, tmp_path, sent=sent)
    finished: list = []

    class Actor:
        def handle_task(self, task):
            raise RuntimeError("model unreachable")

    monkeypatch.setattr(agent_module, "make_agent", lambda **kw: Actor())
    receipt = workers.handle_wake_direct(1, "wake", dict(WAKE_META),
                                         on_finished=lambda tid, ok: finished.append((tid, ok)))
    assert receipt["admitted"] is True
    assert _wait_for(lambda: bool(finished))
    assert finished == [(receipt["task_id"], False)]
    assert get_direct_activity_registry().get(receipt["task_id"]) is None
    args, kwargs = sent[-1]
    assert args[0] == 1 and "model unreachable" in args[1]
    assert kwargs["task_id"] == receipt["task_id"]
    assert kwargs["progress_meta"] == {"task_terminal_status": "failed", "initiator": "consciousness"}
    rows = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["type"] == "direct_chat_error" and row["task_id"] == receipt["task_id"] for row in rows)


def test_owner_turn_still_runs_admission_and_execution_as_one_call(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module

    events = queue.Queue()
    _lane(monkeypatch, tmp_path, event_q=events)

    class Actor:
        def handle_task(self, task):
            self.task = task
            return [{"type": "send_message", "task_id": task["id"], "chat_id": task["chat_id"], "text": "hi"}]

    actor = Actor()
    monkeypatch.setattr(agent_module, "make_agent", lambda **kw: actor)
    workers.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "c-1"})
    assert actor.task["text"] == "hello" and "initiator" not in actor.task.get("metadata", {})
    assert get_direct_activity_registry().snapshot() == []
    drained = events.get(timeout=5)
    assert "initiator" not in drained and drained["_is_direct_chat"] is True


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
    # An empty model slot still honors the role's OWN local flag when the owner set it and it
    # differs from Main's (В25=B: every slot is respected; astra round 4).
    with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_CONSCIOUSNESS": "", "USE_LOCAL_CONSCIOUSNESS": "true",
                                      "USE_LOCAL_MAIN": "false", "OUROBOROS_MODEL": "openai/gpt-5.6-sol"}):
        assert model_role_slot_override(meta) == ("openai/gpt-5.6-sol", True)
    with mock.patch.dict(os.environ, {"OUROBOROS_MODEL_CONSCIOUSNESS": "", "USE_LOCAL_CONSCIOUSNESS": "false",
                                      "USE_LOCAL_MAIN": "true", "OUROBOROS_MODEL": "local/model"}):
        assert model_role_slot_override(meta) == ("local/model", False)
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


def test_prepare_task_context_applies_the_role_slot_through_the_tool_context_seam():
    """The agent pins ``ctx.task_model_override``/``task_use_local_override`` — the
    seam the loop already reads for a subagent's model — from the role slot."""
    import inspect

    from ouroboros import agent as agent_module

    source = inspect.getsource(agent_module.OuroborosAgent._prepare_task_context)
    assert "role_slot = model_role_slot_override(task_metadata)" in source
    assert "ctx.task_model_override, ctx.task_use_local_override = role_slot" in source


def test_handle_task_ledger_category_comes_from_usage_category(monkeypatch, tmp_path):
    from ouroboros import agent as agent_module
    from ouroboros import config, model_wait, subagent_runtime, usage_accounting

    captured: list = []

    @contextlib.contextmanager
    def fake_scope(scope):
        captured.append(scope)
        yield

    monkeypatch.setattr(usage_accounting, "usage_scope", fake_scope)
    monkeypatch.setattr(model_wait, "task_model_wait_scope", lambda **kw: contextlib.nullcontext())
    monkeypatch.setattr(subagent_runtime, "apply_task_start_settings_or_disclose", lambda *a, **k: None)
    monkeypatch.setattr(config, "task_settings_scope", lambda snapshot: contextlib.nullcontext())
    monkeypatch.setattr(agent_module.OuroborosAgent, "_handle_task_scoped", lambda self, task: [])
    agent = object.__new__(agent_module.OuroborosAgent)
    agent.env = SimpleNamespace(drive_root=tmp_path)
    agent._emit_live_log = lambda *a, **k: None
    agent._event_queue = None
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
    in-memory clock facts are named as not read, not guessed. Start and stop keep
    their supervisor path and their owner notice."""
    from ouroboros.tools import control
    from supervisor import events_runtime_controls, state

    (tmp_path / "state").mkdir(parents=True)
    (tmp_path / "locks").mkdir(parents=True)
    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    state.update_state(lambda st: st.update({
        "bg_consciousness_enabled": True, "consciousness_next_wake_at": 1790416800.0,
        "consciousness_last_wake_at": 1790413200.0, "consciousness_next_interval_sec": 900}))
    ctx = SimpleNamespace(task_id="w1", pending_events=[], drive_root=tmp_path, task_metadata={})
    with mock.patch.dict(os.environ, {"OUROBOROS_BG_WAKEUP_MIN": "120", "OUROBOROS_BG_WAKEUP_MAX": "600"}):
        facts = json.loads(control._toggle_consciousness(ctx, "status"))
    assert ctx.pending_events == []  # nothing for the supervisor to publish
    assert facts["source"] == str(tmp_path / "state" / "state.json") and facts["observed_at"]
    assert "source_modified_at" not in facts and "read_gap" not in facts and "not_recorded" not in facts
    assert facts["enabled"] is True and facts["chosen_interval_sec"] == 900
    assert facts["stored_next_wake_at"] == "2026-09-26T10:00:00+00:00"
    assert facts["last_wake_ended_at"] == "2026-09-26T09:00:00+00:00"
    assert facts["configured_bounds_sec"] == {"min": 120, "max": 600, "source": "owner settings, not the state file"}
    assert "last_wake_outcome" not in facts and "pending early-wake reason" in facts["notes"][1]

    def unpublished():
        raise AssertionError("status must not be published")

    sent, clock = [], SimpleNamespace(start=lambda: "enabled", stop=lambda: "disabled", status_snapshot=unpublished)
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
    assert facts["not_recorded"] == ["last_wake_ended_at"] and "last_wake_ended_at" not in facts
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
        assert facts["not_read"] == ["enabled", "stored_next_wake_at", "last_wake_ended_at", "chosen_interval_sec"]
        assert not {"enabled", "stored_next_wake_at", "last_wake_ended_at", "chosen_interval_sec"} & set(facts)
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


def test_wake_affordances_describe_the_real_alarm_and_the_status_audience():
    from ouroboros.tools.control import get_tools

    schemas = {entry.name: entry.schema for entry in get_tools()}
    wake = schemas["set_next_wakeup"]["description"]
    for phrase in ("after a wake-up ends", "OUROBOROS_BG_WAKEUP_MIN/MAX", "already pending keeps its time",
                   "stored for later", "failed wake-up doubles the interval (up to MAX)",
                   "pending event brings the next wake-up forward", "retries after MIN",
                   "exhausted allowance waits for its reset", "sooner than MIN after the last wake-up, boot or skip"):
        assert phrase in wake
    assert len(wake) < 600, "an operational paragraph, not the alarm's whole algorithm"
    assert "answered to you only" in schemas["toggle_consciousness"]["description"]


def test_a_wake_whose_thread_cannot_start_leaves_no_registered_turn(monkeypatch, tmp_path):
    """A registered turn nobody runs would read as a live owner turn forever (opus round 3)."""
    from ouroboros import agent as agent_module

    _lane(monkeypatch, tmp_path)
    monkeypatch.setattr(agent_module, "make_agent", lambda **kw: SimpleNamespace(handle_task=lambda task: []))

    def _refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", _refuse)
    receipt = workers.handle_wake_direct(1, "wake", dict(WAKE_META))
    assert receipt == {"admitted": False, "task_id": "", "reason": "admission_failed"}
    assert get_direct_activity_registry().snapshot() == []
