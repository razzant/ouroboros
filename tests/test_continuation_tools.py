"""Cold resumes converge only after their own frozen prefix has changed."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ouroboros import loop, owner_wait, transcript_prefix as prefix
from ouroboros.tool_policy import initial_tool_schemas
from tests.test_reasoning_binding import signed
from tests.test_route_tool_schema_limit import _catalog, _names, _schema


class Registry:
    def __init__(self, root, schemas, mode):
        self.current = schemas
        self.handlers = {}
        self.omissions = []
        self._ctx = SimpleNamespace(
            task_id="resume", task_attempt=1, task_metadata={}, context_fit_plan=None,
            active_model="anthropic::fixture", active_use_local=False, active_effort="high",
            active_context_mode=mode, drive_root=root, budget_drive_root=root,
            drive_logs=lambda: root / "logs", event_queue=None, owner_wait_resume={},
            owner_wait_callback=lambda *_: "owner_input", model_wait_context=None,
        )

    def schemas(self):
        return deepcopy(self.current)

    def get_schema_by_name(self, name):
        return next((deepcopy(row) for row in self.current if row["function"]["name"] == name), None)

    def capability_omissions(self):
        return self.omissions

    def callable_rows(self, namespace=""):
        return [{"name": row["function"]["name"]} for row in self.current]

    def override_handler(self, name, handler):
        self.handlers[name] = handler

    def policy_hidden_reason(self, name):
        return "not permitted"


class OnDemand(Registry):
    """Its extras stand for an unpinned MCP server: no task starts with their schemas."""

    def on_demand_schema_names(self):
        return frozenset(name for name in _names(self.current) if name.startswith("extra_"))


def prepare(tmp_path, monkeypatch, *, frozen=None, current=None, mode="max", system_changed=True,
            baseline=True, loaded=(), legacy=False, history=None, registry_type=None):
    frozen = deepcopy(frozen if frozen is not None else [_schema("old")])
    current = deepcopy(current if current is not None else [_schema("new")])
    registry = (registry_type or Registry)(tmp_path, current, mode)  # a test may patch this module's Registry
    ctx = registry._ctx
    saved_messages = [{"role": "system", "content": "saved system"},
                      {"role": "user", "content": "continue"},
                      {"role": "assistant", "content": "working", "reasoning_details": [signed()]}]
    saved_messages.extend(history or [])
    if baseline:
        ctx._last_context_observation = {"messages": deepcopy(saved_messages), "tool_schemas": deepcopy(frozen)}
    ctx._actor_loaded_tool_names = set(loaded)
    state = json.loads(json.dumps(owner_wait.continuation_state(ctx, saved_messages, {}, {}, 2, frozen, set())))
    if legacy:
        state.pop("actor_loaded_tool_names")
    del ctx._actor_loaded_tool_names
    # A cold process has no send observer, but discovery already closes over the saved list.
    ctx.__dict__.pop("_last_context_observation", None)
    resident = state["tool_schemas"]
    messages, trace = [], {}
    _, enabled = loop._setup_dynamic_tools(registry, resident, messages, context_mode=mode)
    ctx._execution_trace = trace
    projections = []

    def rebind(plan, tools, messages, **kwargs):
        projections.append(deepcopy(kwargs["tool_schemas"]))
        # A candidate-induced system change is possible only on a second fit.
        messages[0] = {"role": "system", "content": "new system" if system_changed else "saved system"}
        return plan, mode

    monkeypatch.setattr(loop, "_rebind_context_fit_plan", rebind)
    monkeypatch.setattr(loop, "get_context_mode", lambda: mode)
    return registry, state, messages, trace, enabled, projections


def restore(prepared):
    tools, state, messages, trace, _enabled, _projections = prepared
    owner_wait.restore_continuation_state(tools, state, messages, trace, {}, set())
    owner_wait.rebind_restored_route(tools, state, messages)
    return state["tool_schemas"]


def test_intact_own_prefix_keeps_frozen_tools_and_reasoning(tmp_path, monkeypatch):
    prepared = prepare(tmp_path, monkeypatch, system_changed=False)
    tools, state, messages, trace, _, projections = prepared
    frozen = deepcopy(state["tool_schemas"])
    assert restore(prepared) == frozen
    assert len(projections) == 1  # the candidate never causes the first system change
    assert prefix.retire_unbound_reasoning(tools._ctx, messages, frozen) is None
    assert messages[2]["reasoning_details"] and trace == {}
    assert len(messages) == 3


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_changed_system_converges_with_fresh_profile_and_retires_once(tmp_path, monkeypatch, mode):
    prepared = prepare(tmp_path, monkeypatch, mode=mode, current=[_schema("enable_tools"), _schema("new")])
    tools, state, messages, trace, _, projections = prepared
    identity = state["tool_schemas"]
    assert restore(prepared) is identity
    assert prefix.tools_digest(identity) == prefix.tools_digest(initial_tool_schemas(tools, mode))
    assert projections == [[_schema("old")], identity]
    # Adoption leaves reasoning to its existing send owner.
    assert messages[2]["reasoning_details"]
    usage = {}
    assert prefix.prepare_reasoning_for_send(tools._ctx, messages, identity, round_idx=3, usage=usage)
    assert prefix.prepare_reasoning_for_send(tools._ctx, messages, identity, round_idx=3, usage=usage) is None
    assert usage["reasoning_retired"] == 1
    fact = trace["continuation_tools"][0]
    assert fact["from_tools_digest"] == prefix.tools_digest([_schema("old")])
    assert fact["to_tools_digest"] == prefix.tools_digest(identity)
    events = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()]
    assert sum(event.get("checkpoint_kind") == "continuation_tools_changed" for event in events) == 1
    saved = owner_wait.continuation_state(tools._ctx, messages, trace, usage, 3, identity, set())
    assert saved["tool_schemas"] == identity


@pytest.mark.parametrize("trigger", ["no_baseline", "tools_changed", "observer_baseline"])
def test_other_adoption_triggers_use_the_shared_baseline(tmp_path, monkeypatch, trigger):
    prepared = prepare(tmp_path, monkeypatch, system_changed=False, baseline=trigger != "no_baseline")
    tools, state, messages, _, _, _ = prepared
    if trigger == "tools_changed":
        state["tool_schemas"].append(_schema("enabled_since_send"))
    if trigger == "observer_baseline":
        # Neighbour cache-honesty work restores these digests: they win over the observation.
        setattr(tools._ctx, prefix.DIGEST_ATTR, [prefix.message_digest({"role": "system", "content": "older"})])
        setattr(tools._ctx, prefix.TOOLS_DIGEST_ATTR, prefix.tools_digest(state["tool_schemas"]))
    assert restore(prepared) == [_schema("new")]
    assert "tool set changed" in messages[-1]["content"]


@pytest.mark.parametrize("carrier", ["canonical", "native"])
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("carried", [False, True])
def test_history_requires_only_frozen_contracts_after_fitting(tmp_path, monkeypatch, carrier, available, carried):
    row = ({"role": "assistant", "tool_calls": [{"function": {"name": "history_tool"}}]}
           if carrier == "canonical" else
           {"role": "assistant", "content": [{"type": "tool_use", "name": "history_tool", "id": "t", "input": {}}]})
    current = [_schema("new")] + ([_schema("history_tool")] if available else [])
    frozen = [_schema("old")] + ([_schema("history_tool")] if carried else [])
    prepared = prepare(tmp_path, monkeypatch, frozen=frozen, current=current, history=[row])
    tools, state, messages, trace, enabled, projections = prepared
    assert restore(prepared) == (frozen if carried and not available else current)
    if carried and not available:
        assert "saved tool set was kept" in messages[-1]["content"]
        assert "history_tool" in messages[-1]["content"]
        assert trace == {} and len(projections) == 1 and enabled == set()
        assert not hasattr(tools._ctx, "_route_left_out_tool_names")
    else:
        assert len(projections) == 2 and trace["continuation_tools"]
        assert not any("saved tool set was kept" in str(row.get("content")) for row in messages)


@pytest.mark.parametrize("enabled", [False, True])
def test_history_check_is_after_the_route_ceiling(tmp_path, monkeypatch, enabled):
    catalog = _catalog(129)
    tail = [name for name in _names(catalog) if name.startswith("extra_")][-1]
    frozen = [_schema("old"), _schema(tail)]
    prepared = prepare(tmp_path, monkeypatch, frozen=frozen, current=catalog, loaded=[tail] if enabled else (), history=[
        {"role": "assistant", "content": [{"type": "tool_use", "name": tail}]}])
    tools, _, messages, _, _, _ = prepared
    tools._ctx.active_model = "openai::unseen-model"
    # Restore will restore this route too.
    prepared[1]["route"]["active_model"] = tools._ctx.active_model
    resident = restore(prepared)
    # A name this task's history already called is kept first within the ceiling (#2332), enabled or not:
    # the fitted set represents its history, so the current set is adopted.
    assert len(resident) == 128 and tail in _names(resident)
    assert not any("saved tool set was kept" in str(row.get("content")) for row in messages)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("model, count", [("openai::unseen-model", 128), ("anthropic::unseen", 129),
                                          ("openai/unseen-router", 129)])
def test_ceiling_keeps_enabled_or_legacy_resident_without_inventing_intent(tmp_path, monkeypatch, legacy, model, count):
    catalog = _catalog(129)
    tail = [name for name in _names(catalog) if name.startswith("extra_")][-1]
    frozen = [schema for schema in catalog if schema["function"]["name"] != "extra_00"]
    assert len(frozen) == 128
    prepared = prepare(tmp_path, monkeypatch, frozen=frozen, current=catalog, loaded=[tail], legacy=legacy,
                       history=[{"role": "assistant", "tool_calls": [{"function": {"name": "enable_tools"}}]}])
    tools, state, _, _, enabled, _ = prepared
    state["route"]["active_model"] = model
    resident = restore(prepared)
    assert len(resident) == count and tail in _names(resident)
    assert tools._ctx._actor_loaded_tool_names is enabled
    assert enabled == (set() if legacy else {tail})
    # The handler's original set must see restoration and continue to receive new intent.
    tools.current.append(_schema("registered_later"))
    tools.handlers["enable_tools"](ctx=tools._ctx, tools="registered_later")
    assert "registered_later" in enabled and "registered_later" in _names(resident)


def test_a_ceiling_never_forgets_a_selection_across_cold_resumes(tmp_path, monkeypatch):
    catalog = _catalog(131)
    extras = [name for name in _names(catalog) if name.startswith("extra_")]
    revoked, chosen = extras[0], extras[-1]
    current = [schema for schema in catalog if schema["function"]["name"] != revoked]
    # Loaded on a roomy route; resumed on a route that carries 128 schemas at most.
    first = prepare(tmp_path, monkeypatch, frozen=catalog, current=current, loaded=extras, registry_type=OnDemand)
    tools, state, messages, _, enabled, _ = first
    state["route"]["active_model"] = "openai::unseen-model"
    resident = restore(first)
    assert len(resident) == 128 and chosen not in _names(resident)
    assert chosen in enabled and revoked not in enabled  # a ceiling keeps the choice; current policy drops a revoked tool
    saved = owner_wait.continuation_state(tools._ctx, messages, {}, {}, 3, resident, set())
    # The next process resumes on a roomy route and gets the choice back.
    (tmp_path / "second").mkdir()
    second = prepare(tmp_path / "second", monkeypatch, frozen=resident, current=current,
                     loaded=saved["actor_loaded_tool_names"], registry_type=OnDemand)
    second[1]["route"]["active_model"] = "anthropic::unseen"
    names = _names(restore(second))
    assert chosen in names and revoked not in names


def test_nano_extras_use_current_definitions_and_current_policy(tmp_path, monkeypatch):
    extra = _schema("chosen")
    extra["function"]["description"] = "current contract"
    prepared = prepare(tmp_path, monkeypatch, mode="nano", frozen=[_schema("chosen"), _schema("forbidden")],
                       current=[_schema("enable_tools"), extra, _schema("not_selected")], loaded=["chosen", "forbidden"])
    tools, _, messages, trace, enabled, _ = prepared
    resident = restore(prepared)
    assert resident == [_schema("enable_tools"), extra]
    assert enabled == {"chosen"}
    assert "~changed chosen" in messages[-1]["content"]
    assert trace["continuation_tools"][0]["removed"] == ["forbidden"]
    assert tools.handlers["enable_tools"](ctx=tools._ctx, tools="chosen") == "Already loaded: chosen"


@pytest.mark.parametrize("omissions", [False, True])
def test_equal_candidate_is_quiet_but_current_capability_notes_survive_restore(tmp_path, monkeypatch, omissions):
    prepared = prepare(tmp_path, monkeypatch, frozen=[_schema("same")], current=[_schema("same")])
    tools, _, messages, trace, _, projections = prepared
    if omissions:
        tools.omissions = [{"surface": "mcp", "reason": "not_listed_yet", "resource": "current server"}]
    restore(prepared)
    assert not trace and len(projections) == 1
    assert len(messages) == (4 if omissions else 3)
    if omissions:
        assert "current server" in messages[-1]["content"]
    assert not any("tool set changed" in str(row.get("content")) for row in messages)


@pytest.mark.parametrize("rail", ["owner", "budget", "working"])
@pytest.mark.parametrize("system_changed", [False, True])
def test_every_cold_resume_rail_reaches_the_same_decision(tmp_path, monkeypatch, rail, system_changed):
    from ouroboros import budget_pause, working_checkpoint

    prepared = prepare(tmp_path, monkeypatch, system_changed=system_changed)
    tools, state, messages, trace, _, _ = prepared
    if rail == "owner":
        owner_wait.resume_native_loop(tools, state, messages, trace, {}, set())
    elif rail == "working":
        working_checkpoint.resume_from_working(tools, state, messages, trace, {}, set())
    else:
        # Exercise the resume owner while leaving grant CAS/custody to their own suites.
        monkeypatch.setattr(working_checkpoint, "save_resumed", lambda *a, **kw: None)
        monkeypatch.setattr(budget_pause, "set_budget_pause", lambda *a, **kw: None)
        monkeypatch.setattr(budget_pause, "_resume_disclosure", lambda *a, **kw: None)
        budget_pause.resume_paused_loop(tools, state, messages, trace, {}, set(), budget_remaining_usd=None)
    assert state["tool_schemas"] == [_schema("new" if system_changed else "old")]
    assert bool(trace.get("continuation_tools")) == system_changed


def test_real_loop_measures_sends_and_checkpoints_adoption_without_editing_recovered_bytes(tmp_path, monkeypatch):
    from ouroboros import working_checkpoint as wc
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import write_task_result
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("no provider call"))
    registry = _registry(tmp_path, monkeypatch, "t-wait", 1)
    ctx = registry._ctx
    messages = [{"role": "system", "content": "old release system"}, {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "working", "reasoning_details": [signed()]}]
    frozen = [_schema("read_file")]
    ctx._last_context_observation = {"messages": deepcopy(messages), "tool_schemas": deepcopy(frozen)}
    ctx._actor_loaded_tool_names = {"read_file"}
    write_task_result(tmp_path, ctx.task_id, "running")
    saved = SimpleNamespace(tools=registry, messages=messages, llm_trace={}, accumulated_usage={},
                            round_idx=2, tool_schemas=frozen, owner_msg_seen=set())
    wc.save_round(saved, "post_batch")
    exact = wc.checkpoint_path(tmp_path, ctx.task_id, 1).read_bytes()
    task = {"id": ctx.task_id}
    assert wc.attach_recovery(tmp_path, task, source_task_id=ctx.task_id, from_attempt=1, cause="worker_crash")
    handoff = task["_working_recovery"]
    resumed = _registry(tmp_path, monkeypatch, ctx.task_id, 2)
    resumed._ctx.working_recovery = handoff
    expected = prefix.tools_digest(initial_tool_schemas(resumed))
    measured, sent = [], []
    real_measure = loop._measure_round_main_fit

    def measure(call, **kwargs):
        measured.append(prefix.tools_digest(call.tool_schemas))
        return real_measure(call, **kwargs)

    def dispatch(call, disposition, **kwargs):
        sent.append(prefix.tools_digest(call.tool_schemas))
        checkpoint = json.loads(wc.checkpoint_path(tmp_path, ctx.task_id, 2).read_bytes())
        assert prefix.tools_digest(checkpoint["tool_schemas"]) == expected
        assert checkpoint["actor_loaded_tool_names"] == ["read_file"]
        assert read_actor_source_bytes(tmp_path, ctx.task_id, handoff["source_ref"]) == exact
        return {"role": "assistant", "content": "Continued work", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    result, _, _ = loop.run_llm_loop(**_loop_kwargs(tmp_path, resumed, []))
    assert result == "Continued work" and measured == sent == [expected]
    assert read_actor_source_bytes(tmp_path, ctx.task_id, handoff["source_ref"]) == exact
