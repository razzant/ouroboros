"""A live call's start is fitted on the tools the call sends.

In Nano an ordinary turn carries the standing work set (``tool_policy.NANO_RESIDENT_SCHEMA_NAMES``); a call keeps its
own base, the selectors and its feature's ``initial_tools`` (``live_conversation.ConversationHost``). The start plan the
agent builds for the call must price that base, or a window that holds the call is refused for tools it never sends.
"""
from __future__ import annotations

import json
import pathlib
import queue
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as fit
from ouroboros import memory_view
from ouroboros.contracts.realtime import RouteFacts
from ouroboros.tool_policy import NANO_RESIDENT_SCHEMA_NAMES, NANO_SCHEMA_META_NAMES
from tests import _memory_scale_fixture as scale
from tests import _memory_view_synthetic as syn
from tests._live_world import host_world as host_world

FACTS = RouteFacts(conversation_window_tokens=92_000, working_reserve_tokens=10_000, source="synthetic call window")


def _start_arguments(tmp_path, monkeypatch, feature):
    """What the agent hands the context build for one start: a call (``feature``) or an ordinary turn (None)."""
    from ouroboros import agent as agent_module
    from ouroboros import provider_models
    from ouroboros.agent import Env, OuroborosAgent

    seen = {}

    def build(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the context build")

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "nano")
    monkeypatch.setattr(agent_module, "build_llm_messages", build)
    monkeypatch.setattr(provider_models, "model_has_credentials", lambda _model: True)
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    agent = OuroborosAgent(Env(repo_dir=pathlib.Path(__file__).resolve().parents[1], drive_root=tmp_path),
                           event_queue=queue.Queue())
    if feature is not None:
        agent._realtime_turn = SimpleNamespace(resolve_route=lambda *_a, **_kw: None, feature=feature)
    monkeypatch.setattr(agent.llm, "chat", lambda **_kw: pytest.fail("no model call"))
    with pytest.raises(RuntimeError, match="stop after the context build"):
        agent._prepare_task_context({"id": "call-start", "type": "task", "chat_id": 1, "text": "hi"})
    return seen


def _schema(name, chars):
    return {"type": "function", "function": {"name": name, "description": "d" * chars,
                                             "parameters": {"type": "object", "properties": {}}}}


def _plan(root, monkeypatch, nano_base):
    """The real start fit on a 92 000-token call window. Its standing tools are large and the call's own tool small:
    the core with the call's base fits, the same core with the standing set does not."""
    tools = [_schema(name, 200 if name in NANO_SCHEMA_META_NAMES else 6_000)
             for name in sorted(NANO_RESIDENT_SCHEMA_NAMES)]
    tools.append(_schema("query_code", 200))
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=[{**syn.spoken(1, "human", 100), "owner_side": True,
                                                             "immune": True}]))
    core = fit.ContextCore(base_prompt="SYSTEM", bible_md="BIBLE", architecture_md="# Architecture",
        development_md="# Development", docs_need_development=False, semi_stable_text="IDENTITY " + "x" * 300_000,
        dynamic_text="", user_content_json=json.dumps("Open the call."),
        memory_view_json=memory_view.snapshot_json(snapshot), section_root=str(root))
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    evidence = SimpleNamespace(route_fp="synthetic-call", status="confirmed", stale=False, window_tokens=92_000,
                               route_facts=asdict(FACTS), source=FACTS.source)
    return fit.build_context_fit_plan(SimpleNamespace(drive_root=root), core, {"id": "call", "_is_direct_chat": True},
        preferred_mode="nano", tool_schemas=tools, nano_base=nano_base,
        route_resolver=lambda *_a, **_kw: ({"provider": "synthetic", "model": "synthetic"}, evidence))


def test_a_call_start_prices_the_calls_own_base_and_an_ordinary_start_the_standing_set(tmp_path, monkeypatch):
    call = _start_arguments(tmp_path / "call", monkeypatch, {"initial_tools": ("query_code",)})
    assert call["nano_base"] == ("query_code",)
    names = {schema["function"]["name"] for schema in call["tool_schemas"]}
    assert "query_code" in names and not names & (NANO_RESIDENT_SCHEMA_NAMES - NANO_SCHEMA_META_NAMES)
    ordinary = _start_arguments(tmp_path / "turn", monkeypatch, None)
    assert ordinary["nano_base"] is None
    names = {schema["function"]["name"] for schema in ordinary["tool_schemas"]}
    assert {"finish_task", "read_file", "run_command"} <= names and "query_code" not in names


def test_a_window_that_holds_the_call_but_not_the_standing_set_gives_the_call_a_plan(tmp_path, monkeypatch):
    with pytest.raises(fit.MinimumViewUnfit):
        _plan(tmp_path / "standing", monkeypatch, None)  # an ordinary turn on this window is refused, as before
    plan = _plan(tmp_path / "call", monkeypatch, ("query_code",))
    assert plan.initial_mode == "nano"


def test_the_floors_diagnostics_read_the_calls_own_base(tmp_path, monkeypatch):
    """Review 8 (Fable, Astra): the floor note of a selection and a counterfactual start price the call's base, as its
    start did. Its base sends no ``memory_read`` (the note says how to load it) and the standing set's weight is absent
    (the call's window holds the counterfactual start, as it held the real one)."""
    from ouroboros import memory_floor, memory_view

    plan = _plan(tmp_path / "call", monkeypatch, ("query_code",))
    assert plan.start_nano_base == ("query_code",)
    selected = memory_floor.select_view(plan, "nano", [])
    assert "(enable_tools loads memory_read if permitted)" in selected.projection("nano").system_content_json
    counterfactual = memory_floor.counterfactual_starts(
        plan, {"now": memory_view.snapshot_from_json(plan.core.memory_view_json)})
    assert counterfactual["measured"] and counterfactual["arms"]["now"]["mode"] == "nano"


def _fresh_start(tmp_path, monkeypatch, resolve, mode, nano_base):
    from ouroboros import context

    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    env, memory = scale.env_on(scale.installed(tmp_path, "fresh"))
    return context.build_context_fit_plan(env, memory, {"id": "fresh-start", "type": "task", "chat_id": 1, "text": "hi",
                                                        "_is_direct_chat": True},
                                          preferred_mode=mode, tool_schemas=scale.tools(), nano_base=nano_base,
                                          route_resolver=resolve)


@pytest.mark.parametrize("route", [pytest.param(route, id=f"{route[0]}:{route[1]}") for route in scale.call_routes()]
                         + [pytest.param(("text", "85k", None), id="text:85k")])
def test_the_fresh_fixture_starts_on_every_shipped_call_model_and_on_hard_85k_text_at_the_ordinary_builder(
        tmp_path, monkeypatch, route):
    """A new install's start fits every shipped call model's window with its eighth of work room left and no host section
    addressed for the window, and a hard 85,000-token text route keeps the owner's Nano. This prices the ordinary
    builder with the fixture's synthetic schemas and the call's control base; each adapter's own session candidate is
    driven by ``test_each_shipped_adapter_admits_the_fresh_start_at_its_own_session_candidate``."""
    skill, _model, facts = route
    if facts is None:
        resolve, base = scale.route_resolver(85_000, "provider", None), None
    else:
        resolve = scale.route_resolver(facts.conversation_window_tokens, facts.basis, facts.working_reserve_tokens,
                                       provider=skill, facts=facts)
        base = ("list_available_tools", "enable_tools")
    plan = _fresh_start(tmp_path, monkeypatch, resolve, "nano", base)
    shown = plan.projection(plan.initial_mode)
    assert plan.preferred_mode == "nano" and plan.initial_mode == "nano"
    assert "Start room is smaller than requested" not in shown.system_content_json  # the eighth is left
    assert not shown.memory_facts.get("host_sections_addressed"), "no host section addressed for the window"


def _adapter_start(tmp_path, monkeypatch, host_world, skill, model_id, *, window=None, shape="fresh"):
    """The ``shape`` drive's call start (``rich`` with its twelve MCP servers through the real manager) through the adapter's own setup: the real ``ToolRegistry``, the call feature's
    ``INITIAL_TOOLS``, ``call_tools`` and ``session_options``, the ordinary builder for the call's messages, then
    ``realtime_setup.session_candidate`` with the adapter's registered session factory and a ``connect`` that fails the
    test. ``window`` narrows the catalogue's window for the candidate only (the ordinary build keeps the declared one).
    Returns ``(session_candidate's answer, connection attempts)``; the plugin is loaded as a package
    (``extension_loader``)."""
    import asyncio
    import copy
    import importlib.util
    import sys
    from dataclasses import replace

    from ouroboros import capability_evidence as ce, config, context
    from ouroboros.call_peers import LiveCallHub
    from ouroboros.live_conversation import ConversationHost
    from ouroboros.realtime_setup import session_candidate
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_context import ToolContext
    from skills.live_conversation.driver import INITIAL_TOOLS, call_tools, session_options

    path = pathlib.Path(__file__).resolve().parents[1] / "skills" / skill / "plugin.py"
    spec = importlib.util.spec_from_file_location(f"_adapter_start_{skill}", path,
                                                  submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    model = module.CATALOG.model(model_id)
    turn = copy.copy(host_world.host.turn)
    turn.route, turn.catalog_model, turn.native_model = module.ROUTE, model, model_id
    turn.model = module.ROUTE.model_prefix + model_id
    turn.spec = replace(turn.spec, chat_id=1, model=model_id)
    drive = scale.installed(tmp_path, shape)
    if shape == "rich":
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))
        scale.mcp_registry(drive, monkeypatch)
    env, memory = scale.env_on(drive)
    turn.feature = dict(turn.feature, initial_tools=INITIAL_TOOLS, call_tools=call_tools(drive))
    task = dict(host_world.task, chat_id=1, model=turn.model,
                origin_message_ref={"chat_id": 1, "client_message_id": "call:fixture-call:open"})
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=drive, branch_dev="test", task_id=task["id"], current_chat_id=1,
                      is_direct_chat=True, task_metadata=task["metadata"], task_contract={})
    registry = ToolRegistry(env.repo_dir, drive)
    registry.set_context(ctx)
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: drive)
    monkeypatch.setattr(context, "get_context_mode", lambda: "nano")
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    scope = f"{skill}-{shape}-start"
    route = dict(provider=turn.route.accounting_provider, model=turn.model, base_url=turn.route.endpoint,
                 use_local=False, model_role="main", options={})
    evidence = SimpleNamespace(route_fp=scope, status="confirmed", stale=False,
                               window_tokens=model.facts.conversation_window_tokens, source=model.facts.source,
                               route_facts=asdict(model.facts))
    selection = initial_tool_schemas(registry, context_mode="max", standing=False, schema_names=INITIAL_TOOLS)
    rows, _cap = context.build_llm_messages(env, memory, task, ctx=ctx, tool_schemas=selection, tool_registry=registry,
                                            nano_base=INITIAL_TOOLS,
                                            route_resolver=lambda _task, allow_fetch: (route, evidence))
    if window is not None:
        turn.catalog_model = replace(model, facts=replace(model.facts, conversation_window_tokens=window,
                                                          accepted_input_tokens=None))
    host = ConversationHost(turn, ctx, rows, registry, task, "", LiveCallHub(turn.spec))
    factories, attempts = [], []
    api = SimpleNamespace(register_realtime_provider=lambda _provider, **kw: factories.append(kw["session"]),
                          get_settings=lambda *_a, **_kw: pytest.fail("sizing read settings or an account"))
    module.register(api)

    def connect(*_a, **_kw):
        attempts.append(1)
        pytest.fail("sizing tried to connect")

    def factory(*args, **kwargs):
        return factories[0](*args, connect=connect, **kwargs)
    try:
        return session_candidate(host, factory, session_options(host), observation_fp=scope), attempts
    finally:
        asyncio.run(host.finish())
        host.wire.close()


@pytest.mark.parametrize("route", [pytest.param(route, id=f"{route[0]}:{route[1]}") for route in scale.call_routes()])
def test_each_shipped_adapter_admits_the_fresh_start_at_its_own_session_candidate(tmp_path, monkeypatch, host_world,
                                                                                  route):
    """Every shipped call model's own session candidate (its instructions, the call's tools, its serializer) admits a
    new install's start within the catalogue's window, with no work-room deficit and no connection attempt."""
    skill, model_id, facts = route
    (_session, _request, proof), attempts = _adapter_start(tmp_path, monkeypatch, host_world, skill, model_id)
    measured = proof["fit"]["measurement"]
    assert measured["capacity_deficit_tokens"] == 0 and measured["capacity_total_tokens"] == facts.conversation_window_tokens
    assert measured["estimated_input_tokens"] + measured["response_reserve_tokens"] <= facts.conversation_window_tokens
    assert proof.get("work_room", {}).get("deficit_tokens") == 0 and not attempts


def test_a_window_too_small_for_the_native_candidate_is_refused_before_any_connection(tmp_path, monkeypatch,
                                                                                       host_world):
    """The same drive, its ordinary build fitted, on a window its session candidate cannot hold: a typed
    ``context_capacity`` refusal, and nothing tried to connect."""
    from ouroboros.contracts.realtime import RealtimeSetupError

    skill, model_id, _facts = next(route for route in scale.call_routes() if route[0] == "grok_realtime")
    with pytest.raises(RealtimeSetupError) as raised:  # a connection attempt would fail the test instead
        _adapter_start(tmp_path, monkeypatch, host_world, skill, model_id, window=40_000)
    assert raised.value.code == "context_capacity"


# What each shipped call model's own start does with a mature install (``mature``) and the same install with its 39
# running works and twelve MCP servers (``rich``), on the shipped catalogue windows: OpenAI's window measured 27.11.2026
# holds both; Gemini's (both selections) and Grok's still refuse both, typed, before any socket. A route whose window
# comes to hold an install moves to ``ADMITS`` in the change that makes it true.
ADMITS = {"openai_realtime"}


@pytest.mark.parametrize("shape", ["mature", "rich"])
@pytest.mark.parametrize("route", [pytest.param(route, id=f"{route[0]}:{route[1]}") for route in scale.call_routes()])
def test_each_shipped_adapter_admits_or_refuses_a_mature_start_at_its_own_session_candidate(
        tmp_path, monkeypatch, host_world, route, shape):
    skill, model_id, facts = route
    if skill not in ADMITS:
        with pytest.raises(fit.MinimumViewUnfit) as raised:  # a connection attempt would fail the test instead
            _adapter_start(tmp_path, monkeypatch, host_world, skill, model_id, shape=shape)
        assert raised.value.window_tokens == facts.conversation_window_tokens and raised.value.deficit_tokens > 0
        return
    (_session, _request, proof), attempts = _adapter_start(tmp_path, monkeypatch, host_world, skill, model_id,
                                                          shape=shape)
    measured = proof["fit"]["measurement"]
    assert not attempts and measured["capacity_deficit_tokens"] == 0
    assert measured["capacity_total_tokens"] == facts.conversation_window_tokens
    assert measured["estimated_input_tokens"] + measured["response_reserve_tokens"] <= facts.conversation_window_tokens
    assert proof.get("work_room", {}).get("deficit_tokens") == 0  # the call starts with its eighth of work room
