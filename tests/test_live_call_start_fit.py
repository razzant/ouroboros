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
from tests import _memory_view_synthetic as syn

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
