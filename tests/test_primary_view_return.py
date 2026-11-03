"""A return to the primary resumes the view it showed before a configured route acted.

A narrower configured route rewrites the memory of every book in the plan; without the hold a return keeps that
narrowing for the rest of the task. The primary's own view is held at the first rebind that leaves it and resumed
when a rebind returns to it; its own lowering (start fit, real overflow) and my selected forms stay, and a switch
on the same route, or to a primary changed meanwhile, never raises the mode.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import config, context, fallback_cooldown, loop, model_wait
from ouroboros import memory_floor as floor
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from ouroboros.primary_route_observation import hold_selection, record_round_route_result
from ouroboros.tools.registry import ToolRegistry
from tests.test_memory_view_lever import plan as synthetic_plan
from tests.test_memory_view_modes import _built

WINDOWS = {"primary": 1_000_000, "reserve": 65_537}


@pytest.fixture
def routes(monkeypatch):
    """Each model's own window, as the route evidence states it."""
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_kw: (
        {"model": task["model"], "provider": "openrouter", "base_url": "", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=WINDOWS[task["model"]],
                        route_fp=f"{task['model']}-route")))


def _registry(tmp_path):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id, registry._ctx.event_queue = "k04", None
    registry._ctx.primary_route = {"model": "primary", "use_local": False, "role": "main"}
    registry._ctx.active_model, registry._ctx.active_use_local = "primary", False
    return registry


def _rebind(plan, registry, messages, model, mode):
    return loop._rebind_context_fit_plan(plan, registry, messages, model=model, use_local=False,
                                         start_mode=mode, tool_schemas=[])


def _switch(plan, registry, messages, model, mode):
    rebound, mode = _rebind(plan, registry, messages, model, mode)
    registry._ctx.active_model = model  # the walk adopts the route that answered
    return rebound, mode


def _leave(plan, registry, messages, model, mode):
    """The walk's order: the candidate's rebind, then its observed account's rebind, both before adoption."""
    away, mode = _rebind(plan, registry, messages, model, mode)
    return _switch(away, registry, messages, model, mode)


def test_a_return_resumes_the_primary_view_byte_for_byte(tmp_path, monkeypatch, routes):
    _core, build = _built(tmp_path, monkeypatch, ratio=1, page_chars=400_000)
    plan = build(WINDOWS["primary"])
    registry = _registry(tmp_path)
    messages = plan.messages_for(plan.initial_mode)
    shown = messages[0]
    assert plan.initial_mode == "max"
    away, away_mode = _leave(plan, registry, messages, "reserve", "max")
    assert away_mode in {"low", "nano"} and messages[0] != shown
    assert away.projection("max").memory_facts["floor"]["steps"]  # the reserve's cut reached every book
    back, back_mode = _switch(away, registry, messages, "primary", away_mode)
    assert back_mode == "max"
    assert messages[0] == shown  # without the hold the return keeps the reserve's narrow view
    assert back.projection("max").memory_facts["floor"]["steps"] == {}


def test_the_primary_own_low_stays_low_after_a_return(tmp_path, monkeypatch, routes):
    _core, build = _built(tmp_path, monkeypatch, books=600_000, ratio=1)
    plan = build(200_000)  # the primary's own window could not hold Max at the start
    monkeypatch.setitem(WINDOWS, "primary", 200_000)
    registry = _registry(tmp_path)
    messages = plan.messages_for(plan.initial_mode)
    assert plan.initial_mode == "low"
    away, away_mode = _leave(plan, registry, messages, "reserve", plan.initial_mode)
    _back, back_mode = _switch(away, registry, messages, "primary", away_mode)
    assert back_mode == "low"


def test_a_primary_changed_meanwhile_is_a_new_route(tmp_path, monkeypatch, routes):
    _core, build = _built(tmp_path, monkeypatch, ratio=1, page_chars=400_000)
    plan = build(WINDOWS["primary"])
    monkeypatch.setitem(WINDOWS, "successor", WINDOWS["primary"])
    registry = _registry(tmp_path)
    messages = plan.messages_for(plan.initial_mode)
    away, away_mode = _leave(plan, registry, messages, "reserve", "max")
    registry._ctx.primary_route = {"model": "successor", "use_local": False, "role": "main"}  # Settings changed it
    _new, new_mode = _switch(away, registry, messages, "successor", away_mode)
    assert new_mode == away_mode != "max"  # the old primary's view was never this route's own
    assert registry._ctx._primary_view is None


def test_a_switch_on_the_same_route_never_raises_the_mode(tmp_path, monkeypatch, routes):
    _core, build = _built(tmp_path, monkeypatch, ratio=1, page_chars=400_000)
    plan = build(WINDOWS["primary"])
    registry = _registry(tmp_path)
    narrowed = plan.reproject_for_route(window_tokens=WINDOWS["reserve"], known_window=True, ratio=1,
                                        output_reserve=65_536, tool_schemas=None, start_mode="max")
    messages = narrowed.messages_for(narrowed.initial_mode)
    # An account rotation or a wait's reprepare rebinds the acting primary itself: D2 keeps the shown mode.
    _same, mode = _switch(narrowed, registry, messages, "primary", narrowed.initial_mode)
    assert mode == narrowed.initial_mode != "max"
    assert getattr(registry._ctx, "_primary_view", None) is None


def _windows_by(monkeypatch, key, windows):
    """Route evidence whose window follows one task field: the model, or the role whose account serves it."""
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **_kw: (
        {"model": task["model"], "provider": "openrouter", "base_url": "", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=windows[task[key]], route_fp=task[key])))


def _primary_on(model, tmp_path, monkeypatch):
    _core, build = _built(tmp_path, monkeypatch, ratio=1, page_chars=400_000)
    plan = replace(build(1_000_000), model=model, model_role="main")
    registry = _registry(tmp_path)
    registry._ctx.primary_route["model"] = registry._ctx.active_model = model
    return plan, registry, plan.messages_for("max")


@pytest.mark.parametrize("reserve", ["openai/gpt-reserve", "anthropic/claude-reserve"])
def test_a_failed_primary_row_leaves_the_acting_view(tmp_path, monkeypatch, reserve):
    primary = "openai/gpt-primary"  # the same family as the first reserve: both rows share one transcript list
    _windows_by(monkeypatch, "model", {primary: 1_000_000, reserve: 65_537})
    monkeypatch.setattr(config, "fallback_candidate_targets", lambda **_kw: [])
    fallback_cooldown.reset_for_tests()
    plan, registry, messages = _primary_on(primary, tmp_path, monkeypatch)
    shown = messages[0]
    away, away_mode = _leave(plan, registry, messages, reserve, "max")
    acting_view, sent = messages[0], {}

    def failing_round(call):
        sent["view"] = call.messages[0]
        call.accumulated_usage["_last_llm_error_kind"] = "transport_unavailable"
        return None, 0.0, call.active_context_mode

    monkeypatch.setattr(loop, "_call_round_model", failing_round)
    msg, model, _local, plan_after, mode_after = loop._run_cross_model_fallback_chain(
        llm=SimpleNamespace(), ctx=registry._ctx, tools=registry, messages=messages, active_model=reserve,
        active_use_local=False, tool_schemas=[], active_effort="medium", max_retries=3, drive_logs=tmp_path,
        task_id="k04", round_idx=2, event_queue=None, accumulated_usage={"_last_llm_error_kind": "transport_unavailable"},
        task_type="", emit_progress=lambda *_a, **_kw: None, context_fit_plan=away, active_context_mode=away_mode)
    assert sent["view"] == shown  # the primary row was offered its own view
    assert (msg, model, plan_after, mode_after) == (None, reserve, away, away_mode)
    assert messages[0] == acting_view  # the reserve keeps showing its own view, not the primary's Max


@pytest.mark.parametrize("pin", ["", "account-a"], ids=["auto", "pinned"])
def test_a_wait_card_choosing_the_primary_resumes_its_view(tmp_path, monkeypatch, pin):
    primary, reserve = "claudexor::codex=gpt-main", "claudexor::codex=gpt-reserve"
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": pin, "fallback": [pin]}))
    waiter = SimpleNamespace(overrides={})
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    _windows_by(monkeypatch, "model", {primary: 1_000_000, reserve: 65_537})
    plan, registry, messages = _primary_on(primary, tmp_path, monkeypatch)
    shown = messages[0]
    away, mode = loop._rebind_context_fit_plan(plan, registry, messages, model=reserve, use_local=False,
                                              model_role="fallback:0", start_mode="max", tool_schemas=[])
    registry._ctx.active_model = reserve
    # The fallback's wait card picks the primary: the card writes its choice for the waiting role before the rebind.
    waiter.overrides["fallback:0"] = {"model": primary, "use_local": False, "model_account_override": pin}
    _back, back_mode = loop._rebind_context_fit_plan(away, registry, messages, model=primary, use_local=False,
                                                     model_role="fallback:0", start_mode=mode, tool_schemas=[])
    assert mode != "max" and back_mode == "max" and messages[0] == shown


def test_an_earlier_main_account_choice_keeps_the_hold_on_the_primary_view(tmp_path, monkeypatch):
    primary, reserve = "claudexor::codex=gpt-main", "claudexor::codex=gpt-reserve"
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a", "fallback": ["account-b"]}))
    waiter = SimpleNamespace(overrides={"main": {"model_account_override": "account-b"}})  # an earlier card for Main
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    _windows_by(monkeypatch, "model", {primary: 1_000_000, reserve: 65_537})
    plan, registry, messages = _primary_on(primary, tmp_path, monkeypatch)
    shown = messages[0]
    # The walk candidate's rebind, then its observed account's rebind, both before the walk adopts it.
    away, mode = loop._rebind_context_fit_plan(plan, registry, messages, model=reserve, use_local=False,
                                              model_role="fallback:0", start_mode="max", tool_schemas=[])
    away, mode = loop._rebind_context_fit_plan(away, registry, messages, model=reserve, use_local=False,
                                              model_role="fallback:0", start_mode=mode, tool_schemas=[])
    registry._ctx.active_model = reserve
    _back, back_mode = loop._rebind_context_fit_plan(away, registry, messages, model=primary, use_local=False,
                                                     model_role="main", start_mode=mode, tool_schemas=[])
    assert mode != "max" and back_mode == "max" and messages[0] == shown  # never the candidate's view


def test_a_primary_account_changed_meanwhile_keeps_its_view(tmp_path, monkeypatch):
    model = "claudexor::codex=gpt-test"  # one model shows one view; the new account's window still fits it
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-a"}))
    _windows_by(monkeypatch, "model", {model: 1_000_000, "other/model": 65_537})
    plan, registry, messages = _primary_on(model, tmp_path, monkeypatch)
    shown = messages[0]
    away, away_mode = _leave(plan, registry, messages, "other/model", "max")
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "account-b"}))  # the owner moves Main to another account
    _back, back_mode = _switch(away, registry, messages, model, away_mode)
    assert away_mode != "max" and back_mode == "max" and messages[0] == shown


def test_a_fallback_slot_a_card_moved_off_the_primary_model_is_not_the_primary(tmp_path, monkeypatch):
    primary, card, reserve = "claudexor::codex=gpt-main", "claudexor::codex=gpt-card", "claudexor::codex=gpt-reserve"
    waiter = SimpleNamespace(overrides={})
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    _windows_by(monkeypatch, "model", {primary: 1_000_000, card: 1_000_000, reserve: 65_537})
    plan, registry, messages = _primary_on(primary, tmp_path, monkeypatch)
    away, mode = _leave(plan, registry, messages, reserve, "max")
    acting_view = messages[0]
    # A slot configured with the primary's model, sent to another model by an earlier card: the walk still names it P.
    waiter.overrides["fallback:0"] = {"model": card, "use_local": False}
    _slot, slot_mode = loop._rebind_context_fit_plan(away, registry, messages, model=primary, use_local=False,
                                                     model_role="fallback:0", start_mode=mode, tool_schemas=[])
    assert mode != "max" and slot_mode == mode and messages[0] == acting_view


def test_a_card_moving_main_to_another_model_drops_the_hold(tmp_path, monkeypatch):
    primary, card, reserve = "claudexor::codex=gpt-main", "claudexor::codex=gpt-card", "claudexor::codex=gpt-reserve"
    waiter = SimpleNamespace(overrides={})
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    _windows_by(monkeypatch, "model", {primary: 1_000_000, card: 1_000_000, reserve: 65_537})
    plan, registry, messages = _primary_on(primary, tmp_path, monkeypatch)
    away, mode = _leave(plan, registry, messages, reserve, "max")
    waiter.overrides["main"] = {"model": card, "use_local": False}  # the owner moved Main to another model
    _row, row_mode = loop._rebind_context_fit_plan(away, registry, messages, model=primary, use_local=False,
                                                   model_role="main", start_mode=mode, tool_schemas=[])
    assert mode != "max" and row_mode == mode and registry._ctx._primary_view is None


def test_forms_i_select_away_hold_on_the_primary_view():
    held, snap = synthetic_plan()
    owner = SimpleNamespace(_primary_view=(("primary", False), held))
    hold_selection(owner, ["F5:page-000"], owner_note=False)
    home, view = owner._primary_view
    assert home == ("primary", False) and "Page 0." in held.projection("max").system_content_json
    assert "Page 0." not in view.projection("max").system_content_json
    assert dict(floor.kept_level(snap, view.projection("max").memory_facts).addressed)["F5"] == ("page-000",)
    hold_selection(owner, ["core:BIBLE"], owner_note=False)  # a form the held view refuses releases the hold
    assert owner._primary_view is None


def test_the_primary_answering_releases_the_hold(tmp_path):
    owner = SimpleNamespace(primary_route={"model": "primary", "use_local": False, "role": "main"},
                            _primary_view=object(), _primary_return_requested=None, _execution_trace={})
    usage = {"_last_round_cache_usage": {"round": 3, "model": "primary"}, "_model_route": {}}
    call = SimpleNamespace(tools=SimpleNamespace(_ctx=owner), active_model="primary", active_use_local=False,
                           accumulated_usage=usage, round_idx=3)
    record_round_route_result(call, "main", accepted=True)
    assert owner._primary_view is None
