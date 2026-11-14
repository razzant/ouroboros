"""The owner's Remote switch (``OUROBOROS_REMOTE_EXECUTION``, ``remote_placement``) through its real consumers.

The settings save that writes it; Main as following work runs it, moving at the next logical call through
``round_route`` and its notice, and a refused call's wait that yields when Off moves it back; the account pin and the
window that stay their own engine's; the ``/api/state`` projection; and the delegated-start selectors, a parent that
began before the press and the four continuation cases included. No host, account or model is contacted.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import remote_placement, route_choice
from ouroboros.remote_placement import REMOTE_EXECUTION_KEY
from tests._task_settings_helpers import isolated_runtime, save_owner  # noqa: F401

LOCAL, REMOTE = "claudexor::codex=gpt-6", "claudexor@vps::codex=gpt-6"
ROWS = [{"name": "vps", "label": "VDSina", "ssh_host": "vds", "remote_root": "/home/o/runs"},
        {"name": "lab", "label": "Lab box", "ssh_host": "lab", "enabled": False}]


def _save(values):
    """One owner save of ``values`` over the document as it is (the test writer writes whole documents)."""
    from ouroboros import config

    current = json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8")) if config.SETTINGS_PATH.exists() else {}
    save_owner({**current, **values})


def _switch(on, engine="vps"):
    _save({REMOTE_EXECUTION_KEY: remote_placement.remote_execution_text({"on": on, "engine": engine})})


@pytest.fixture
def owner(isolated_runtime, monkeypatch):  # noqa: F811
    save_owner({"OUROBOROS_MODEL": LOCAL, "OUROBOROS_ENGINE_CONNECTIONS": ROWS,
                "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": "work"}),
                "OUROBOROS_MODEL_CONTEXT_WINDOWS": json.dumps({"main": 400000})})
    return isolated_runtime


# --- the save --------------------------------------------------------------------------------------------------------

def _post(monkeypatch, path, initial, payload):
    """One ``POST /api/settings`` through the real handler over ``initial`` (written first), with the process-wide
    side effects a save fires (environment projection, supervisor start, MCP reload) left out."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import settings as handlers

    for name in ("_start_supervisor_if_needed_for_request", "_apply_settings_to_env", "_apply_settings_save_side_effects"):
        monkeypatch.setattr(handlers, name, lambda *_a, **_k: None)
    monkeypatch.setattr(handlers, "apply_runtime_provider_defaults", lambda document: (document, False, []))
    path.write_text(json.dumps(initial), encoding="utf-8")
    app = Starlette(routes=[Route("/api/settings", endpoint=handlers.api_settings_post, methods=["POST"])])
    app.state.drive_root = app.state.repo_dir = path.parent
    response = TestClient(app).post("/api/settings", json=payload)
    return response, json.loads(path.read_text(encoding="utf-8"))


def test_the_save_keeps_canonical_text_and_refuses_an_engine_that_takes_no_new_work(isolated_runtime, monkeypatch):  # noqa: F811
    from ouroboros import config

    path = config.SETTINGS_PATH
    initial = {"OUROBOROS_ENGINE_CONNECTIONS": ROWS}
    response, saved = _post(monkeypatch, path, initial, {REMOTE_EXECUTION_KEY: {"on": True, "engine": "vps"}})
    assert response.status_code == 200, response.text
    assert saved[REMOTE_EXECUTION_KEY] == '{"engine":"vps","on":true}'
    for value, fragment in (({"on": True, "engine": "gone"}, "No remote engine named 'gone' is saved"),
                            ({"on": True, "engine": "lab"}, "Lab box is switched off in Settings"),
                            ({"on": True}, "on needs the remote engine"),
                            ({"on": "yes", "engine": "vps"}, "on must be true or false"),
                            ({"on": True, "engine": "vps", "account": "x"}, "unknown field(s) account")):
        response, saved = _post(monkeypatch, path, initial, {REMOTE_EXECUTION_KEY: value})
        assert response.status_code == 400 and fragment in response.json()["error"], (value, response.text)
        assert response.json()["code"] == "remote_execution_invalid" and REMOTE_EXECUTION_KEY not in saved
    # Off keeps the remembered engine; a save re-sending the saved switch is not judged again (a removed engine's
    # Remove saves; the work the switch places there is then refused, never moved).
    response, saved = _post(monkeypatch, path, initial, {REMOTE_EXECUTION_KEY: '{"on": false, "engine": "gone"}'})
    assert response.status_code == 200 and saved[REMOTE_EXECUTION_KEY] == '{"engine":"gone","on":false}'
    on = {**initial, REMOTE_EXECUTION_KEY: '{"engine":"vps","on":true}'}
    response, saved = _post(monkeypatch, path, on, {"OUROBOROS_ENGINE_CONNECTIONS": [],
                                                                  REMOTE_EXECUTION_KEY: '{"engine":"vps","on":true}'})
    assert response.status_code == 200 and saved["OUROBOROS_ENGINE_CONNECTIONS"] == []


# --- Main as following work runs it ----------------------------------------------------------------------------------

def _following_ctx(model=LOCAL):
    ctx = SimpleNamespace(task_model_override=None, task_use_local_override=None, task_route_source=None,
                          task_metadata={"lead": {"source": "main"}}, context_fit_plan=None)
    opened = route_choice.open_route(ctx, None)
    ctx.route_choice, ctx.active_model, ctx.active_use_local = opened[2], opened[0], opened[1]
    ctx.primary_route = {"model": opened[0], "use_local": opened[1], "role": "main"}
    return ctx


def test_main_moves_at_the_next_call_and_back_with_its_notice_and_the_state_names_it(owner):
    from ouroboros.gateway.state_publication import produced_fields  # noqa: F401  (the published field's producer)

    ctx = _following_ctx()
    assert (ctx.active_model, route_choice.round_route(ctx, None, (LOCAL, False))) == (LOCAL, None)
    _switch(True)
    moved = route_choice.round_route(ctx, None, (LOCAL, False))
    assert moved == (REMOTE, False), "the next logical call of a running following turn moves"
    route_choice.note_settings_change(ctx, LOCAL, moved[0], "main")
    assert ctx._route_facts_pending.startswith(
        "[ROUTE FACTS] Main moved with my human's Remote switch to the remote engine vps (its own Auto account): ")
    assert route_choice.configured_references()["main"] == {"model": REMOTE, "use_local": False}
    view = remote_placement.state_view()
    assert view["on"] and view["engine"] == "vps" and view["main"]["scope"] == "follows"
    assert view["main"]["effective"] == {"model": REMOTE, "use_local": False} and view["main"]["model"] == LOCAL
    assert [row["name"] for row in view["engines"]] == ["vps", "lab"] and view["engines"][1]["enabled"] is False
    _switch(False)
    back = route_choice.round_route(ctx, None, moved)
    assert back == (LOCAL, False), "Off restores this computer at the next call; nothing else is touched"
    route_choice.note_settings_change(ctx, REMOTE, LOCAL, "main")
    assert "moved with my human's Remote switch to this computer" in ctx._route_facts_pending


@pytest.mark.parametrize("main,local,scope", [("openrouter/anthropic/claude-x", False, "api"),
                                              ("qwen-local", True, "local"),
                                              ("claudexor@lab::codex=gpt-6", False, "engine")])
def test_a_main_the_switch_cannot_move_stays_exactly_as_configured(owner, main, local, scope):
    _save({"OUROBOROS_MODEL": main, "USE_LOCAL_MAIN": local})
    _switch(True)
    assert route_choice.following_route("main") == (main, local)
    view = remote_placement.state_view()
    assert view["main"]["scope"] == scope and view["main"]["effective"] == {"model": main, "use_local": local}


def test_explicit_choices_and_other_roles_keep_their_routes_while_empty_slots_use_main_as_it_runs(owner):
    _save({"OUROBOROS_MODEL_LIGHT": "openai::light", "OUROBOROS_MODEL_CONSCIOUSNESS": ""})
    _switch(True)
    explicit = SimpleNamespace(task_model_override=LOCAL, task_use_local_override=False, task_route_source="room",
                               task_metadata={}, context_fit_plan=None)
    assert route_choice.open_route(explicit, None)[:2] == (LOCAL, False), "a room's own model stays explicit"
    assert route_choice.following_route("light") == ("openai::light", False)
    assert route_choice.following_route("consciousness") == (REMOTE, False), "an empty slot uses Main as it runs"


def test_a_pin_and_a_window_stay_on_their_engine_for_every_reader(owner):
    from ouroboros.llm_substitution import send_account
    from ouroboros.model_slots import (MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, model_role_option,
                                       route_binding, task_model_binding)

    _switch(True)
    assert model_role_option(MODEL_ACCOUNTS_KEY, "main") == "work", "the saved pin stays saved"
    assert model_role_option(MODEL_ACCOUNTS_KEY, "main", model=LOCAL) == "work"
    assert model_role_option(MODEL_ACCOUNTS_KEY, "main", model=REMOTE) == ""
    assert model_role_option(MODEL_CONTEXT_WINDOWS_KEY, "main", model=REMOTE) == 0
    assert model_role_option(MODEL_CONTEXT_WINDOWS_KEY, "main", model=LOCAL) == 400000
    assert route_binding(REMOTE, False, "main")[2] == "" and route_binding(LOCAL, False, "main")[2] == "work"
    local, remote = ({"source": "codex", "resolved_model": "gpt-6", "connection": c, "usage_model": m}
                     for c, m in (("", LOCAL), ("vps", REMOTE)))
    assert send_account(local, [], {"model_role": "main"})[0] == {"mode": "pin", "profileId": "work"}
    assert send_account(remote, [], {"model_role": "main"})[0] == {"mode": "auto"}, "the remote engine's own Auto"
    # A nanny following its parent's Main: the role's pin, read on the engine that role runs on now.
    from ouroboros.subagent_runtime import select_subagent_snapshot

    actor = select_subagent_snapshot({"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [SESSION]})},
                                     subagent_id="builder")[0]
    cognition = {"schema": 2, "model": LOCAL, "effort": "high", "use_local_model": False, "source_role": "main",
                 "credential_profile_id": "", "processing_preference": "", "chosen_by": "inherited", "follows_role": "main"}
    nanny = {"task_metadata": {"configured_subagent": actor, "parent_cognitive_route": cognition}}
    assert task_model_binding(nanny)[1] == ""
    _switch(False)
    assert task_model_binding(nanny)[1] == "work"
    # A Main the owner set to that engine keeps its own pin there and nowhere else.
    _save({"OUROBOROS_MODEL": REMOTE, "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": "remote-work"})})
    assert model_role_option(MODEL_ACCOUNTS_KEY, "main", model=REMOTE) == "remote-work"
    assert model_role_option(MODEL_ACCOUNTS_KEY, "main", model=LOCAL) == ""


def test_the_round_handover_moves_a_running_turn_and_leaves_the_other_engines_turn_behind(owner, monkeypatch):
    """The real round boundary (``loop_routes.bind_round_route``): after the press the next logical call of a turn that
    follows Main is bound to the remote route with its route notice; the conversation's continuations that this
    computer's engine issued stay behind at the switch seam (never sent there), so the send asks that engine for its
    own Auto; Off brings the next call back. The call already in flight is not this boundary's: it finished before."""
    from ouroboros import loop
    from ouroboros.llm_substitution import send_account
    from ouroboros.loop_routes import bind_round_route
    from ouroboros.tools.registry import ToolRegistry

    tools = ToolRegistry(repo_dir=owner, drive_root=owner)
    ctx = tools._ctx
    ctx.task_id, ctx.task_attempt, ctx.task_metadata = "work", 1, {"lead": {"source": "main"}}
    ctx.budget_drive_root, ctx.event_queue = owner, queue.Queue()
    route_choice.bind_task_route(ctx, ctx.task_metadata, presence=False)
    loop._reset_turn_state(ctx)
    model, local, ctx.route_choice = route_choice.open_route(ctx, None)
    ctx.active_model, ctx.active_use_local, ctx.active_effort = model, local, "high"
    ctx.primary_route = {"model": model, "use_local": local, "role": "main"}
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda plan, *_args, **_kwargs: (plan, "max"))
    local_answer = {"source": "codex", "model": "gpt-6", "credentialProfileId": "work", "accountFingerprint": "fp"}
    messages = [{"role": "user", "content": "start"},
                {"role": "assistant", "content": "a", "nativeContinuation": {"route": dict(local_answer)}}]
    assert bind_round_route(tools, messages, (LOCAL, False, "high"), None, "max", "max", [])[:2] == (LOCAL, False)
    _switch(True)
    assert bind_round_route(tools, messages, (LOCAL, False, "high"), None, "max", "max", [])[:2] == (REMOTE, False)
    assert ctx._route_facts_pending.startswith("[ROUTE FACTS] Main moved with my human's Remote switch")
    assert not any("nativeContinuation" in message for message in messages), "this computer's turn stays behind"
    remote = {"source": "codex", "resolved_model": "gpt-6", "connection": "vps", "usage_model": REMOTE}
    assert send_account(remote, messages, {"model_role": "main"})[0] == {"mode": "auto"}
    _switch(False)
    assert bind_round_route(tools, messages, (REMOTE, False, "high"), None, "max", "max", [])[:2] == (LOCAL, False)


def test_a_moved_turn_sizes_on_its_new_engines_own_account_never_one_observed_on_the_old(owner, monkeypatch):
    """Astra review (14.11.2026), through the real round boundary and the real rebind (``_rebind_context_fit_plan``,
    the production route question and probe; only the engines' catalogs are faked, each engine holding a "work" of its
    own). A turn running here carries the account observed serving it ("work", its fingerprint); the switch moves it to
    vps, where the role's pin is cleared, so the vps engine is asked for its own Auto and that window sizes the turn:
    the observation from this computer, matching in source and model, is not restored there. An account observed on
    an engine stays that engine's: a rebind on the same engine keeps it, an explicit receipt is used as given, and Off
    carries nothing of vps's back here, where the role's own pin binds again."""
    from ouroboros import capability_evidence as ce, context, loop
    from ouroboros.context_fit import ContextFitPlan, ContextFitProjection
    from ouroboros.llm import LLMClient
    from ouroboros.loop_routes import bind_round_route
    from ouroboros.tools.registry import ToolRegistry

    accounts = {"": {"work": "fp-mac"}, "vps": {"work": "fp-vps-work", "vps-auto": "fp-vps"}}
    asked, carried = [], []

    def catalog(source, credential_profile_id=None, *, requested_model=None, engine=""):
        asked.append((engine, credential_profile_id))
        profile = credential_profile_id or ("vps-auto" if engine else "work")
        return {"source": source, "credentialProfileId": profile, "accountFingerprint": accounts[engine][profile],
                "observedAt": ce.utc_now_iso(), "provenance": "engine catalog",
                "models": [{"id": requested_model, "contextWindow": 200_000 if engine else 300_000}]}

    question = context._context_fit_route

    def route_question(task, **kwargs):  # the route the probe is asked about, with the observation it carries
        carried.append(dict(task["model_route"] or {}))
        return question(task, **kwargs)

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog), raising=False)
    monkeypatch.setattr(context, "_context_fit_route", route_question)
    tools = ToolRegistry(repo_dir=owner, drive_root=owner)
    ctx = tools._ctx
    ctx.task_id, ctx.task_attempt, ctx.task_metadata = "work", 1, {"lead": {"source": "main"}}
    ctx.budget_drive_root, ctx.event_queue = owner, queue.Queue()
    route_choice.bind_task_route(ctx, ctx.task_metadata, presence=False)
    loop._reset_turn_state(ctx)
    model, local, ctx.route_choice = route_choice.open_route(ctx, None)
    ctx.active_model, ctx.active_use_local, ctx.active_effort = model, local, "high"
    ctx.primary_route = {"model": model, "use_local": local, "role": "main"}
    start = ContextFitPlan(
        core_sha256="a" * 64, preferred_mode="max", initial_mode="max", model=LOCAL, provider="claudexor",
        route_fp="route-a", status="confirmed", stale=False, window_tokens=400_000,
        output_reserve_tokens=65_536, user_content_json=json.dumps("go"),
        max_projection=ContextFitProjection(mode="max", system_content_json=json.dumps("MAX_SYSTEM"),
                                            estimated_tokens=10, calibrated_tokens=10, calibration_ratio=1.0),
        low_projection=ContextFitProjection(mode="low", system_content_json=json.dumps("LOW_SYSTEM"),
                                            estimated_tokens=10, calibrated_tokens=10, calibration_ratio=1.0),
    )
    messages = start.messages_for("max")
    here = {"source": "codex", "model": "gpt-6", "credentialProfileId": "work", "accountFingerprint": "fp-mac"}
    there = {"source": "codex", "model": "gpt-6", "credentialProfileId": "vps-auto", "accountFingerprint": "fp-vps"}
    plan, mode = loop._rebind_context_fit_plan(start, tools, messages, model=LOCAL, use_local=False, tool_schemas=[],
                                               start_mode="max")
    assert (asked, plan.model_route) == ([("", "work")], here), "the turn runs here on the account observed serving it"
    again, mode = loop._rebind_context_fit_plan(plan, tools, messages, model=LOCAL, use_local=False, tool_schemas=[],
                                                start_mode=mode)
    assert (carried[-1], again.model_route) == (here, here), "a rebind on the same engine keeps its observation"
    _switch(True)
    moved, moved_local, _effort, plan, mode = bind_round_route(tools, messages, (LOCAL, False, "high"), again, mode,
                                                               "max", [])
    assert (moved, moved_local) == (REMOTE, False)
    assert (asked[-1], plan.status, plan.window_tokens) == (("vps", None), ce.STATUS_CONFIRMED, 200_000), \
        "the moved turn asks vps for its own Auto and plans by its window, never this computer's account there"
    assert ("vps", "work") not in asked and carried[-1] == {} and (plan.model, plan.model_route) == (REMOTE, there)
    kept, _mode = loop._rebind_context_fit_plan(plan, tools, messages, model=REMOTE, use_local=False, tool_schemas=[],
                                                start_mode=mode)
    assert (carried[-1], kept.model_route) == (there, there), "vps's own observation stays on vps"
    loop._rebind_context_fit_plan(again, tools, messages, model=REMOTE, use_local=False, tool_schemas=[],
                                  start_mode=mode, model_route=dict(there))
    assert carried[-1] == there, "an explicit receipt is the caller's and is used as given"
    _switch(False)
    back, _local, _effort, plan, mode = bind_round_route(tools, messages, (REMOTE, False, "high"), plan, mode, "max", [])
    assert (back, carried[-1], plan.model_route) == (LOCAL, {}, here), "Off: this computer's own pin, nothing of vps's"
    assert ("vps", "work") not in asked and ("", "vps-auto") not in asked, asked


# --- a refused call's wait -------------------------------------------------------------------------------------------

def test_a_refused_main_call_parked_on_its_remote_engine_yields_when_off_moves_it(owner, monkeypatch):
    """The actual wait consumer (``TaskModelWait.wait``, an ordinary round's ``owner_model_wake``): a following Main
    call refused by its remote engine (``connection_unavailable``) waits; Off moves the role, the wait yields
    ``route_moved`` through the existing owner-model wake seam, the loop rejoins its round (no second scheduler) and
    the next round's handover puts the call on this computer. A sent call never polls this."""
    from ouroboros import loop
    from ouroboros.model_wait import ModelWaitInterrupted, TaskModelWait
    from ouroboros.owner_model import waiting_wake

    from ouroboros.task_results import write_task_result

    write_task_result(owner, "work", "scheduled", type="task", chat_id=7, root_task_id="work",
                      metadata={"lead": {"source": "main"}})
    write_task_result(owner, "work", "running")
    _switch(True)
    ctx = _following_ctx()
    assert ctx.active_model == REMOTE
    ctx._route_binding = (REMOTE, False)
    waiter = TaskModelWait(task={"id": "work", "metadata": {}}, drive_root=owner, event_queue=queue.Queue(),
                           worker_slot_held=True, owner_control=lambda: None)
    waiter.tool_context = ctx
    monkeypatch.setattr(waiter, "pre_dispatch_pause", lambda: None)
    assert waiting_wake(waiter) is None, "nothing moved: the refused call keeps waiting"
    polls = []

    def flip_off_once():
        polls.append(True)
        if len(polls) == 1:
            _switch(False)  # the owner's press while the call is parked
        return "test_wait_limit" if len(polls) > 5 else None

    from ouroboros import config
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    def refused(*_args, **_kwargs):  # the engine the call waits for takes no new work: its catalog refuses too
        raise ClaudexorUnavailable("remote_engine_disabled", "The engine connection 'vps' is switched off.")

    monkeypatch.setattr(config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.01)
    llm = SimpleNamespace(claudexor_model_sources=refused, claudexor_model_catalog=refused)
    refusal = RuntimeError("remote engine switched off")
    refusal.problem = {"code": "connection_unavailable"}
    with waiter.register_reprepare("main", lambda values: values, owner_model_wake=True), \
            pytest.raises(ModelWaitInterrupted) as interrupted:
        waiter.wait(llm, refusal, {"model": REMOTE, "model_role": "main", "model_poll_control": flip_off_once})
    assert interrupted.value.control_reason == route_choice.ROUTE_MOVED_WAKE
    assert waiter.control_reason() is None, "a sent call's own poll never yields to the move: it finishes where sent"
    assert loop._handle_model_wait_control(SimpleNamespace(), interrupted.value) is None
    assert route_choice.round_route(ctx, None, (REMOTE, False)) == (LOCAL, False)
    # A turn on its own explicit route is not moved by the switch, so its wait is not woken by it.
    explicit = SimpleNamespace(route_choice=route_choice.explicit("room"), active_model=REMOTE, active_use_local=False)
    assert route_choice.following_moved(explicit) is False


# --- the delegated-start selectors -----------------------------------------------------------------------------------

SESSION = {"subagent_id": "builder", "recommended_use": "Builds.", "effort": "high",
           "route": {"kind": "agent_session", "target_id": "codex=gpt-6", "credential_profile_id": "work"}}
EXPLICIT = {"subagent_id": "remote-builder", "recommended_use": "Builds there.", "effort": "high",
            "route": {"kind": "agent_session", "target_id": "codex=gpt-6", "credential_profile_id": "work",
                      "engine": "vps"}}
OTHER = {"subagent_id": "other", "recommended_use": "Another pin.", "effort": "high",
         "route": {"kind": "agent_session", "target_id": "codex=gpt-6", "credential_profile_id": "spare"}}
API = {"subagent_id": "api", "recommended_use": "Thinks.", "effort": "high",
       "route": {"kind": "api_model", "target_id": LOCAL, "credential_profile_id": "work"}}
REVIEWER = {**API, "subagent_id": "reviewer", "review_eligible": True}


def _roster(*rows):
    _save({"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": list(rows)})})


def _scheduled(root, subagent):
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=root.parent / "repo", drive_root=root, is_direct_chat=True)
    ctx.task_id, ctx.task_depth, ctx.current_chat_id = "main", 0, 1
    ctx.event_queue, ctx.task_metadata = queue.Queue(), {"root_task_id": "main"}
    answer = _schedule_task(ctx, subagent_id=subagent, objective="Build", expected_output="A patch")
    assert "error" not in answer.lower(), answer
    return ctx.event_queue.get_nowait()["configured_subagent"]


def test_a_parent_that_began_before_the_press_places_the_helpers_it_starts_after_it(owner, monkeypatch):
    from ouroboros.config import load_settings
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot

    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")
    _roster(SESSION, EXPLICIT, API, REVIEWER)
    before = load_settings()  # the parent's task snapshot: taken while Remote was Off
    with task_settings_scope(task_settings_snapshot(before, {})):
        assert _scheduled(owner, "builder")["route"] == SESSION["route"], "Off: the row as saved"
        _switch(True)
        placed = _scheduled(owner, "builder")
        assert placed["route"] == {"kind": "agent_session", "target_id": "codex=gpt-6", "credential_profile_id": "",
                                   "engine": "vps"}
        assert placed["placement"] == {"by": "remote_switch", "engine": "vps", "row_account": "work"}
        assert _scheduled(owner, "api")["route"]["target_id"] == REMOTE, "an API row on a local subscription route"
        explicit = _scheduled(owner, "remote-builder")
        assert explicit["route"] == EXPLICIT["route"] and "placement" not in explicit, "a row naming its engine keeps it"
    # The saved roster and its handles are never rewritten, and a reviewer's selection is not this overlay's.
    from ouroboros.subagent_runtime import select_subagent_snapshot

    saved = json.loads(load_settings()["OUROBOROS_SUBAGENTS"])
    assert saved["items"][0]["route"] == SESSION["route"]
    assert select_subagent_snapshot(load_settings(), subagent_id="reviewer")[0]["route"]["target_id"] == LOCAL
    from ouroboros.subagent_history import snapshot_handle

    assert snapshot_handle(placed) == "codex=gpt-6/on:vps/high", "history names where the helper ran"
    assert "on:" not in snapshot_handle({**placed, "route": SESSION["route"]})


def test_the_agents_catalog_says_where_a_new_start_runs_and_what_that_engine_refuses(owner):
    from ouroboros.subagent_runtime import current_model_visible_subagent_catalog

    _roster(SESSION, EXPLICIT)
    assert "remote_switch" not in current_model_visible_subagent_catalog()
    _switch(True)
    catalog = current_model_visible_subagent_catalog()
    rows = {row["subagent_id"]: row for row in catalog["rows"]}
    assert rows["codex=gpt-6/high/@work"]["runs_on"] == "vps" and "runs_on" not in rows["codex=gpt-6/on:vps/high/@work"]
    assert catalog["remote_switch"]["refused_there"] == list(remote_placement.REMOTE_SESSION_LIMITS)


def _selected(root, monkeypatch, subagent, continue_from=""):
    """``delegate_start``'s exact selection (``subagent_runtime.exact_start``) up to its frozen snapshot."""
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    seen = {}

    def captured(*_args, **_kwargs):
        from ouroboros.delegate_shared import delegate_result

        seen.update(runtime._EXACT_START_SELECTION.get() or {})
        return delegate_result({"status": "started", "run_id": "next"})

    monkeypatch.setattr(delegate, "_delegate_start", captured)
    monkeypatch.setattr("ouroboros.delegate_plan.inherited_mode", lambda *_a: None)
    monkeypatch.setattr("ouroboros.delegate_continuation.inherited_access", lambda *_a: None)
    ctx = ToolContext(repo_dir=root.parent / "repo", drive_root=root)
    ctx.task_id = "parent"
    spec = {"subagent_id": subagent, **({"continue_from": continue_from} if continue_from else {})}
    answer = json.loads(runtime.exact_start(ctx, "Build", spec).text)
    assert answer["status"] == "started", answer
    snapshot = seen["snapshot"]
    route = runtime.exact_session_binding(snapshot)[1]
    return snapshot, {"engine": route.engine, "profile_id": route.profile_id}


def _predecessor(root, run_id, *, connection="", profile=""):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import run

    ref = ({"kind": "remote", "connection": connection, "ssh_host": "vds", "engine_home": "~/.claudexor/v3",
            "store": "store-1", "remote_root": "/home/o/runs"} if connection else {})
    assert custody.record_started(root, custody.RunCustody(run_id=run_id, task_id="parent", invocation_id=f"inv-{run_id}",
                                                           route_id="codex", model="gpt-6", profile_id=profile,
                                                           engine_ref=ref))
    assert run(root, run_id).engine_ref == ref and run(root, run_id).profile_id == profile  # the recorded route truth


def test_a_fresh_direct_start_follows_the_switch_and_a_continuation_never_does(owner, monkeypatch):
    """The four continuation cases, both directions, with the SAME profile name ("work") on both engines: a
    continuation runs where its predecessor ran (``execution_domain.start_engine``), so the switch never places it;
    only a predecessor the switch placed remotely makes a local row ask there for what the predecessor asked
    (Auto), never this computer's "work"."""
    _roster(SESSION, EXPLICIT, OTHER)
    _switch(True)
    assert _selected(owner, monkeypatch, "builder")[1] == {"engine": "vps", "profile_id": ""}, "a fresh start"
    # (1) a local predecessor and a row saved local: the row's own pin, On or Off.
    _predecessor(owner, "local-run", profile="work")
    assert _selected(owner, monkeypatch, "builder", "local-run")[1] == {"engine": "", "profile_id": "work"}
    _switch(False)
    assert _selected(owner, monkeypatch, "builder", "local-run")[1] == {"engine": "", "profile_id": "work"}
    # (2) another row deliberately chosen on the same engine: its own pin, as before.
    assert _selected(owner, monkeypatch, "other", "local-run")[1] == {"engine": "", "profile_id": "spare"}
    # (3) a predecessor the switch placed on vps (it asked Auto there) and a row saved local: never the local "work",
    # Off or On.
    _predecessor(owner, "placed-run", connection="vps", profile="")
    snapshot, route = _selected(owner, monkeypatch, "builder", "placed-run")
    assert route == {"engine": "vps", "profile_id": ""}
    assert snapshot["placement"] == {"by": "predecessor", "engine": "vps", "row_account": "work"}
    _switch(True)
    assert _selected(owner, monkeypatch, "builder", "placed-run")[1] == {"engine": "vps", "profile_id": ""}
    # (4) a row saved on that remote engine: its own pin there.
    assert _selected(owner, monkeypatch, "remote-builder", "placed-run")[1] == {"engine": "vps", "profile_id": "work"}
    # A remote predecessor that asked for a remote account keeps that request, not the local row's same-named one.
    _predecessor(owner, "pinned-run", connection="vps", profile="vps-account")
    assert _selected(owner, monkeypatch, "builder", "pinned-run")[1] == {"engine": "vps", "profile_id": "vps-account"}
    # A row naming ANOTHER engine than its predecessor ran on, either way: the continuation runs where the predecessor
    # ran, so it asks there for what the predecessor asked, never the other engine's pin ("work" on vps).
    _predecessor(owner, "lab-run", connection="lab", profile="")
    snapshot, route = _selected(owner, monkeypatch, "remote-builder", "lab-run")
    assert route == {"engine": "lab", "profile_id": ""}
    assert snapshot["placement"] == {"by": "predecessor", "engine": "lab", "row_account": "work", "row_engine": "vps"}
    _predecessor(owner, "home-run", profile="home")
    snapshot, route = _selected(owner, monkeypatch, "remote-builder", "home-run")
    assert route == {"engine": "", "profile_id": "home"} and "engine" not in snapshot["route"]
    # Every case: the snapshot's engine is the one the start keeps, the predecessor's recorded binding
    # (``execution_domain.start_engine``), so an account never crosses to another engine.
    from ouroboros.delegate_custody_current import run
    from ouroboros.execution_domain import recorded_ref

    for subagent, run_id in (("builder", "local-run"), ("other", "local-run"), ("builder", "placed-run"),
                             ("remote-builder", "placed-run"), ("builder", "pinned-run"),
                             ("remote-builder", "lab-run"), ("remote-builder", "home-run")):
        kept = str(recorded_ref(run(owner, run_id)).get("connection") or "")
        assert _selected(owner, monkeypatch, subagent, run_id)[1]["engine"] == kept, (subagent, run_id)


def test_a_main_saved_while_on_is_answered_where_it_runs_and_saved_as_chosen(owner):
    """``POST /api/owner/chat-model`` answers Main where following work runs it (the composer shows that at once)
    and saves the owner's route itself: the switch never writes an engine into Main."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config
    from ouroboros.gateway.owner_effort import api_owner_chat_model

    _switch(True)
    app = Starlette(routes=[Route("/api/owner/chat-model", api_owner_chat_model, methods=["POST"])])
    app.state.drive_root = owner
    reply = TestClient(app).post("/api/owner/chat-model", json={"kind": "model", "model": "claudexor::codex=gpt-6-luna"})
    assert reply.status_code == 200, reply.text
    assert reply.json()["main_route"] == {"model": "claudexor@vps::codex=gpt-6-luna", "use_local": False}
    assert reply.json()["previous_main_route"] == {"model": REMOTE, "use_local": False}
    assert json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))["OUROBOROS_MODEL"] == "claudexor::codex=gpt-6-luna"
