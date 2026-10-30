"""The room's chat model (DESIGN "Composer owner controls"; ``composer_cognition``, ``route_choice``).

A room saves its own choice — the configured Main's or first Fallback's reference, or one raw route — through
``POST /api/owner/chat-model``: Main in its own settings key (never a role slot, never exported to an environment), a
Project on its registry row, neither copied to the other, Light and every role slot untouched. New owner work
snapshots it once at admission, Main's reference recorded as such: a Continue and a same-room ``start_task`` keep the
snapshot, a later picker change never retargets admitted work, delegated and Presence work and call turns never carry
one. A chosen route binds as the task's own route on role ``room`` — Auto account, route-default processing, Auto
window, from the first context plan on — never Main's pins; the Fallback reference runs on ``fallback:0`` with that
entry's options.
"""
from __future__ import annotations

import json
import os
import types

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import config, projects_registry as projects
from ouroboros.composer_cognition import (
    MAIN_CHAT_MODEL_KEY, normalize_model_choice, read_model_choice,
)
from ouroboros.gateway.owner_effort import api_owner_chat_model
from ouroboros.model_slots import (
    MODEL_CONTEXT_WINDOWS_KEY, model_role_option, task_model_binding, task_processing_preference,
)
from ouroboros import route_choice
from tests.test_task_settings_snapshot import isolated_runtime, save_owner  # noqa: F401 -- the shared isolated root

ASTRA = "claudexor::codex=gpt-6-astra"
SOL = "claudexor::codex=gpt-6.1-sol"
LUNA = "claudexor::codex=gpt-6-luna"
OPUS = "anthropic::claude-opus-5-5"
OWNER = {
    "OUROBOROS_MODEL": SOL, "OUROBOROS_MODEL_LIGHT": "openai::light-model", "OUROBOROS_MODEL_FALLBACKS": LUNA,
    # Main's own pins, processing and window: none of them may reach a route the room chose for itself.
    "OUROBOROS_MODEL_ACCOUNTS": json.dumps({"main": "acc-main", "fallback": ["acc-fallback"]}),
    "OUROBOROS_MODEL_PROCESSING_PREFERENCES": json.dumps({"main": "fast", "fallback": ["economy"]}),
    "OUROBOROS_MODEL_CONTEXT_WINDOWS": json.dumps({"main": 123456, "fallback": [654321]}),
}
CHOSEN = {"kind": "model", "model": ASTRA, "use_local": False}


@pytest.fixture
def room(isolated_runtime):  # noqa: F811 -- the imported fixture, used by name
    save_owner(OWNER)
    for pid in ("alpha", "beta"):
        projects.create_project(isolated_runtime, pid, origin="owner_ui")
    app = Starlette(routes=[Route("/api/owner/chat-model", api_owner_chat_model, methods=["POST"])])
    app.state.drive_root = isolated_runtime
    return isolated_runtime, TestClient(app)


def _saved() -> dict:
    return json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))


def _registry_row(root, pid: str) -> dict:
    data = json.loads((root / "state/projects.json").read_text(encoding="utf-8"))
    return next(row for row in data["projects"] if row["id"] == pid)


def test_the_choice_is_checked_by_the_route_grammar_never_by_a_catalog():
    assert normalize_model_choice("") == normalize_model_choice(None) == {"kind": "main"}
    assert normalize_model_choice({"kind": "fallback", "model": "ignored"}) == {"kind": "fallback"}
    # An unlisted model, a named connection and a bare OpenRouter id are routes like any other.
    for model in ("claudexor::codex=gpt-7-unreleased", "openai-compatible@work::my/model", "vendor/model"):
        assert normalize_model_choice({"kind": "model", "model": f" {model} "}) == {"kind": "model", "model": model, "use_local": False}
    assert normalize_model_choice({"kind": "model", "model": "qwen3 (local)"}) == {"kind": "model", "model": "qwen3", "use_local": True}
    for bad in ({"kind": "pin"}, {"kind": "model", "model": ""}, {"kind": "model", "model": "claudexor::codex"},
                {"kind": "model", "model": "claudexor@x::codex=m"}, {"kind": "model", "model": "a\nb"},
                {"kind": "model", "model": ASTRA, "use_local": True}, {"kind": "model", "model": ASTRA, "use_local": "yes"},
                "not json", [CHOSEN]):
        with pytest.raises(ValueError):
            normalize_model_choice(bad)


def test_main_and_each_project_keep_their_own_choice_and_no_role_slot_moves(room):
    root, client = room
    reply = client.post("/api/owner/chat-model", json={"kind": "model", "model": ASTRA})
    assert reply.status_code == 200, reply.text
    assert reply.json() == {"ok": True, "project_id": "", "model_choice": CHOSEN, "previous_model_choice": {"kind": "main"}}
    saved = _saved()
    assert json.loads(saved[MAIN_CHAT_MODEL_KEY]) == CHOSEN
    assert {key: saved[key] for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS")} == {
        "OUROBOROS_MODEL": SOL, "OUROBOROS_MODEL_LIGHT": "openai::light-model", "OUROBOROS_MODEL_FALLBACKS": LUNA}
    assert MAIN_CHAT_MODEL_KEY not in os.environ and MAIN_CHAT_MODEL_KEY not in config.settings_env_keys()
    assert config.get_light_model() == "openai::light-model"

    reply = client.post("/api/owner/chat-model", json={"kind": "fallback", "project_id": "alpha"})
    assert reply.status_code == 200 and reply.json()["model_choice"] == {"kind": "fallback"}
    assert _registry_row(root, "alpha")["model_choice"] == {"kind": "fallback"}
    assert "model_choice" not in _registry_row(root, "beta")
    assert read_model_choice(root) == CHOSEN and read_model_choice(root, "alpha") == {"kind": "fallback"}
    assert read_model_choice(root, "beta") == {"kind": "main"}
    # A new room starts on Main's reference: Main chat's own choice is never copied into it.
    projects.create_project(root, "gamma", origin="owner_ui")
    assert read_model_choice(root, "gamma") == {"kind": "main"}
    rows = {row["id"]: row["model_choice"] for row in projects.projects_summary(root, limit=10)}
    assert rows == {"alpha": {"kind": "fallback"}, "beta": {"kind": "main"}, "gamma": {"kind": "main"}}

    # The reset is Main's reference again: the row forgets, the settings key goes blank.
    assert client.post("/api/owner/chat-model", json={"kind": "main", "project_id": "alpha"}).json()["previous_model_choice"] == {"kind": "fallback"}
    assert "model_choice" not in _registry_row(root, "alpha")
    assert client.post("/api/owner/chat-model", json={"kind": "main"}).json()["model_choice"] == {"kind": "main"}
    assert _saved()[MAIN_CHAT_MODEL_KEY] == ""


@pytest.mark.parametrize("body,status,code", [
    ({"kind": "pinned"}, 400, "chat_model_invalid"),
    ({"kind": "model", "model": "claudexor::codex"}, 400, "chat_model_invalid"),
    ({"kind": "model", "model": ASTRA, "project_id": "../alpha"}, 400, "chat_model_invalid"),
    ({"kind": "model", "model": ASTRA, "project_id": "nowhere"}, 404, "project_unavailable"),
])
def test_a_refused_choice_says_nothing_was_saved(room, body, status, code):
    root, client = room
    before = (_saved(), json.loads((root / "state/projects.json").read_text(encoding="utf-8")))
    reply = client.post("/api/owner/chat-model", json=body)
    assert reply.status_code == status and reply.json()["saved"] is False and reply.json()["code"] == code, reply.text
    assert (_saved(), json.loads((root / "state/projects.json").read_text(encoding="utf-8"))) == before


def test_admission_snapshots_the_room_once_and_records_mains_reference(room, monkeypatch):
    root, client = room
    main_task, project_task = {"id": "m1"}, {"id": "p1", "project_id": "alpha"}
    route_choice.capture_room_route(main_task, root)
    assert main_task["metadata"]["room_route"] == {"kind": "main"}  # recorded, so nothing later re-reads the room
    client.post("/api/owner/chat-model", json={"kind": "model", "model": ASTRA})
    client.post("/api/owner/chat-model", json={"kind": "fallback", "project_id": "alpha"})
    route_choice.capture_room_route(main_task, root)  # already admitted: the newer choice never reaches it
    assert main_task["metadata"]["room_route"] == {"kind": "main"}
    later = {"id": "m2"}
    route_choice.capture_room_route(later, root)
    route_choice.capture_room_route(project_task, root)
    assert later["metadata"]["room_route"] == CHOSEN
    assert project_task["metadata"]["room_route"] == {"kind": "fallback", "model": LUNA, "use_local": False}
    client.post("/api/owner/chat-model", json={"kind": "model", "model": OPUS})
    assert later["metadata"]["room_route"] == CHOSEN  # a snapshot, never a live reference to the picker

    save_owner({**OWNER, "OUROBOROS_MODEL_FALLBACKS": ""})  # the first Fallback is gone: the reference admits on Main
    no_fallback = {"id": "p2", "project_id": "alpha"}
    route_choice.capture_room_route(no_fallback, root)
    assert no_fallback["metadata"]["room_route"] == {"kind": "main"}

    gone = {"id": "p3", "project_id": "nowhere"}  # a room that cannot be read admits the work on Main's reference
    route_choice.capture_room_route(gone, root)
    assert gone["metadata"]["room_route"] == {"kind": "main"}

    def broken(*_args, **_kwargs):
        raise TypeError("a programming error is never mistaken for an unreadable room")
    monkeypatch.setattr("ouroboros.composer_cognition.read_model_choice", broken)
    with pytest.raises(TypeError):
        route_choice.capture_room_route({"id": "m3"}, root)


def test_continue_and_a_same_room_start_task_keep_the_admitted_snapshot(room, monkeypatch):
    root, client = room
    from ouroboros.tools.control import _start_task
    from supervisor.continuation_admission import _successor_task
    import supervisor.workers as workers

    for admitted in ({"kind": "main"}, CHOSEN):
        successor = _successor_task(None, "previous", {"objective": "The work", "metadata": {"room_route": admitted}},
                                    {"successor_task_id": "next", "chat_id": 1}, {"cause": "budget"},
                                    {"original": {"content": "The work"}}, {})
        assert successor["metadata"]["room_route"] == admitted
    client.post("/api/owner/chat-model", json={"kind": "model", "model": OPUS})  # the picker moves after admission

    monkeypatch.setattr("ouroboros.tools.control_events._wait_for_promotion_admission",
                        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": ""})
    events = []
    ctx = types.SimpleNamespace(pending_events=events, event_queue=None, current_chat_id=1, drive_root=root,
                                project_id="", task_id="owner-turn", task_metadata={"room_route": {"kind": "main"}})
    assert _start_task(ctx, "Do the long part", title="Long part", workspace="none").startswith("OK: task")
    assert events[0]["room_route"] == {"kind": "main"}
    _start_task(ctx, "Elsewhere", title="Elsewhere", project_id="beta")
    assert "room_route" not in events[1]  # another room's new work starts on that room's choice

    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    enqueued = []
    host = types.SimpleNamespace(enqueue_task=lambda task: enqueued.append(task) or True,
                                 persist_queue_snapshot=lambda **_kwargs: True, load_state=lambda: {"owner_chat_id": 1})
    assert workers.promote_chat_to_task(events[0], host)["status"] == "scheduled"
    assert enqueued[-1]["metadata"]["room_route"] == {"kind": "main"}  # the same request, the model it was admitted on
    fresh = {key: value for key, value in events[0].items() if key != "room_route"}
    assert workers.promote_chat_to_task({**fresh, "task_id": "host-started", "routing_token": "host-token"},
                                        host)["status"] == "scheduled"
    assert enqueued[-1]["metadata"]["room_route"] == {"kind": "model", "model": OPUS, "use_local": False}


def test_a_call_turn_keeps_its_engine_while_a_typed_turn_takes_the_rooms_choice(room, monkeypatch):
    # A live call is admitted through the same direct door, already carrying its engine's route: the room's chat
    # model never tags it, so nothing downstream reads the room's route as the one the call runs on.
    import queue
    import threading

    from ouroboros import call_protocol as cp
    from ouroboros.call_peers import close_hub, open_hub
    from ouroboros.live_conversation import LiveConversationTurn
    from supervisor import worker_chat_lane, workers

    _root, client = room
    client.post("/api/owner/chat-model", json={"kind": "model", "model": ASTRA})
    turn = object.__new__(LiveConversationTurn)  # no provider/configuration reads (test_call_task_lifecycle)
    turn.spec = cp.CallSpec("room-call", 1, "synthetic", "synthetic")
    turn.model, turn.request = "synthetic::voice", {"kind": "live_conversation", "call_id": "room-call"}
    turn._lock, turn._closed = threading.Lock(), False
    open_hub(turn.spec)
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "")
    monkeypatch.setattr(workers, "get_event_q", queue.Queue)
    monkeypatch.setattr(workers, "send_with_budget", lambda *args, **kwargs: None)
    agent = types.SimpleNamespace(handle_task=lambda task=None: [])
    try:
        call = worker_chat_lane._admit_chat_task(agent, 1, "Opening", task_metadata={
            "execution_transport": turn.request}, realtime_turn=turn)
        assert call is not None and worker_chat_lane._execute_chat_task(call)
        typed = worker_chat_lane._admit_chat_task(agent, 1, "Hello")
        assert typed is not None and worker_chat_lane._execute_chat_task(typed)
    finally:
        close_hub(turn.spec.call_id, "fixture_cleanup")
    assert call["task"]["model"] == "synthetic::voice"
    assert route_choice.ROOM_ROUTE_KEY not in call["task"]["metadata"]
    assert typed["task"]["metadata"][route_choice.ROOM_ROUTE_KEY] == CHOSEN


def _ctx(metadata: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(task_metadata=metadata, task_model_override=None, task_use_local_override=None,
                                 task_route_source=None, primary_route=None)


def test_a_chosen_route_runs_on_auto_and_route_defaults_from_its_first_plan(room):
    from ouroboros.agent_dispatch import main_model_named_effort
    from ouroboros.context_fit import _context_route

    metadata = {"room_route": CHOSEN}
    ctx = _ctx(metadata)
    route_choice.bind_task_route(ctx, metadata, presence=False)
    assert (ctx.task_model_override, ctx.task_use_local_override, ctx.task_route_source) == (ASTRA, False, "room_choice")
    assert route_choice.open_route(ctx, None) == (ASTRA, False, {"kind": "explicit", "source": "room_choice"})
    # Main's account pin, processing and window stay Main's: the room's route has none of its own.
    assert task_model_binding({"task_metadata": metadata}) == ("room", "")
    assert task_processing_preference({"task_metadata": metadata}, model_role="room") == ""
    assert model_role_option(MODEL_CONTEXT_WINDOWS_KEY, "room") == 0
    plan_route, _settings = _context_route({"metadata": metadata})
    assert (plan_route["model"], plan_route["model_role"], plan_route["options"]["credential_profile_id"]) == (ASTRA, "room", "")
    assert main_model_named_effort({"room_route": {**CHOSEN, "model": "claudexor::cursor=gpt-5-high"}}) == "high"
    assert main_model_named_effort(metadata) == ""  # the carried model names no level; Main's is never consulted

    fallback = {"room_route": {"kind": "fallback", "model": LUNA, "use_local": False}}
    assert task_model_binding({"task_metadata": fallback}) == ("fallback:0", None)  # its entry's own account
    assert task_processing_preference({"task_metadata": fallback}, model_role="fallback:0") == "economy"
    plan_route, _settings = _context_route({"metadata": fallback})
    assert (plan_route["model"], plan_route["model_role"], plan_route["options"]["credential_profile_id"]) == (
        LUNA, "fallback:0", "acc-fallback")
    assert model_role_option(MODEL_CONTEXT_WINDOWS_KEY, "fallback:0") == 654321

    main = {"room_route": {"kind": "main"}}
    ctx = _ctx(main)
    route_choice.bind_task_route(ctx, main, presence=False)
    assert ctx.task_model_override is None and task_model_binding({"task_metadata": main}) == ("main", None)
    assert _context_route({"metadata": main})[0]["model"] == SOL

    # Delegated and Presence work never take a room's route, whatever their metadata carries.
    for meta, presence in (({**metadata, "delegation_role": "subagent", "model": OPUS}, False), (metadata, True)):
        ctx = _ctx(meta)
        route_choice.bind_task_route(ctx, meta, presence=presence)
        assert ctx.task_route_source != "room_choice"
    assert task_model_binding({"task_metadata": {**metadata, "delegation_role": "subagent"}})[0] == "main"


def test_a_connected_direct_openai_route_chosen_in_a_room_runs_through_the_openai_connection(room):
    # The picker offers OpenAI only when its key lists models (owner 30.10: connected sources only); once chosen, the
    # room's route is the existing provider path's, with no picker-specific transport.
    from ouroboros.context_fit import _context_route
    from ouroboros.llm_routing import resolve_target

    root, client = room
    save_owner({**OWNER, "OPENAI_API_KEY": "sk-test-not-a-real-key"})
    chosen = {"kind": "model", "model": "openai::gpt-6.1", "use_local": False}
    assert client.post("/api/owner/chat-model", json={**chosen, "project_id": "alpha"}).json()["model_choice"] == chosen
    task = {"id": "p-openai", "project_id": "alpha"}
    route_choice.capture_room_route(task, root)
    assert task["metadata"]["room_route"] == chosen
    plan_route, settings = _context_route({"metadata": task["metadata"]})
    assert (plan_route["provider"], plan_route["model"], plan_route["model_role"]) == ("openai", "openai::gpt-6.1", "room")
    target = resolve_target(plan_route["model"], settings)
    assert (target["base_url"], target["api_key"]) == ("https://api.openai.com/v1", "sk-test-not-a-real-key")


def test_a_room_route_on_a_wait_card_is_switched_for_the_task_never_persisted_into_settings():
    from ouroboros.model_slots import apply_model_role_override

    with pytest.raises(ValueError, match="composer"):
        apply_model_role_override(dict(OWNER), role="room", model=OPUS, credential_profile_id="", use_local=False)


def test_both_references_resolve_from_the_live_settings(room):
    assert route_choice.configured_references() == {
        "main": {"model": SOL, "use_local": False}, "fallback": {"model": LUNA, "use_local": False}}
    save_owner({**OWNER, "OUROBOROS_MODEL_FALLBACKS": ""})
    assert route_choice.configured_references()["fallback"] is None


def test_the_main_chat_key_is_never_exported_and_an_unreadable_value_reads_as_mains_reference():
    from ouroboros.composer_cognition import main_model_choice, model_choice_setting

    assert MAIN_CHAT_MODEL_KEY in config.SETTINGS_KEYS_NOT_EXPORTED_TO_ENV
    assert json.loads(model_choice_setting({"kind": "model", "model": ASTRA})) == CHOSEN
    assert model_choice_setting({"kind": "main"}) == ""
    for unreadable in ("garbage", '{"kind": "pin"}', "{'kind': 'model'}", ""):
        assert main_model_choice({MAIN_CHAT_MODEL_KEY: unreadable}) == {"kind": "main"}


def test_stars_are_bounded_route_strings_and_never_checked_against_a_catalog():
    from ouroboros.gateway.ui_preferences import DEFAULT_UI_PREFERENCES, _normalize_preferences

    assert DEFAULT_UI_PREFERENCES["model_favorites"] == []
    assert _normalize_preferences({"model_favorites": [f" {ASTRA} ", ASTRA, OPUS, "", "unknown/model"]},
                                  fill_defaults=False) == {"model_favorites": [ASTRA, OPUS, "unknown/model"]}
    assert _normalize_preferences({"model_favorites": None}, fill_defaults=False) == {"model_favorites": []}
    for bad in ("one", [1], ["a\tb"], ["x" * 401], [f"vendor/model-{i}" for i in range(201)]):
        with pytest.raises(ValueError):
            _normalize_preferences({"model_favorites": bad}, fill_defaults=False)


@pytest.mark.parametrize("carrier", [{"presence": {"binding_id": "fixture-place"}},
                                    {"presence_binding_authority": {"binding_id": "fixture-place"}}])
@pytest.mark.parametrize("carried", [None, CHOSEN])
def test_presence_promotions_never_copy_or_capture_the_main_chat_route(room, monkeypatch, carrier, carried):
    from supervisor import workers

    root, client = room
    client.post("/api/owner/chat-model", json={"kind": "model", "model": OPUS})
    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    enqueued = []
    host = types.SimpleNamespace(enqueue_task=lambda task: enqueued.append(task) or True,
                                 persist_queue_snapshot=lambda **_kwargs: True, load_state=lambda: {"owner_chat_id": 1})
    event = {"task_id": "presence-promotion", "routing_token": "fixture-token", "objective": "Inspect the report",
             "chat_id": 1, "workspace": "none", **carrier}
    if carried is not None:
        event[route_choice.ROOM_ROUTE_KEY] = carried
    assert workers.promote_chat_to_task(event, host)["status"] == "scheduled"
    metadata = enqueued[-1].get("metadata", {})
    assert route_choice.ROOM_ROUTE_KEY not in metadata
    ctx = _ctx(metadata)
    route_choice.bind_task_route(ctx, metadata, presence=False)
    assert route_choice.open_route(ctx, None)[:2] == (SOL, False)


@pytest.mark.parametrize("address,metadata,expected", [
    ({"project_id": "alpha"}, {}, CHOSEN),
    ({"project_id": "beta"}, {}, {"kind": "main"}),
    ({"chat_id": 1, "source": "web"}, {}, {"kind": "model", "model": OPUS, "use_local": False}),
    ({}, {}, None),  # A hidden, unbound API task has no chat whose choice to borrow.
    ({"project_id": "alpha"}, {"model_role": "light"}, None),
    ({"project_id": "alpha"}, {"room_route": {"kind": "main"}}, {"kind": "main"}),
])
def test_api_room_admission_reads_the_canonical_choice_not_the_fork(room, tmp_path, monkeypatch, address, metadata, expected):
    from ouroboros.gateway.tasks import api_tasks_create
    from ouroboros.task_results import load_task_result
    from supervisor import workers

    root, client = room
    client.post("/api/owner/chat-model", json={"kind": "model", "model": OPUS})
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    repo = tmp_path / "api-repo"
    repo.mkdir()
    app = Starlette(routes=[Route("/api/tasks", api_tasks_create, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = root, repo
    monkeypatch.setattr(workers, "WORKERS", {0: types.SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    admitted = []
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: admitted.append(task) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    response = TestClient(app).post("/api/tasks", json={"description": "Read the report", "memory_mode": "empty",
                                                     "metadata": metadata, **address})
    assert response.status_code == 200, response.text
    task = admitted[-1]
    assert task["drive_root"] != str(root), "exercise the distinct empty execution drive"
    assert task["metadata"].get(route_choice.ROOM_ROUTE_KEY) == expected
    stored = load_task_result(root, response.json()["task_id"])
    assert stored["metadata"].get(route_choice.ROOM_ROUTE_KEY) == expected
    # Later room edits do not retarget the admission or its durable continuation source.
    client.post("/api/owner/chat-model", json={"kind": "main", "project_id": "alpha"})
    assert task["metadata"].get(route_choice.ROOM_ROUTE_KEY) == expected
