"""Room choices, Main's writer and one-time canonical lead admission.

A chosen primary uses an option-less role from the first plan; a recorded follow
tracks Main live. Every separately started owner root reads its own destination.
"""
from __future__ import annotations

import json
import types

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import config, projects_registry as projects
from ouroboros.composer_cognition import (
    normalize_model_choice, read_model_choice,
)
from ouroboros.gateway.owner_effort import api_owner_chat_model
from ouroboros.model_slots import (
    MODEL_CONTEXT_WINDOWS_KEY, model_role_option, task_model_binding, task_processing_preference,
)
from ouroboros import route_choice
from tests._task_settings_helpers import isolated_runtime, save_owner  # noqa: F401 -- the shared isolated root

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
LEAD = {"source": "room", "model": ASTRA, "use_local": False}


@pytest.fixture
def room(isolated_runtime, monkeypatch):  # noqa: F811 -- the imported fixture, used by name
    from supervisor import workers

    monkeypatch.setattr(workers, "WORKERS", {0: types.SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "_worker_pool_execution_state",
                        lambda *_a, **_kw: {"available": True, "disabled_reason": "", "worker_count": 1})
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
    # An unlisted model, a named connection and a bare OpenRouter id are routes like any other.
    for model in ("claudexor::codex=gpt-7-unreleased", "openai-compatible@work::my/model", "vendor/model"):
        assert normalize_model_choice({"kind": "model", "model": f" {model} "}) == {"kind": "model", "model": model, "use_local": False}
    assert normalize_model_choice({"kind": "model", "model": "qwen3 (local)"}) == {"kind": "model", "model": "qwen3", "use_local": True}
    for bad in ({"kind": "fallback"}, {"kind": "pin"}, {"kind": "model", "model": ""}, {"kind": "model", "model": "claudexor::codex"},
                {"kind": "model", "model": "claudexor@x::codex=m"}, {"kind": "model", "model": "a\nb"},
                {"kind": "model", "model": ASTRA, "use_local": True}, {"kind": "model", "model": ASTRA, "use_local": "yes"},
                "not json", [CHOSEN]):
        with pytest.raises(ValueError):
            normalize_model_choice(bad)


def test_main_and_each_project_keep_their_own_choice(room):
    root, client = room
    reply = client.post("/api/owner/chat-model", json={"kind": "model", "model": ASTRA})
    assert reply.status_code == 200, reply.text
    assert reply.json() == {"ok": True, "project_id": "", "main_route": {"model": ASTRA, "use_local": False},
                            "previous_main_route": {"model": SOL, "use_local": False},
                            "account": "kept", "window_reset": True}
    saved = _saved()
    assert saved["OUROBOROS_MODEL"] == ASTRA
    assert saved["OUROBOROS_MODEL_LIGHT"] == "openai::light-model"
    assert saved["OUROBOROS_MODEL_FALLBACKS"] == LUNA
    assert config.get_light_model() == "openai::light-model"
    reply = client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    assert reply.status_code == 200 and reply.json()["model_choice"] == CHOSEN
    assert _registry_row(root, "alpha")["model_choice"] == CHOSEN
    assert "model_choice" not in _registry_row(root, "beta")
    assert read_model_choice(root) == {"kind": "main"}
    assert read_model_choice(root, "alpha") == CHOSEN
    projects.create_project(root, "gamma", origin="owner_ui")
    rows = {row["id"]: row["model_choice"] for row in projects.projects_summary(root, limit=10)}
    assert rows == {"alpha": CHOSEN, "beta": {"kind": "main"}, "gamma": {"kind": "main"}}
    assert client.post("/api/owner/chat-model", json={"kind": "main", "project_id": "alpha"}).json()["previous_model_choice"] == CHOSEN
    assert "model_choice" not in _registry_row(root, "alpha")


@pytest.mark.parametrize("model,local,account,pin,reset", [
    (ASTRA, False, "kept", "acc-main", True),
    ("claudexor::claude=some-model", False, "auto", "", True),
    (OPUS, False, "none", "", True),
    ("local-model", True, "none", "", True),
    (SOL, False, "kept", "acc-main", False),
])
def test_main_write_keeps_only_same_subscription_source_pin_and_resets_only_changed_model(
        room, model, local, account, pin, reset):
    _root, client = room
    response = client.post("/api/owner/chat-model", json={"kind": "model", "model": model, "use_local": local})
    assert response.status_code == 200, response.text
    assert response.json()["account"] == account and response.json()["window_reset"] is reset
    saved = _saved()
    assert saved["OUROBOROS_MODEL"] == model and saved["USE_LOCAL_MAIN"] is local
    assert json.loads(saved["OUROBOROS_MODEL_ACCOUNTS"])["main"] == pin
    windows = json.loads(saved[MODEL_CONTEXT_WINDOWS_KEY])
    assert ("main" not in windows) is reset
    assert windows["fallback"] == [654321]
    if not reset:
        assert windows["main"] == 123456


def test_main_locality_only_keeps_the_manual_window_and_records_the_writers_receipt(room, monkeypatch):
    root, client = room
    save_owner({**OWNER, "OUROBOROS_MODEL": "owner/model", "OUROBOROS_MODEL_ACCOUNTS": "{}"})
    changes = []
    monkeypatch.setattr("ouroboros.gateway.owner_effort.record_settings_change", lambda request, receipt, **kw: changes.append(receipt))
    reply = client.post("/api/owner/chat-model", json={"kind": "model", "model": "owner/model", "use_local": True})
    assert reply.status_code == 200 and reply.json()["window_reset"] is False
    assert json.loads(_saved()[MODEL_CONTEXT_WINDOWS_KEY])["main"] == 123456
    assert changes[0]["USE_LOCAL_MAIN"]["new"] is True
    assert "OUROBOROS_MODEL" not in changes[0]
    recents = json.loads((root / "state/ui_preferences.json").read_text(encoding="utf-8"))["model_recents"]
    assert recents == [{"model": "owner/model", "use_local": True}]


@pytest.mark.parametrize("refusal,status,code", [
    ("DeploymentPolicyRefusal", 403, "managed_by_deployment"),
    ("SettingsPreconditionFailed", 409, "settings_precondition_failed"),
    ("SettingsLockUnavailable", 503, "settings_locked"),
])
def test_main_writer_refusals_pass_through_without_recording_a_recent(room, monkeypatch, refusal, status, code):
    from ouroboros import deployment_policy
    from ouroboros.gateway import owner_settings

    root, client = room
    error = getattr(deployment_policy if refusal == "DeploymentPolicyRefusal" else owner_settings, refusal)
    before = _saved()
    def refuse(*args, **kwargs):
        raise error("OUROBOROS_MODEL")
    monkeypatch.setattr("ouroboros.gateway.owner_effort._owner_update_settings", refuse)
    response = client.post("/api/owner/chat-model", json=CHOSEN)
    assert response.status_code == status and response.json()["code"] == code
    assert response.json()["saved"] is False and _saved() == before
    assert not (root / "state/ui_preferences.json").exists()


@pytest.mark.parametrize("body,status,code", [
    ({"kind": "pinned"}, 400, "chat_model_invalid"),
    ({"kind": "main"}, 400, "chat_model_invalid"),
    ({"kind": "fallback", "project_id": "alpha"}, 400, "chat_model_invalid"),
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
    main, follower = {"id": "main"}, {"id": "follower", "project_id": "alpha"}
    for task in (main, follower):
        route_choice.capture_lead(task, root)
        assert task["metadata"]["lead"] == {"source": "main"}
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    chosen = {"id": "chosen", "project_id": "alpha"}
    route_choice.capture_lead(chosen, root)
    assert chosen["metadata"]["lead"] == LEAD
    for task in (main, follower):
        route_choice.capture_lead(task, root)
        assert task["metadata"]["lead"] == {"source": "main"}
    # Move/detach and retry keep the same work's primary, even if the room changes.
    for pid in ("beta", "", "alpha"):
        chosen["project_id"] = pid
        route_choice.capture_lead(chosen, root)
        assert chosen["metadata"]["lead"] == LEAD
    gone = {"id": "gone", "project_id": "nowhere"}
    route_choice.capture_lead(gone, root)
    assert "lead" not in gone["metadata"]
    def broken(*_args, **_kwargs):
        raise TypeError("programming error")
    monkeypatch.setattr("ouroboros.composer_cognition.read_model_choice", broken)
    with pytest.raises(TypeError):
        route_choice.capture_lead({"id": "new"}, root)


@pytest.mark.parametrize("extra,stamp", [
    ({}, True),
    ({"metadata": {"schedule_writer_kind": "owner"}}, True),
    ({"actor_id": "consciousness", "metadata": {"schedule_writer_kind": "consciousness", "work_kind": "consciousness"}}, True),
    ({"metadata": {"schedule_writer_kind": "owner", "presence_binding_authority": {"binding_id": "place"}}}, True),
    ({"_presence_turn": True}, False),
    ({"metadata": {"presence_binding_authority": {"binding_id": "place"}}}, False),
    ({"metadata": {"work_kind": "consciousness"}}, False),
    ({"actor_id": "consciousness"}, False),
    ({"type": "evolution"}, False),
    ({"metadata": {"evolution_transaction": "transaction"}}, False),
    ({"parent_task_id": "parent"}, False),
    ({"metadata": {"delegation_role": "subagent"}}, False),
    ({"metadata": {"model_role": "light"}}, False),
    ({"model_role": "consciousness"}, False),
    ({"model": "synthetic::voice"}, False),
])
def test_admission_stamps_owner_roots_only(room, extra, stamp):
    root, client = room
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    task = {"id": "admission", "project_id": "alpha", **extra}
    route_choice.capture_lead(task, root)
    assert task.get("metadata", {}).get("lead") == (LEAD if stamp else None)


def test_a_separately_started_root_reads_its_destination_on_its_own_admission(room, monkeypatch):
    from ouroboros.tools.control import _start_task
    from supervisor import queue, workers

    root, client = room
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    monkeypatch.setattr("ouroboros.tools.control_events._wait_for_promotion_admission",
                        lambda *_a, **_k: {"status": "scheduled", "effective_project_id": "alpha"})
    events = []
    ctx = types.SimpleNamespace(pending_events=events, event_queue=None, current_chat_id=1, drive_root=root,
                                project_id="alpha", task_id="owner-turn",
                                task_metadata={"origin_suppressed": True, "lead": {"source": "main"}})
    assert _start_task(ctx, "Do the long part", title="Long part", workspace="none").startswith("OK: task")
    assert "lead" not in events[0]
    client.post("/api/owner/chat-model", json={"kind": "model", "model": OPUS, "project_id": "alpha"})
    host = types.SimpleNamespace(enqueue_task=queue.enqueue_task, persist_queue_snapshot=lambda **_kwargs: True,
                                 load_state=lambda: {"owner_chat_id": 1})
    reservation = queue.reserve_task_admission(events[0]["task_id"], events[0]["routing_token"], drive_root=root)
    assert reservation["status"] == "reserved", reservation
    outcome = workers.promote_chat_to_task(events[0], host)
    assert outcome["status"] == "scheduled", outcome
    assert queue.PENDING[-1]["metadata"]["lead"] == {"source": "room", "model": OPUS, "use_local": False}


def test_a_call_turn_keeps_its_engine_while_a_project_turn_takes_the_rooms_choice(room):
    from supervisor import worker_chat_lane

    root, client = room
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    project = projects.get_project(root, "alpha")
    call = worker_chat_lane._seed_task(project["chat_id"], "Opening", None, None, {
        "project_id": "alpha", "execution_transport": {"kind": "live_conversation", "call_id": "room-call"}})
    call["model"], call["use_local_model"] = "synthetic::voice", False
    typed = worker_chat_lane._seed_task(project["chat_id"], "Hello", None, None, {"project_id": "alpha"})
    for task in (call, typed):
        task["project_id"] = "alpha"
        task["_conversation_prepare"] = {"quiet": False}
        worker_chat_lane.prepare_conversation_task(types.SimpleNamespace(), task, root)
    assert call["model"] == "synthetic::voice" and "lead" not in call["metadata"]
    assert typed["metadata"]["lead"] == LEAD


def _ctx(metadata: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(task_metadata=metadata, task_model_override=None, task_use_local_override=None,
                                 task_route_source=None, primary_route=None)


def test_a_chosen_route_runs_on_auto_and_route_defaults_from_its_first_plan(room):
    from ouroboros.agent_dispatch import main_model_named_effort
    from ouroboros.context_fit import _context_route

    metadata = {"lead": LEAD}
    ctx = _ctx(metadata)
    route_choice.bind_task_route(ctx, metadata, presence=False)
    assert (ctx.task_model_override, ctx.task_use_local_override, ctx.task_route_source) == (ASTRA, False, "room")
    assert route_choice.open_route(ctx, None) == (ASTRA, False, {"kind": "explicit", "source": "room"})
    # Main's account pin, processing and window stay Main's: the room's route has none of its own.
    assert task_model_binding({"task_metadata": metadata}) == ("lead", "")
    assert task_processing_preference({"task_metadata": metadata}, model_role="lead") == ""
    assert model_role_option(MODEL_CONTEXT_WINDOWS_KEY, "lead") == 0
    plan_route, _settings = _context_route({"metadata": metadata})
    assert (plan_route["model"], plan_route["model_role"], plan_route["options"]["credential_profile_id"]) == (ASTRA, "lead", "")
    assert main_model_named_effort({"lead": {**LEAD, "model": "claudexor::cursor=gpt-5-high"}}) == "high"
    assert main_model_named_effort(metadata) == ""  # the carried model names no level; Main's is never consulted

    main = {"lead": {"source": "main"}}
    ctx = _ctx(main)
    route_choice.bind_task_route(ctx, main, presence=False)
    assert ctx.task_model_override is None and task_model_binding({"task_metadata": main}) == ("main", None)
    assert _context_route({"metadata": main})[0]["model"] == SOL

    # Delegation has its own route; an integrating place turn binds the same primary carrier.
    for meta, presence in (({**metadata, "delegation_role": "subagent", "model": OPUS}, False), (metadata, True)):
        ctx = _ctx(meta)
        route_choice.bind_task_route(ctx, meta, presence=presence)
        assert (ctx.task_route_source == "room") is presence
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
    route_choice.capture_lead(task, root)
    assert task["metadata"]["lead"] == {"source": "room", "model": chosen["model"], "use_local": False}
    plan_route, settings = _context_route({"metadata": task["metadata"]})
    assert (plan_route["provider"], plan_route["model"], plan_route["model_role"]) == ("openai", "openai::gpt-6.1", "lead")
    target = resolve_target(plan_route["model"], settings)
    assert (target["base_url"], target["api_key"]) == ("https://api.openai.com/v1", "sk-test-not-a-real-key")


def test_a_lead_on_a_wait_card_is_switched_for_the_task_never_persisted_into_settings():
    from ouroboros.model_slots import apply_model_role_override

    with pytest.raises(ValueError, match="work"):
        apply_model_role_override(dict(OWNER), role="lead", model=OPUS, credential_profile_id="", use_local=False)


def test_main_reference_resolves_from_live_settings(room):
    assert route_choice.configured_references() == {"main": {"model": SOL, "use_local": False}}
    save_owner({**OWNER, "OUROBOROS_MODEL": ASTRA})
    assert route_choice.configured_references() == {"main": {"model": ASTRA, "use_local": False}}


@pytest.mark.parametrize("value", [{"kind": "fallback"}, "garbage", {"kind": "pin"}])
def test_stored_invalid_room_choice_reads_as_main_with_warning(value, caplog):
    from ouroboros.composer_cognition import project_model_choice

    assert project_model_choice({"id": "alpha", "model_choice": value}) == {"kind": "main"}
    assert "Unreadable model choice" in caplog.text


@pytest.mark.parametrize("source", ["room", "owner", "owner_words", "mind"])
def test_lead_sources_are_validated_and_a_stray_root_model_stays_inert(source):
    metadata = {"lead": {**LEAD, "source": source}, "model": OPUS}
    ctx = _ctx(metadata)
    route_choice.bind_task_route(ctx, metadata, presence=False)
    assert ctx.task_model_override == ASTRA and ctx.task_route_source == source
    for bad in ("unknown", "switch_model", "fallback"):
        metadata["lead"]["source"] = bad
        route_choice.bind_task_route(ctx, metadata, presence=False)
        assert ctx.task_model_override is None


@pytest.mark.parametrize("address,metadata,expected", [
    ({"project_id": "alpha"}, {}, LEAD),
    ({"project_id": "beta"}, {}, {"source": "main"}),
    ({"chat_id": 1, "source": "web"}, {}, {"source": "main"}),
    ({}, {}, {"source": "main"}),
    ({"project_id": "alpha"}, {"model_role": "light"}, None),
])
def test_api_room_admission_reads_the_canonical_choice_not_the_fork(room, tmp_path, monkeypatch, address, metadata, expected):
    from ouroboros.gateway.tasks import api_tasks_create
    from ouroboros.task_results import load_task_result
    from supervisor import queue, workers

    root, client = room
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    repo = tmp_path / "api-repo"
    repo.mkdir()
    app = Starlette(routes=[Route("/api/tasks", api_tasks_create, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = root, repo
    monkeypatch.setattr(workers, "WORKERS", {0: types.SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    response = TestClient(app).post("/api/tasks", json={"description": "Read the report", "memory_mode": "empty",
                                                     "metadata": metadata, **address})
    assert response.status_code == 200, response.text
    task = queue.PENDING[-1]
    assert task["drive_root"] != str(root), "exercise the distinct empty execution drive"
    assert task["metadata"].get("lead") == expected
    stored = load_task_result(root, response.json()["task_id"])
    assert stored["metadata"].get("lead") == expected
    client.post("/api/owner/chat-model", json={"kind": "main", "project_id": "alpha"})
    assert task["metadata"].get("lead") == expected


@pytest.mark.parametrize("metadata", [{"lead": LEAD}, {"model_role": "lead", "model": OPUS, "use_local_model": True}])
def test_api_clients_cannot_mint_a_lead(room, monkeypatch, metadata):
    from ouroboros.gateway.tasks import api_tasks_create

    root, client = room
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    app = Starlette(routes=[Route("/api/tasks", api_tasks_create, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = root, config.REPO_DIR
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    response = TestClient(app).post("/api/tasks", json={"description": "Read", "project_id": "alpha", "metadata": metadata})
    assert response.status_code == 200, response.text
    from supervisor import queue
    admitted = queue.PENDING[-1]["metadata"]
    assert admitted["lead"] == LEAD
    ctx = _ctx(admitted)
    route_choice.bind_task_route(ctx, admitted, presence=False)
    assert (ctx.task_model_override, ctx.task_use_local_override) == (ASTRA, False)


def test_queue_admission_keeps_an_existing_lead_and_restart_never_recaptures(room):
    from supervisor import queue

    root, client = room
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    for n, lead in enumerate(({"source": "main"}, {**LEAD, "source": "owner"})):
        task = {"id": f"queue-{n}", "type": "task", "project_id": "alpha", "metadata": {"lead": lead}}
        admitted = queue.enqueue_task(task)
        assert "_admission_blocked" not in admitted
        assert admitted["metadata"]["lead"] == lead
        queue.PENDING.clear()
        assert queue.enqueue_task(admitted, restoring_snapshot=True)["metadata"]["lead"] == lead
        queue.PENDING.clear()
    restored = queue.enqueue_task({"id": "legacy", "type": "task", "metadata": {}}, restoring_snapshot=True)
    assert "lead" not in restored["metadata"]


def test_schedule_api_refuses_a_carrier_and_an_ordinary_occurrence_takes_its_room(room, monkeypatch):
    from ouroboros.gateway.schedules import api_schedules_upsert
    from supervisor.queue_schedules import _task_from_schedule

    root, client = room
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    app = Starlette(routes=[Route("/api/schedules", api_schedules_upsert, methods=["POST"])])
    app.state.drive_root = root
    body = {"id": "model-choice", "trigger": {"type": "once", "run_at": "2099-01-01T12:00:00Z"},
            "task": {"description": "Read", "project_id": "alpha", "metadata": {"lead": LEAD}}}
    with TestClient(app) as schedules:
        refused = schedules.post("/api/schedules", json=body)
        assert refused.status_code == 400 and "reserved" in refused.text
        body["task"]["metadata"] = {}
        accepted = schedules.post("/api/schedules", json=body)
    assert accepted.status_code == 200, accepted.text
    task = _task_from_schedule(accepted.json()["schedule"])
    route_choice.capture_lead(task, root)
    assert task["metadata"]["lead"] == LEAD


@pytest.mark.parametrize("presence", [False, True])
def test_a_bare_lead_selector_cannot_override_a_local_followers_route(room, presence):
    save_owner({"OUROBOROS_MODEL": "local-only", "USE_LOCAL_MAIN": True})
    metadata = {"model_role": "lead", "use_local_model": False}
    ctx = _ctx(metadata)
    route_choice.bind_task_route(ctx, metadata, presence=presence)
    assert ctx.task_model_override is None and ctx.task_use_local_override is None
    assert route_choice.open_route(ctx, None) == ("local-only", True, route_choice.following("main"))
    assert task_model_binding({"metadata": metadata}) == ("main", None)


def test_conversation_capture_uses_canonical_registry_even_with_an_execution_copy(room, tmp_path):
    root, client = room
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    execution = tmp_path / "execution"
    projects.create_project(execution, "alpha", origin="owner_ui")
    task = {"id": "split-turn", "project_id": "alpha", "budget_drive_root": str(root)}
    route_choice.capture_lead(task, execution)
    assert task["metadata"]["lead"] == LEAD


def test_unreadable_room_leaves_no_follow_fact_and_a_later_door_can_stamp_it(room, monkeypatch, caplog):
    root, client = room
    client.post("/api/owner/chat-model", json={**CHOSEN, "project_id": "alpha"})
    task = {"id": "unreadable-room", "project_id": "alpha"}
    with monkeypatch.context() as patch:
        patch.setattr("ouroboros.composer_cognition.read_model_choice",
                      lambda *_a, **_k: (_ for _ in ()).throw(OSError("unreadable registry")))
        route_choice.capture_lead(task, root)
    assert "lead" not in task["metadata"]
    assert "Room model choice unreadable" in caplog.text
    ctx = _ctx(task["metadata"])
    route_choice.bind_task_route(ctx, task["metadata"], presence=False)
    assert ctx.task_model_override is None and route_choice.opening_role(ctx) == "main"
    route_choice.capture_lead(task, root)
    assert task["metadata"]["lead"] == LEAD


def test_an_owner_lead_uses_no_options_from_a_previously_followed_role(room):
    metadata = {"lead": {**LEAD, "source": "owner"}, "model_role": "light"}
    assert task_model_binding({"metadata": metadata}) == ("lead", "")
    assert task_processing_preference({"metadata": metadata}) == ""


@pytest.mark.parametrize("source", [None, "initial_user", "live_call"])
def test_promotion_keeps_the_primary_and_only_owner_choices_enter_recents(room, monkeypatch, source):
    from ouroboros.tools.control_routing import _start_task
    from supervisor import queue, workers

    root, client = room
    cited = source is not None
    if cited:
        client.post('/api/owner/chat-model', json={**CHOSEN, 'project_id': 'alpha'})
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'cyber_pro')
    monkeypatch.setattr('ouroboros.tools.control_events._wait_for_promotion_admission',
                        lambda *a, **kw: {'status': 'scheduled'})
    events = []
    ctx = types.SimpleNamespace(pending_events=events, event_queue=None, current_chat_id=1, drive_root=root,
        project_id='alpha', task_id='owner-turn', task_metadata={} if source == 'live_call' else {'origin_suppressed': True},
        _owner_directives=[{'source': source or 'initial_user', 'content': 'Keep the report whole. Use the other route.'}])
    text = _start_task(ctx, 'Do the work', title='Report', workspace='none', model=OPUS,
                       owner_words='Use the other route' if cited else '')
    assert text.startswith('OK: task')
    event = events[0]
    host = types.SimpleNamespace(enqueue_task=queue.enqueue_task, persist_queue_snapshot=lambda **kw: True,
                                 load_state=lambda: {'owner_chat_id': 1})
    assert queue.reserve_task_admission(event['task_id'], event['routing_token'], drive_root=root)['status'] == 'reserved'
    assert workers.promote_chat_to_task(event, host)['status'] == 'scheduled'
    task = queue.PENDING[-1]
    assert task['metadata']['lead'] == {'source': 'owner_words' if cited else 'mind', 'model': OPUS, 'use_local': False,
                                        **({'cited': 'Use the other route'} if cited else {})}
    prefs = root / 'state/ui_preferences.json'
    if cited:
        assert json.loads(prefs.read_text(encoding='utf-8'))['model_recents'][0]['model'] == OPUS
    else:
        assert not prefs.exists() or not json.loads(prefs.read_text(encoding='utf-8')).get('model_recents')
    if source == 'live_call':
        from ouroboros.agent_task_pipeline import _store_task_result
        from ouroboros.owner_continue import owner_sources
        from ouroboros.owner_words import task_owner_words
        from ouroboros.task_results import load_task_result
        from supervisor.continuation_admission import _successor_task

        assert not task['metadata']['owner_corpus']
        assert task_owner_words(root, task['id'], task=task)[0] == []
        env = types.SimpleNamespace(drive_root=root, repo_dir=root, drive_path=lambda rel: root / rel)
        _store_task_result(env, task, 'Done.', {'rounds': 1, 'cost': 0}, {'tool_calls': [], 'reasoning_notes': []})
        stored = load_task_result(root, task['id'])
        sources = owner_sources(root, stored, task['id'])
        assert sources['original'] is None and not sources['later'] and not sources['gaps']
        successor = _successor_task(types.SimpleNamespace(DRIVE_ROOT=root), task['id'], stored,
            {'successor_task_id': 'next', 'chat_id': 1}, {'cause': 'context_overflow'}, sources, {})
        assert successor['metadata']['lead'] == task['metadata']['lead'] and not successor['metadata']['owner_corpus']
    elif cited:
        assert task['metadata']['owner_corpus'][-1]['content'] == ctx._owner_directives[0]['content']


def test_a_minds_primary_uses_the_same_carrier_and_cannot_be_recaptured(room):
    from ouroboros.route_choice import capture_lead

    root, client = room
    task = {'id': 'cyber-start', 'source': 'promote_chat_to_task', 'project_id': 'alpha',
            'metadata': {'lead': {'source': 'mind', 'model': OPUS, 'use_local': False}}}
    capture_lead(task, root)
    assert task['metadata']['lead']['source'] == 'mind'
    client.post('/api/owner/chat-model', json={**CHOSEN, 'project_id': 'alpha'})
    capture_lead(task, root, admitted=True, project=projects.get_project(root, 'alpha'))
    assert task['metadata']['lead'] == {'source': 'mind', 'model': OPUS, 'use_local': False}
    ctx = _ctx(task['metadata'])
    route_choice.bind_task_route(ctx, task['metadata'], presence=False)
    assert ctx.task_model_override == OPUS and ctx.task_route_source == 'mind'
    legacy_pair = {'model_role': 'lead', 'model': OPUS, 'use_local_model': False}
    route_choice.bind_task_route(ctx, legacy_pair, presence=False)
    assert ctx.task_model_override is None
    route_choice.bind_task_route(ctx, legacy_pair, presence=True)
    assert ctx.task_model_override == OPUS and ctx.task_route_source == 'presence'
