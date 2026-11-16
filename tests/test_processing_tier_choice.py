"""The owner's processing tier beside the model, and the routes the owner keeps pinned.

A room's tier (``processing_choice``) and Main's own tier (its role option) are written by
``POST /api/owner/chat-processing`` and change nothing else; ``/api/state`` says both. Admission stamps
a root's tier once into ``metadata.processing`` (the room's, or the decided Default) independently of its
lead, and the carried tier answers for the root's own route. Pins change one route at a time under the
preferences lock, are announced to every window, and never choose a model.
"""
from __future__ import annotations

import asyncio
import copy
import json
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import config, route_choice
from ouroboros import projects_registry as projects
from ouroboros.gateway.owner_effort import api_owner_chat_model, api_owner_chat_processing
from ouroboros.gateway.state import api_state
from ouroboros.gateway.ui_preferences import api_ui_preferences_get, api_ui_preferences_post, record_model_recent
from ouroboros.model_slots import (
    MODEL_ACCOUNTS_KEY,
    MODEL_CONTEXT_WINDOWS_KEY,
    processing_tiers,
    task_processing_preference,
)
from ouroboros.model_slots import MODEL_PROCESSING_PREFERENCES_KEY as ROLES
from ouroboros.model_slots import PROCESSING_PREFERENCE_KEY as GLOBAL
from tests._task_settings_helpers import isolated_runtime, save_owner  # noqa: F401 -- the shared isolated root

SOL = "claudexor::codex=gpt-6.1-sol"
ASTRA = "claudexor::codex=gpt-6-astra"
LUNA = "claudexor::codex=gpt-6-luna"
OPUS = "anthropic::claude-opus-5-5"
# Main's model, pins and windows: a tier write of any room, Main's own included, must leave each exactly as it was.
OWNER = {
    "OUROBOROS_MODEL": SOL, "OUROBOROS_MODEL_FALLBACKS": LUNA,
    MODEL_ACCOUNTS_KEY: json.dumps({"main": "acc-main", "fallback": ["acc-fallback"]}),
    MODEL_CONTEXT_WINDOWS_KEY: json.dumps({"main": 123456, "fallback": [654321]}),
    ROLES: json.dumps({"light": "economy", "fallback": ["fast"]}),
    GLOBAL: "economy",
}
CHOSEN = {"kind": "model", "model": ASTRA, "use_local": False}
ROOM_ECONOMY = {"source": "room", "tier": "economy"}
DEFAULT = {"source": "default"}
CHAT_PROCESSING, CHAT_MODEL = "/api/owner/chat-processing", "/api/owner/chat-model"
PREFS = "/api/ui/preferences"
R1 = {"model": "openai::gpt-6.1", "use_local": False}
R2 = {"model": ASTRA, "use_local": False}
R3 = {"model": "qwen3", "use_local": True}


def _saved() -> dict:
    return json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))


def _row(root, pid: str) -> dict:
    data = json.loads((root / "state/projects.json").read_text(encoding="utf-8"))
    return next(row for row in data["projects"] if row["id"] == pid)


def _registry(root) -> dict:
    return json.loads((root / "state/projects.json").read_text(encoding="utf-8"))


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
    app = Starlette(routes=[Route(CHAT_MODEL, api_owner_chat_model, methods=["POST"]),
                            Route(CHAT_PROCESSING, api_owner_chat_processing, methods=["POST"]),
                            Route("/api/state", api_state)])
    app.state.drive_root = isolated_runtime
    app.state.app_start = 0.0
    return isolated_runtime, TestClient(app)


def _state(client) -> dict:
    response = client.get("/api/state")
    assert response.status_code == 200, response.text
    return response.json()


# --- Pins ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def prefs(isolated_runtime, monkeypatch):  # noqa: F811
    from ouroboros.gateway import ws

    frames: list = []
    monkeypatch.setattr(ws, "broadcast_ws_sync", lambda msg: frames.append(copy.deepcopy(msg)) or True)
    save_owner({"OUROBOROS_MODEL": SOL})
    app = Starlette(routes=[Route(PREFS, api_ui_preferences_get, methods=["GET"]),
                            Route(PREFS, api_ui_preferences_post, methods=["POST"])])
    app.state.drive_root = isolated_runtime
    return isolated_runtime, TestClient(app), frames


def _pins(client) -> list:
    response = client.get(PREFS)
    assert response.status_code == 200, response.text
    return response.json()["model_pins"]


def test_two_windows_pin_one_route_each_and_both_pins_survive(prefs):
    root, client, frames = prefs
    record_model_recent(root, "vendor/recent-model")
    assert client.post(PREFS, json={"sidebar_width": 250}).status_code == 200
    settings = config.SETTINGS_PATH.read_bytes()
    frames.clear()
    # Two windows that never read each other's pins: each names only its own route.
    first = client.post(PREFS, json={"model_pins": {"pin": [R1]}})
    assert first.status_code == 200, first.text
    assert first.json()["model_pins"] == [R1]
    second = client.post(PREFS, json={"model_pins": {"pin": [R2]}})
    assert second.status_code == 200 and second.json()["model_pins"] == [R1, R2]
    assert _pins(client) == [R1, R2]
    # Pinning a pinned route again keeps its place; a `` (local)`` label is the same route as its locality field.
    again = client.post(PREFS, json={"model_pins": {"pin": [R1, {"model": "qwen3 (local)"}]}})
    assert again.json()["model_pins"] == [R1, R2, R3]
    # Unpin removes exactly its route; unpinning a route never pinned changes nothing else.
    unpinned = client.post(PREFS, json={"model_pins": {"unpin": [R2, {"model": "never/pinned"}]}})
    assert unpinned.status_code == 200 and unpinned.json()["model_pins"] == [R1, R3]
    current = client.get(PREFS).json()
    assert current["model_pins"] == [R1, R3]
    # A pin is no owner model act: the recents, the other keys and the settings document stay as they were.
    assert current["model_recents"] == [{"model": "vendor/recent-model", "use_local": False}]
    assert current["sidebar_width"] == 250
    assert config.SETTINGS_PATH.read_bytes() == settings
    assert not (root / "state/projects.json").exists()
    assert frames == [{"type": "ui_preferences_changed", "keys": ["model_pins"]}] * 4


def test_a_pin_beside_a_contributions_change_names_both_groups(prefs):
    _root, client, frames = prefs
    response = client.post(PREFS, json={"model_pins": {"pin": [R1]}, "contributions": {"hide": ["a:one"]}})
    assert response.status_code == 200, response.text
    assert response.json()["model_pins"] == [R1] and response.json()["contributions"]["hidden"] == ["a:one"]
    assert frames == [{"type": "ui_preferences_changed", "keys": ["contributions", "model_pins"]}]
    frames.clear()
    assert client.post(PREFS, json={"widget_order": ["w"]}).status_code == 200
    assert frames == [], "a write that names no announced group tells no window"


@pytest.mark.parametrize("body", [
    {"model_pins": [R1]},  # the whole list is never posted: it would drop another window's pins
    {"model_pins": {}},
    {"model_pins": {"pin": [R2], "replace": [R1]}},
    {"model_pins": {"pin": R2}},
    {"model_pins": {"pin": [ASTRA]}},
    {"model_pins": {"pin": [{"model": ""}]}},
    {"model_pins": {"pin": [{"model": "a\nb"}]}},
    {"model_pins": {"pin": [{"model": "claudexor::codex"}]}},
    {"model_pins": {"pin": [{"model": ASTRA, "use_local": True}]}},  # a subscription route is never local
    {"model_pins": {"pin": [{"model": "vendor/model", "use_local": "yes"}]}},
    {"model_pins": {"unpin": [{"model": ""}]}},
    {"model_pins": {"pin": [R2]}, "widget_order": "bad"},  # one refused key refuses the whole write
])
def test_a_refused_pin_write_changes_nothing_and_tells_no_window(prefs, body):
    root, client, frames = prefs
    assert client.post(PREFS, json={"model_pins": {"pin": [R1]}}).status_code == 200
    frames.clear()
    path = root / "state/ui_preferences.json"
    before = path.read_bytes()
    response = client.post(PREFS, json=body)
    assert response.status_code == 400, response.text
    assert path.read_bytes() == before and _pins(client) == [R1]
    assert frames == []


def test_the_twenty_fifth_pin_is_refused_until_one_is_unpinned(prefs):
    root, client, frames = prefs
    routes = [{"model": f"vendor/model-{index}", "use_local": False} for index in range(25)]
    assert client.post(PREFS, json={"model_pins": {"pin": routes[:24]}}).json()["model_pins"] == routes[:24]
    path = root / "state/ui_preferences.json"
    before = path.read_bytes()
    frames.clear()
    refused = client.post(PREFS, json={"model_pins": {"pin": [routes[24]]}})
    assert refused.status_code == 400 and "unpin one first" in refused.json()["error"], refused.text
    assert client.post(PREFS, json={"model_pins": {"pin": routes}}).status_code == 400  # 25 in one list
    assert path.read_bytes() == before and _pins(client) == routes[:24] and frames == []
    # A route already pinned is no new pin at the bound; unpin-and-pin in one write fits.
    assert client.post(PREFS, json={"model_pins": {"pin": [routes[3]]}}).json()["model_pins"] == routes[:24]
    swapped = client.post(PREFS, json={"model_pins": {"unpin": [routes[0]], "pin": [routes[24]]}})
    assert swapped.status_code == 200 and swapped.json()["model_pins"] == routes[1:]


def test_one_write_naming_a_route_twice_pins_it_once(prefs):
    root, client, _frames = prefs
    response = client.post(PREFS, json={"model_pins": {"pin": [R1, R1, {"model": "qwen3 (local)"}, R3]}})
    assert response.status_code == 200, response.text
    assert response.json()["model_pins"] == [R1, R3]
    assert json.loads((root / "state/ui_preferences.json").read_text(encoding="utf-8"))["model_pins"] == [R1, R3]
    assert _pins(client) == [R1, R3]


def test_twenty_three_pins_take_one_more_route_named_twice(prefs):
    _root, client, _frames = prefs
    routes = [{"model": f"vendor/model-{index}", "use_local": False} for index in range(24)]
    assert client.post(PREFS, json={"model_pins": {"pin": routes[:23]}}).status_code == 200
    response = client.post(PREFS, json={"model_pins": {"pin": [routes[23], routes[23]]}})
    assert response.status_code == 200, response.text
    assert response.json()["model_pins"] == routes


def test_stored_pins_read_through_the_route_grammar_and_junk_is_dropped(prefs):
    root, client, _frames = prefs
    path = root / "state/ui_preferences.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    junk = [R1, ASTRA, {"model": ""}, R1, {"model": "vendor/x", "use_local": "yes"}, None, 7,
            {"model": "qwen3 (local)"}, {"model": ASTRA, "use_local": True}, R2]
    path.write_text(json.dumps({"model_pins": junk, "sidebar_width": 250}), encoding="utf-8")
    assert _pins(client) == [R1, R3, R2]
    # The next pin is applied over the clean read: junk never comes back.
    assert client.post(PREFS, json={"model_pins": {"pin": [{"model": "vendor/new"}]}}).status_code == 200
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["model_pins"] == [R1, R3, R2, {"model": "vendor/new", "use_local": False}]
    assert stored["sidebar_width"] == 250
    path.write_text(json.dumps({"model_pins": "junk"}), encoding="utf-8")
    assert _pins(client) == []
    many = [{"model": f"vendor/model-{index}", "use_local": False} for index in range(30)]
    path.write_text(json.dumps({"model_pins": many}), encoding="utf-8")
    assert _pins(client) == many[:24]


def test_concurrent_windows_pin_under_the_lock_and_every_pin_survives(isolated_runtime, monkeypatch):  # noqa: F811
    from ouroboros.gateway import ws

    monkeypatch.setattr(ws, "broadcast_ws_sync", lambda msg: True)
    routes = [{"model": f"vendor/model-{index}", "use_local": False} for index in range(6)]
    barrier = threading.Barrier(len(routes))

    def pin(route):
        async def body():
            barrier.wait(timeout=10)
            return {"model_pins": {"pin": [route]}}

        request = types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace(
            drive_root=isolated_runtime)), json=body)
        return asyncio.run(api_ui_preferences_post(request)).status_code

    with ThreadPoolExecutor(max_workers=len(routes)) as pool:
        statuses = list(pool.map(pin, routes))
    assert statuses == [200] * len(routes)
    stored = json.loads((isolated_runtime / "state/ui_preferences.json").read_text(encoding="utf-8"))["model_pins"]
    assert sorted(route["model"] for route in stored) == sorted(route["model"] for route in routes)
    assert len(stored) == len(routes)


# --- Main's own tier and a room's tier ----------------------------------------------------------------------------


def _without(document: dict, key: str) -> dict:
    return {name: value for name, value in document.items() if name != key}


def _effective() -> dict:
    """The settings every reader sees (defaults included): a save materializes defaults into the document.

    The factory-default subagent row is derived at read time from Main's whole binding, its tier included
    (``review_pool_migration``), so it follows Main's tier by design and is left out of the comparison."""
    return {key: value for key, value in copy.deepcopy(config.load_settings()).items() if key != "OUROBOROS_SUBAGENTS"}


def test_mains_own_tier_moves_only_its_role_option(room):
    root, client = room
    before, raw = _effective(), _saved()
    registry = _registry(root)
    reply = client.post(CHAT_PROCESSING, json={"processing": "fast"})
    assert reply.status_code == 200, reply.text
    assert reply.json() == {"ok": True, "project_id": "", "main_processing": "fast", "previous_main_processing": "",
                            "processing_default": "economy"}
    saved = _saved()
    assert json.loads(saved[ROLES]) == {"main": "fast", "light": "economy", "fallback": ["fast"]}
    # Main's model, locality, account pin and context window stay exactly as they were.
    for key in ("OUROBOROS_MODEL", MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, GLOBAL, "OUROBOROS_MODEL_FALLBACKS"):
        assert saved[key] == raw[key], key
    assert json.loads(saved[MODEL_ACCOUNTS_KEY]) == {"main": "acc-main", "fallback": ["acc-fallback"]}
    assert json.loads(saved[MODEL_CONTEXT_WINDOWS_KEY]) == {"main": 123456, "fallback": [654321]}
    assert _effective()["USE_LOCAL_MAIN"] == before["USE_LOCAL_MAIN"]
    assert _without(_effective(), ROLES) == _without(before, ROLES)
    assert _registry(root) == registry, "Main's tier is not a room's"
    assert _state(client)["processing_tiers"] == {"main": "fast", "default": "economy"}
    # An explicit Standard is a tier of Main's own, distinct from Default.
    reply = client.post(CHAT_PROCESSING, json={"processing": "standard"})
    assert reply.status_code == 200
    assert (reply.json()["main_processing"], reply.json()["previous_main_processing"]) == ("standard", "fast")
    assert json.loads(_saved()[ROLES])["main"] == "standard"
    assert task_processing_preference({"metadata": {"lead": {"source": "main"}}}) == "standard"
    assert _state(client)["processing_tiers"] == {"main": "standard", "default": "economy"}
    # Default removes Main's own: Main's work pays the default tier again.
    reply = client.post(CHAT_PROCESSING, json={"processing": ""})
    assert reply.json() == {"ok": True, "project_id": "", "main_processing": "", "previous_main_processing": "standard",
                            "processing_default": "economy"}
    assert json.loads(_saved()[ROLES]) == {"light": "economy", "fallback": ["fast"]}
    assert task_processing_preference({"metadata": {"lead": {"source": "main"}}}) == "economy"
    assert _state(client)["processing_tiers"] == {"main": "", "default": "economy"}
    again = client.post(CHAT_PROCESSING, json={"processing": ""})
    assert again.status_code == 200 and again.json()["previous_main_processing"] == ""
    assert _without(_effective(), ROLES) == _without(before, ROLES)
    assert all(_saved()[key] == raw[key] for key in ("OUROBOROS_MODEL", MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY))


def test_a_rooms_tier_is_its_own_and_both_axes_survive_each_others_writes(room):
    root, client = room
    settings = _saved()
    reply = client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    assert reply.status_code == 200, reply.text
    assert reply.json() == {"ok": True, "project_id": "alpha", "processing_choice": "economy",
                            "previous_processing_choice": ""}
    assert _row(root, "alpha")["processing_choice"] == "economy"
    assert "processing_choice" not in _row(root, "beta")
    # An explicit Standard is stored, distinct from Default.
    reply = client.post(CHAT_PROCESSING, json={"processing": "standard", "project_id": "alpha"})
    assert (reply.json()["processing_choice"], reply.json()["previous_processing_choice"]) == ("standard", "economy")
    assert _row(root, "alpha")["processing_choice"] == "standard"
    # A model choice after a tier keeps the tier, and a tier after a model choice keeps the model.
    assert client.post(CHAT_MODEL, json={**CHOSEN, "project_id": "alpha"}).status_code == 200
    assert (_row(root, "alpha")["processing_choice"], _row(root, "alpha")["model_choice"]) == ("standard", CHOSEN)
    assert client.post(CHAT_PROCESSING, json={"processing": "fast", "project_id": "alpha"}).status_code == 200
    assert (_row(root, "alpha")["processing_choice"], _row(root, "alpha")["model_choice"]) == ("fast", CHOSEN)
    assert client.post(CHAT_MODEL, json={"kind": "main", "project_id": "alpha"}).status_code == 200
    assert _row(root, "alpha")["processing_choice"] == "fast" and "model_choice" not in _row(root, "alpha")
    payload = _state(client)
    rooms = {row["id"]: row for row in payload["projects"]}
    assert (rooms["alpha"]["processing_choice"], rooms["beta"]["processing_choice"]) == ("fast", "")
    assert rooms["alpha"]["model_choice"] == {"kind": "main"}
    assert payload["processing_tiers"] == {"main": "", "default": "economy"}
    # Default removes the room's own tier and moves nothing else.
    assert client.post(CHAT_MODEL, json={**CHOSEN, "project_id": "alpha"}).status_code == 200
    reply = client.post(CHAT_PROCESSING, json={"processing": "", "project_id": "alpha"})
    assert (reply.json()["processing_choice"], reply.json()["previous_processing_choice"]) == ("", "fast")
    assert "processing_choice" not in _row(root, "alpha") and _row(root, "alpha")["model_choice"] == CHOSEN
    assert _saved() == settings, "a room's choices never write Main"


def test_mains_tier_and_mains_model_survive_each_others_writes(room):
    _root, client = room
    assert client.post(CHAT_PROCESSING, json={"processing": "economy"}).status_code == 200
    assert client.post(CHAT_MODEL, json=CHOSEN).status_code == 200
    saved = _saved()
    assert saved["OUROBOROS_MODEL"] == ASTRA and json.loads(saved[ROLES])["main"] == "economy"
    assert client.post(CHAT_PROCESSING, json={"processing": "standard"}).status_code == 200
    saved = _saved()
    assert saved["OUROBOROS_MODEL"] == ASTRA and json.loads(saved[ROLES])["main"] == "standard"


@pytest.mark.parametrize("body,status,code", [
    ({"processing": "turbo"}, 400, "chat_processing_invalid"),
    ({"processing": "auto", "project_id": "alpha"}, 400, "chat_processing_invalid"),
    ({"processing": 1}, 400, "chat_processing_invalid"),
    ({"processing": None}, 400, "chat_processing_invalid"),  # the declared grammar, as the card's tier writer's
    ({"processing": "  FAST "}, 400, "chat_processing_invalid"),
    ({"processing": None, "project_id": "alpha"}, 400, "chat_processing_invalid"),
    ({}, 400, "chat_processing_invalid"),
    ({"project_id": "alpha"}, 400, "chat_processing_invalid"),
    ({"processing": "fast", "project_id": "../alpha"}, 400, "chat_processing_invalid"),
    ({"processing": "fast", "project_id": "nowhere"}, 404, "project_unavailable"),
    ({"processing": "fast", "project_id": "closing"}, 404, "project_unavailable"),
])
def test_a_refused_tier_says_nothing_was_saved(room, body, status, code):
    root, client = room
    projects.create_project(root, "closing", origin="owner_ui")
    projects.begin_project_deletion(root, "closing")
    before = (_saved(), _registry(root))
    reply = client.post(CHAT_PROCESSING, json=body)
    assert reply.status_code == status, reply.text
    assert reply.json()["saved"] is False and reply.json()["code"] == code
    assert (_saved(), _registry(root)) == before


def test_a_room_takes_ultrafast_while_mains_own_tier_refuses_it(room):
    root, client = room
    settings, registry = _saved(), _registry(root)
    refused = client.post(CHAT_PROCESSING, json={"processing": "ultrafast"})
    assert refused.status_code == 400 and refused.json()["code"] == "chat_processing_invalid", refused.text
    assert refused.json()["saved"] is False and (_saved(), _registry(root)) == (settings, registry)
    taken = client.post(CHAT_PROCESSING, json={"processing": "ultrafast", "project_id": "alpha"})
    assert taken.status_code == 200, taken.text
    assert taken.json() == {"ok": True, "project_id": "alpha", "processing_choice": "ultrafast",
                            "previous_processing_choice": ""}
    assert _row(root, "alpha")["processing_choice"] == "ultrafast"
    rooms = {row["id"]: row for row in _state(client)["projects"]}
    assert rooms["alpha"]["processing_choice"] == "ultrafast"
    task = {"id": "roomed", "project_id": "alpha"}
    route_choice.capture_lead(task, root)
    assert task["metadata"]["processing"] == {"source": "room", "tier": "ultrafast"}
    assert task_processing_preference({"metadata": task["metadata"]}) == "ultrafast"
    assert _saved() == settings, "a room's Ultrafast never reaches the settings document"


@pytest.mark.parametrize("values", [
    {ROLES: json.dumps({"main": "ultrafast"})},
    {ROLES: json.dumps({"light": "economy", "fallback": ["ultrafast"]})},
    {GLOBAL: "ultrafast"},
])
def test_a_settings_write_of_ultrafast_is_refused_and_saves_nothing(room, values):
    _root, client = room
    before = config.SETTINGS_PATH.read_bytes()
    with pytest.raises(ValueError):
        save_owner(values)
    assert config.SETTINGS_PATH.read_bytes() == before
    assert processing_tiers(config.load_settings()) == {"main": "", "default": "economy"}
    assert task_processing_preference({"metadata": {"lead": {"source": "main"}}}) == "economy"


def test_a_non_object_body_is_refused(room):
    root, client = room
    before = (_saved(), _registry(root))
    reply = client.post(CHAT_PROCESSING, json=["fast"])
    assert reply.status_code == 400 and reply.json()["code"] == "chat_processing_invalid"
    assert (_saved(), _registry(root)) == before


@pytest.mark.parametrize("settings,expected", [
    ({}, {"main": "", "default": ""}),
    ({ROLES: '{"main":"standard"}', GLOBAL: "fast"}, {"main": "standard", "default": "fast"}),
    ({ROLES: {"main": "economy"}}, {"main": "economy", "default": ""}),
    ({ROLES: '{"main":"turbo"}', GLOBAL: "turbo"}, {"main": "", "default": ""}),  # unreadable is none, never a tier
    ({ROLES: "not json", GLOBAL: 3}, {"main": "", "default": ""}),
    ({ROLES: '{"nobody":"fast"}', GLOBAL: " Fast "}, {"main": "", "default": "fast"}),
])
def test_processing_tiers_read_tolerantly(settings, expected):
    assert processing_tiers(settings) == expected


def test_an_unreadable_stored_room_tier_reads_as_default(room, caplog):
    from ouroboros.composer_cognition import _update_room

    root, client = room
    _update_room(root, "alpha", lambda row: row.update(processing_choice="turbo"))  # a hand-edited row
    assert _row(root, "alpha")["processing_choice"] == "turbo"
    rooms = {row["id"]: row for row in _state(client)["projects"]}
    assert rooms["alpha"]["processing_choice"] == ""
    task = {"id": "unreadable-tier", "project_id": "alpha"}
    route_choice.capture_lead(task, root)
    assert task["metadata"]["processing"] == DEFAULT
    assert "Unreadable processing choice" in caplog.text


def test_the_agent_browser_cannot_press_the_tier_control(monkeypatch):
    from ouroboros import browser_policy

    monkeypatch.setattr(browser_policy, "browser_url_block_reason", lambda *a, **k: "")
    monkeypatch.setattr(browser_policy, "runtime_service_kind", lambda *a: "main")
    for path in (CHAT_PROCESSING, "/api/owner/chat-%70rocessing", CHAT_PROCESSING + "/"):
        request = types.SimpleNamespace(method="POST", url="http://example.invalid" + path)
        assert "BROWSER_OWNER_CONTROL_BLOCKED" in browser_policy.browser_request_block_reason(
            request, None, restricted=False, runtime_mode="standard"), path
        request.method = "GET"
        assert browser_policy.browser_request_block_reason(request, None, restricted=False,
                                                           runtime_mode="standard") == ""


# --- Admission: the tier is stamped once, beside the lead and independent of it ----------------------------------


def test_admission_stamps_the_rooms_tier_once_and_the_decided_default_otherwise(room):
    root, client = room
    client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    roomed, plain, main = {"id": "roomed", "project_id": "alpha"}, {"id": "plain", "project_id": "beta"}, {"id": "main"}
    for task in (roomed, plain, main):
        route_choice.capture_lead(task, root)
    assert roomed["metadata"] == {"lead": {"source": "main"}, "processing": ROOM_ECONOMY}
    assert plain["metadata"] == {"lead": {"source": "main"}, "processing": DEFAULT}
    assert main["metadata"] == {"lead": {"source": "main"}, "processing": DEFAULT}
    # Main's own tier is Main's role option, never a carrier: Main's work keeps the decided Default and pays it.
    client.post(CHAT_PROCESSING, json={"processing": "fast"})
    later_main = {"id": "later-main"}
    route_choice.capture_lead(later_main, root)
    assert later_main["metadata"]["processing"] == DEFAULT
    assert task_processing_preference({"metadata": later_main["metadata"]}) == "fast"
    # A later change of a room reaches only work admitted after it: nothing is ever re-stamped.
    client.post(CHAT_PROCESSING, json={"processing": "fast", "project_id": "alpha"})
    client.post(CHAT_PROCESSING, json={"processing": "standard", "project_id": "beta"})
    for task in (roomed, plain, main):
        before = copy.deepcopy(task["metadata"])
        route_choice.capture_lead(task, root)
        route_choice.capture_lead(task, root, admitted=True, project=projects.get_project(root, "alpha"))
        assert task["metadata"] == before
    later = {"id": "later", "project_id": "alpha"}
    route_choice.capture_lead(later, root)
    assert later["metadata"]["processing"] == {"source": "room", "tier": "fast"}
    queued = {"id": "queued", "project_id": "beta"}
    route_choice.capture_lead(queued, root, admitted=True, project=projects.get_project(root, "beta"))
    assert queued["metadata"]["processing"] == {"source": "room", "tier": "standard"}
    main_queued = {"id": "main-queued"}
    route_choice.capture_lead(main_queued, root, admitted=True, project=None)
    assert main_queued["metadata"]["processing"] == DEFAULT
    gone = {"id": "gone", "project_id": "nowhere"}
    route_choice.capture_lead(gone, root)
    assert "processing" not in gone["metadata"] and "lead" not in gone["metadata"]


@pytest.mark.parametrize("given", [
    {"metadata": {"lead": {"source": "owner_words", "model": OPUS, "use_local": False, "cited": "Use the other route"}}},
    {"metadata": {"lead": {"source": "mind", "model": OPUS, "use_local": False}}},
    {"metadata": {"lead": {"source": "main"}}},
    {"model": OPUS},
])
def test_a_root_arriving_with_its_own_model_still_takes_its_rooms_tier(room, given):
    root, client = room
    client.post(CHAT_MODEL, json={**CHOSEN, "project_id": "alpha"})
    client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    task = {"id": "own-model", "project_id": "alpha", **copy.deepcopy(given)}
    route_choice.capture_lead(task, root)
    assert task["metadata"]["processing"] == ROOM_ECONOMY
    assert task["metadata"].get("lead") == given.get("metadata", {}).get("lead"), "its own model, never the room's"


@pytest.mark.parametrize("carrier", [DEFAULT, {"source": "owner", "tier": "standard"}, ROOM_ECONOMY,
                                     {"source": "room", "tier": "turbo"}])
def test_a_root_that_already_carries_a_tier_is_never_restamped(room, carrier):
    root, client = room
    client.post(CHAT_PROCESSING, json={"processing": "fast", "project_id": "alpha"})
    task = {"id": "carried", "project_id": "alpha", "metadata": {"processing": copy.deepcopy(carrier)}}
    route_choice.capture_lead(task, root)
    route_choice.capture_lead(task, root, admitted=True, project=projects.get_project(root, "alpha"))
    assert task["metadata"]["processing"] == carrier
    assert task["metadata"]["lead"] == {"source": "main"}, "the lead is still stamped on its own"


@pytest.mark.parametrize("extra,stamp", [
    ({}, True),
    ({"metadata": {"schedule_writer_kind": "owner"}}, True),
    ({"actor_id": "consciousness", "metadata": {"schedule_writer_kind": "consciousness", "work_kind": "consciousness"}},
     True),
    ({"_presence_turn": True}, False),
    ({"metadata": {"presence_binding_authority": {"binding_id": "place"}}}, False),
    ({"metadata": {"work_kind": "consciousness"}}, False),
    ({"actor_id": "consciousness"}, False),
    ({"type": "evolution"}, False),
    ({"metadata": {"evolution_transaction": "transaction"}}, False),
    ({"parent_task_id": "parent"}, False),
    ({"metadata": {"delegation_role": "subagent"}}, False),
    ({"delegation_role": "subagent"}, False),
    ({"metadata": {"model_role": "light"}}, False),
    ({"model_role": "consciousness"}, False),
])
def test_only_owner_roots_take_a_tier(room, extra, stamp):
    root, client = room
    client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    task = {"id": "admission", "project_id": "alpha", **copy.deepcopy(extra)}
    route_choice.capture_lead(task, root)
    metadata = task.get("metadata", {})
    assert metadata.get("processing") == (ROOM_ECONOMY if stamp else None)
    if not stamp:
        assert "lead" not in metadata


def test_an_unreadable_room_stamps_neither_and_a_later_door_can(room, monkeypatch, caplog):
    root, client = room
    client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    cited = {"source": "owner_words", "model": OPUS, "use_local": False, "cited": "Use the other route"}
    for task in ({"id": "unreadable", "project_id": "alpha"},
                 {"id": "unreadable-cited", "project_id": "alpha", "metadata": {"lead": dict(cited)}}):
        lead = copy.deepcopy(task.get("metadata", {}).get("lead"))
        with monkeypatch.context() as patch:
            patch.setattr("ouroboros.composer_cognition.read_room",
                          lambda *_a, **_k: (_ for _ in ()).throw(OSError("unreadable registry")))
            route_choice.capture_lead(task, root)
        assert "processing" not in task["metadata"] and task["metadata"].get("lead") == lead
        route_choice.capture_lead(task, root)
        assert task["metadata"]["processing"] == ROOM_ECONOMY
    assert "Room model choice unreadable" in caplog.text


# --- The carried tier answers for the root's own route -------------------------------------------------------------


def test_a_roots_carried_tier_answers_for_its_own_route_and_nothing_else(monkeypatch):
    from ouroboros.subagent_runtime import select_subagent_snapshot
    from tests.test_processing_preferences import actor_settings
    from tests.test_supervisor_route import selected_task

    monkeypatch.setenv(GLOBAL, "economy")
    monkeypatch.setenv(ROLES, json.dumps({"main": "standard", "fallback": ["", "standard"]}))
    lead = {"lead": {"source": "room", "model": ASTRA, "use_local": False}}
    for carried in ({"processing": {"source": "room", "tier": "fast"}}, {"processing": {"source": "owner", "tier": "fast"}}):
        for metadata in (carried, {**carried, **lead}, {**carried, "lead": {"source": "main"}}):
            for task in ({"metadata": metadata}, {"task_metadata": metadata}):
                assert task_processing_preference(task) == "fast", metadata
                assert task_processing_preference(task, model_role="main") == "fast"
                assert task_processing_preference(task, model_role="lead") == "fast"
                # On a fallback row too, configured or not: the work's own tier outranks the row's authored one.
                assert task_processing_preference(task, model_role="fallback:0") == "fast"
                assert task_processing_preference(task, model_role="fallback:1") == "fast"
                assert task_processing_preference(task, override="economy") == "economy"
    # Without a carrier, with the decided Default or with an unreadable one, the default resolution stands:
    # Main's own tier for work that follows Main, the default for a lead (never Main's).
    for metadata in ({}, {"processing": DEFAULT}, {"processing": {"source": "room", "tier": "turbo"}},
                     {"processing": {"source": "mind", "tier": "fast"}}, {"processing": {"source": "room", "tier": ""}},
                     {"processing": "fast"}):
        assert task_processing_preference({"metadata": metadata}) == "standard", metadata
        assert task_processing_preference({"metadata": metadata}, model_role="fallback:0") == "standard"
        assert task_processing_preference({"metadata": {**metadata, **lead}}) == "economy"
        assert task_processing_preference({"metadata": {**metadata, **lead}}, model_role="fallback:0") == "economy"
        # The row's authored tier stands only here, where the work has no tier of its own.
        assert task_processing_preference({"metadata": {**metadata, **lead}}, model_role="fallback:1") == "standard"
    # A delegated child never reads a root's carrier.
    child = {"processing": {"source": "room", "tier": "fast"}, "delegation_role": "subagent"}
    assert task_processing_preference({"metadata": child}) == "standard"
    # A configured row and a supervisor's frozen cognition keep their own tier.
    snapshot, _ = select_subagent_snapshot(actor_settings("economy"), subagent_id="critic")
    configured = {"task_metadata": {"configured_subagent": snapshot, "processing": {"source": "owner", "tier": "fast"}}}
    assert task_processing_preference(configured, model_role="subagent:critic") == "economy"
    assert task_processing_preference(configured, model_role="fallback:0") == "economy"
    nanny = {**selected_task(), "task_metadata": {"processing": {"source": "owner", "tier": "fast"}}}
    assert task_processing_preference(nanny) == "economy"
    assert task_processing_preference(nanny, model_role="fallback:0") == "economy"


@pytest.mark.parametrize("row_tier", ["fast", "standard", "economy"])
def test_a_works_own_tier_outranks_a_fallback_rows_and_default_leaves_the_rows_standing(monkeypatch, row_tier):
    """The card promises its tier from the next call on whatever route serves it; a fallback row's authored tier is a
    Settings default the work's own choice (a room's or the owner's, Standard included) outranks."""
    monkeypatch.setenv(GLOBAL, "")
    monkeypatch.setenv(ROLES, json.dumps({"main": "standard", "fallback": [row_tier]}))
    for primary in ({}, {"lead": {"source": "main"}}, {"lead": {"source": "room", "model": ASTRA, "use_local": False}}):
        for source in ("room", "owner"):
            for tier in ("standard", "fast", "economy", "ultrafast"):
                metadata = {**primary, "processing": {"source": source, "tier": tier}}
                assert task_processing_preference({"task_metadata": metadata}, model_role="fallback:0") == tier
        # No tier of its own (none, or the decided Default): the row's authored tier stands.
        for metadata in (dict(primary), {**primary, "processing": DEFAULT}):
            assert task_processing_preference({"task_metadata": metadata}, model_role="fallback:0") == row_tier


# --- Doors: API and schedules cannot mint a tier; Continue and promotion keep the stored one -----------------------


@pytest.mark.parametrize("project_id,expected", [("alpha", ROOM_ECONOMY), ("", DEFAULT)])
def test_a_conversation_turn_takes_its_rooms_tier_and_its_row_learns_it_by_frame(room, project_id, expected):
    from supervisor import worker_chat_lane

    root, client = room
    client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    chat_id = projects.get_project(root, "alpha")["chat_id"] if project_id else 1
    turn = worker_chat_lane._seed_task(chat_id, "Hello", None, None, {"project_id": project_id} if project_id else {})
    turn["project_id"] = project_id
    turn["_conversation_prepare"] = {"quiet": False}
    frames: list = []
    worker_chat_lane.prepare_conversation_task(types.SimpleNamespace(), turn, root, broadcast=frames.append)
    assert turn["metadata"]["processing"] == expected
    # A room turn's decided Default reaches its row too, so a step rebuilt from the row never reads the room's tier as
    # new; a Main turn has no room tier to drift to and sends no frame.
    assert [frame for frame in frames if "processing" in (frame.get("facts") or {})] == ([{
        "type": "direct_turn_bound", "task_id": turn["id"], "task_attempt": 1, "facts": {"processing": expected}}]
        if project_id else [])


def test_a_turn_rebuilt_from_its_row_keeps_the_decided_default_after_its_room_takes_a_tier(room):
    """A first step that re-enters preparation from its row (its attempt unchanged) keeps the Default it was
    admitted with: the row learned it by frame, so the room's later Flex reaches only turns admitted after it."""
    from supervisor import conversation_lane, worker_chat_lane

    root, client = room
    chat_id = projects.get_project(root, "alpha")["chat_id"]
    turn = worker_chat_lane._seed_task(chat_id, "Hello", None, None, {"project_id": "alpha"})
    turn.update(project_id="alpha", _conversation_prepare={"quiet": False})
    frames: list = []
    worker_chat_lane.prepare_conversation_task(types.SimpleNamespace(), turn, root, broadcast=frames.append)
    row = {"id": turn["id"], "type": "task", "chat_id": chat_id, "project_id": "alpha", "metadata": {"project_id": "alpha"}}
    running = {turn["id"]: {"task": row, "attempt": 1}}
    for frame in frames:
        if frame.get("type") == "direct_turn_bound":
            conversation_lane.handle_direct_turn_bound(frame, types.SimpleNamespace(RUNNING=running))
    assert row["metadata"]["processing"] == DEFAULT
    client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    again = {**copy.deepcopy(row), "_conversation_prepare": {"quiet": False}}
    frames.clear()
    worker_chat_lane.prepare_conversation_task(types.SimpleNamespace(), again, root, broadcast=frames.append)
    assert again["metadata"]["processing"] == DEFAULT
    assert not [frame for frame in frames if "processing" in (frame.get("facts") or {})], frames


def test_a_resumed_turn_keeps_what_it_carries_and_never_reads_its_rooms_current_tier(room):
    """#2547's rule for the basis holds for the tier: a resumed first step admits nothing new and sends no frame."""
    from supervisor import worker_chat_lane

    root, client = room
    client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    chat_id = projects.get_project(root, "alpha")["chat_id"]
    for carried in ({}, {"processing": {"source": "room", "tier": "fast"}}):
        turn = worker_chat_lane._seed_task(chat_id, "Hello again", None, None, {"project_id": "alpha", **carried})
        turn.update(project_id="alpha", _attempt=2, _conversation_prepare={"quiet": False})
        frames: list = []
        worker_chat_lane.prepare_conversation_task(types.SimpleNamespace(), turn, root, broadcast=frames.append)
        assert turn["metadata"].get("processing") == carried.get("processing"), turn["metadata"]
        assert not [frame for frame in frames if "processing" in (frame.get("facts") or {})], frames


@pytest.mark.parametrize("address,expected", [
    ({"project_id": "alpha"}, ROOM_ECONOMY),
    ({"project_id": "beta"}, DEFAULT),
    ({}, DEFAULT),
])
def test_api_clients_cannot_mint_a_tier_and_the_stored_row_keeps_the_admitted_one(room, monkeypatch, address,
                                                                                    expected):
    from ouroboros.gateway.tasks import api_tasks_create
    from ouroboros.task_results import load_task_result
    from supervisor import queue

    root, client = room
    client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    app = Starlette(routes=[Route("/api/tasks", api_tasks_create, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = root, config.REPO_DIR
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])
    minted = {"processing": {"source": "owner", "tier": "fast"}, "lead": {"source": "owner", "model": OPUS, "use_local": False}}
    response = TestClient(app).post("/api/tasks", json={"description": "Read", "metadata": minted, **address})
    assert response.status_code == 200, response.text
    admitted = queue.PENDING[-1]["metadata"]
    assert admitted["processing"] == expected
    assert load_task_result(root, response.json()["task_id"])["metadata"]["processing"] == expected


def test_a_schedule_template_cannot_carry_a_tier_and_its_occurrence_takes_the_room(room):
    from ouroboros.gateway.schedules import api_schedules_upsert
    from supervisor.queue_schedules import _task_from_schedule

    root, client = room
    client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    app = Starlette(routes=[Route("/api/schedules", api_schedules_upsert, methods=["POST"])])
    app.state.drive_root = root
    body = {"id": "tier-choice", "trigger": {"type": "once", "run_at": "2099-01-01T12:00:00Z"},
            "task": {"description": "Read", "project_id": "alpha", "metadata": {"processing": {"source": "owner",
                                                                                               "tier": "fast"}}}}
    with TestClient(app) as schedules:
        refused = schedules.post("/api/schedules", json=body)
        assert refused.status_code == 400 and "reserved" in refused.text, refused.text
        body["task"]["metadata"] = {}
        accepted = schedules.post("/api/schedules", json=body)
    assert accepted.status_code == 200, accepted.text
    task = _task_from_schedule(accepted.json()["schedule"])
    route_choice.capture_lead(task, root)
    assert task["metadata"]["processing"] == ROOM_ECONOMY


@pytest.mark.parametrize("carried", [None, {"source": "owner", "tier": "standard"}, DEFAULT])
def test_a_promoted_root_reads_its_rooms_tier_anew(room, monkeypatch, carried):
    """A promotion starts a separate root (start_task's promote event carries at most a chosen lead): its admission
    reads the destination's tier, and a tier put on the event by anything else is not its carrier."""
    from ouroboros.tools.control_routing import _start_task
    from supervisor import queue, workers

    root, client = room
    client.post(CHAT_PROCESSING, json={"processing": "economy", "project_id": "alpha"})
    monkeypatch.setattr("ouroboros.tools.control_events._wait_for_promotion_admission",
                        lambda *a, **kw: {"status": "scheduled"})
    events = []
    ctx = types.SimpleNamespace(pending_events=events, event_queue=None, current_chat_id=1, drive_root=root,
                                project_id="alpha", task_id="owner-turn", task_metadata={"origin_suppressed": True},
                                _owner_directives=[{"source": "initial_user", "content": "Do the work."}])
    assert _start_task(ctx, "Do the work", title="Report", workspace="none").startswith("OK: task")
    event = events[0]
    assert "processing" not in (event.get("metadata") or {}), "the start's promote event carries no tier"
    if carried is not None:
        event["metadata"] = {**(event.get("metadata") or {}), "processing": dict(carried)}
    host = types.SimpleNamespace(enqueue_task=queue.enqueue_task, persist_queue_snapshot=lambda **kw: True,
                                 load_state=lambda: {"owner_chat_id": 1})
    assert queue.reserve_task_admission(event["task_id"], event["routing_token"], drive_root=root)["status"] == "reserved"
    assert workers.promote_chat_to_task(event, host)["status"] == "scheduled"
    assert queue.PENDING[-1]["metadata"]["processing"] == ROOM_ECONOMY
