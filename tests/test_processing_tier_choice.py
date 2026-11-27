"""The owner's processing tier beside the model, and the routes the owner keeps pinned.

A room's tier (``processing_choice``) and the Main chat's own (its setting) are written by
``POST /api/owner/chat-processing`` and change nothing else; ``/api/state`` says both, beside Main the role's own
tier, which Settings › Models writes on the binding shown (``MODELS``). Admission stamps a root's tier once into
``metadata.processing`` (the room's, the Main chat's, or the decided Default) independently of its lead, and the
carried tier answers for the root's own route. Pins change one route at a time under the preferences lock, are
announced to every window, and never choose a model.
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
from ouroboros.gateway.owner_effort import api_owner_chat_model, api_owner_chat_processing, chat_consent_word
from ouroboros.gateway.state import api_state
from ouroboros.gateway.ui_preferences import api_ui_preferences_get, api_ui_preferences_post, record_model_recent
from ouroboros.model_slots import (
    MODEL_ACCOUNTS_KEY,
    MODEL_CONTEXT_WINDOWS_KEY,
    processing_tiers,
    send_processing_preference,
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
MAIN_CHAT = {"scope": "main_chat"}  # how a composer write names the Main chat; a room is named by its project_id
MODELS = "/api/owner/safety-model"  # Settings › Models' writer: Main the role's model and tier, on the binding shown
MAIN_CHAT_TIER = "OUROBOROS_MAIN_CHAT_PROCESSING"
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


def _observe(monkeypatch, read) -> None:
    """What the catalog says of a pinned account, for the check Main the role's own Ultrafast makes in Settings ›
    Models (a gateway handshake in production). The Main chat's Ultrafast follows a profile's name and never asks."""
    monkeypatch.setattr("ouroboros.gateway.models.observed_profile", read)


@pytest.fixture
def room(isolated_runtime, monkeypatch):  # noqa: F811 -- the imported fixture, used by name
    from supervisor import workers

    monkeypatch.setattr(workers, "WORKERS", {0: types.SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "_worker_pool_execution_state",
                        lambda *_a, **_kw: {"available": True, "disabled_reason": "", "worker_count": 1})
    _observe(monkeypatch, lambda source, pin:
             ("fixture-store", "fixture-account") if (source, pin) == ("codex", "acc-main") else None)
    from ouroboros.gateway.safety_settings import api_owner_safety_model

    _booted(OWNER)
    for pid in ("alpha", "beta"):
        projects.create_project(isolated_runtime, pid, origin="owner_ui")
    app = Starlette(routes=[Route(CHAT_MODEL, api_owner_chat_model, methods=["POST"]),
                            Route(CHAT_PROCESSING, api_owner_chat_processing, methods=["POST"]),
                            Route(MODELS, api_owner_safety_model, methods=["POST"]),
                            Route("/api/state", api_state)])
    app.state.drive_root = isolated_runtime
    app.state.app_start = 0.0
    app.state.bind_host = "127.0.0.1"  # the server's default bind; an unset host reads as a wildcard and probes the LAN
    return isolated_runtime, TestClient(app)


def _state(client) -> dict:
    response = client.get("/api/state")
    assert response.status_code == 200, response.text
    return response.json()


def _booted(values: dict) -> None:
    """A saved document as a running install holds it: its locked boot read saved the one-time upgrades, which a
    Settings writer's own locked read would otherwise save before any refusal."""
    save_owner(values)
    config.load_settings(migrate=True)


def _main_tier(client, tier: str, tokens=None):
    """Settings › Models writes Main the role's own tier (``""`` removes it), on the read tokens shown."""
    roles = json.loads(_saved().get(ROLES) or "{}")
    roles.pop("main", None)
    if tier:
        roles["main"] = tier
    return client.post(MODELS, json={"settings": {ROLES: roles}, **({} if tokens is None else {"if_unchanged": tokens})})


def _main_model(client, model: str, tokens=None):
    """Settings › Models writes Main the role's own model, on the read tokens shown."""
    return client.post(MODELS, json={"settings": {"OUROBOROS_MODEL": model, "USE_LOCAL_MAIN": False},
                                     **({} if tokens is None else {"if_unchanged": tokens})})


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


def test_the_main_chats_tier_is_its_own_and_moves_no_role_option(room):
    root, client = room
    before, raw = _effective(), _saved()
    registry = _registry(root)
    reply = client.post(CHAT_PROCESSING, json={"processing": "fast", **MAIN_CHAT})
    assert reply.status_code == 200, reply.text
    assert reply.json() == {"ok": True, "project_id": "", "processing_choice": "fast", "previous_processing_choice": ""}
    saved = _saved()
    assert saved[MAIN_CHAT_TIER] == "fast"
    # Main the role's tier, model, locality, account pin and context window stay exactly as they were.
    for key in (ROLES, "OUROBOROS_MODEL", MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, GLOBAL, "OUROBOROS_MODEL_FALLBACKS"):
        assert saved[key] == raw[key], key
    assert _without(_effective(), MAIN_CHAT_TIER) == _without(before, MAIN_CHAT_TIER)
    assert _registry(root) == registry, "the Main chat's tier is not a room's"
    state = _state(client)
    assert state["processing_tiers"] == {"main": "", "default": "economy"}
    assert state["main_chat"] == {"model_choice": {"kind": "main"}, "processing_choice": "fast"}
    # The owner's work in Main carries it from admission, on the route it follows; nothing names a model for it.
    turn = {"id": "turn"}
    route_choice.capture_lead(turn, root)
    assert turn["metadata"] == {"lead": {"source": "main"}, "processing": {"source": "room", "tier": "fast"}}
    assert task_processing_preference({"metadata": turn["metadata"]}) == "fast"
    # An explicit Standard is a tier of its own, distinct from Default.
    reply = client.post(CHAT_PROCESSING, json={"processing": "standard", **MAIN_CHAT})
    assert (reply.json()["processing_choice"], reply.json()["previous_processing_choice"]) == ("standard", "fast")
    assert _saved()[MAIN_CHAT_TIER] == "standard"
    # Default removes the chat's own: its new work pays what Main pays again.
    reply = client.post(CHAT_PROCESSING, json={"processing": "", **MAIN_CHAT})
    assert reply.json() == {"ok": True, "project_id": "", "processing_choice": "", "previous_processing_choice": "standard"}
    assert _saved()[MAIN_CHAT_TIER] == ""
    later = {"id": "later"}
    route_choice.capture_lead(later, root)
    assert later["metadata"]["processing"] == DEFAULT
    again = client.post(CHAT_PROCESSING, json={"processing": "", **MAIN_CHAT})
    assert again.status_code == 200 and again.json()["previous_processing_choice"] == ""
    assert _without(_effective(), MAIN_CHAT_TIER) == _without(before, MAIN_CHAT_TIER)


def test_mains_role_tier_moves_only_its_role_option(room):
    root, client = room
    raw, registry = _saved(), _registry(root)
    assert _main_tier(client, "fast").status_code == 200
    saved = _saved()
    assert json.loads(saved[ROLES]) == {"main": "fast", "light": "economy", "fallback": ["fast"]}
    for key in ("OUROBOROS_MODEL", MODEL_ACCOUNTS_KEY, MODEL_CONTEXT_WINDOWS_KEY, GLOBAL, "OUROBOROS_MODEL_FALLBACKS"):
        assert saved[key] == raw[key], key
    assert _registry(root) == registry and saved.get(MAIN_CHAT_TIER, "") == ""
    assert _state(client)["processing_tiers"] == {"main": "fast", "default": "economy"}
    assert task_processing_preference({"metadata": {"lead": {"source": "main"}}}) == "fast"
    # Default removes Main's own: Main's work pays the default tier again.
    assert _main_tier(client, "").status_code == 200
    assert json.loads(_saved()[ROLES]) == {"light": "economy", "fallback": ["fast"]}
    assert task_processing_preference({"metadata": {"lead": {"source": "main"}}}) == "economy"
    assert _state(client)["processing_tiers"] == {"main": "", "default": "economy"}


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


def test_the_main_chats_tier_and_model_survive_each_others_writes_and_main_keeps_both_of_its_own(room):
    _root, client = room
    assert _main_tier(client, "fast").status_code == 200
    assert client.post(CHAT_PROCESSING, json={"processing": "economy", **MAIN_CHAT}).status_code == 200
    assert client.post(CHAT_MODEL, json={**CHOSEN, **MAIN_CHAT}).status_code == 200
    saved = _saved()
    assert (saved["OUROBOROS_MAIN_CHAT_MODEL"], saved[MAIN_CHAT_TIER]) == (ASTRA, "economy")
    assert client.post(CHAT_PROCESSING, json={"processing": "standard", **MAIN_CHAT}).status_code == 200
    saved = _saved()
    assert (saved["OUROBOROS_MAIN_CHAT_MODEL"], saved[MAIN_CHAT_TIER]) == (ASTRA, "standard")
    assert saved["OUROBOROS_MODEL"] == SOL and json.loads(saved[ROLES])["main"] == "fast"
    assert _state(client)["main_chat"] == {"model_choice": CHOSEN, "processing_choice": "standard"}
    assert _state(client)["main_route"] == {"model": SOL, "use_local": False}


def _asked(model: str = SOL, account: str = "acc-main", *, local: bool = False, connection: str = "") -> dict:
    """What the page asked the owner about, by name: the model, its locality, the subscription profile (``""`` on
    Auto and where a route has none) and, on an API route, its connection's revision as read. By default the route
    new work in the Main chat runs on while it follows Main."""
    from ouroboros.gateway.owner_settings import CONNECTION_CONDITION, connection_revision

    return {"model": model, "use_local": local, "account": account,
            **({CONNECTION_CONDITION: connection_revision(_saved(), connection)} if connection else {})}


def _chat_ultrafast(client, confirmed="main"):
    confirmed = _asked() if confirmed == "main" else confirmed
    return client.post(CHAT_PROCESSING, json={"processing": "ultrafast", **MAIN_CHAT,
                                              **({} if confirmed is None else {"confirmed": confirmed})})


def _never_asked(*_args):
    raise AssertionError("the Main chat's Ultrafast follows a profile by its name: nothing asks who is signed in to it")


def test_the_main_chats_ultrafast_is_saved_with_the_route_it_was_asked_for_without_writing_main(room, monkeypatch):
    _root, client = room
    _observe(monkeypatch, _never_asked)
    roles = _saved()[ROLES]
    taken = _chat_ultrafast(client)
    assert taken.status_code == 200, taken.text
    assert taken.json() == {"ok": True, "project_id": "", "processing_choice": "ultrafast", "previous_processing_choice": ""}
    # Saved with one word for what was confirmed: the model, its locality and the profile's name. None of it is a
    # key, made from one, or anything of the account signed in to that profile.
    assert _saved()[MAIN_CHAT_TIER] == "ultrafast@" + chat_consent_word([SOL, False, "acc-main"])
    assert _saved()[ROLES] == roles
    assert _state(client)["main_chat"] == {"model_choice": {"kind": "main"}, "processing_choice": "ultrafast"}
    assert _state(client)["processing_tiers"] == {"main": "", "default": "economy"}


@pytest.mark.parametrize("asked", [
    None,  # nothing confirmed: nobody was asked
    {"model": ASTRA},  # another model
    {"account": ""},  # the same model, asked for on Auto while Main pins a profile
    {"account": "acc-other"},  # another profile
    {"local": True},
    {"connection": "anthropic"},  # a connection named for a route that has none
])
def test_ultrafast_asked_for_another_route_than_the_main_chats_saves_nothing(room, asked):
    _root, client = room
    save_owner({**_saved(), "ANTHROPIC_API_KEY": "sk-fictional-one"})
    before = _saved()
    reply = _chat_ultrafast(client, None if asked is None else _asked(**asked))
    assert reply.status_code == 409, reply.text
    assert reply.json()["saved"] is False and reply.json()["code"] == "settings_changed_elsewhere"
    assert _saved() == before


@pytest.mark.parametrize("confirmed", [
    {"model": SOL}, {"model": SOL, "use_local": "no", "account": "acc-main"}, {"model": 7, "use_local": False, "account": ""},
    {"ultrafast": json.dumps([SOL, False, "acc-main", "", ""])},  # an earlier spelling, with an account's identity
    "acc-main", {**{"model": SOL, "use_local": False, "account": "acc-main"}, "connection": 7},
])
def test_a_confirmation_that_does_not_name_its_route_is_refused_by_the_grammar(room, confirmed):
    _root, client = room
    before = _saved()
    reply = _chat_ultrafast(client, confirmed)
    assert reply.status_code == 400 and reply.json()["code"] == "chat_processing_invalid", reply.text
    assert reply.json()["saved"] is False and _saved() == before


@pytest.mark.parametrize("model,connection,first,changed", [
    (OPUS, "anthropic", {"ANTHROPIC_API_KEY": "sk-fictional-one"}, {"ANTHROPIC_API_KEY": "sk-fictional-two"}),
    ("openai-compatible::fictional-gamma", "openai-compatible",
     {"OPENAI_COMPATIBLE_API_KEY": "sk-fictional-one", "OPENAI_COMPATIBLE_BASE_URL": "https://one.fictional.invalid/v1"},
     {"OPENAI_COMPATIBLE_BASE_URL": "https://two.fictional.invalid/v1"}),
], ids=["another-key", "another-address"])
def test_a_changed_api_connection_under_the_same_model_does_not_inherit_the_main_chats_ultrafast(room, model, connection, first, changed):
    """The same model on a connection whose key or address changed is another destination: the consent given on the
    connection as it was shown starts no new work on Ultrafast there, and the work admitted on it stops asking it too.
    A work carries a random saved revision, not the key or a derivative of it. A chip pick does not re-price it."""
    from ouroboros.lead_fact import processing_fields
    from ouroboros.provider_connections import live_connection_account
    from supervisor.message_ingress import new_work_choice
    from ouroboros.composer_cognition import saved_choice

    root, client = room
    save_owner({**_saved(), **first})
    own = {"kind": "model", "model": model, "use_local": False}
    assert client.post(CHAT_MODEL, json={**own, **MAIN_CHAT}).status_code == 200
    # An API route's Ultrafast stands on its connection as shown: without that revision nothing is saved.
    before = _saved()
    assert _chat_ultrafast(client, _asked(model, "")).status_code == 409 and _saved() == before
    shown = _asked(model, "", connection=connection)
    taken = _chat_ultrafast(client, shown)
    assert taken.status_code == 200, taken.text
    kept = _saved()[MAIN_CHAT_TIER]
    old_connection = live_connection_account(model)
    assert old_connection.startswith(f"api:{connection}#")
    assert _saved() == {**before, MAIN_CHAT_TIER: "ultrafast@" + chat_consent_word([model, False, ""], old_connection)}
    assert _state(client)["main_chat"] == {"model_choice": own, "processing_choice": "ultrafast"}
    work = _main_work(root, "on-the-shown-connection")
    assert work == {"lead": {"source": "room", "model": model, "use_local": False},
                    "processing": {**_confirmed(model, ""), "connection": old_connection}}
    accepted = new_work_choice(root, 1, "New work on API", None, saved_choice(root, "")["choice_token"])
    assert route_choice.new_work_carriers(json.loads(json.dumps(accepted))) == work
    assert send_processing_preference(work, model, False, "lead") == "ultrafast"
    on_route = {"model": model, "use_local": False, "basis": "room"}
    assert processing_fields({"metadata": work}, on_route)["processing"] == {
        "choice": "ultrafast", "source": "room", "bound": True, "resolved": "ultrafast"}
    assert client.post(CHAT_PROCESSING, json={"processing": "fast", **MAIN_CHAT}).status_code == 200
    assert send_processing_preference(work, model, False, "lead") == "ultrafast"
    assert _main_work(root, "on-fast")["processing"] == {"source": "room", "tier": "fast"}
    assert _chat_ultrafast(client, shown).status_code == 200
    assert live_connection_account(model) == old_connection and _saved()[MAIN_CHAT_TIER] == kept
    # An ordinary save of that connection, under the same model and the same connection name, ends the answer:
    # for new work, and for the calls of the work admitted before it, which now go through the changed connection.
    save_owner({**_saved(), **changed})
    assert _saved()[MAIN_CHAT_TIER] == "" and _state(client)["main_chat"] == {"model_choice": own, "processing_choice": ""}
    assert send_processing_preference(work, model, False, "lead") == "economy"
    assert processing_fields({"metadata": work}, on_route)["processing"] == {"choice": "", "withheld": "ultrafast", "resolved": "economy"}
    assert _main_work(root, "on-the-changed-connection") == {
        "lead": {"source": "room", "model": model, "use_local": False}, "processing": {"source": "default"}}
    assert "processing_choice" not in projects.create_project(root, "after-the-connection-changed", origin="owner_ui")
    # The answer given on the connection as it was is stale for the connection as it is, and the old window is told so.
    stale = _chat_ultrafast(client, shown)
    assert stale.status_code == 409 and stale.json()["code"] == "settings_changed_elsewhere", stale.text
    assert "connection this model is sent through changed" in stale.json()["error"]
    assert _saved()[MAIN_CHAT_TIER] == ""
    again = _chat_ultrafast(client, _asked(model, "", connection=connection))
    assert again.status_code == 200, again.text
    assert _saved()[MAIN_CHAT_TIER] != kept and _state(client)["main_chat"]["processing_choice"] == "ultrafast"
    new_work = _main_work(root, "reconfirmed")
    assert new_work["processing"]["connection"] != old_connection
    assert send_processing_preference(work, model, False, "lead") == "economy"
    assert send_processing_preference(new_work, model, False, "lead") == "ultrafast"
    # A setting of another connection or the chip moves nothing of this one's accepted work.
    save_owner({**_saved(), "OPENROUTER_API_KEY": "sk-fictional-unrelated"})
    assert _state(client)["main_chat"]["processing_choice"] == "ultrafast"
    assert client.post(CHAT_MODEL, json={**CHOSEN, **MAIN_CHAT}).status_code == 200
    assert send_processing_preference(work, model, False, "lead") == "economy"
    assert send_processing_preference(new_work, model, False, "lead") == "ultrafast"
    # A rollback to the old payer's fields is a new revision, not authority to revive the old work's tier.
    save_owner({**_saved(), **first})
    assert live_connection_account(model) not in (old_connection, new_work["processing"]["connection"])
    assert send_processing_preference(work, model, False, "lead") == "economy"
    assert send_processing_preference(new_work, model, False, "lead") == "economy"


def test_legacy_api_carriers_cannot_assert_the_current_payer(room):
    root, _client = room
    save_owner({**_saved(), "ANTHROPIC_API_KEY": "fictional-key",
                "OUROBOROS_MAIN_CHAT_MODEL": OPUS,
                MAIN_CHAT_TIER: "ultrafast@" + chat_consent_word([OPUS, False, ""])})
    assert _main_work(root, "old-setting")["processing"] == DEFAULT
    assert send_processing_preference({"processing": _confirmed(OPUS, "")}, OPUS, False, "lead") == "economy"


def _confirmed(model: str = SOL, account: str = "acc-main", local: bool = False) -> dict:
    """The carrier of work admitted on the Main chat's Ultrafast: the tier with the one destination it was confirmed
    for, by value (its calls are another process's)."""
    return {"source": "room", "tier": "ultrafast", "route": [model, local, account]}


def _main_work(root, name: str) -> dict:
    task = {"id": name}
    route_choice.capture_lead(task, root)
    return task["metadata"]


def test_a_changed_system_route_does_not_inherit_the_main_chats_ultrafast(room):
    """Only the tier moved to Ultrafast while the Main chat follows Main; then Main the role's own route is changed
    on purpose, as a separate act. The consent names the old route: the new one starts no work on Ultrafast."""
    root, client = room
    assert _chat_ultrafast(client).status_code == 200
    kept = _saved()[MAIN_CHAT_TIER]
    assert _main_work(root, "on-the-asked-route") == {"lead": {"source": "main"}, "processing": _confirmed()}
    # Settings › Models moves Main's own model. Main's own tier is not Ultrafast, so that write asks nothing, and it
    # lands with the chat's answer ended: the route under it moved.
    moved = _main_model(client, ASTRA)
    assert moved.status_code == 200, moved.text
    assert kept.startswith("ultrafast@") and _saved()[MAIN_CHAT_TIER] == ""
    assert _state(client)["main_route"] == {"model": ASTRA, "use_local": False}
    assert _state(client)["main_chat"] == {"model_choice": {"kind": "main"}, "processing_choice": ""}
    assert _main_work(root, "on-the-new-route") == {"lead": {"source": "main"}, "processing": {"source": "default"}}
    assert task_processing_preference({"task_metadata": _main_work(root, "again")}, model_role="main") == "economy"
    # Ultrafast on the new route is a new question: the old answer names another route.
    assert _chat_ultrafast(client).status_code == 409
    asked = _chat_ultrafast(client, _asked(ASTRA))
    assert asked.status_code == 200, asked.text
    assert _saved()[MAIN_CHAT_TIER].startswith("ultrafast@") and _saved()[MAIN_CHAT_TIER] != kept  # another destination
    assert _state(client)["main_chat"]["processing_choice"] == "ultrafast"


def test_the_main_chats_ultrafast_ends_when_its_route_moves_and_a_return_asks_again(room, monkeypatch):
    """Ordinary Settings saves move Main's pinned profile away and back. The save that moves the route ends the
    answer for new work and the return brings nothing back. Asked again, Ultrafast is saved for the profile by its
    name: whoever is signed in to it then is not asked for."""
    root, client = room
    _observe(monkeypatch, _never_asked)
    assert _chat_ultrafast(client).status_code == 200
    # Another pinned profile under the same model is another route.
    save_owner({**_saved(), MODEL_ACCOUNTS_KEY: json.dumps({"main": "acc-other", "fallback": ["acc-fallback"]})})
    assert _saved()[MAIN_CHAT_TIER] == "" and _state(client)["main_chat"]["processing_choice"] == ""
    assert _main_work(root, "other-account")["processing"] == {"source": "default"}
    assert _chat_ultrafast(client).status_code == 409  # asked for the profile Main left
    save_owner({**_saved(), MODEL_ACCOUNTS_KEY: OWNER[MODEL_ACCOUNTS_KEY]})
    assert _saved()[MAIN_CHAT_TIER] == "" and _state(client)["main_chat"]["processing_choice"] == ""
    assert _main_work(root, "the-first-label-again")["processing"] == {"source": "default"}
    asked = _chat_ultrafast(client)
    assert asked.status_code == 200, asked.text
    assert _main_work(root, "asked-again")["processing"] == _confirmed()
    # A model picked in the capsule is another route too: the same press never carries the consent along.
    assert client.post(CHAT_MODEL, json={**CHOSEN, **MAIN_CHAT}).status_code == 200
    assert _saved()[MAIN_CHAT_TIER] == "" and _state(client)["main_chat"] == {"model_choice": CHOSEN, "processing_choice": ""}
    assert _main_work(root, "own-model")["processing"] == {"source": "default"}
    assert _chat_ultrafast(client).status_code == 409  # asked for Main's route, which the chat left
    assert _chat_ultrafast(client, _asked(ASTRA)).status_code == 409  # its own model runs on Auto, never on Main's profile
    own = _chat_ultrafast(client, _asked(ASTRA, ""))
    assert own.status_code == 200, own.text
    assert _main_work(root, "own-ultrafast")["processing"] == _confirmed(ASTRA, "")
    # Main the role moving under a chat with its own model changes nothing of that chat's route.
    assert _main_model(client, LUNA).status_code == 200
    assert _state(client)["main_chat"]["processing_choice"] == "ultrafast"


def test_a_new_room_never_copies_the_main_chats_ultrafast(room):
    root, client = room
    assert _chat_ultrafast(client).status_code == 200
    made = projects.create_project(root, "after-ultrafast", origin="owner_ui")
    assert "processing_choice" not in made and "model_choice" not in made
    assert client.post(CHAT_PROCESSING, json={"processing": "fast", **MAIN_CHAT}).status_code == 200
    assert projects.create_project(root, "after-fast", origin="owner_ui")["processing_choice"] == "fast"


def test_an_ultrafast_no_confirmed_write_saved_is_default(room):
    """Only the chat's confirmed write sets Ultrafast: a settings write that merely names the tier lands as Default,
    and a word no tier spells reads as Default."""
    root, client = room
    for stored in ("ultrafast", "ultrafast@", "ultrafast@0000", "fast@" + "0" * 24):
        save_owner({**_saved(), MAIN_CHAT_TIER: stored})
        assert _state(client)["main_chat"]["processing_choice"] == "", stored
        assert _main_work(root, f"stored-{stored}")["processing"] == {"source": "default"}
    assert not _saved()[MAIN_CHAT_TIER].startswith("ultrafast")
    # Even the right word for this destination, saved by anything but the chat's confirmed write.
    right = "ultrafast@" + chat_consent_word([SOL, False, "acc-main"])
    for stored in ("ultrafast", right):
        save_owner({**_saved(), MAIN_CHAT_TIER: stored})
        assert _saved()[MAIN_CHAT_TIER] == "" and _state(client)["main_chat"]["processing_choice"] == ""
    assert _chat_ultrafast(client).status_code == 200 and _saved()[MAIN_CHAT_TIER] == right


def test_the_main_chats_ultrafast_is_judged_on_one_locked_read_of_the_destination(room, monkeypatch):
    """The destination is compared under the lock with that write's one read, a tier already saved included, and
    by its names alone: no engine is asked who is signed in to the profile, before the lock or under it."""
    root, client = room
    _observe(monkeypatch, _never_asked)
    assert _chat_ultrafast(client).status_code == 200
    # Another window gave the Main chat its own model: only the chat's default changed. The old window's answer names
    # Main's route, and is told exactly that, whether or not Ultrafast is already saved for the new one.
    assert client.post(CHAT_MODEL, json={**CHOSEN, **MAIN_CHAT}).status_code == 200
    for saved_first in (False, True):
        if saved_first:
            assert _chat_ultrafast(client, _asked(ASTRA, "")).status_code == 200
        before = _saved()
        old = _chat_ultrafast(client)
        assert old.status_code == 409 and old.json()["code"] == "settings_changed_elsewhere", old.text
        assert old.json()["error"].startswith("The model or profile new work runs on here is not the one Ultrafast was asked for")
        assert _saved() == before
    # The same answer for the destination as it is saves nothing new and is no refusal.
    assert _chat_ultrafast(client, _asked(ASTRA, "")).status_code == 200 and _saved() == before


def test_a_saved_ultrafast_is_in_force_only_while_new_work_runs_on_the_route_it_names(room, monkeypatch):
    """The consent names the model, its locality and the profile. A new sign-in to that profile changes nothing the
    host reads, so the tier stays in force. A destination that moved without the settings writer (a document edited
    by hand here; a shipped default or an operator's variable does the same) is another route: new work takes the
    ordinary mode, the state says the saved Ultrafast is withheld and nothing is written for it; the same route
    again puts it back in force."""
    root, client = room
    assert _chat_ultrafast(client).status_code == 200
    saved = _saved()
    in_force = {"model_choice": {"kind": "main"}, "processing_choice": "ultrafast"}
    withheld = {"model_choice": {"kind": "main"}, "processing_choice": "", "processing_withheld": "ultrafast"}
    _observe(monkeypatch, _never_asked)  # whoever is signed in to the profile now
    assert _state(client)["main_chat"] == in_force
    assert _main_work(root, "after-a-new-sign-in")["processing"] == _confirmed()
    for index, moved in enumerate(({"OUROBOROS_MODEL": ASTRA}, {MODEL_ACCOUNTS_KEY: json.dumps({"main": "acc-other"})},
                                   {MODEL_ACCOUNTS_KEY: "{}"})):
        config.SETTINGS_PATH.write_text(json.dumps({**saved, **moved}), encoding="utf-8")  # no writer ended it
        assert _state(client)["main_chat"] == withheld, moved
        assert _main_work(root, f"withheld-{index}") == {"lead": {"source": "main"}, "processing": {"source": "default"}}
        assert _saved()[MAIN_CHAT_TIER] == saved[MAIN_CHAT_TIER]
    config.SETTINGS_PATH.write_text(json.dumps(saved), encoding="utf-8")
    assert _state(client)["main_chat"] == in_force
    assert _main_work(root, "the-same-route-again")["processing"] == _confirmed()
    # An unrelated settings save keeps the saved consent as it is: nothing moved under it.
    save_owner({**_saved(), GLOBAL: "fast"})
    assert _saved()[MAIN_CHAT_TIER] == saved[MAIN_CHAT_TIER] and _state(client)["main_chat"] == in_force


def test_the_consent_word_is_made_of_the_names_the_page_was_shown_and_of_no_key():
    """A pure rule: the same model, locality and profile name are the same word; any one of them changed is another.
    The word has no input but those three, so neither a connection's key or address nor an account's identity can be
    read back from it."""
    word = chat_consent_word([SOL, False, "acc-main"])
    assert word == chat_consent_word([SOL, False, "acc-main", "ignored-revision"]) == chat_consent_word((SOL, False, "acc-main"))
    assert len(word) == 24 and int(word, 16) >= 0
    others = [chat_consent_word([ASTRA, False, "acc-main"]), chat_consent_word([SOL, True, "acc-main"]),
              chat_consent_word([SOL, False, "acc-other"]), chat_consent_word([SOL, False, ""])]
    assert word not in others and len(set(others)) == len(others)


@pytest.mark.parametrize("url,body,code", [
    (CHAT_PROCESSING, {"processing": "fast"}, "chat_processing_invalid"),
    (CHAT_PROCESSING, {"processing": ""}, "chat_processing_invalid"),
    (CHAT_MODEL, CHOSEN, "chat_model_invalid"),
])
def test_a_write_that_names_no_scope_is_an_older_pages_and_saves_nothing(room, url, body, code):
    """A page loaded before the Main chat kept its own model and tier meant Main the role by a body without
    ``project_id``, and its plain Standard, Fast or Default press carried no binding tokens to tell it by. It is told
    to reload, and nothing is saved under another setting's name; a room's write, named by its project_id as ever,
    still saves."""
    root, client = room
    before = config.SETTINGS_PATH.read_bytes(), _registry(root)
    reply = client.post(url, json=body)
    assert reply.status_code == 400 and reply.json()["code"] == code, reply.text
    assert reply.json()["saved"] is False and "Reload the page" in reply.json()["error"]
    both = client.post(url, json={**body, "project_id": "alpha", **MAIN_CHAT})  # two scopes name none
    assert both.status_code == 400 and both.json()["code"] == code and both.json()["saved"] is False, both.text
    assert (config.SETTINGS_PATH.read_bytes(), _registry(root)) == before
    assert client.post(url, json={**body, "project_id": "alpha"}).status_code == 200
    assert config.SETTINGS_PATH.read_bytes() == before[0]  # the room's own row, never the settings document
    assert client.post(url, json={**body, **MAIN_CHAT}).status_code == 200


def test_a_tier_confirmed_for_one_destination_is_asked_of_that_destination_alone(room):
    """Pure policy on fictional routes over the fixture's ordinary save: the carrier names where its tier was
    confirmed, and a send bound to that model, locality and profile asks it. Anywhere else, and for a reader that
    names no send (a helper taking the work's tier), the default resolution stands. Carriers without a destination
    answer as they always did, and a destination without a named profile also needs the chat's word to stand."""
    root, client = room
    assert _chat_ultrafast(client).status_code == 200
    bound = {"lead": {"source": "main"}, "processing": _confirmed()}

    def asked(metadata, role, route):
        return task_processing_preference({"task_metadata": metadata}, model_role=role, route=route)

    assert asked(bound, "main", (SOL, False, "acc-main")) == "ultrafast"
    assert asked(bound, "main", lambda: [SOL, False, "acc-main"]) == "ultrafast"  # read only for such a carrier
    for elsewhere in ((ASTRA, False, "acc-main"), (SOL, False, "acc-other"), (SOL, False, ""), (SOL, True, "acc-main"), None):
        assert asked(bound, "main", elsewhere) == "economy", elsewhere  # the default resolution: what this work takes there
    # A fallback row is another destination: its authored tier, else the work's primary role's; never the confirmed one.
    assert asked(bound, "fallback:0", (LUNA, False, "acc-fallback")) == "fast"
    assert asked(bound, "fallback:1", (OPUS, False, "")) == "economy"
    # The same model and profile reached as a fallback row are the destination that was confirmed.
    assert asked(bound, "fallback:1", (SOL, False, "acc-main")) == "ultrafast"
    # No send named: a helper inheriting the work's tier, a nanny's frozen route, any older reader.
    assert route_choice.task_processing(bound) == ""
    assert route_choice.task_processing(bound, (SOL, False, "acc-main")) == "ultrafast"
    assert task_processing_preference({"task_metadata": bound}) == task_processing_preference({"metadata": bound}) == "economy"
    # A room's own Ultrafast and the owner's card act name no destination: they answer on every route, as before,
    # whatever the Main chat confirmed.
    for source in ("room", "owner"):
        free = {"lead": {"source": "main"}, "processing": {"source": source, "tier": "ultrafast"}}
        for role, route in (("main", None), ("main", (ASTRA, False, "")), ("fallback:0", (LUNA, False, "acc-fallback"))):
            assert asked(free, role, route) == "ultrafast", (source, role)
        assert route_choice.task_processing(free) == "ultrafast"
    # The carrier is host-made and validated whole: one that names a destination it cannot state confirms nothing.
    assert route_choice.processing_snapshot(_confirmed()) == _confirmed()
    assert route_choice.processing_snapshot({**_confirmed(), "extra": 1}) == _confirmed()
    for broken in ([SOL, False], [SOL, "no", "acc-main"], ["", False, ""], [SOL, False, None], "route", None):
        assert route_choice.processing_snapshot({"source": "room", "tier": "ultrafast", "route": broken}) is None, broken
        assert asked({"processing": {"source": "room", "tier": "ultrafast", "route": broken}}, "main", (SOL, False, "acc-main")) == "economy"
    # New work's carriers keep the destination its acceptance assigned; a broken one starts on Default.
    assigned = {"model_choice": {"kind": "main"}, "processing_choice": "ultrafast", "processing_route": [SOL, False, "acc-main"]}
    assert route_choice.new_work_carriers(assigned) == {"lead": {"source": "main"}, "processing": _confirmed()}
    assert route_choice.new_work_carriers({**assigned, "processing_route": [SOL]})["processing"] == DEFAULT
    assert route_choice.new_work_carriers({"model_choice": {"kind": "main"}, "processing_choice": "fast"})["processing"] == {
        "source": "room", "tier": "fast"}
    # The chat's own model runs on Auto: confirmed with no profile, and asked on Auto alone. That later choice is
    # new work's: the work admitted on Main's named profile carries its own answer.
    assert client.post(CHAT_MODEL, json={**CHOSEN, **MAIN_CHAT}).status_code == 200
    assert _chat_ultrafast(client, _asked(ASTRA, "")).status_code == 200
    own = {"lead": {"source": "room", "model": ASTRA, "use_local": False}, "processing": _confirmed(ASTRA, "")}
    assert asked(own, "lead", (ASTRA, False, "")) == "ultrafast"
    assert asked(own, "lead", (ASTRA, False, "acc-main")) == asked(own, "lead", (LUNA, False, "")) == "economy"
    assert asked(bound, "main", (SOL, False, "acc-main")) == "ultrafast"
    # Later tier picks apply only to new work even on the Auto pool.
    assert client.post(CHAT_PROCESSING, json={"processing": "fast", **MAIN_CHAT}).status_code == 200
    assert asked(own, "lead", (ASTRA, False, "")) == "ultrafast"
    assert asked(bound, "main", (SOL, False, "acc-main")) == "ultrafast"


def test_each_send_names_its_own_destination_and_a_moved_wait_asks_again(room):
    """The round's call, its wrap-up and a letter's candidate ask through one reader, which names the destination the
    send is bound to by the existing resolver: model, locality and account, a wait card's live choice included."""
    from ouroboros.model_slots import moved_send_processing_options

    _root, client = room
    assert _chat_ultrafast(client).status_code == 200
    bound = {"lead": {"source": "main"}, "processing": _confirmed()}

    def sent(model, role="main", local=False, **more):
        return send_processing_preference(bound, model, local, role, **more)

    assert sent(SOL) == "ultrafast"
    assert sent(ASTRA) == "economy"  # Main's model moved since admission
    assert sent(SOL, local=True) == "economy"
    assert sent(LUNA, "fallback:0") == "fast"
    # A wait card's choice for this work replaces the route exactly as its send applies it: another model, another
    # profile and Auto are not the profile that was confirmed.
    for chosen in ({"model": ASTRA, "use_local": False}, {"model_account_override": "acc-other"}, {"model_account_override": ""}):
        assert sent(SOL, overrides={"main": chosen}) == "economy", chosen
    assert sent(SOL, overrides={"main": {"model_account_override": "acc-main"}}) == "ultrafast"

    # A waiting send keeps the tier it captured; only a tier confirmed for one destination is asked again when the
    # wait card moves that send, and then for the route it now takes.
    def waiting(metadata, tier, model, **more):
        return moved_send_processing_options(metadata, tier, model, False, "main", **more)["processing_preference"]

    assert waiting(bound, "ultrafast", ASTRA) == "economy"
    assert waiting(bound, "ultrafast", SOL) == "ultrafast"
    assert waiting(bound, "ultrafast", SOL, overrides={"main": {"model_account_override": "acc-other"}}) == "economy"
    free = {"lead": {"source": "main"}, "processing": {"source": "owner", "tier": "ultrafast"}}
    assert waiting(free, "ultrafast", ASTRA) == "ultrafast" and waiting({}, "fast", ASTRA) == "fast"
    assert waiting(None, "fast", ASTRA) == "fast"


def test_the_card_says_a_confirmed_tier_only_where_its_calls_go_to_that_destination(room, monkeypatch):
    """What ``/api/state`` publishes for a work's card: the tier is the work's own choice while its calls are bound
    to the confirmed model and profile, said as bound to them (its helpers and calls elsewhere take the default); on
    a fallback, another model or another profile the card says no choice, names the tier as withheld and resolves
    what those calls take. What the chat chooses later is not the card's."""
    from ouroboros.lead_fact import processing_fields

    _root, client = room
    assert _chat_ultrafast(client).status_code == 200
    _observe(monkeypatch, _never_asked)
    row = {"metadata": {"lead": {"source": "main"}, "processing": _confirmed()}}
    on_main = {"model": SOL, "use_local": False, "basis": "main"}
    own = {"processing": {"choice": "ultrafast", "source": "room", "bound": True, "resolved": "ultrafast"}}
    held = {"processing": {"choice": "", "withheld": "ultrafast", "resolved": "economy"}}
    assert processing_fields(row, on_main) == own
    fallback = {"model": LUNA, "use_local": False, "basis": "fallback", "role": "fallback:0", "primary": on_main}
    assert processing_fields(row, fallback) == {"processing": {"choice": "", "withheld": "ultrafast", "resolved": "fast"}}
    moved = {"model": ASTRA, "use_local": False, "basis": "main"}  # Main the role changed under the running work
    assert processing_fields(row, moved) == held
    # The owner's own tier act on the card is this work's everywhere, as it always was; its receipt rides beside it.
    last = {"requested": "ultrafast", "submitted": "ultrafast", "observed": "ultrafast", "reason": None}
    chosen = {"metadata": {"lead": {"source": "main"}, "processing": {"source": "owner", "tier": "ultrafast"}}, "processing_last": last}
    for lead in (on_main, fallback, moved):
        assert processing_fields(chosen, lead) == {"processing": {"choice": "ultrafast", "source": "owner", "last": last, "resolved": "ultrafast"}}
    withheld_last = processing_fields({**row, "processing_last": {**last, "requested": "fast"}}, fallback)["processing"]
    assert (withheld_last["withheld"], withheld_last["last"]["requested"]) == ("ultrafast", "fast")
    # Main's pin moved to another profile under the running work: another destination, by its name. Back on the
    # profile it was confirmed for, the tier is the work's own again.
    save_owner({**_saved(), MODEL_ACCOUNTS_KEY: json.dumps({"main": "acc-other", "fallback": ["acc-fallback"]})})
    assert processing_fields(row, on_main) == held
    save_owner({**_saved(), MODEL_ACCOUNTS_KEY: OWNER[MODEL_ACCOUNTS_KEY]})
    assert processing_fields(row, on_main) == own
    # Those saves ended the chat's own Ultrafast, and another tier chosen there is new work's alone: the card of the
    # work admitted on it still says it, and the card's own act stays.
    assert _state(client)["main_chat"]["processing_choice"] == ""
    assert client.post(CHAT_PROCESSING, json={"processing": "fast", **MAIN_CHAT}).status_code == 200
    assert processing_fields(row, on_main) == own
    assert processing_fields(chosen, on_main)["processing"]["choice"] == "ultrafast"


def test_work_admitted_on_the_main_chats_ultrafast_asks_it_from_its_first_call_whatever_the_chat_chooses_later(room, monkeypatch):
    """The ordinary path with fictional settings: the owner confirms Ultrafast in the Main chat for Main's model on
    its pinned profile, a turn is accepted, and its calls are made by another process that holds what the turn
    carries. Each call bound to that model and profile asks Ultrafast, the first one included: no receipt, catalog
    reading or engine handshake is waited for, and nothing asks who is signed in to the profile. Another model,
    another profile and a fallback take what they would without this choice. What the chat chooses afterwards is new
    work's alone, in both directions. That is the named profile's rule alone: work admitted on Auto reads the choice
    as it stands."""
    from supervisor import worker_chat_lane
    from supervisor.message_ingress import new_work_choice
    from ouroboros.composer_cognition import saved_choice

    root, client = room
    assert _chat_ultrafast(client).status_code == 200
    seed = worker_chat_lane._seed_task(1, "Compare the two drafts", None, None, {})
    worker_chat_lane._stamp_admitted_choice(seed, root)  # the turn's route and tier are accepted here, once
    carried = json.loads(json.dumps(seed["metadata"]))  # what crosses to the carrier's process, by value
    assert carried == {"lead": {"source": "main"}, "processing": _confirmed()}
    _observe(monkeypatch, _never_asked)

    def sent(model=SOL, role="main", metadata=carried, **more):
        return send_processing_preference(metadata, model, False, role, **more)

    assert sent() == "ultrafast"  # the first call: the turn's own carrier is all it needs
    assert sent(LUNA, "fallback:0") == "fast"  # a fallback row: its own authored tier
    assert sent(ASTRA) == "economy"  # another model given to the work
    assert sent(overrides={"main": {"model_account_override": "acc-other"}}) == "economy"  # another profile, by a wait card
    # Main's pinned profile moves under the running work: its calls go to another profile, at its usual tier.
    save_owner({**_saved(), MODEL_ACCOUNTS_KEY: json.dumps({"main": "acc-other", "fallback": ["acc-fallback"]})})
    assert sent() == "economy"
    # Back on the profile it was admitted on, by the same name: its tier again. That save ended the chat's own
    # saved Ultrafast, which is new work's question; the admitted work reads its carrier.
    save_owner({**_saved(), MODEL_ACCOUNTS_KEY: OWNER[MODEL_ACCOUNTS_KEY]})
    assert _saved()[MAIN_CHAT_TIER] == "" and sent() == "ultrafast"
    later = worker_chat_lane._seed_task(1, "And a third draft", None, None, {})
    worker_chat_lane._stamp_admitted_choice(later, root)
    assert later["metadata"] == {"lead": {"source": "main"}, "processing": {"source": "default"}}
    assert sent(metadata=later["metadata"]) == "economy"
    # The chat chooses again: Ultrafast, another tier, Default. None of it re-prices work already admitted: the one
    # admitted on Default stays there, the one admitted on Ultrafast keeps it.
    assert _chat_ultrafast(client).status_code == 200
    assert sent(metadata=later["metadata"]) == "economy"
    for tier in ("fast", ""):
        assert client.post(CHAT_PROCESSING, json={"processing": tier, **MAIN_CHAT}).status_code == 200
        assert sent() == "ultrafast"
    # A model picked in the capsule is another route for new work; the admitted work stays on Main's, on its tier.
    picked = client.post(CHAT_MODEL, json={**CHOSEN, **MAIN_CHAT})
    assert picked.status_code == 200 and _state(client)["main_chat"] == {"model_choice": CHOSEN, "processing_choice": ""}
    assert sent() == "ultrafast"
    # Ultrafast for the picked model is asked on its own, on Auto: words sent as new work are assigned the tier with
    # that destination and ask it on Auto alone, never on a profile nobody was asked about.
    assert _chat_ultrafast(client, _asked(ASTRA, "")).status_code == 200
    token = saved_choice(root, "")["choice_token"]
    assert token == picked.json()["choice_token"]
    assigned = new_work_choice(root, 1, "Now on the other model", None, token)
    assert assigned == {"model_choice": CHOSEN, "processing_choice": "ultrafast", "processing_route": [ASTRA, False, ""]}
    new_work = json.loads(json.dumps(route_choice.new_work_carriers(assigned)))
    assert new_work == {"lead": {"source": "room", "model": ASTRA, "use_local": False}, "processing": _confirmed(ASTRA, "")}
    assert sent(ASTRA, "lead", new_work) == "ultrafast"
    assert sent(ASTRA, "lead", new_work, overrides={"lead": {"model_account_override": "acc-main"}}) == "economy"
    assert sent(LUNA, "fallback:0", new_work) == "fast"
    assert sent() == "ultrafast"  # the earlier work, still on Main's route and its own tier
    assert _state(client)["main_chat"] == {"model_choice": CHOSEN, "processing_choice": "ultrafast"}, "no destination is published"
    # Default chosen here is new work's, even when the earlier work is on the source's Auto pool.
    assert client.post(CHAT_PROCESSING, json={"processing": "", **MAIN_CHAT}).status_code == 200
    assert sent(ASTRA, "lead", new_work) == sent() == "ultrafast"
    after = _main_work(root, "after-default")
    assert sent(ASTRA, "lead", after) == "economy"
    assert client.post(CHAT_MODEL, json={"kind": "main", **MAIN_CHAT}).status_code == 200
    assert sent(ASTRA, "lead", new_work) == sent() == "ultrafast"


def test_the_main_chat_follows_mains_destination_as_mains_own_rule_resolves_it(room):
    """One resolver names Main's destination for both Ultrafast rules: a model the document labels local is local
    whatever its box says, and the pinned account belongs to a subscription route alone."""
    from ouroboros.gateway.owner_effort import _chat_destination, _main_destination
    from ouroboros.gateway.owner_settings import settings_in_place_of

    for values, expected in (
            ({}, [SOL, False, "acc-main"]),
            ({"OUROBOROS_MODEL": "qwen3 (local)", "USE_LOCAL_MAIN": False}, ["qwen3 (local)", True, ""]),
            ({"OUROBOROS_MODEL": OPUS}, [OPUS, False, ""])):
        settings = settings_in_place_of({**_saved(), **values})
        assert _main_destination(settings) == _chat_destination(settings) == expected
    own = settings_in_place_of({**_saved(), "OUROBOROS_MAIN_CHAT_MODEL": ASTRA})
    assert _chat_destination(own) == [ASTRA, False, ""] and _main_destination(own) == [SOL, False, "acc-main"]


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


def test_a_room_takes_ultrafast_without_writing_main(room):
    root, client = room
    settings = _saved()
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


MAIN_TIER_KEYS = ("OUROBOROS_MODEL", "USE_LOCAL_MAIN", MODEL_ACCOUNTS_KEY, ROLES)


def _binding(connection: str = "", *, model=None) -> dict:
    """The four read tokens, confirmed destination and its API revision if applicable."""
    from ouroboros.gateway.owner_settings import CONNECTION_CONDITION, connection_revision, setting_tokens

    saved = _saved()
    target = model or saved["OUROBOROS_MODEL"]
    pin = json.loads(saved[MODEL_ACCOUNTS_KEY]).get("main", "") if target.startswith("claudexor::codex=") else ""
    confirmed = json.dumps([target, False, pin, "fixture-store" if pin else "", "fixture-account" if pin else ""])
    return {"ultrafast": confirmed, **setting_tokens(saved, MAIN_TIER_KEYS), **({CONNECTION_CONDITION: connection_revision(saved, connection)} if connection else {})}


def test_mains_own_ultrafast_never_moves_to_another_system_route_unconfirmed(room):
    """Only Main the role's tier moved to Ultrafast, confirmed for its route. A separate change of that route
    (Settings › Models, or the owner's cited words through the role's writer) is refused until Ultrafast is
    confirmed for the new one or the tier is changed first: the old consent allows no other route."""
    from ouroboros.gateway.owner_effort import BindingRefused, _write_main_route

    _, client = room
    assert _main_tier(client, "ultrafast", _binding()).status_code == 200
    before = config.SETTINGS_PATH.read_bytes()
    moved = _main_model(client, ASTRA)
    assert 400 <= moved.status_code < 500 and moved.json()["saved"] is False, moved.text
    assert "Confirm Ultrafast" in moved.json()["error"] and "Settings → Models" in moved.json()["error"]
    with pytest.raises(BindingRefused, match="Confirm Ultrafast"):
        _write_main_route(None, {"kind": "model", "model": ASTRA, "use_local": False})
    assert config.SETTINGS_PATH.read_bytes() == before
    assert _state(client)["main_route"] == {"model": SOL, "use_local": False}
    confirmed = _main_model(client, ASTRA, _binding(model=ASTRA))  # asked again, for the new route
    assert confirmed.status_code == 200, confirmed.text
    assert _state(client)["main_route"] == {"model": ASTRA, "use_local": False}
    assert _state(client)["processing_tiers"]["main"] == "ultrafast"


def test_a_composer_write_naming_mains_binding_is_refused_and_saves_nothing(room):
    """A page loaded before the Main chat kept its own model and tier still sends Main's read tokens."""
    root, client = room
    tokens = _binding()
    before = config.SETTINGS_PATH.read_bytes(), _registry(root)
    for url, body, code in ((CHAT_PROCESSING, {"processing": "ultrafast"}, "chat_processing_invalid"),
                            (CHAT_PROCESSING, {"processing": "standard", "project_id": "alpha"}, "chat_processing_invalid"),
                            (CHAT_MODEL, CHOSEN, "chat_model_invalid")):
        for condition in (tokens, {}, None):
            reply = client.post(url, json={**body, "if_unchanged": condition})
            assert reply.status_code == 400 and reply.json()["code"] == code, reply.text
            assert reply.json()["saved"] is False and "Reload the page" in reply.json()["error"]
            assert (config.SETTINGS_PATH.read_bytes(), _registry(root)) == before


@pytest.mark.parametrize("changed", MAIN_TIER_KEYS)
def test_main_ultrafast_compares_every_read_token_before_writing(room, changed):
    from ouroboros.gateway.owner_settings import setting_tokens

    _, client = room
    tokens = _binding()
    # A token from a different read of just this field, with every other token current.
    other = {"OUROBOROS_MODEL": ASTRA, "USE_LOCAL_MAIN": True,
             MODEL_ACCOUNTS_KEY: '{"main":"another"}', ROLES: '{"main":"standard"}'}
    tokens.update(setting_tokens({**_saved(), changed: other[changed]}, [changed]))
    before = config.SETTINGS_PATH.read_bytes()
    reply = _main_tier(client, "ultrafast", tokens)
    assert reply.status_code == 409 and reply.json()["code"] == "settings_changed_elsewhere", reply.text
    assert reply.json()["saved"] is False and config.SETTINGS_PATH.read_bytes() == before


def test_main_ultrafast_conditions_survive_an_unrelated_settings_save(room):
    _, client = room
    before = _saved()
    tokens = _binding()
    _booted({**before, GLOBAL: "fast"})
    reply = _main_tier(client, "ultrafast", tokens)
    assert reply.status_code == 200, reply.text
    saved = _saved()
    assert saved[GLOBAL] == "fast" and json.loads(saved[ROLES]) == {**json.loads(before[ROLES]), "main": "ultrafast"}


@pytest.mark.parametrize("new_identity", [("fixture-store", "replacement"), ("replacement-store", "fixture-account"), None])
def test_replaced_pinned_identity_refuses_the_settings_writer(room, monkeypatch, new_identity):
    _, client = room
    assert _saved()["OUROBOROS_CONSCIOUSNESS_SAFETY_MODE"] == "full"  # the booted install the fixture saved
    seen = _binding()
    before = config.SETTINGS_PATH.read_bytes()
    monkeypatch.setattr("ouroboros.gateway.models.observed_profile", lambda source, pin: new_identity)
    reply = client.post(MODELS, json={"settings": {ROLES: {"main": "ultrafast"}}, "if_unchanged": seen})
    assert reply.status_code == 409 and not reply.json()["saved"]
    assert config.SETTINGS_PATH.read_bytes() == before


def test_safety_model_writer_returns_shared_invalid_main_binding_code(room):
    _, client = room
    before = config.SETTINGS_PATH.read_bytes()
    reply = client.post(MODELS, json={"settings": {ROLES: {"main": "ultrafast"}}})
    assert reply.status_code == 400, reply.text
    assert reply.json()["code"] == "main_binding_invalid" and reply.json()["saved"] is False
    assert config.SETTINGS_PATH.read_bytes() == before


@pytest.mark.parametrize("writer", ["tier", "model"])
def test_a_choice_refused_for_a_replaced_pinned_identity_is_saved_when_made_again_on_the_one_observed(room, monkeypatch, writer):
    """The owner's second choice after that 409. The refusal wrote nothing and moved no token, so the same read with
    the store and fingerprint a catalog reread names is saved; the identity first shown, sent again, is refused again."""
    _, client = room
    if writer == "model":  # Main already holds Ultrafast: its model change carries the tier to the new destination
        assert _main_tier(client, "ultrafast", _binding()).status_code == 200

    def write(tokens):
        return _main_model(client, ASTRA, tokens) if writer == "model" else _main_tier(client, "ultrafast", tokens)

    shown = _binding(model=ASTRA) if writer == "model" else _binding()
    observed = ("replacement-store", "replacement-account")
    monkeypatch.setattr("ouroboros.gateway.models.observed_profile", lambda source, pin: observed)
    before = config.SETTINGS_PATH.read_bytes()
    for _ in range(2):
        refused = write(shown)
        assert refused.status_code == 409 and refused.json()["code"] == "settings_changed_elsewhere", refused.text
        assert refused.json()["saved"] is False and config.SETTINGS_PATH.read_bytes() == before
    again = {**shown, "ultrafast": json.dumps([*json.loads(shown["ultrafast"])[:3], *observed])}
    reply = write(again)
    assert reply.status_code == 200, reply.text
    saved = _saved()
    assert saved["OUROBOROS_MODEL"] == (ASTRA if writer == "model" else SOL)
    assert json.loads(saved[ROLES])["main"] == "ultrafast" and json.loads(saved[MODEL_ACCOUNTS_KEY])["main"] == "acc-main"


def test_auto_ultrafast_confirms_the_model_without_freezing_account_membership(room, monkeypatch):
    _, client = room
    _booted({**OWNER, MODEL_ACCOUNTS_KEY: "{}"})
    monkeypatch.setattr("ouroboros.gateway.models.observed_profile", lambda *_: pytest.fail("Auto never reads an account"))
    assert _main_tier(client, "ultrafast", _binding()).status_code == 200
    before = config.SETTINGS_PATH.read_bytes()
    assert _main_model(client, ASTRA).status_code == 400
    assert config.SETTINGS_PATH.read_bytes() == before
    assert _main_model(client, ASTRA, _binding(model=ASTRA)).status_code == 200


WORK = {"kind": "openai", "name": "work", "api_key": "sk-test-first-synthetic-key"}
KEYED = {**OWNER, "OUROBOROS_MODEL": "openai@work::gpt-6.1", MODEL_ACCOUNTS_KEY: json.dumps({"fallback": ["acc-fallback"]}),
         "OUROBOROS_CONNECTIONS": json.dumps([WORK])}


def _connections(*rows) -> dict:
    return {"OUROBOROS_CONNECTIONS": json.dumps(list(rows))}


@pytest.mark.parametrize("edit", [
    _connections({**WORK, "api_key": "sk-test-second-synthetic-key"}),  # the named key rotated under the same route
    _connections({**WORK, "api_surface": "responses"}),  # the named connection's API surface
], ids=["named-credential", "api-surface"])
def test_main_ultrafast_binds_the_named_connection_under_the_same_route(room, edit):
    """Window 1 reads Main's binding; window 2 changes the connection Main's route names, not its route: window 1's
    Ultrafast is refused. The revision says no secret, and another connection's or setting's edit never moves it."""
    from ouroboros.gateway.owner_settings import BindingRefused, connection_revisions

    _, client = room
    _booted(KEYED)
    tokens = _binding("openai@work")
    assert "sk-test" not in json.dumps(tokens) and tokens["connection"].startswith("openai@work#")
    other = {"kind": "openai", "name": "home", "api_key": "sk-test-home-synthetic-key"}
    _booted({**KEYED, "OUROBOROS_BG_WAKEUP_MIN": 750, **_connections(WORK, other)})  # unrelated edits
    assert _binding("openai@work") == tokens
    rows = connection_revisions([{"id": "openai@work"}, {"id": "openai@home"}], _saved())
    assert rows[0]["revision"] == tokens["connection"] and rows[1]["revision"].startswith("openai@home#")
    assert "sk-test" not in json.dumps(rows)
    _booted({**KEYED, "OUROBOROS_BG_WAKEUP_MIN": 750, **edit})
    assert _binding("openai@work")["connection"] != tokens["connection"]
    before = config.SETTINGS_PATH.read_bytes()
    reply = _main_tier(client, "ultrafast", tokens)
    assert reply.status_code == 409 and reply.json()["code"] == "settings_changed_elsewhere", reply.text
    assert reply.json()["saved"] is False and config.SETTINGS_PATH.read_bytes() == before
    selected = _main_tier(client, "ultrafast", _binding("openai@work"))
    assert selected.status_code == 200, selected.text
    saved = _saved()
    selected_binding = _binding("openai@work")
    with pytest.raises(BindingRefused, match="Standard or Default"):
        save_owner({**saved, **_connections(WORK)})
    assert _saved() == saved
    save_owner({**saved, **_connections(*json.loads(edit["OUROBOROS_CONNECTIONS"]), other)})
    assert _binding("openai@work") == selected_binding


def test_confirming_ultrafast_again_checks_the_same_unauthored_main_the_page_read(room):
    from ouroboros.gateway.owner_settings import connection_revision
    from ouroboros.gateway.settings import api_settings_get

    _, client = room
    _booted({"OPENAI_API_KEY": "fixture-openai-only"})
    client.app.router.routes.append(Route("/api/settings", api_settings_get, methods=["GET"]))

    def read_choice():
        result = client.get("/api/settings")
        assert result.status_code == 200, result.text
        shown = result.json()
        model, meta = shown["OUROBOROS_MODEL"], shown["_meta"]
        assert "OUROBOROS_MODEL" not in _saved() and model.startswith("openai::")
        revision = next(item["revision"] for item in meta["connections"] if item["id"] == "openai")
        return {**{key: meta["tokens"][key] for key in MAIN_TIER_KEYS}, "connection": revision,
                "ultrafast": json.dumps([model, False, "", "", ""])}

    def choose(seen):
        return _main_tier(client, "ultrafast", seen)

    first = choose(read_choice())
    assert first.status_code == 200 and json.loads(_saved()[ROLES])["main"] == "ultrafast", first.text
    unchanged = config.SETTINGS_PATH.read_bytes()
    again = read_choice()  # fresh tokens for the now-saved tier, still no authored model
    same = choose(again)
    assert same.status_code == 200 and config.SETTINGS_PATH.read_bytes() == unchanged, same.text
    # Another connection's revision is no word on this route: a write that would move the tier there is refused.
    assert _main_tier(client, "", again).status_code == 200
    wrong = choose({**read_choice(), "connection": connection_revision(_saved(), "openrouter")})
    assert wrong.status_code == 400 and wrong.json()["saved"] is False, wrong.text
    assert "main" not in json.loads(_saved()[ROLES])


def test_main_ultrafast_on_an_api_connection_needs_its_revision(room):
    _, client = room
    _booted(KEYED)
    before = config.SETTINGS_PATH.read_bytes()
    bare = _main_tier(client, "ultrafast", _binding())
    assert bare.status_code == 400 and bare.json()["code"] == "main_binding_invalid", bare.text
    assert config.SETTINGS_PATH.read_bytes() == before
    reply = _main_tier(client, "ultrafast", _binding("openai@work"))
    assert reply.status_code == 200, reply.text


def test_mains_model_change_stands_on_the_tier_it_was_shown(room):
    """Window 1 shows Main with its tier and asks to change the model; window 2 sets Main's Ultrafast meanwhile. Window
    1's write is refused (nothing of it lands) instead of carrying Ultrafast to the new model unasked; read again, it
    writes, and an unrelated save between read and write does not refuse it."""
    _, client = room
    shown = _binding()
    assert _main_tier(client, "ultrafast", _binding()).status_code == 200
    before = config.SETTINGS_PATH.read_bytes()
    refused = _main_model(client, ASTRA, shown)
    assert refused.status_code == 409 and refused.json()["code"] == "settings_changed_elsewhere", refused.text
    assert refused.json()["saved"] is False and config.SETTINGS_PATH.read_bytes() == before
    again = _binding(model=ASTRA)
    _booted({**_saved(), "OUROBOROS_BG_WAKEUP_MIN": 750})
    reply = _main_model(client, ASTRA, again)
    assert reply.status_code == 200, reply.text
    saved = _saved()
    assert saved["OUROBOROS_MODEL"] == ASTRA and json.loads(saved[ROLES])["main"] == "ultrafast"
    assert saved["OUROBOROS_BG_WAKEUP_MIN"] == 750


def test_a_model_change_carrying_ultrafast_onto_a_key_stands_on_that_key_as_shown(room):
    """The picked route's connection, not Main's old one: its revision as window 1 showed it is the condition."""
    _, client = room
    _booted({**OWNER, **_connections(WORK)})
    assert _main_tier(client, "ultrafast", _binding()).status_code == 200
    picked = "openai@work::gpt-6.1"
    bare = _main_model(client, picked, _binding())  # the old route's confirmation says nothing of the picked one
    assert bare.status_code == 400 and bare.json()["code"] == "main_binding_invalid", bare.text
    other = {**_binding(), "connection": _binding("openai@work", model=picked)["connection"].replace("openai@work", "openai")}
    refused = _main_model(client, picked, other)  # a revision of another connection is no current read of this one
    assert refused.status_code in (400, 409) and refused.json()["saved"] is False, refused.text
    assert _saved()["OUROBOROS_MODEL"] == SOL
    shown = _binding("openai@work", model=picked)
    _booted({**_saved(), **_connections({**WORK, "api_key": "sk-test-rotated-synthetic-key"})})
    rotated = _main_model(client, picked, {**_binding(), "connection": shown["connection"]})
    assert rotated.status_code == 409 and rotated.json()["saved"] is False, rotated.text
    assert _saved()["OUROBOROS_MODEL"] == SOL, "nothing of the refused change landed"
    reply = _main_model(client, picked, _binding("openai@work", model=picked))
    assert reply.status_code == 200, reply.text


@pytest.mark.parametrize("body", [
    {**CHOSEN, "project_id": "alpha", "if_unchanged": {}},  # a room's model names no binding of Main
    {**CHOSEN, "if_unchanged": {"OUROBOROS_MODEL": "x"}},
    {**CHOSEN, "if_unchanged": {**{key: "x" for key in MAIN_TIER_KEYS}, GLOBAL: "x"}},
    {**CHOSEN, "if_unchanged": "tokens"},
    {**CHOSEN, "if_unchanged": {**{key: "x" for key in MAIN_TIER_KEYS}, "connection": "openai#x"}},  # a subscription has none
])
def test_a_model_choice_with_a_malformed_binding_saves_nothing(room, body):
    root, client = room
    before = config.SETTINGS_PATH.read_bytes(), _registry(root)
    reply = client.post(CHAT_MODEL, json=body)
    assert reply.status_code == 400 and reply.json()["code"] == "chat_model_invalid", reply.text
    assert reply.json()["saved"] is False and (config.SETTINGS_PATH.read_bytes(), _registry(root)) == before


@pytest.mark.parametrize("reset", ["", "standard"], ids=["default", "standard"])
def test_main_ultrafast_round_trips_until_the_owner_resets_it_for_rollback(room, reset):
    from ouroboros.settings_setup_contract import build_initial_setup_state

    _root, client = room
    before = config.SETTINGS_PATH.read_bytes()
    refused = _main_tier(client, "ultrafast")
    assert refused.status_code == 400 and not refused.json()["saved"] and config.SETTINGS_PATH.read_bytes() == before
    selected = _main_tier(client, "ultrafast", _binding())
    assert selected.status_code == 200, selected.text
    roles = {**json.loads(OWNER[ROLES]), "main": "ultrafast"}
    loaded = config.load_settings(strict=True)
    assert json.loads(_saved()[ROLES]) == json.loads(loaded[ROLES]) == roles
    assert build_initial_setup_state(loaded)["modelProcessingPreferences"] == roles
    assert _state(client)["processing_tiers"] == {"main": "ultrafast", "default": "economy"}
    work = {"metadata": {"lead": {"source": "main"}, "processing": DEFAULT}}
    assert task_processing_preference(work) == "ultrafast"
    own_model = {"source": "room", "model": ASTRA, "use_local": False}
    assert task_processing_preference({"metadata": {"lead": own_model}}) == "economy"
    assert task_processing_preference(work, model_role="fallback:0") == "fast"
    assert task_processing_preference(work, model_role="fallback:1") == "ultrafast"

    # An unrelated Settings edit goes through a different reader and writer.
    config.save_settings({**loaded, "OUROBOROS_BG_WAKEUP_MIN": 750})
    assert json.loads(config.load_settings(strict=True)[ROLES]) == roles
    saved = _main_tier(client, reset)
    assert saved.status_code == 200, saved.text
    compatible = json.loads(OWNER[ROLES])
    if reset:
        compatible["main"] = reset
    assert json.loads(_saved()[ROLES]) == compatible
    assert task_processing_preference(work) == (reset or "economy")
    assert _saved()["OUROBOROS_BG_WAKEUP_MIN"] == 750


@pytest.mark.parametrize("values", [
    {ROLES: json.dumps({"main": "ultrafast", "light": "ultrafast"})},
    {ROLES: json.dumps({"light": "economy", "fallback": ["ultrafast"]})},
    {GLOBAL: "ultrafast"},
])
def test_a_settings_write_of_ultrafast_outside_main_is_refused_atomically(room, values):
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
    assert _main_tier(client, "fast").status_code == 200
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
    # The admitted tier reaches the row beside the admitted model, a decided Default and a Main turn's too (the Main
    # chat has its own choice to drift to), so a step rebuilt from the row never reads either as new.
    assert [frame for frame in frames if "processing" in (frame.get("facts") or {})] == [{
        "type": "direct_turn_bound", "task_id": turn["id"], "task_attempt": 1,
        "facts": {"lead": {"source": "main"}, "processing": expected}}]


@pytest.mark.parametrize("project_id", ["alpha", ""])
@pytest.mark.parametrize("tier", ["", "fast"])
def test_a_turn_rebuilt_from_its_row_keeps_the_model_and_tier_it_was_admitted_with(room, project_id, tier):
    """A Pause and its Resume, a park or a restore rebuild a turn from its queue row, in another process than the
    carrier that admitted it. The row learned the admitted model and tier by frame (the decided Default and the Main
    chat's own choice included), so what the chip chose since reaches only turns admitted after it."""
    from supervisor import conversation_lane, worker_chat_lane

    root, client = room
    where = {"project_id": project_id} if project_id else {}
    scope = where or MAIN_CHAT
    assert client.post(CHAT_MODEL, json={**CHOSEN, **scope}).status_code == 200
    if tier:
        assert client.post(CHAT_PROCESSING, json={"processing": tier, **scope}).status_code == 200
    chat_id = projects.get_project(root, "alpha")["chat_id"] if project_id else 1
    row = worker_chat_lane._seed_task(chat_id, "Hello", None, None, dict(where))
    row.update(project_id=project_id, _conversation_prepare={"quiet": False})
    admitted = {"lead": {"source": "room", "model": ASTRA, "use_local": False},
                "processing": {"source": "room", "tier": tier} if tier else DEFAULT}
    held = lambda task: {key: (task.get("metadata") or {}).get(key) for key in admitted}  # noqa: E731
    running = {row["id"]: {"task": row, "attempt": 1}}
    carried = copy.deepcopy(row)  # the carrier's own copy, as its process receives it
    frames: list = []
    worker_chat_lane.prepare_conversation_task(types.SimpleNamespace(), carried, root, broadcast=frames.append)
    assert held(carried) == admitted and held(row) == {"lead": None, "processing": None}
    for frame in frames:
        if frame.get("type") == "direct_turn_bound":
            conversation_lane.handle_direct_turn_bound(copy.deepcopy(frame), types.SimpleNamespace(RUNNING=running))
    assert held(row) == admitted
    # The owner moves the chip while the turn is paused: the next task's choice, never this turn's.
    assert client.post(CHAT_MODEL, json={"kind": "model", "model": LUNA, "use_local": False, **scope}).status_code == 200
    assert client.post(CHAT_PROCESSING, json={"processing": "economy", **scope}).status_code == 200
    again = copy.deepcopy(row)  # its Resume hands the row to a carrier again, the same attempt
    frames.clear()
    worker_chat_lane.prepare_conversation_task(types.SimpleNamespace(), again, root, broadcast=frames.append)
    assert held(again) == admitted
    assert not [frame for frame in frames if set(admitted) & set(frame.get("facts") or {})], frames


@pytest.mark.parametrize("project_id", ["alpha", ""])
@pytest.mark.parametrize("tier", ["", "fast"])
def test_a_turn_takes_its_model_and_tier_once_as_it_is_admitted(room, project_id, tier):
    """Words accepted as work of their own take the model and tier saved then, the decided Default included, and keep
    them on the seed its queue row is made from. What the chip chooses while the turn waits for a carrier, is
    stopped, paused or resumed reaches only work accepted after it: the carrier's first step admits nothing new."""
    from supervisor import worker_chat_lane

    root, client = room
    scope = {"project_id": project_id} if project_id else MAIN_CHAT
    assert client.post(CHAT_MODEL, json={**CHOSEN, **scope}).status_code == 200
    if tier:
        assert client.post(CHAT_PROCESSING, json={"processing": tier, **scope}).status_code == 200
    chat_id = projects.get_project(root, "alpha")["chat_id"] if project_id else 1
    seed = worker_chat_lane._seed_task(chat_id, "Hello", None, None, {"project_id": project_id} if project_id else {})
    worker_chat_lane._stamp_admitted_choice(seed, root)
    admitted = {"lead": {"source": "room", "model": ASTRA, "use_local": False},
                "processing": {"source": "room", "tier": tier} if tier else DEFAULT}
    assert {key: seed["metadata"].get(key) for key in admitted} == admitted
    # The chip moves before any carrier took the turn.
    assert client.post(CHAT_MODEL, json={"kind": "model", "model": LUNA, "use_local": False, **scope}).status_code == 200
    assert client.post(CHAT_PROCESSING, json={"processing": "economy", **scope}).status_code == 200
    carried = {**copy.deepcopy(seed), "_conversation_prepare": {"quiet": False}}  # the carrier's copy of the row
    frames: list = []
    worker_chat_lane.prepare_conversation_task(types.SimpleNamespace(), carried, root, broadcast=frames.append)
    assert {key: carried["metadata"].get(key) for key in admitted} == admitted
    assert not [frame for frame in frames if set(admitted) & set(frame.get("facts") or {})], frames
    # The next turn is the one that starts on what the chip says now.
    later = worker_chat_lane._seed_task(chat_id, "Again", None, None, {"project_id": project_id} if project_id else {})
    worker_chat_lane._stamp_admitted_choice(later, root)
    assert later["metadata"]["lead"]["model"] == LUNA and later["metadata"]["processing"] == ROOM_ECONOMY


def test_the_admission_stamp_never_refuses_a_turn_and_leaves_the_capture_to_its_carrier(room, monkeypatch):
    """A room that cannot be read, or a fault in the capture itself, stamps nothing at admission and never raises:
    the turn is admitted as before, and its carrier's first step is the one that captures then (the frame test
    above)."""
    from supervisor import worker_chat_lane

    root, _client = room
    seed = worker_chat_lane._seed_task(1, "Hello", None, None, {})

    def broken(*_args, **_kwargs):
        raise TypeError("a fault of the capture's own")

    with monkeypatch.context() as patched:
        patched.setattr("ouroboros.route_choice.capture_lead", broken)
        worker_chat_lane._stamp_admitted_choice(seed, root)
    assert "lead" not in seed.get("metadata", {}) and "processing" not in seed.get("metadata", {})
    gone = worker_chat_lane._seed_task(7, "Hello", None, None, {"project_id": "nowhere"})
    worker_chat_lane._stamp_admitted_choice(gone, root)
    assert "lead" not in gone["metadata"] and "processing" not in gone["metadata"]
    # Words accepted as new work already carry what their acceptance assigned: nothing is chosen again.
    assigned = {"model_choice": CHOSEN, "processing_choice": "fast"}
    new_work = worker_chat_lane._seed_task(1, "Hello", None, None, {"new_work": assigned})
    before = copy.deepcopy(new_work["metadata"])
    worker_chat_lane._stamp_admitted_choice(new_work, root)
    assert new_work["metadata"] == before == {"lead": {"source": "room", "model": ASTRA, "use_local": False},
                                              "processing": {"source": "room", "tier": "fast"}}


def test_the_row_keeps_a_newer_model_or_tier_than_the_admission_frame_brings(room):
    """An owner's act on the work (its model, its tier) and new work's own carriers reach the row first: the
    admission frame that lands after them changes neither."""
    from supervisor import conversation_lane

    lead, tier = {"source": "room", "model": ASTRA, "use_local": False}, {"source": "room", "tier": "fast"}
    newer = {"lead": {"source": "owner", "model": LUNA, "use_local": False}, "processing": {"source": "owner", "tier": "economy"}}
    frame = {"type": "direct_turn_bound", "task_id": "turn", "task_attempt": 1, "facts": {"lead": lead, "processing": tier}}
    for row, expected in (
            ({"id": "turn", "metadata": {}}, {"lead": lead, "processing": tier}),
            ({"id": "turn"}, {"lead": lead, "processing": tier}),
            ({"id": "turn", "metadata": dict(newer), "owner_model": {"request_id": "r"}, "owner_processing": {"request_id": "p"}}, newer),
            ({"id": "turn", "metadata": {"lead": {"source": "main"}}}, {"lead": {"source": "main"}, "processing": tier})):
        conversation_lane.handle_direct_turn_bound(copy.deepcopy(frame), types.SimpleNamespace(RUNNING={"turn": {"task": row, "attempt": 1}}))
        assert {key: row["metadata"][key] for key in ("lead", "processing") if key in row["metadata"]} == expected
    # A frame of another attempt, or one that names no model the host could have admitted, teaches the row nothing.
    row = {"id": "turn", "metadata": {}}
    for stale in ({**frame, "task_attempt": 2}, {**frame, "facts": {"lead": {"source": "elsewhere", "model": ASTRA, "use_local": False}}},
                  {**frame, "facts": {"lead": {"source": "room", "model": "", "use_local": False}}}):
        conversation_lane.handle_direct_turn_bound(copy.deepcopy(stale), types.SimpleNamespace(RUNNING={"turn": {"task": row, "attempt": 1}}))
    assert row["metadata"] == {}


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


def test_a_stale_binding_is_409_before_any_destination_rule(room):
    """Window 1 was shown Fast and picks a key model, so it rightly names no connection; window 2 sets Ultrafast while
    the question stands. The lock judges staleness first: 409 and nothing saved, never a 400 for a missing revision."""
    _, client = room
    _booted({**OWNER, ROLES: json.dumps({"main": "fast"}), **_connections(WORK)})
    shown = _binding()
    assert _main_tier(client, "ultrafast", _binding()).status_code == 200
    before = config.SETTINGS_PATH.read_bytes()
    reply = _main_model(client, "openai@work::gpt-6.1", shown)
    assert reply.status_code == 409 and reply.json()["code"] == "settings_changed_elsewhere", reply.text
    assert reply.json()["saved"] is False and config.SETTINGS_PATH.read_bytes() == before


@pytest.mark.parametrize("direction", ["api_to_subscription", "subscription_to_api"])
def test_main_ultrafast_read_on_one_route_is_409_after_main_moved_to_the_other_kind(room, direction):
    """The question was shown for Main on a key (with its revision) or on a subscription (with none); window 2 moves
    Main to the other kind before the write. Its tokens are stale: 409, whatever the new route would need."""
    _, client = room
    keyed = {**OWNER, "OUROBOROS_MODEL": "openai@work::gpt-6.1", MODEL_ACCOUNTS_KEY: json.dumps({}), **_connections(WORK)}
    start, then = (keyed, {**keyed, "OUROBOROS_MODEL": SOL}) if direction == "api_to_subscription" else ({**OWNER, **_connections(WORK)}, keyed)
    _booted(start)
    shown = _binding("openai@work" if direction == "api_to_subscription" else "")
    _booted(then)
    before = config.SETTINGS_PATH.read_bytes()
    reply = _main_tier(client, "ultrafast", shown)
    assert reply.status_code == 409 and reply.json()["code"] == "settings_changed_elsewhere", reply.text
    assert reply.json()["saved"] is False and config.SETTINGS_PATH.read_bytes() == before


def test_an_openrouter_revision_moves_with_the_routing_its_requests_apply(room):
    """The global provider filter decides which endpoint carries a tier, so it is part of the key's revision; an
    unrelated setting is not. One effective view serves a whole GET's rows."""
    from ouroboros.gateway.owner_settings import connection_revision, connection_revisions

    save_owner({**OWNER, "OPENROUTER_API_KEY": "sk-or-test-synthetic"})
    first = connection_revision(_saved(), "openrouter")
    save_owner({**_saved(), "OUROBOROS_BG_WAKEUP_MIN": 750})
    assert connection_revision(_saved(), "openrouter") == first
    save_owner({**_saved(), "OUROBOROS_OR_PROVIDER": json.dumps({"only": ["synthetic-provider"]})})
    filtered = connection_revision(_saved(), "openrouter")
    assert filtered != first and "sk-or" not in filtered
    assert connection_revisions([{"id": "openrouter"}], _saved()) == [{"id": "openrouter", "revision": filtered}]
