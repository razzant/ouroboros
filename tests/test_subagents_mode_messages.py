"""The Swarm annotation is an acceptance fact, shared by live echo and history."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import config, projects_registry as projects
from ouroboros.composer_cognition import update_project_cognition
from ouroboros.subagent_runtime import subagents_availability
from ouroboros.gateway.history import make_chat_history_endpoint
from supervisor import message_ingress
from tests.test_task_settings_snapshot import document, isolated_runtime, save_owner  # noqa: F401
from tests.test_chat_attachments import _web_bridge


def mode_settings(mode, availability):
    values = {**document("old"), "OUROBOROS_SUBAGENTS_MODE": mode}
    if availability == "disabled":
        values["OUROBOROS_SUBAGENTS"] = '{"enabled":false,"items":[]}'
    elif availability == "unsaved":
        values = {"OUROBOROS_SUBAGENTS_MODE": mode, "OUROBOROS_MODEL_HEAVY": "owner/unsaved-candidate"}
    return values


@pytest.mark.parametrize("availability", ["available", "disabled", "unsaved"])
@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("accepted,later", [("swarm", "off"), ("auto", "swarm"), ("off", "swarm")])
def test_acceptance_label_survives_room_switch_and_duplicate_echo(monkeypatch, project, accepted, later, availability):
    values = mode_settings(accepted, availability)
    save_owner(values)
    settings_reader = config.load_settings
    if availability == "unsaved":
        # Isolate the undecided loader result from this install's factory proposals.
        monkeypatch.setattr(config, "load_settings", lambda: dict(values))
    assert subagents_availability(config.load_settings()) == availability
    room = projects.create_project(config.DATA_DIR, "alpha") if project else {"chat_id": 1, "id": ""}
    bridge, echoes = _web_bridge(config.DATA_DIR, monkeypatch)
    sent = dict(broadcast=True, sender_session_id="surface", client_message_id="accepted",
                chat_id=room["chat_id"], project_id=room["id"])
    bridge.ui_send("Owner words", **sent)
    row = message_ingress.accepted_chat_message(config.DATA_DIR, room["chat_id"], "accepted")
    monkeypatch.setattr(config, "load_settings", settings_reader)
    if project:
        update_project_cognition(config.DATA_DIR, "alpha", subagents_mode=later)
    save_owner(mode_settings("off" if project else later, "available"))
    bridge.ui_send("Owner words", **sent)
    assert message_ingress.accepted_chat_message(config.DATA_DIR, room["chat_id"], "accepted") == row
    assert row.get("subagents_mode") == ("swarm" if accepted == "swarm" and availability == "available" else None)
    assert len(echoes) == 2
    assert all(echo.get("subagents_mode") == row.get("subagents_mode") for echo in echoes)
    response = asyncio.run(make_chat_history_endpoint(config.DATA_DIR)(SimpleNamespace(
        query_params={"chat_id": str(room["chat_id"]), "limit": "10"})))
    rows = json.loads(response.body)["messages"]
    replay = next(item for item in rows if item.get("client_message_id") == "accepted")
    assert replay.get("subagents_mode") == row.get("subagents_mode")


def test_unannotated_legacy_history_is_readable_and_has_no_inferred_label():
    save_owner({**document("old"), "OUROBOROS_SUBAGENTS_MODE": "swarm"})
    path = config.DATA_DIR / "logs/chat.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"direction": "in", "chat_id": 1, "user_id": 1,
                               "text": "old row", "ts": "2026-10-01T01:00:00Z"}) + "\n")
    response = asyncio.run(make_chat_history_endpoint(config.DATA_DIR)(SimpleNamespace(query_params={})))
    replay = next(item for item in json.loads(response.body)["messages"] if item["text"] == "old row")
    assert "subagents_mode" not in replay


@pytest.mark.parametrize("room_state", ["readable", "deleting", "registry", "bindings", "settings"])
@pytest.mark.parametrize("surface", ["round", "message"])
def test_room_annotations_survive_unavailable_project(room_state, surface, monkeypatch, tmp_path):
    from ouroboros.loop_messages import append_context_facts
    from ouroboros.settings_integrity import live_subagents_mode, task_settings_scope
    from tests.test_project_cognition_runtime import snapshot
    from tests.test_round_facts_line import _facts_ctx

    save_owner(mode_settings("swarm", "available"))
    room = projects.create_project(config.DATA_DIR, "alpha")
    actor = snapshot({"id": "parent", "project_id": "alpha", "chat_id": room["chat_id"]})
    ctx = _facts_ctx(tmp_path, monkeypatch)
    if room_state == "deleting":
        projects.begin_project_deletion(config.DATA_DIR, "alpha")
    elif room_state in {"registry", "bindings"}:
        path = projects._registry_path if room_state == "registry" else projects._bindings_path
        path(config.DATA_DIR).write_text("{unreadable", encoding="utf-8")
    elif room_state == "settings":
        def unreadable():
            raise OSError("settings unavailable")
        monkeypatch.setattr(config, "load_settings", unreadable)

    with task_settings_scope(actor):
        if room_state != "readable":
            with pytest.raises((ValueError, OSError)):
                live_subagents_mode()  # only annotations degrade; admission reads remain strict
        if surface == "round":
            assert append_context_facts(ctx, money={})
            fact = ctx.messages[-1]["content"]
            assert "[CONTEXT_FACTS]" in fact and "own cost" in fact
            assert ("Subagents: Swarm" if room_state == "readable" else "Subagents: unknown") in fact
        else:
            row = message_ingress.log_chat("in", room["chat_id"], 1, "Owner words",
                drive_root=config.DATA_DIR, source="web", client_message_id="annotation")
            assert row is not None and row["text"] == "Owner words"
            assert row.get("subagents_mode") == ("swarm" if room_state == "readable" else None)
            assert message_ingress.accepted_chat_message(config.DATA_DIR, room["chat_id"], "annotation") == row


@pytest.mark.parametrize("mode", ["off", "auto", "swarm"])
@pytest.mark.parametrize("availability", ["available", "disabled", "unsaved"])
@pytest.mark.parametrize("project", [False, True])
def test_direct_owner_root_records_effective_admission_not_client_metadata(monkeypatch, mode, availability, project):
    from ouroboros import agent as agent_module
    from supervisor import workers
    from tests.test_native_conversation_activity import _lane

    values = mode_settings(mode, availability)
    save_owner(values)
    room = projects.create_project(config.DATA_DIR, "alpha") if project else {"chat_id": 1, "id": ""}
    if project:
        values = mode_settings("off" if mode == "swarm" else "swarm", availability)
        save_owner(values)
    if availability == "unsaved":
        monkeypatch.setattr(config, "load_settings", lambda: dict(values))
    assert subagents_availability(config.load_settings()) == availability
    _lane(monkeypatch, config.DATA_DIR)
    seen = []
    monkeypatch.setattr(agent_module, "make_agent", lambda **kwargs: SimpleNamespace(
        handle_task=lambda task: seen.append(task) or []))
    workers.handle_chat_direct(room["chat_id"], "Work", task_metadata={
        "subagents_mode": "forged", "force_plan": True, "project_id": room["id"]})
    assert len(seen) == 1
    assert seen[0]["metadata"]["subagents_mode"] == (mode if availability == "available" else "off")
    assert "subagents_mode" not in seen[0]["task_contract"]


def test_legacy_force_plan_frame_is_accepted_and_ignored(monkeypatch):
    from starlette.websockets import WebSocketDisconnect
    from ouroboros.gateway import ws
    from supervisor import message_bus

    delivered, warnings = [], []
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: SimpleNamespace(
        ui_send=lambda text, **kwargs: delivered.append((text, kwargs))))

    class Socket:
        app = SimpleNamespace(state=SimpleNamespace(drive_root=config.DATA_DIR))
        first = True

        async def accept(self):
            pass

        async def receive_text(self):
            if self.first:
                self.first = False
                return json.dumps({"type": "chat", "content": "ordinary request", "force_plan": True})
            for _ in range(100):
                if delivered:
                    break
                await asyncio.sleep(0.01)
            raise WebSocketDisconnect()

        async def send_text(self, text):
            warnings.append(text)

    asyncio.run(ws.ws_endpoint(Socket()))
    assert len(delivered) == 1 and not warnings
    assert delivered[0][0] == "ordinary request"
    assert "force_plan" not in delivered[0][1]["task_metadata"]
    assert "force_plan_source" not in delivered[0][1]["task_metadata"]


def test_state_serves_stored_main_mode_with_settings_lock():
    from ouroboros.gateway.state import api_state

    save_owner({**document("old"), "OUROBOROS_SUBAGENTS_MODE": "swarm",
                "OUROBOROS_SUBAGENTS": '{"enabled":false,"items":[]}'})
    response = asyncio.run(api_state(SimpleNamespace(query_params={}, app=SimpleNamespace(
        state=SimpleNamespace(drive_root=config.DATA_DIR)))))
    assert response.status_code == 200, response.body
    payload = json.loads(response.body)
    assert payload["subagents_mode"] == "swarm"
    assert payload["subagents_availability"] == "disabled"
