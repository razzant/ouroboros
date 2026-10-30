"""A connector opens a meeting call through its connected place, and only where a focus would start now."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from ouroboros import places as p
from ouroboros import presence_call as door


def _event(**extra):
    return {"source_event_id": "e1", "provider": "meet", "account_id": "a", "conversation_id": "room",
            "thread_id": "", "conversation_key": "ignored", "actor": {"id": "u1", "display_name": "Ann"},
            "conversation": {}, "message": {}, "text": "joined the call", **extra}


@pytest.fixture
def place(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance",
                        lambda _root: {"status": "available", "remaining_usd": 2})
    monkeypatch.setattr("ouroboros.config.get_consciousness_autonomy", lambda: "act")
    return p.connect_place(tmp_path, connector="meet_bridge", provider="meet", account_id="a",
                           conversation_id="room", wakes=["mention"])


@pytest.fixture
def opened(monkeypatch):
    """Fakes past the door: the feature, the hub and the turn thread; records what opened."""
    record = {"hubs": [], "closed": [], "started": []}

    @contextlib.contextmanager
    def configuration(_name):
        yield object(), SimpleNamespace(enabled=record.get("enabled", True), provider="gemini_live",
                                        model="m", voice="v")

    class Turn:
        def __init__(self, _feature, _config, spec, publish=None):
            self.spec = spec

    class Lease:
        def release(self):
            record["released"] = True

    monkeypatch.setattr("ouroboros.live_conversation.feature_configuration", configuration)
    monkeypatch.setattr("ouroboros.live_conversation.LiveConversationTurn", Turn)
    monkeypatch.setattr("ouroboros.call_peers.open_hub", lambda spec: record["hubs"].append(spec))
    monkeypatch.setattr("ouroboros.call_peers.close_hub", lambda call_id, reason: record["closed"].append(reason))
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: SimpleNamespace(broadcast=lambda *_: None))
    monkeypatch.setattr(door, "_configured_gate", lambda _root: SimpleNamespace(try_acquire=lambda _key: Lease()))
    monkeypatch.setattr(door, "presence_turn_replay", lambda *args: None)
    monkeypatch.setattr(door.PROACTIVE_TURNS, "start_thread",
                        lambda task_id, **kw: (record["started"].append(task_id), (object(), True))[1])
    return record


def _open(tmp_path, **kw):
    return door.open_presence_call(drive_root=tmp_path, repo_dir=tmp_path, skill_name=kw.pop("skill", "meet_bridge"),
                                   event=kw.pop("event", _event(facts=["mention"])), **kw)


def _heard(tmp_path):
    from ouroboros import dialogue_store

    with dialogue_store.transaction(tmp_path, write=False) as conn:
        rows = conn.execute("SELECT body, dispatch_input FROM dialogue WHERE stream='chat'").fetchall()
    return [(dialogue_store.decode(row["body"]).get("direction"), row["dispatch_input"]) for row in rows]


def test_a_waking_active_place_opens_the_call_under_its_place(tmp_path, place, opened):
    result = _open(tmp_path)
    assert result.call_id and opened["started"] == [result.task_id]
    assert [spec.presence_binding_id for spec in opened["hubs"]] == [place.place_id]
    assert opened["closed"] == []
    # The request was heard in the place's room before the decision, and no attention verdict settled it.
    heard = _heard(tmp_path)
    assert [direction for direction, _ in heard] == ["in"] and "place_attention" not in (heard[0][1] or "")
    # Naming the place by id opens it the same way.
    assert _open(tmp_path, binding_id=place.place_id, event=_event(source_event_id="e2", facts=["mention"])).call_id


@pytest.mark.parametrize("change, facts, autonomy, code", [
    (("set_state", {"state": "silent"}), ["mention"], "act", "place_silent"),
    (("set_state", {"state": "paused"}), ["mention"], "act", "place_paused"),
    (None, ["direct_message"], "act", "place_no_wake_match"),
    (None, None, "act", "place_no_wake_match"),
    (None, ["mention"], "observe", "place_advises"),
])
def test_a_call_never_opens_where_a_focus_would_not_start(tmp_path, place, opened, monkeypatch,
                                                          change, facts, autonomy, code):
    if change:
        p.edit_place(tmp_path, place.place_id, change[0], author="owner", **change[1])
    monkeypatch.setattr("ouroboros.config.get_consciousness_autonomy", lambda: autonomy)
    event = _event(**({"facts": facts} if facts is not None else {}))
    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path, event=event)
    assert refused.value.code == code
    assert opened["hubs"] == [] and opened["started"] == []
    # The refused request stays a heard arrival carrying the reason, as a text event would.
    (direction, facts_json), = _heard(tmp_path)
    assert direction == "in" and code.removeprefix("place_") in facts_json


def test_consciousness_off_and_an_exhausted_allowance_refuse_before_any_hub(tmp_path, place, opened, monkeypatch):
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: False)
    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path)
    assert refused.value.code == "place_off"
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "exhausted"})
    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path)
    assert refused.value.code == "consciousness_allowance_exhausted"
    assert opened["hubs"] == []


def test_only_the_places_own_connector_opens_and_calls_must_be_on(tmp_path, place, opened):
    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path, skill="other_bridge")
    assert refused.value.code == "place_not_connected"
    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path, skill="other_bridge", binding_id=place.place_id)
    assert refused.value.code == "presence_binding_wrong_transport"
    opened["enabled"] = False
    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path)
    assert refused.value.code == "calls_disabled"
    assert opened["hubs"] == []


def test_event_facts_must_be_declared_wake_facts(tmp_path, place, opened):
    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path, event=_event(facts=["shouting"]))
    assert refused.value.code == "presence_event_invalid"


def test_host_door_checks_the_connected_place_of_the_connector(tmp_path, place):
    from ouroboros.gateway import host_call

    ctx = SimpleNamespace(data_dir=tmp_path)
    assert host_call._place(ctx, "meet_bridge", place.place_id).place_id == place.place_id
    with pytest.raises(ValueError):
        host_call._place(ctx, "other_bridge", place.place_id)
    with pytest.raises(ValueError):
        host_call._place(ctx, "meet_bridge", "unknown")


THUMB = {"type": "name", "value": "thumbsup"}
ANON = {"kind": "anonymous", "authenticated": False}


def _reaction(kind="add", actor=None, text="", **fact):
    values = {"add": {"new": [THUMB]}, "count": {"counts": [{**THUMB, "count": 3}]}}[kind]
    reaction = {"schema_version": 1, "change": kind, **values, "target": {"message_id": "m1", "resolution": "resolved"}}
    actor = actor or (ANON if kind == "count" else {"kind": "user", "platform_actor_id": "u1", "display_name": "Ann"})
    return _event(actor=actor, message={"reaction": {**reaction, **fact}}, text=text, facts=["mention"])


@pytest.mark.parametrize("event", [
    _reaction("count", counts=3),  # a shape /presence/turn refuses; a reader of its count would raise
    _reaction(change="noted"),
    _reaction(text="thumbs up!"),  # a reaction carries no words
    _reaction("count", actor={"kind": "user", "platform_actor_id": "u1"}),  # an aggregate has no individual
])
def test_an_invalid_typed_reaction_is_refused_before_the_room_hears_it(tmp_path, place, opened, event):
    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path, event=event)
    assert refused.value.code == "presence_reaction_invalid"
    assert _heard(tmp_path) == [] and opened["hubs"] == [] and opened["started"] == []


@pytest.mark.parametrize("event", [_reaction(), _reaction("count")])
def test_a_valid_typed_reaction_is_heard_as_context_and_opens_no_call(tmp_path, place, opened, event):
    from ouroboros import dialogue_store

    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path, event=event)
    assert refused.value.code == "place_reaction_context_only"
    (direction, facts_json), = _heard(tmp_path)
    assert direction == "in" and "reaction_context_only" in facts_json
    row, = [entry.row for entry in dialogue_store.iter_rows(tmp_path, "chat")]
    assert row["text"] == "" and row["transport"]["message"] == event["message"]
    assert opened["hubs"] == [] and opened["started"] == []


def test_an_older_connectors_own_reaction_key_keeps_the_call_path(tmp_path, place, opened):
    assert _open(tmp_path, event=_event(message={"reaction": "thumbsup"}, facts=["mention"])).call_id
    assert opened["started"]


def test_the_skill_door_answers_an_invalid_reaction_as_the_turn_door_does(tmp_path, place, opened, monkeypatch):
    from starlette.testclient import TestClient

    from ouroboros.gateway import host_call, host_service

    async def authenticated(*_args):
        return "meet_bridge", {}
    monkeypatch.setattr(host_call, "_authenticated", authenticated)
    monkeypatch.setattr(host_service.HostServiceContext, "require_permission", lambda *_args: None)
    client = TestClient(host_service.create_host_service_app(tmp_path))
    refused = client.post("/call/open", json={"role": "meeting", "event": _reaction("count", counts=3)})
    assert (refused.status_code, refused.json()["code"]) == (400, "presence_reaction_invalid")
    assert _heard(tmp_path) == []
    heard = client.post("/call/open", json={"role": "meeting", "event": _reaction()})
    assert heard.json()["code"] == "place_reaction_context_only" and len(_heard(tmp_path)) == 1
    assert opened["hubs"] == []
