"""A connector opens a meeting call through its connected place, and only where a focus would start now."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from ouroboros import places as p
from ouroboros import presence_call as door
from tests.test_place_focus import focus as focus  # noqa: F401  (fixture)


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
    assert refused.value.code == "presence_payload_invalid"


ME = {"sends_as": {"id": "me", "label": "ouroboros-agent", "kind": "user"}}


def test_a_call_event_carries_the_connectors_identity_and_its_own_account_opens_nothing(tmp_path, place, opened):
    # Another person's event with the negotiated identity opens the call, as it does without one.
    assert _open(tmp_path, event=_event(identity=ME, facts=["mention"])).call_id
    # The account this place sends as is heard, not answered: no call, no hub, the room keeps it with its reason.
    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path, event=_event(source_event_id="e2", actor={"id": "me"}, identity=ME, facts=["mention"]))
    assert refused.value.code == "place_own_account"
    assert len(opened["hubs"]) == 1 and "own_account" in _heard(tmp_path)[-1][1]
    # A malformed identity is refused before the room hears it, with the code /presence/turn answers.
    with pytest.raises(door.PresenceCallRefused) as refused:
        _open(tmp_path, event=_event(source_event_id="e3", identity={"sends_as": {"id": "me"}}, facts=["mention"]))
    assert refused.value.code == "presence_identity_invalid"
    assert len(_heard(tmp_path)) == 2


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


def test_the_skill_door_answers_reactions_and_identities_as_the_turn_door_does(tmp_path, place, opened, monkeypatch):
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
    assert (heard.status_code, heard.json()["code"]) == (403, "place_reaction_context_only")
    assert len(_heard(tmp_path)) == 1
    assert opened["hubs"] == []
    own = client.post("/call/open", json={"role": "meeting",
                                          "event": _event(source_event_id="e4", actor={"id": "me"}, identity=ME,
                                                          facts=["mention"])})
    assert (own.status_code, own.json()["code"]) == (403, "place_own_account")
    malformed = client.post("/call/open", json={"role": "meeting",
                                                "event": _event(source_event_id="e5", identity=[], facts=["mention"])})
    assert (malformed.status_code, malformed.json()["code"]) == (400, "presence_identity_invalid")


# One event parser for both connector doors (#2867): the same wire event through ``/presence/turn`` and
# ``/call/open`` of one Host Service app. Real: routing, admission, places.json, capture, attention, the
# own-account and owner rules, the text door's carrier re-admission and ``focus_metadata``. Doubles: skill-token
# authentication, the call's Live Conversation feature/hub/thread (``opened``), the model and the live projection.
SIGNAL, OWNER = "signal.words", "owner-account"


def _wire(source_event_id, **over):
    return {"source_event_id": source_event_id, "provider": "chat", "account_id": "a", "conversation_id": "r",
            "thread_id": "", "conversation_key": "ignored-by-both-doors",
            "actor": {"platform_actor_id": "person", "display_name": "Person"},
            "conversation": {}, "message": {}, "text": "hello", "facts": ["mention"], **over}


@pytest.fixture
def doors(focus, opened, monkeypatch):
    from starlette.testclient import TestClient

    import ouroboros.presence_events as presence_events
    from ouroboros.extension_contributions import project_contributions
    from ouroboros.extension_ui_validation import validate_contributions
    from ouroboros.gateway import host_call, host_service
    from ouroboros.presence_runner import run_presence_turn
    from ouroboros.task_results import write_task_result
    from tests.test_extension_icons import PLACE
    from tests.test_skill_contributions import manifest

    root, place, _admission, _event = focus

    async def authenticated(*_args):
        return "bridge", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    monkeypatch.setattr(host_call, "_authenticated", authenticated)
    monkeypatch.setattr(host_service.HostServiceContext, "require_permission", lambda *_args: None)
    monkeypatch.setattr("ouroboros.config.get_consciousness_autonomy", lambda: "act")
    clean, errors = validate_contributions(manifest([{
        **PLACE, "label": "Chat", "destination": "vendor:Chat", "wakes": [],
        "facts": ["mention", "any_message", "from_bot", {"id": SIGNAL, "label": "Words you chose appear"}],
        "settings": "connection"}], kind="extension", permissions=["presence"]))
    assert not errors
    rows = project_contributions("bridge", clean, "reviewed")
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: rows)
    place = p.edit_place(root, place.place_id, "add_wake", author="owner", fact=SIGNAL)
    place = p.edit_place(root, place.place_id, "confirm_owner_account", author="owner", platform_actor_id=OWNER)
    assert place.state == "active"

    class Hub:
        def __init__(self, spec):
            self.spec, self.closed = spec, False

        def grant_peer(self, *_args, **_kwargs):
            return None
    monkeypatch.setattr(host_call, "get_hub",
                        lambda call_id: next((Hub(s) for s in opened["hubs"] if s.call_id == call_id), None))
    captured, reparses, real_capture, real_parse = {}, [], door.capture_presence_event, presence_events.presence_turn_event

    def capture(admission, event, **kwargs):
        captured[event.source_event_id] = real_capture(admission, event, **kwargs)
        return captured[event.source_event_id]

    def reparse(payload, admission, **kwargs):  # the carrier imports the parser when it runs; the call door did at load
        event = real_parse(payload, admission, **kwargs)
        reparses.append((event.source_event_id, list(kwargs.get("vouched_facts") or ())))
        return event
    monkeypatch.setattr(door, "capture_presence_event", capture)
    monkeypatch.setattr(presence_events, "presence_turn_event", reparse)

    class Model:
        def handle_task(self, task):
            write_task_result(root, task["id"], "completed", metadata={
                **task["metadata"], "presence_outcome": "silent", "presence_result_text": ""},
                terminal_origin="model_final", result="")
            return [{"type": "presence_result", "outcome": "silent", "text": ""}]

    with TestClient(host_service.create_host_service_app(root)) as client:
        client.app.state.host_service_context.presence_runner = lambda **kw: run_presence_turn(
            repo_dir=root, drive_root=root, agent_factory=lambda **_kw: Model(), **kw)
        yield SimpleNamespace(root=root, place=place, client=client, captured=captured, reparses=reparses)


FOCUS, OPENED = (200, "focus"), (200, "opened")  # the text door's turn ran its carrier; the call door opened the call
INVALID_ACTOR, INVALID_PAYLOAD = (400, "presence_actor_invalid"), (400, "presence_payload_invalid")
# name, event overrides, /presence/turn answer, /call/open answer. An own-account event is heard by both doors and
# answered as each door answers a heard event: silence for a turn, a refused call; that difference is the doors'.
DOOR_CASES = [
    ("plain", {}, FOCUS, OPENED),
    ("authenticated_true", {"actor": {"platform_actor_id": "person", "authenticated": True}}, FOCUS, OPENED),
    ("authenticated_false", {"actor": {"platform_actor_id": "person", "authenticated": False}}, FOCUS, OPENED),
    ("authenticated_string", {"actor": {"platform_actor_id": "person", "authenticated": "false"}},
     INVALID_ACTOR, INVALID_ACTOR),
    ("authenticated_int", {"actor": {"platform_actor_id": "person", "authenticated": 0}}, INVALID_ACTOR, INVALID_ACTOR),
    ("actor_not_object", {"actor": "person"}, INVALID_ACTOR, INVALID_ACTOR),
    ("actor_empty", {"actor": {}}, (400, "presence_identity_missing"), (400, "presence_identity_missing")),
    ("owner", {"actor": {"platform_actor_id": OWNER}}, FOCUS, OPENED),
    ("owner_unverified", {"actor": {"platform_actor_id": OWNER, "authenticated": False}}, FOCUS, OPENED),
    ("owner_authenticated_string", {"actor": {"platform_actor_id": OWNER, "authenticated": "false"}},
     INVALID_ACTOR, INVALID_ACTOR),
    ("declared_signal", {"facts": [SIGNAL]}, FOCUS, OPENED),
    ("declared_signal_with_core", {"facts": ["mention", SIGNAL]}, FOCUS, OPENED),
    ("undeclared_signal", {"facts": ["signal.other"]}, INVALID_PAYLOAD, INVALID_PAYLOAD),
    ("undeclared_word", {"facts": ["shouting"]}, INVALID_PAYLOAD, INVALID_PAYLOAD),
    ("nonstring_fact", {"facts": [1]}, INVALID_PAYLOAD, INVALID_PAYLOAD),
    ("unhashable_fact", {"facts": [["mention"]]}, INVALID_PAYLOAD, INVALID_PAYLOAD),
    ("facts_not_list", {"facts": "mention"}, INVALID_PAYLOAD, INVALID_PAYLOAD),
    ("source_order_valid", {"message": {"source_order": 7}}, FOCUS, OPENED),
    ("source_order_negative", {"message": {"source_order": -1}}, INVALID_PAYLOAD, INVALID_PAYLOAD),
    ("source_order_string", {"message": {"source_order": "7"}}, INVALID_PAYLOAD, INVALID_PAYLOAD),
    ("transport_queue_malformed", {"conversation": {"transport_queue": {"schema_version": 2}}},
     INVALID_PAYLOAD, INVALID_PAYLOAD),
    ("identity_malformed", {"identity": []}, (400, "presence_identity_invalid"), (400, "presence_identity_invalid")),
    ("identity_other_actor", {"identity": ME}, FOCUS, OPENED),
    ("own_account", {"actor": {"platform_actor_id": "me"}, "identity": ME}, (200, "heard"), (403, "place_own_account")),
    ("own_account_unverified", {"actor": {"platform_actor_id": "me", "authenticated": False}, "identity": ME},
     FOCUS, OPENED),
    ("own_account_authenticated_string", {"actor": {"platform_actor_id": "me", "authenticated": "false"},
                                          "identity": ME}, INVALID_ACTOR, INVALID_ACTOR),
    ("extra_event_key", {"surprise": 1}, (400, "presence_event_invalid"), (400, "presence_event_invalid")),
]


def _rooms_heard(root, source_event_id):
    from ouroboros import dialogue_store

    return len([row for row in dialogue_store.rows_by_client_id(root, source_event_id) if row.row.get("direction") == "in"])


def test_both_connector_doors_parse_one_event_with_one_set_of_answers(doors, opened):
    observed, expected = {}, {}
    for name, over, text_answer, call_answer in DOOR_CASES:
        hubs, carried = len(opened["hubs"]), len(doors.reparses)
        text = doors.client.post("/presence/turn", json={"event": _wire(f"t-{name}", **over), "staged_files": [],
                                                         "continuation_version": 1, "delivery_reporting_version": 1})
        ran = any(sid == f"t-{name}" for sid, _facts in doors.reparses[carried:])
        call = doors.client.post("/call/open", json={"role": "meeting", "event": _wire(f"c-{name}", **over)})
        text_body, call_body = text.json(), call.json()
        observed[name] = {
            "text": (text.status_code, ("focus" if ran else "heard") if text.status_code == 200 else text_body["code"]),
            "call": (call.status_code, "opened" if call.status_code == 200 else call_body["code"]),
            "heard": (_rooms_heard(doors.root, f"t-{name}"), _rooms_heard(doors.root, f"c-{name}")),
            "hub": len(opened["hubs"]) - hubs,
        }
        refused = text_answer[0] == 400
        expected[name] = {"text": text_answer, "call": call_answer, "heard": (0, 0) if refused else (1, 1),
                          "hub": int(call_answer == OPENED)}
        if refused:  # the text door keeps its refusal envelope
            assert text_body["disposition"] == "rejected", name
    assert observed == expected
    assert doors.client.post("/presence/turn", json={"event": _wire("words", facts=["shouting"])}).json()["error"] \
        == "event.facts must be a list of declared wake facts"
    # The call door hears the connector's declared signal; the carrier re-admits the Host's facts by value.
    assert ("t-declared_signal", [SIGNAL]) in doors.reparses
    assert ("t-declared_signal_with_core", ["mention", SIGNAL]) in doors.reparses
    # Owner provenance follows the boolean alone: an unverified owner address never confirms the owner.
    owner = doors.captured["c-owner"]["metadata"]["focus"]["actor"]
    unverified = doors.captured["c-owner_unverified"]["metadata"]["focus"]["actor"]
    assert owner["owner_confirmed"] is True and "owner_address_unverified" not in owner
    assert unverified["owner_confirmed"] is False and unverified["owner_address_unverified"] is True


def test_a_long_calls_heard_utterances_keep_the_account_its_place_sends_as(doors):
    """``recent_identity`` reads the place's newest 50 arrivals; a call's heard utterances are arrivals, so they carry
    the identity its focus admitted (``focus.identity``) instead of pushing the declaration out of that window."""
    import json

    from ouroboros import dialogue_store, place_arrivals, place_guard

    root, place = doors.root, doors.place
    sends = {"sends_as": {"id": OWNER, "label": "Owner account", "kind": "user"}}
    assert doors.client.post("/call/open", json={"role": "meeting", "event": _wire("call", identity=sends)}).status_code == 200
    task = json.loads(json.dumps(doors.captured["call"]))  # the call's own task, as capture built it (``_build_task``)
    task["metadata"]["execution_transport"] = {"kind": "live_conversation"}  # as run_presence_turn marks it
    assert task["metadata"]["focus"]["identity"] == sends
    for number in range(60):
        assert door.record_call_utterance(root, task, direction="in", text=f"utterance {number}",
                                          speaker_label="Ann", utterance_id=f"item-{number}")
    assert door.record_call_utterance(root, task, direction="out", text="spoken answer", utterance_id="say-1")
    assert place_arrivals.recent_identity(root, place.place_id) == sends
    # Safety's facts for a reply from work outside this focus read that window.
    facts = place_guard.call_place_facts(SimpleNamespace(drive_root=root, task_metadata={}), "place_reply", {
        "place_id": place.place_id, "destination": {"account_id": "a", "conversation_id": "r", "thread_id": ""}})
    row, = facts["places"]
    assert row["sends_as"] == sends["sends_as"] and row["sends_as_owner"] is True
    heard, = dialogue_store.rows_by_client_id(root, "call:heard:item-0")
    assert heard.row["transport"]["identity"] == sends
    assert heard.row["presence_event_identity"] == task["metadata"]["presence_event_identity"]
