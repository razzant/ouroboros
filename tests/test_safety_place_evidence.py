"""The occasion is separate from destinations and their standing delegation."""
import json

import pytest

from ouroboros import place_guard as _place_guard
from ouroboros import tool_effects as _tool_effects
from tests._safety_fixtures import supervisor as supervisor_fixture
from tests.test_tool_effects import published_tool as _published_tool
supervisor, published_tool = supervisor_fixture, _published_tool
REAL_CALL_PLACE_FACTS = _place_guard.call_place_facts  # bound before the fixture stubs it
REAL_EFFECT_OF = _tool_effects.effect_of


def snapshot(rows):
    """One read of the place store as ``places.place_records`` answers it: these places, no circles."""
    from ouroboros import places
    return places.PlaceRecords(0, {row.place_id: row for row in rows}, (), {})


def bundle(h, name, arguments):
    assert h.module.check_safety(name, arguments, ctx=h.ctx)[0]
    return json.loads(h.calls[-1]["user_prompt"].split(
        "Host evidence (provenance is not consent):\n", 1)[1].split("\nProposed tool call:", 1)[0])


@pytest.mark.parametrize("name", ["start_task", "remote_write"])
def test_focus_and_arbitrary_place_arguments_are_not_destinations(supervisor, monkeypatch, name):
    from ouroboros import places, place_guard
    h = supervisor
    h.ctx.task_metadata["focus"] = {"place_id": "focus-room"}
    record = places.Place("target", "connector", "chat", label="Target",
                          notes=(places.NoteEntry("For this target only", "owner"),))
    reads = []
    monkeypatch.setattr(places, "place_records", lambda root: reads.append(root) or snapshot([record]))
    data = bundle(h, name, {"place_id": "target", "text": "unrelated work"})
    assert data["subject"]["place"] == {} and not data["standing_delegation"]["place_notes"]
    assert data["occasion"]["focus"]["place_id"] == "focus-room" and not reads
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *_a, **_k: {"places": [{"place_id": "target"}]})
    data = bundle(h, name, {"text": "resolved work"})
    assert data["subject"]["place"]["places"][0]["place_id"] == "target"
    assert data["standing_delegation"]["place_notes"][0]["standing_delegation"] is True


@pytest.mark.parametrize(("name", "canonical_source"), [
    ("configure_presence", "context"),
    ("place_reply", "metadata"),
    ("remote_write", "context"),
])
def test_destination_notes_use_canonical_root_and_keep_each_place(supervisor, monkeypatch, tmp_path,
                                                                canonical_source, name):
    from ouroboros import places, place_guard
    h = supervisor
    root = tmp_path / "canonical"
    h.ctx.drive_root = tmp_path / "child"
    h.ctx.task_metadata["focus"] = {"place_id": "elsewhere"}
    if canonical_source == "metadata":
        h.ctx.task_metadata["budget_drive_root"] = str(root)
        h.ctx.budget_drive_root = str(tmp_path / "wrong-context")
    else:
        h.ctx.budget_drive_root = str(root)
    records = [places.Place(id, "connector", "chat", label=id, state=state, notes=(
        places.NoteEntry(f"{id} owner's note", "owner"), places.NoteEntry(f"{id} migration", "migration"),
        places.NoteEntry(f"{id} mind's note", "ouroboros"))) for id, state in (("a", "active"), ("b", "silent"))]
    reads = []
    monkeypatch.setattr(places, "place_records", lambda path: reads.append(path) or snapshot(records))
    args = {"place_id": "a", "action": "inspect", "destination": {"conversation_id": "chat"}}
    if name == "remote_write":
        monkeypatch.setattr(place_guard, "call_place_facts", lambda *_a, **_k: {
            "places": [{"place_id": p.place_id, "place_state": p.state} for p in records],
            "targets": {"to": ["a", "b"]}})
    data = bundle(h, name, args)
    notes = data["standing_delegation"]["place_notes"]
    assert reads == [root] and not h.ctx.drive_root.exists()
    assert {n["place_id"] for n in notes} == ({"a", "b"} if name == "remote_write" else {"a"})
    assert all(n["standing_delegation"] is (n["author"] in {"owner", "migration"}) for n in notes)
    assert "elsewhere" not in json.dumps(data["subject"])
    if name == "remote_write":
        assert data["subject"]["place"]["places"][1]["place_state"] == "silent"
    else:
        assert data["subject"]["place"]["place_state"] == "active"


def test_place_reply_bundle_carries_this_turns_own_sends_in_its_conversation(supervisor, monkeypatch, tmp_path):
    """The Supervisor judges a focus's final with what the turn already said there: receipts and the run's own
    receipt-less message calls (06.11 I2: an honest "I already replied" was flagged as unverified without them)."""
    from ouroboros import places, place_guard
    from ouroboros.utils import append_jsonl
    h = supervisor
    monkeypatch.setattr(place_guard, "call_place_facts", REAL_CALL_PLACE_FACTS)
    monkeypatch.setattr(places, "load_places", lambda root: [places.Place("p", "connector", "chat", "a", "r", label="Team")])
    key = "chat:a:r:0"
    h.ctx.task_id = "focus"
    h.ctx.task_metadata = {"wake_cause": "focus", "focus": {"place_id": "p"}, "presence": {
        "place_id": "p", "delivery_reporting_version": 1, "transport_skill": "connector",
        "event": {"conversation_key": key, "provider": "chat", "account_id": "a", "conversation_id": "r", "thread_id": ""}}}
    h.ctx._place_message_calls = [{"tool": "ext_send", "at": "2026-11-06T02:18:50+00:00", "conversation_keys": [key], "text": "queued"}]
    for row in ({"task_id": "focus", "direction": "in", "text": "hi"},
                {"task_id": "focus", "type": "presence_delivery", "text": "I replied",
                 "transport": {"conversation_key": key, "delivery": {"state": "delivered", "delivery_id": "d", "part_id": "0"}}},
                {"task_id": "quiet-focus", "direction": "in", "text": "hi again"}):
        append_jsonl(tmp_path / "logs" / "chat.jsonl", row)
    args = {"place_id": "p", "text": "final", "destination": {"account_id": "a", "conversation_id": "r", "thread_id": ""}}
    data = bundle(h, "place_reply", args)
    assert data["subject"]["place"]["this_turn_sends"] == {
        "confirmed": '"I replied"', "calls_without_receipt": [{"tool": "ext_send", "at": "2026-11-06T02:18:50+00:00", "text": "queued"}]}
    h.ctx.task_id, h.ctx._place_message_calls = "quiet-focus", []  # a turn that said nothing there: none, not unknown
    assert bundle(h, "place_reply", args)["subject"]["place"]["this_turn_sends"] == {"confirmed": "none", "calls_without_receipt": []}


def test_an_mcp_call_carries_whose_credential_it_acts_with(supervisor, monkeypatch):
    """The bundle names whose credential an MCP call acts with (a server signed in as my human); a server the host
    does not know carries no claim."""
    from types import SimpleNamespace
    from ouroboros import mcp_client, mcp_oauth
    h = supervisor
    cfg = SimpleNamespace(id="slack", name="Slack (user)", carries_credentials=lambda: False, env_from_settings={})
    monkeypatch.setattr(mcp_client, "get_manager", lambda: SimpleNamespace(server_config=lambda sid: cfg if sid == "slack" else None))
    monkeypatch.setattr(mcp_oauth, "status", lambda _cfg: {"state": mcp_oauth.SIGNED_IN, "issuer": "https://slack.com"})
    data = bundle(h, "mcp_slack__conversations_add_message", {"channel_id": "C1", "payload": "hi"})
    assert data["subject"]["sends_as"] == {"mcp_server": "slack", "label": "Slack (user)",
                                           "acts_with": "your human's sign-in at https://slack.com"}
    assert "sends_as" not in bundle(h, "mcp_other__post", {"x": 1})["subject"]


def test_a_speech_refusal_names_the_words_the_rule_and_the_destinations_notes(supervisor, monkeypatch):
    """A flagged message returns to the mind with the exact words, the rule they break and the destination's label
    and my human's notes there (resolved once, for the bundle); a SAFE message says nothing."""
    from ouroboros import places, place_guard, tool_effects
    h = supervisor
    monkeypatch.setattr(tool_effects, "effect_of", lambda *a, **kw: tool_effects.ToolEffect("message", declared=True))
    monkeypatch.setattr(place_guard, "call_place_facts", lambda *_a, **_k: {
        "places": [{"place_id": "fb", "place_label": "Telegram · Studio"}]})
    monkeypatch.setattr(places, "place_records", lambda root: snapshot([places.Place("fb", "connector", "chat", label="Telegram · Studio",
        notes=(places.NoteEntry("Do not disclose private work here.", "owner"), places.NoteEntry("They like brevity.", "ouroboros")))]))
    h.replies.append({"status": "DANGEROUS", "reason": "owner rule 1 keeps work private", "speech": True,
                      "quote": "The confidential launch is next week"})
    ok, message = h.module.check_safety("connector_send", {"chat_id": "-100", "text": "Alex: The confidential launch is next week."}, ctx=h.ctx)
    assert not ok and "owner rule 1 keeps work private" in message and 'Flagged words: "The confidential launch is next week"' in message
    assert ('[DESTINATION] Telegram · Studio: your human\'s notes for this place: "Do not disclose private work here."'
            in message) and "They like brevity" not in message
    assert h.module.check_safety("connector_send", {"chat_id": "-100", "text": "The public example checks passed."}, ctx=h.ctx) == (True, "")


def test_a_sending_identity_is_my_humans_where_their_bot_or_app_confirms_them(supervisor, monkeypatch, published_tool):
    """A destination I write to from a user account counts my human's confirmation from another room of the same
    connector account, as every other reader does; a room of another account does not."""
    from ouroboros import place_arrivals, places
    h = supervisor
    monkeypatch.setattr(_place_guard, "call_place_facts", REAL_CALL_PLACE_FACTS)
    monkeypatch.setattr(_tool_effects, "effect_of", REAL_EFFECT_OF)
    root = h.ctx.drive_root
    first = places.connect_place(root, connector="connector", provider="service", account_id="account", conversation_id="C1")
    places.connect_place(root, connector="connector", provider="service", account_id="account", conversation_id="C2")
    places.connect_place(root, connector="connector", provider="service", account_id="other", conversation_id="C3")
    places.edit_place(root, first.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="U-owner", account_id="account")
    monkeypatch.setattr(place_arrivals, "recent_identity",
                        lambda *args: {"sends_as": {"id": "U-owner", "label": "Owner", "kind": "user"}})
    name, _tool = published_tool("message", target="room", conversation="room", account_id="account_id")
    sibling = _place_guard.call_place_facts(h.ctx, name, {"room": "C2", "account_id": "account", "text": "Hello"})
    elsewhere = _place_guard.call_place_facts(h.ctx, name, {"room": "C3", "account_id": "other", "text": "Hello"})
    assert sibling["places"][0]["sends_as_owner"] is True
    assert elsewhere["places"][0]["sends_as_owner"] is False


def test_a_circle_call_is_judged_with_the_circle_it_changes(supervisor, monkeypatch, tmp_path):
    """A configure_presence call naming a circle carries that circle's current words and its places into the bundle,
    read now from the canonical store; a placement call also carries the place's own guidance."""
    from ouroboros import places
    h = supervisor
    h.ctx.budget_drive_root = str(tmp_path)
    room = places.connect_place(tmp_path, connector="bridge", provider="chat", conversation_id="r", label="Friends chat")
    circle = places.create_circle(tmp_path, name="Friends", text="Keep work out of here.", author="owner")
    places.edit_place(tmp_path, room.place_id, "set_circle", author="owner", circle_id=circle["circle_id"])
    data = bundle(h, "configure_presence", {"action": "edit_circle", "circle_id": circle["circle_id"],
                                           "text": "Talk about anything."})
    shown = data["subject"]["place"]["circle"]
    assert shown["text"] == "Keep work out of here." and shown["text_author"] == "owner"
    assert shown["places"] == [{"place_id": room.place_id, "place_label": "Friends chat", "place_state": "active",
                                "placed_by": "owner"}]
    other = places.connect_place(tmp_path, connector="bridge", provider="chat", conversation_id="w", label="Work chat")
    data = bundle(h, "configure_presence", {"action": "set_circle", "place_id": other.place_id,
                                           "circle_id": circle["circle_id"]})
    assert data["subject"]["place"]["place_id"] == other.place_id and data["subject"]["place"]["circle"]["name"] == "Friends"
