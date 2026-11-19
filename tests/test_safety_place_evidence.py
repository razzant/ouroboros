"""The occasion is separate from destinations and their standing delegation."""
import json
from dataclasses import replace
from types import SimpleNamespace

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


def bundle(h, name, arguments, *, ext_tool=None):
    before = len(h.calls)
    assert h.module.check_safety(name, arguments, ctx=h.ctx, ext_tool=ext_tool)[0]
    assert len(h.calls) > before
    return model_bundle(h)


def model_bundle(h):
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


@pytest.fixture
def telegram_publication(monkeypatch, supervisor):
    """The two declarations of the paired Telegram connector, bound by the real registry reader."""
    from ouroboros import extension_registry_state as registry
    from ouroboros.extension_surface_names import extension_surface_name

    skill = "telegram-bot"
    name = extension_surface_name(skill, "telegram_send")
    reply_route = {"provider": "telegram", "confirmation": "owner_confirmation"}
    route = {**reply_route, "account": "direct_account_id"}
    tool = {"name": "telegram_send", "effect": "message", "target": "chat_id", "conversation": "chat_id",
            "thread": "topic_id", "direct_recipient": route}
    manifest = SimpleNamespace(raw_extra={"tools": [tool], "place_reply": {"direct_recipient": reply_route}},
                               plugin_api={"version": "2.5"})
    publication = registry._ExtensionRegistrations(
        generation_digest="published", content_hash="reviewed-payload", loaded_skill=SimpleNamespace(manifest=manifest),
        contributions=[{"provides": "place", "skill": skill, "id": "place"}])
    monkeypatch.setitem(registry._extensions, skill, publication)
    schema = {"properties": {"direct_account_id": {"type": "string"}}}
    monkeypatch.setitem(registry._tools, name, {"skill": skill, "extension_generation": "published", "schema": schema})
    monkeypatch.setattr(_place_guard, "call_place_facts", REAL_CALL_PLACE_FACTS)
    monkeypatch.setattr(_tool_effects, "effect_of", REAL_EFFECT_OF)
    return name, manifest


def test_only_a_single_account_pinned_send_to_the_confirmed_user_identifies_the_owner(supervisor, telegram_publication):
    from ouroboros import extension_registry_state as registry, places

    name, manifest = telegram_publication
    root = supervisor.ctx.drive_root
    group = places.connect_place(root, connector="telegram-bot", provider="telegram",
                                 account_id="7001", conversation_id="-401", label="Owner DM")
    places.edit_place(root, group.place_id, "confirm_owner_account", author="owner", platform_actor_id="4101")
    bound = registry.get_tool_stamped(name)
    args = {"direct_account_id": "7001", "chat_id": "4101", "topic_id": "24", "text": "Synthetic example"}
    assert _place_guard.bind_direct_send(supervisor.ctx, name, bound, args)
    assert "owner_confirmation" not in args, "a confirmation source does not supply delivery policy"
    assert "confirmed_owner_dm" not in bundle(supervisor, name, args, ext_tool=bound)["subject"]["place"]
    # The account-wide Place supplies policy without requiring a separately connected DM.
    delivery = places.connect_place(root, connector="telegram-bot", provider="telegram", account_id="7001")
    assert _place_guard.bind_direct_send(supervisor.ctx, name, bound, args)
    condition = args["owner_confirmation"]
    assert condition["place_id"] == group.place_id and condition["delivery_place"]["place_id"] == delivery.place_id
    assert condition["thread_id"] == condition["delivery_place"]["thread_id"] == "24"
    shown = bundle(supervisor, name, args, ext_tool=bound)["subject"]["place"]
    assert shown["places"][0]["place_id"] == delivery.place_id and "sends_as" not in shown["places"][0]
    assert shown["confirmed_owner_dm"] == {"connector": "telegram-bot", "provider": "telegram",
                                          "account_id": "7001", "user_id": "4101"}
    for override in ({"chat_id": "-401"}, {"chat_id": "4102"}, {"direct_account_id": "7002"},
                     {"direct_account_id": ""}, {"direct_account_id": "*"}, {"chat_id": "unknown"},
                     {"chat_id": ["4101", "4102"]}, {"chat_id": True}):
        place = bundle(supervisor, name, {**args, **override}, ext_tool=bound)["subject"]["place"]
        assert "confirmed_owner_dm" not in place, override
    unpinned = {"chat_id": "4101", "text": "Synthetic example", "account_id": "7001"}
    assert "confirmed_owner_dm" not in bundle(supervisor, name, unpinned, ext_tool=bound)["subject"]["place"]
    for schema in ({}, {"properties": {"direct_account_id": {"type": "string", "default": "7002"}}}):
        place = bundle(supervisor, name, args, ext_tool={**bound, "schema": schema})["subject"]["place"]
        assert place["confirmed_owner_dm"] == shown["confirmed_owner_dm"]
    # The generation dispatched supplies the contract. A later or undeclared tool inherits none of it.
    declaration = manifest.raw_extra["tools"][0]
    del declaration["direct_recipient"]
    assert "confirmed_owner_dm" not in bundle(supervisor, name, args, ext_tool=registry.get_tool_stamped(name))["subject"]["place"]
    assert "confirmed_owner_dm" in bundle(supervisor, name, args, ext_tool=bound)["subject"]["place"]
    places.edit_place(root, group.place_id, "revoke_owner_account", author="owner", platform_actor_id="4101")
    assert "confirmed_owner_dm" not in bundle(supervisor, name, args, ext_tool=bound)["subject"]["place"]


def test_dm_evidence_and_destination_notes_use_the_same_sending_bot(supervisor, telegram_publication):
    from ouroboros import extension_registry_state as registry, places

    name, _ = telegram_publication
    root = supervisor.ctx.drive_root
    for bot in ("7002", "7001"):
        room = places.connect_place(root, connector="telegram-bot", provider="telegram", account_id=bot,
                                    conversation_id="4101", label="Owner DM", first={"note": f"Words for bot {bot}"})
        places.edit_place(root, room.place_id, "confirm_owner_account", author="owner", platform_actor_id="4101")
    args = {"direct_account_id": "7001", "chat_id": "4101", "text": "A synthetic message"}
    assert _place_guard.bind_direct_send(supervisor.ctx, name, registry.get_tool_stamped(name), args)
    data = bundle(supervisor, name, args, ext_tool=registry.get_tool_stamped(name))
    assert data["subject"]["place"]["confirmed_owner_dm"]["account_id"] == "7001"
    assert data["subject"]["place"]["places"][0]["conversation_keys"] == ["telegram:7001:4101:0"]
    assert [note["text"] for note in data["standing_delegation"]["place_notes"]] == ["Words for bot 7001"]


def test_normal_send_binds_the_current_owner_association_before_safety_and_dispatch(
        supervisor, telegram_publication, monkeypatch):
    from ouroboros import extension_loader, extension_process_runner, extension_registry_state as registry, owner_pause, places
    from ouroboros.tools.extension_dispatch import _dispatch_extension_tool_result, bound_dispatch_candidate

    name, _ = telegram_publication
    ctx = supervisor.ctx
    room = places.connect_place(ctx.drive_root, connector="telegram-bot", provider="telegram", account_id="7001")
    places.edit_place(ctx.drive_root, room.place_id, "confirm_owner_account", author="owner", platform_actor_id="4101")
    raw = registry._tools[name]
    monkeypatch.setitem(registry._tools, name, {**raw, "out_of_process": True})
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *a, **kw: True)
    monkeypatch.setattr(owner_pause, "admit_tool_dispatch", lambda _ctx: None)
    delivered = []
    monkeypatch.setattr(extension_process_runner, "dispatch_extension_tool_subprocess",
                        lambda _tool, _ctx, args: delivered.append(dict(args)) or "queued")
    def dispatch(arguments):
        prepared = dict(arguments)
        tool, unavailable = bound_dispatch_candidate(ctx, name, args=prepared)
        assert tool and not unavailable
        return _dispatch_extension_tool_result(ctx, name, tool, prepared)

    original = {"chat_id": "4101", "text": "Hello"}
    assert dispatch(original).status == "ok"
    assert "direct_account_id" not in original
    judged = json.loads(supervisor.calls[-1]["user_prompt"].split("\nArguments:\n")[1].split("\nIs this safe?")[0])
    assert judged == delivered[-1]
    assert judged == {**original, "direct_account_id": "7001", "owner_confirmation": judged["owner_confirmation"]}
    assert judged["owner_confirmation"]["place_id"] == room.place_id
    assert model_bundle(supervisor)["subject"]["place"]["confirmed_owner_dm"]["user_id"] == "4101"

    supervisor.replies.append({"status": "SUSPICIOUS", "reason": "Check the synthetic instruction"})
    assert dispatch({**original, "text": "Another message"}).status != "ok"
    assert len(delivered) == 1
    for chat in ("-401", "4102"):
        other = {**original, "chat_id": chat}
        assert dispatch(other).status == "ok"
        assert "confirmed_owner_dm" not in model_bundle(supervisor)["subject"]["place"]
        expected = other if chat == "-401" else {**other, "direct_account_id": "7001"}
        assert delivered[-1] == expected
    places.edit_place(ctx.drive_root, room.place_id, "revoke_owner_account", author="owner", platform_actor_id="4101")
    assert dispatch(original).status == "ok"
    assert delivered[-1] == {**original, "direct_account_id": "7001"}
    assert "confirmed_owner_dm" not in model_bundle(supervisor)["subject"]["place"]


def test_another_provider_or_a_second_target_cannot_supply_the_proof(supervisor, telegram_publication):
    from ouroboros import extension_registry_state as registry, places

    name, manifest = telegram_publication
    root = supervisor.ctx.drive_root
    for skill, provider, account in (("telegram-bot", "other", "7001"), ("user-session", "telegram", "7001")):
        room = places.connect_place(root, connector=skill, provider=provider, account_id=account, conversation_id="4101")
        places.edit_place(root, room.place_id, "confirm_owner_account", author="owner", platform_actor_id="4101")
    unscoped = places.connect_place(root, connector="telegram-bot", provider="telegram", conversation_id="4101")
    places.edit_place(root, unscoped.place_id, "confirm_owner_account", author="owner", platform_actor_id="4101")
    args = {"direct_account_id": "7001", "chat_id": "4101", "text": "Synthetic example"}
    assert "confirmed_owner_dm" not in bundle(supervisor, name, args, ext_tool=registry.get_tool_stamped(name))["subject"]["place"]
    places.edit_place(root, unscoped.place_id, "confirm_owner_account", author="owner", account_id="7001", platform_actor_id="4101")
    assert _place_guard.bind_direct_send(supervisor.ctx, name, registry.get_tool_stamped(name), args)
    tool = manifest.raw_extra["tools"][0]
    for changes in ({"target": ["chat_id", "copy_to"]},
                    {"touches": [{"effect": "message", "conversation": "copy_to"}]}):
        manifest.raw_extra["tools"] = [{**tool, **changes}]
        data = bundle(supervisor, name, {**args, "copy_to": "4102"}, ext_tool=registry.get_tool_stamped(name))
        assert "confirmed_owner_dm" not in data["subject"]["place"]


@pytest.mark.parametrize(("chat", "user", "account", "redirect", "extra", "declares", "confirmed"), [
    ("4101", "4101", "7001", {}, False, True, True),
    ("4101", "4101", "7001", {}, None, True, False),  # selected message predates the explicit audience
    ("-401", None, "7001", {}, False, True, False),  # the confirmed owner spoke in a group
    ("4102", "4102", "7001", {}, False, True, False),
    ("4101", None, "7001", {}, False, True, False),  # event from before the connector contract
    ("4101", "4101", "7001", {}, False, False, False),
    ("4101", "4101", "7002", {}, False, True, False),
    ("4102", "4102", "7001", {"conversation_id": "4101"}, False, True, False),
    ("4101", "4101", "7001", {"account_id": "7002"}, False, True, False),
    ("4101", "4101", "7001", {"thread_id": "8"}, False, True, False),
    ("4101", "4101", "7001", {"transport": "elsewhere"}, False, True, False),
    ("4101", "4101", "7001", {}, True, True, False),
])
def test_automatic_reply_checks_the_retained_route_and_all_its_addresses(
        supervisor, telegram_publication, chat, user, account, redirect, extra, declares, confirmed):
    from ouroboros import places
    from ouroboros.dialogue_provenance import presence_root_carrier
    from ouroboros.place_reply import prepare_focus_reply
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_events import parse_presence_event
    from ouroboros.presence_runner import _build_task

    _, manifest = telegram_publication
    if not declares:
        del manifest.raw_extra["place_reply"]
    root = supervisor.ctx.drive_root
    room = places.connect_place(root, connector="telegram-bot", provider="telegram", label="Owner DM")
    places.edit_place(root, room.place_id, "confirm_owner_account", author="owner", account_id="7001", platform_actor_id="4101")
    admission = admit_presence_turn(drive_root=root, authenticated_transport_skill="telegram-bot", provider="telegram",
                                    account_id=account, conversation_id=chat)
    admission = replace(admission, destination=replace(admission.destination, **redirect),
                        allowance={"status": "available", "remaining_usd": 2})
    address = {"conversation": "-402", "effect": "write", "match": "scope"}
    wire = {"source_event_id": "synthetic-update", "provider": "telegram", "account_id": account, "conversation_id": chat,
            "thread_id": "", "conversation_key": "", "actor": {"platform_actor_id": "4101"}, "text": "Hello",
            "conversation": {"title": "Owner DM", **({"direct_reply_user_id": user} if user else {})},
            "message": {"message_id": "17", **({"reply_targets": [address] if extra else []} if extra is not None else {})}}
    event = parse_presence_event(wire, authenticated_publication=(admission.transport_skill, "reviewed-payload"))
    assert not parse_presence_event(wire, authenticated_publication=(admission.transport_skill, "earlier-payload")).direct_reply_key
    # A pooled reader need not have the gateway's live publication. An old event gains nothing on upgrade.
    if declares:
        del manifest.raw_extra["place_reply"]
    else:
        manifest.raw_extra["place_reply"] = {"direct_recipient": {"provider": "telegram"}}
    task = _build_task(admission, event, drive_root=root, staged_files=())
    # A root started from this focus keeps its automatic reply even when the focus itself speaks by tool.
    supervisor.ctx.task_metadata = presence_root_carrier(task["metadata"])
    assert prepare_focus_reply(supervisor.ctx, "A synthetic reply") == (True, "")
    if user == chat and declares and redirect:
        assert supervisor.ctx._presence_completion["outcome"] == "silent"
        assert "destination differs" in supervisor.ctx._focus_reply_hold
        assert not supervisor.calls
        return
    data = model_bundle(supervisor)
    assert data["subject"]["tool"] == "place_reply"
    place = data["subject"]["place"]
    if confirmed:
        assert place["confirmed_owner_dm"] == {"connector": "telegram-bot", "provider": "telegram",
                                               "account_id": "7001", "user_id": "4101"}
        before = len(supervisor.calls)
        assert prepare_focus_reply(supervisor.ctx, "A synthetic reply") == (True, "") and len(supervisor.calls) == before
        places.edit_place(root, room.place_id, "revoke_owner_account", author="owner", account_id="7001", platform_actor_id="4101")
        assert prepare_focus_reply(supervisor.ctx, "A synthetic reply") == (True, "")
        assert len(supervisor.calls) == before and supervisor.ctx._presence_completion["outcome"] == "silent"
        assert "same owner confirmation" in supervisor.ctx._focus_reply_hold
    else:
        assert "confirmed_owner_dm" not in place
    if extra:
        assert place["touches"][0]["conversation"] == "-402"
