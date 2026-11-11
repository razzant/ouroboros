"""One author fact for a place row: the account its place confirms now is my human's, read by every reader.

Each guard is checked both ways: a row from the confirmed account reads as my human's, also on a place where I
write from that same account, and the same row from an unauthenticated actor, another account, an imported
message, a participant or a bot account of mine does not — on the full stored row and on the facts the memory
inventory keeps beside it (``row_meta``).
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ouroboros import dialogue_store, places
from ouroboros.chat_chain import row_address
from ouroboros.chronicle_import import row_lineage
from ouroboros.dialogue_provenance import dialogue_speaker, row_author, row_class
from ouroboros.memory_inventory import row_meta
from ouroboros.memory_view import _spoken
from ouroboros.owner_words import owner_row
from ouroboros.places import owner_confirmed_row, place_owner_accounts

ROOM = 2 ** 41 + 7


@pytest.fixture
def place(tmp_path):
    connected = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="qa", label="Team chat")
    places.edit_place(tmp_path, connected.place_id, "confirm_owner_account", author="owner",
                      platform_actor_id="anton", account_id="qa")
    return connected


def _place_row(place_id, actor_id="anton", *, text="Ship the long answer, the end changes its meaning.",
               sends_as="", sends_kind="user", authenticated=None, account="qa", message=None, **extra):
    actor = {"platform_actor_id": actor_id, "display_name": actor_id.title()}
    if authenticated is not None:
        actor["authenticated"] = authenticated
    transport = {"provider": "chat", "account_id": account, "conversation_id": "design", "actor": actor,
                 **({"message": message} if message else {})}
    if sends_as:
        transport["identity"] = {"sends_as": {"id": sends_as, "label": "Shared", "kind": sends_kind}}
    return {"ts": "2026-11-04T10:00:00Z", "direction": "in", "chat_id": ROOM, "text": text, "source": "presence:chat",
            "sender_label": actor_id.title(), "sender_session_id": actor_id, "client_message_id": f"ev-{actor_id}",
            "transport": transport, "presence_provenance": {"place_id": place_id}, "place_id": place_id, **extra}


def test_the_confirmed_account_is_my_human_on_the_row_and_on_its_stored_facts(tmp_path, place):
    accounts = place_owner_accounts(tmp_path)
    owner, participant = _place_row(place.place_id), _place_row(place.place_id, "sasha")
    assert owner_confirmed_row(owner, accounts) and owner_confirmed_row(row_meta(owner), accounts)
    assert not owner_confirmed_row(participant, accounts) and not owner_confirmed_row(row_meta(participant), accounts)
    assert not owner_confirmed_row(owner, {}), "no snapshot of the place: no confirmation"


def test_one_confirmation_counts_in_every_room_of_its_bot_or_app_and_nowhere_else(tmp_path, place):
    """My human confirmed their account once, in ``place``: it is theirs in another room of the same connector account,
    on its rows and at an admission there, never on another account of that connector or through another connector,
    and nothing is copied into the other room's record (derived at every read)."""
    from types import SimpleNamespace

    from ouroboros.presence_context import focus_metadata
    from ouroboros.presence_runner import PresenceTurnEvent

    sibling = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="qa", conversation_id="ops")
    other_account = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="other",
                                         conversation_id="ops")
    other_connector = places.connect_place(tmp_path, connector="mailer", provider="chat", account_id="qa")
    accounts = place_owner_accounts(tmp_path)
    assert owner_confirmed_row(_place_row(sibling.place_id), accounts), "one confirmation per bot or app"
    assert owner_confirmed_row(row_meta(_place_row(sibling.place_id)), accounts)
    assert not owner_confirmed_row(_place_row(sibling.place_id, "sasha"), accounts), "a participant stays a participant"
    assert not owner_confirmed_row(_place_row(other_account.place_id, account="other"), accounts), "another account"
    assert not owner_confirmed_row(_place_row(other_connector.place_id), accounts), "another connector"

    def admitted(target, account):
        event = PresenceTurnEvent("ev-1", "chat", account, "ops", "", f"chat:{account}:ops:0",
                                  {"platform_actor_id": "anton"}, {}, {"message_id": "ev-1"}, "Ship it")
        admission = SimpleNamespace(place=places.get_place(tmp_path, target.place_id), binding_id=target.place_id,
                                    allowance={}, circle=None)
        return focus_metadata(admission, event, "presence-1", 7, tmp_path)["focus"]["actor"]["owner_confirmed"]

    assert admitted(sibling, "qa") is True and admitted(other_account, "other") is False
    assert places.get_place(tmp_path, sibling.place_id).owner_accounts == (), "the sibling's record stays as stored"
    from ouroboros.place_projection import place_view  # the sheet's view says so, and where it was confirmed
    shared = place_view(tmp_path, places.get_place(tmp_path, sibling.place_id))["shared_owner_accounts"]
    assert [(row["platform_actor_id"], row["place_id"], row["place_label"]) for row in shared] == [
        ("anton", place.place_id, "Team chat")]
    assert place_view(tmp_path, places.get_place(tmp_path, other_account.place_id))["shared_owner_accounts"] == []


def test_my_post_through_my_humans_account_is_never_their_words_and_their_same_words_still_are(tmp_path, place):
    """A message I posted through a tool signed in as my human (an MCP send whose result named its conversation and
    message id) comes back through the connector from their confirmed account: by its provider identity it is mine,
    on the row, on its stored facts, at an admission and at a cite of words a turn captured before the receipt
    landed; my human's own message in the very same words stays theirs."""
    from types import SimpleNamespace

    from ouroboros import own_sends
    from ouroboros.loop_messages import _record_owner_directive
    from ouroboros.owner_words import PLACE_CONFIRMED, cited_owner_basis, place_row_facts
    from ouroboros.presence_context import focus_metadata
    from ouroboros.presence_runner import PresenceTurnEvent

    words = "Ship the long answer, the end changes its meaning."
    receipt = ("MsgID,UserID,UserName,RealName,Channel,ThreadTs,Text,Time\n"
               f'1793991732.265319,anton,anton,Anton,design,,"{words}",2026-11-07T10:00:00Z\n')
    turn = SimpleNamespace(drive_root=tmp_path, task_metadata={})  # its echo opened a turn before the receipt
    _record_owner_directive(turn, source=PLACE_CONFIRMED, content=words, carried=True, msg_id="chat 7 / echo",
                            **place_row_facts("design", {"message_id": "1793991732.265319"}))
    assert cited_owner_basis(turn, words)[0]["source"] == "place_owner", "before the receipt nothing says it is mine"
    assert own_sends.record_result(tmp_path, {"annotations": {"readOnlyHint": False}}, receipt,
                                   via="mcp:slack/conversations_add_message")
    assert cited_owner_basis(turn, words)[0] is None, "the receipt now names it: my echo is no order of theirs"
    _record_owner_directive(turn, source=PLACE_CONFIRMED, content=words, carried=True, msg_id="chat 7 / theirs",
                            **place_row_facts("design", {"message_id": "1793991800.000100"}))
    assert cited_owner_basis(turn, words)[0]["source"] == "place_owner", "their own message in the same words"
    accounts = place_owner_accounts(tmp_path)
    echo = _place_row(place.place_id, text=words, message={"message_id": "1793991732.265319"})
    theirs = _place_row(place.place_id, text=words, message={"message_id": "1793991800.000100"})
    assert not owner_confirmed_row(echo, accounts) and not owner_confirmed_row(row_meta(echo), accounts)
    assert owner_confirmed_row(theirs, accounts) and owner_confirmed_row(row_meta(theirs), accounts)
    assert not row_author(echo, owner_accounts=accounts)["label"].startswith("my human")
    assert row_author(theirs, owner_accounts=accounts)["label"].startswith("my human")

    def admitted(message_id):
        event = PresenceTurnEvent(message_id, "chat", "qa", "design", "", "chat:qa:design:0",
                                  {"platform_actor_id": "anton"}, {}, {"message_id": message_id}, words)
        admission = SimpleNamespace(place=places.get_place(tmp_path, place.place_id), binding_id=place.place_id,
                                    allowance={}, circle=None)
        return focus_metadata(admission, event, "presence-1", 7, tmp_path)["focus"]["actor"]["owner_confirmed"]

    assert admitted("1793991732.265319") is False and admitted("1793991800.000100") is True


def test_a_place_where_i_write_from_my_humans_account_keeps_their_rows_theirs(tmp_path, place):
    """The owner's "This is my account" confirms the account I send as: their rows there stay theirs, as at
    admission (``presence_context.focus_metadata``), on the full row and its stored facts."""
    accounts = place_owner_accounts(tmp_path)
    shared = _place_row(place.place_id, sends_as="anton")
    assert owner_confirmed_row(shared, accounts) and owner_confirmed_row(row_meta(shared), accounts)
    assert not owner_confirmed_row(_place_row(place.place_id, "ouro-bot", sends_as="ouro-bot", sends_kind="bot"),
                                   accounts), "a bot of mine is never my human"


@pytest.mark.parametrize("variant", ["unauthenticated", "other_account", "legacy_form"])
def test_a_row_the_confirmation_cannot_vouch_for_is_not_my_human(tmp_path, place, variant):
    accounts = place_owner_accounts(tmp_path)
    row = {
        "unauthenticated": lambda: _place_row(place.place_id, authenticated=False),
        "other_account": lambda: _place_row(place.place_id, account="other"),
        "imported": lambda: _place_row(place.place_id, historical_import={"authorship": "platform_person"}),
        "legacy_form": lambda: {"direction": "in", "text": "old", "sender_label": "Anton",
                                "presence": {"place_id": place.place_id, "actor_id": "anton"}},
    }[variant]()
    assert owner_confirmed_row(row, accounts) is False
    assert owner_confirmed_row(row_meta(row), accounts) is False
    assert owner_confirmed_row(_place_row(place.place_id), accounts), "the same place still confirms its account"


def test_a_stored_row_says_the_same_author_as_its_whole_row(tmp_path, place):
    """The stored tail keeps only the transport facts its readers read: the author fact is one of those readers, so a
    row's stored facts name the same author as the row (a bot, the account I send as, an unauthenticated actor)."""
    accounts = place_owner_accounts(tmp_path)
    bot = _place_row(place.place_id, "ci-bot")
    bot["transport"]["actor"]["is_bot"] = True
    rows = [bot, _place_row(place.place_id, "ouro-bot", sends_as="ouro-bot", sends_kind="bot"),
            _place_row(place.place_id, authenticated=False), _place_row(place.place_id),
            _place_row(place.place_id, message={"forward_origin": {"sender_user_name": "Someone else"}})]
    for row in rows:
        assert row_author(row_meta(row), owner_accounts=accounts) == row_author(row, owner_accounts=accounts)
    labels = [row_author(row_meta(row), owner_accounts=accounts)["label"] for row in rows]
    assert labels[0].startswith("a bot: ") and labels[1].startswith("the account I send as: ")  # my own account
    assert not labels[2].startswith("my human") and labels[3].startswith("my human: ")


def test_a_forward_from_my_humans_account_stays_theirs_as_an_account_and_never_as_words(tmp_path, place):
    """The account stays confirmed wherever identity is read (the row's author, the admitted focus, the Safety
    occasion); the forwarded words are never my human's wherever words or authority are read (owner_speech,
    owner_row, the quotable row, confirmed place words, the wake's owner count)."""
    from types import SimpleNamespace
    from ouroboros.consciousness_wake import _row_kind
    from ouroboros.dialogue_provenance import owner_speech, recorded_person
    from ouroboros.owner_relay import quotable_owner_row
    from ouroboros.owner_words import place_confirmed_words
    from ouroboros.presence_context import focus_metadata, focus_owner_now
    from ouroboros.presence_runner import PresenceTurnEvent

    accounts = place_owner_accounts(tmp_path)
    message = {"message_id": 41, "forward_origin": {"sender_user_name": "Someone else"}}
    row = _place_row(place.place_id, message=message)
    for source in (row, row_meta(row)):
        assert owner_confirmed_row(source, accounts), "the account is still my human's"
        author = row_author(source, owner_accounts=accounts)
        assert author["label"].startswith("Forwarded by my human: Anton") and author["owner_confirmed"] is True
        assert author["forwarded"] is True and recorded_person(author) and not owner_speech(source, author)
        assert not owner_row(source, accounts=accounts), "its words are another author's"
    assert not quotable_owner_row(row, accounts=accounts)
    assert _row_kind(row, accounts) == "correspondent_message" and _row_kind(_place_row(place.place_id), accounts) == "owner_message"
    event = PresenceTurnEvent("ev-1", "chat", "qa", "design", "", "chat:qa:design:0",
                              {"platform_actor_id": "anton"}, {}, message, row["text"])
    admission = SimpleNamespace(place=places.get_place(tmp_path, place.place_id), binding_id=place.place_id, allowance={}, circle=None)
    metadata = focus_metadata(admission, event, "presence-1", 7, tmp_path)
    assert metadata["focus"]["actor"]["owner_confirmed"] is True
    turn = SimpleNamespace(drive_root=tmp_path, task_metadata={**metadata, "presence": {"event": {
        "actor": {"platform_actor_id": "anton"}, "account_id": "qa", "conversation_id": "design", "message": message}}})
    assert focus_owner_now(turn) is True, "the Safety occasion still names my human's confirmed account"
    assert place_confirmed_words(focus_owner_now(turn), row["text"], message) == "", "never their words"
    plain = _place_row(place.place_id)
    assert owner_speech(plain, row_author(plain, owner_accounts=accounts)) and owner_row(plain, accounts=accounts)


@pytest.mark.parametrize("authenticated,mine", [(False, False), (True, True), (None, True)],
                         ids=["unauthenticated", "authenticated", "no-fact"])
def test_a_sender_equal_to_my_account_is_mine_only_when_the_connector_did_not_deny_it(tmp_path, place, authenticated,
                                                                                    mine):
    """An unauthenticated actor is never my account: a mail From equal to the mailbox address proves nothing
    (``places.own_account_event``); the same id authenticated, or with no such fact, is my own account's event; on
    the whole row and on its stored facts."""
    accounts = place_owner_accounts(tmp_path)
    row = _place_row(place.place_id, "box@example.com", sends_as="box@example.com", sends_kind="user",
                     authenticated=authenticated)
    for read in (row, row_meta(row)):
        author = row_author(read, owner_accounts=accounts)
        assert (author["kind"] == "ouroboros" and author.get("via") == "own_account") is mine, author
        assert author["label"].startswith("the account I send as: ") is mine, author["label"]
        assert row_class(read, owner_accounts=accounts)["lane"] == 1


def test_the_snapshot_reads_the_stored_places_without_migrating_them(tmp_path):
    path = places.places_path(tmp_path)
    path.parent.mkdir(parents=True)
    unmigrated = {"schema_version": 1, "revision": 1, "migration": {}, "places": {"p1": {
        "place_id": "p1", "connector": "bridge", "provider": "chat", "account_id": "qa",
        "owner_accounts": [{"platform_actor_id": "anton", "account_id": "qa"}]}}}
    path.write_text(json.dumps(unmigrated), encoding="utf-8")
    before = path.read_bytes()
    assert [a.platform_actor_id for a in place_owner_accounts(tmp_path)["p1"]] == ["anton"]
    assert path.read_bytes() == before, "the reader never migrates"
    places.load_places(tmp_path)
    assert path.read_bytes() != before, "the migrating reader would have written"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(places.PlaceError):
        place_owner_accounts(tmp_path)
    assert "owner_accounts" not in row_lineage(tmp_path), "unreadable places name no owner"


def test_row_author_names_my_human_only_from_the_current_confirmation(tmp_path, place):
    accounts = row_lineage(tmp_path)["owner_accounts"]
    owner = _place_row(place.place_id)
    plain = row_author(owner)
    assert plain["kind"] == "human" and not plain["label"].startswith("my human") and "owner_confirmed" not in plain
    named = row_author(owner, owner_accounts=accounts)
    assert named["label"].startswith("my human: Anton [provider=chat") and named["owner_confirmed"] is True
    assert row_class(owner, owner_accounts=accounts)["lane"] == 1
    sasha = row_author(_place_row(place.place_id, "sasha"), owner_accounts=accounts)
    assert sasha["label"].startswith("Sasha [") and "owner_confirmed" not in sasha
    shared = row_author(_place_row(place.place_id, sends_as="anton"), owner_accounts=accounts)
    assert shared["label"].startswith("my human: Anton [") and shared["owner_confirmed"] is True
    # The account I send as, unconfirmed: my own account's event, never a person. Its task_id is the turn the arrival
    # woke, never its sender: no work of mine is named for it.
    own = _place_row(place.place_id, "ouro-bot", sends_as="ouro-bot", sends_kind="bot", task_id="presence-woken")
    bot = row_author(own, owner_accounts=accounts)
    assert bot["kind"] == "ouroboros" and bot["via"] == "own_account" and row_class(own, owner_accounts=accounts)["lane"] == 1
    assert bot["label"].startswith("the account I send as: Ouro-Bot [") and "task" not in bot["label"], bot["label"]
    assert row_author(row_meta(own), owner_accounts=accounts)["label"] == bot["label"]  # on its stored facts too
    unknown = row_author(own)  # without the owner snapshot a shared account cannot be told apart: it stays its fact
    assert unknown["kind"] == "human" and unknown["label"] == bot["label"]  # one label in both readers
    sent = {**own, "direction": "out", "type": "presence_delivery", "task_id": "root-sent"}  # my delivery from that account
    assert row_author(sent, owner_accounts=accounts)["label"].startswith("Ouroboros (task root-sent) [")
    places.edit_place(tmp_path, place.place_id, "revoke_owner_account", author="owner",
                      platform_actor_id="anton", account_id="qa")
    withdrawn = row_author(owner, owner_accounts=place_owner_accounts(tmp_path))
    assert not withdrawn["label"].startswith("my human"), "a withdrawn confirmation unnames the earlier row"


def test_a_nameless_place_row_has_no_recorded_author_and_main_stays_my_door():
    nameless = {"direction": "in", "text": "hi", "source": "presence:chat", "presence_provenance": {"place_id": "p"},
                "transport": {"provider": "chat", "actor": {}}}
    assert dialogue_speaker(nameless) == "author not recorded"
    assert dialogue_speaker({"direction": "in", "text": "hi", "chat_id": 1}) == "User"


def test_the_memory_view_keeps_my_humans_place_words_on_their_side(tmp_path, place):
    accounts = place_owner_accounts(tmp_path)

    def side(row):
        meta = row_meta(row)
        entry = (row_address(row), meta, 5)
        return _spoken(entry, row_class(meta, pos=5, owner_accounts=accounts)["author"], {})["owner_side"]

    assert side(_place_row(place.place_id)) is True
    assert side(_place_row(place.place_id, "sasha")) is False
    assert side(_place_row(place.place_id, sends_as="anton")) is True, "my human's account, which I also write from"
    assert side(_place_row(place.place_id, "ouro-bot", sends_as="ouro-bot", sends_kind="bot")) is False
    assert side({"ts": "2026-11-04T10:00:00Z", "direction": "in", "chat_id": 1, "text": "Main words"}) is True


def test_owner_word_carriers_take_place_words_only_from_the_confirmed_account(tmp_path, place):
    assert owner_row(_place_row(place.place_id), tmp_path)
    assert not owner_row(_place_row(place.place_id, "sasha"), tmp_path), "a participant is never my human's words"
    assert not owner_row(_place_row(place.place_id), None), "no readable places: not confirmed"
    assert owner_row({"direction": "in", "text": "Main words", "chat_id": 1}, None)
    assert not owner_row({"direction": "in", "text": "fix", "chat_id": 1, "source": "skill_repair"}, tmp_path)


def test_owner_source_admits_my_humans_place_row_and_refuses_a_participant(tmp_path, place):
    from ouroboros.owner_source import resolve_owner_source
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id, ctx.current_chat_id = "place-task", ROOM

    def source(actor_id):
        row = _place_row(place.place_id, actor_id, text=f"Grant it, says {actor_id}.")
        dialogue_store.append(tmp_path, "chat", row)
        ref = build_owner_message_ref(chat_id=ROOM, client_message_id=row["client_message_id"], ts=row["ts"],
                                      text=row["text"])
        return {"kind": "chat", "ref": ref}

    assert resolve_owner_source(ctx, source("anton"))["text"] == "Grant it, says anton."
    assert resolve_owner_source(ctx, source("sasha")) == {}


def test_room_history_reads_the_same_fact_for_a_shared_account(tmp_path, monkeypatch):
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_runner import capture_presence_event
    from tests.place_history_scenario import scenario

    client, _place, task, event = scenario(tmp_path, monkeypatch)
    admission = admit_presence_turn(drive_root=tmp_path, authenticated_transport_skill="bridge",
                                    provider="chat", account_id="qa", conversation_id="design")
    for source_id, actor, identity in (
            ("own-sent", "owner", {"sends_as": {"id": "owner", "label": "Anton", "kind": "user"}}),
            ("owner-typed", "owner", None),
            ("bot-sent", "ouro-bot", {"sends_as": {"id": "ouro-bot", "label": "Ouroboros", "kind": "bot"}}),
            ("owner-forwarded", "owner", None)):  # a forward keeps its sender: the account reads (you) too
        message = {"forward_origin": {"sender_user_name": "Someone else"}} if source_id == "owner-forwarded" else {}
        capture_presence_event(admission, replace(event, source_event_id=source_id, identity=identity, text=source_id,
                                                  actor={"platform_actor_id": actor, "display_name": actor},
                                                  message=message), drive_root=tmp_path)
    page = client.get("/api/chat/history", params={"chat_id": task["chat_id"]}).json()
    confirmed = {row["text"]: row["place_history"].get("owner_confirmed") for row in page["messages"]
                 if row["text"] in {"own-sent", "owner-typed", "bot-sent", "owner-forwarded"}}
    assert confirmed == {"own-sent": True, "owner-typed": True, "bot-sent": False, "owner-forwarded": True}, (
        "the owner's account reads (you)")
    client.close()


@pytest.mark.parametrize("actor_id,mine", [("anton", True), ("sasha", False)])
def test_review_evidence_signs_a_place_row_by_the_snapshot_without_an_activated_chronicle(tmp_path, place, actor_id,
                                                                                         mine):
    """No activated chronicle, so no row has a stream position: the places' owner snapshot still signs the row
    (``dialogue_evidence._row_projection``), as every other reader does."""
    from ouroboros.chronicle_import import ChronicleStore
    from ouroboros.dialogue_evidence import _row_projection, read_room_source

    row = _place_row(place.place_id, actor_id)
    lineage = row_lineage(tmp_path)
    assert "lineage_epoch" not in lineage and not ChronicleStore(tmp_path).log_path.exists()
    author = _row_projection(row, "chat", 1, tmp_path, pos=None, lineage=lineage)["author"]
    assert author.startswith("my human: Anton [") is mine and (actor_id.title() in author)
    dialogue_store.append(tmp_path, "chat", row)
    source = read_room_source(tmp_path, ROOM)
    assert source is not None
    [stored] = [line for line in source["rows"] if line.get("text") == row["text"]]
    assert stored["author"] == author, "the room source reads the same author"
