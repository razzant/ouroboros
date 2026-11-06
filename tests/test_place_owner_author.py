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
               sends_as="", sends_kind="user", authenticated=None, account="qa", **extra):
    actor = {"platform_actor_id": actor_id, "display_name": actor_id.title()}
    if authenticated is not None:
        actor["authenticated"] = authenticated
    transport = {"provider": "chat", "account_id": account, "conversation_id": "design", "actor": actor}
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
            _place_row(place.place_id, authenticated=False), _place_row(place.place_id)]
    for row in rows:
        assert row_author(row_meta(row), owner_accounts=accounts) == row_author(row, owner_accounts=accounts)
    labels = [row_author(row_meta(row), owner_accounts=accounts)["label"] for row in rows]
    assert labels[0].startswith("a bot: ") and labels[1].startswith("the account I send as: ")  # my own account
    assert not labels[2].startswith("my human") and labels[3].startswith("my human: ")


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
            ("bot-sent", "ouro-bot", {"sends_as": {"id": "ouro-bot", "label": "Ouroboros", "kind": "bot"}})):
        capture_presence_event(admission, replace(event, source_event_id=source_id, identity=identity, text=source_id,
                                                  actor={"platform_actor_id": actor, "display_name": actor}),
                               drive_root=tmp_path)
    page = client.get("/api/chat/history", params={"chat_id": task["chat_id"]}).json()
    confirmed = {row["text"]: row["place_history"].get("owner_confirmed") for row in page["messages"]
                 if row["text"] in {"own-sent", "owner-typed", "bot-sent"}}
    assert confirmed == {"own-sent": True, "owner-typed": True, "bot-sent": False}, "the owner's account reads (you)"
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
