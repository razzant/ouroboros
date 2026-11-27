"""A late delivery note stays with the place and bot that sent it (#4091). For Telegram, core writes it only where the
sending task's own stored route proves that place; the bridge's bot today and the places left today prove nothing,
since a settings change or a disconnect can come between the send and its receipt. The canonical receipt is stored
either way. Fictitious places and tasks in a temporary data folder; nothing leaves the machine."""
from __future__ import annotations

import pytest

from ouroboros import places
from ouroboros import telegram_owner_bot as owner
from ouroboros.place_reply import record_late_delivery
from ouroboros.presence_bindings import conversation_key
from ouroboros.presence_delivery import PresenceDeliveryRecorder, _payload_from_row
from tests._dialogue import chat_rows
from tests._store_shared import seed_result
from tests.test_presence_delivery import isolated_supervisor  # noqa: F401 -- the autouse isolation of receipt writes

A, B, CHAT, ELSEWHERE = "123456", "777000", "-1001500", "-1002000"
NOTE = "Delivery reported while {} is Paused: Hello from the group"


def _today(monkeypatch, bot):
    """The bridge's bot today. A late note never asks it: every read is recorded."""
    asked = []

    def read(*_args, **_kwargs):
        asked.append(bot)
        return bot
    monkeypatch.setattr(owner, "owner_bridge_bot_id", read)
    monkeypatch.setattr(owner, "owner_bot_of", read)
    return asked


def _place(root, *, account=A, chat=CHAT, contribution=owner.OWNER_BOT_CONTRIBUTION, label="Owner group"):
    return places.connect_place(root, connector="telegram-bot", provider="telegram", label=label, account_id=account,
                                conversation_id=chat, contribution_id=contribution,
                                scope="conversation" if chat else "account", state=places.STATE_PAUSED)


def _ordinary(root, *, account=B, chat="", contribution="place", label="Connector group"):
    """The old pair's connector bot: a place of its real account that covers every chat and topic of it."""
    return _place(root, account=account, chat=chat, contribution=contribution, label=label)


def _turn(root, place, *, route=None, task_id="turn-1", transport="telegram-bot", admitted=True):
    """The origin a receipt names: a stored Presence turn admitted in ``place`` for the event ``route`` (account, chat,
    topic), which its task keeps as ``presence.event`` and the place as ``presence.place`` (``place_summary``: the
    place's revision then, never its connection)."""
    account, chat, thread = route or (place.account_id, place.conversation_id, "")
    seed_result(root, task_id, {"_schema_version": 1, "task_id": task_id, "status": "completed", "metadata": {"presence": {
        "transport_skill": transport, "binding_id": place.place_id, "place_id": place.place_id,
        **({"place": places.place_summary(place)} if admitted else {}),
        "event": {"source_event_id": "in-1", "provider": "telegram", "account_id": account, "conversation_id": chat,
                  "thread_id": thread, "conversation_key": conversation_key("telegram", account, chat, thread)}}}})
    return {"kind": "tool", "task_id": task_id, "source_event_id": "in-1"}


def _receipt(account=A, origin=None, provider="telegram", chat=CHAT, thread=""):
    """What the connector reports for one send: the bot, the chat and its origin, never the place."""
    return {"schema_version": 1, "delivery_id": f"send:{account}", "part_id": "0", "state": "delivered",
            "provider": provider, "account_id": account, "conversation_id": chat, "thread_id": thread,
            "text": "Hello from the group", "format": "markdown", "message": {"message_id": 7},
            "origin": origin or {"kind": "automatic"}}


def _rows(root, kind):
    return [row for row in chat_rows(root) if row.get("type") == kind]


def _record(root, payload):
    """One receipt through the Host's writer, then the canonical row it must always leave exactly as reported."""
    assert PresenceDeliveryRecorder(root).record("telegram-bot", payload) == {
        "ok": True, "recorded": True, "duplicate": False, "history_coverage": "indexed"}
    kept, = _rows(root, "presence_delivery")
    assert _payload_from_row(kept) == ("telegram-bot", payload) and kept["direction"] == "out"
    return [row["place_id"] for row in _rows(root, "place_notice")], _rows(root, "place_notice")


def _layout(root, name):
    """The places of one layout, stored, and the bot a receipt in it names."""
    if name == "connector place alone":
        _ordinary(root)
        return B
    mine = _place(root)
    if name in ("both kinds", "owner place disconnected"):
        _ordinary(root, account="", chat=CHAT)  # a place of the same chat that names no bot
    if name == "owner place disconnected":
        places.edit_place(root, mine.place_id, "disconnect", author=places.AUTHOR_OWNER)
    return A


@pytest.mark.parametrize("today", [A, B, ""])
@pytest.mark.parametrize("layout", ["owner place alone", "connector place alone", "both kinds",
                                    "owner place disconnected"])
def test_a_receipt_without_a_proven_origin_leaves_no_note_however_few_places_are_left(
        tmp_path, monkeypatch, today, layout):
    asked = _today(monkeypatch, today)
    assert _record(tmp_path, _receipt(_layout(tmp_path, layout)))[0] == [] and not asked


@pytest.mark.parametrize("today", [A, ""])
@pytest.mark.parametrize("beside", [False, True])
@pytest.mark.parametrize("sender", ["owner", "connector"])
def test_the_sending_task_keeps_the_note_with_its_own_place(tmp_path, monkeypatch, today, beside, sender):
    asked = _today(monkeypatch, today)
    account = A if sender == "owner" else B
    sent_from = _place(tmp_path) if sender == "owner" else _ordinary(tmp_path)
    if beside:  # the other kind covers the same bot and chat: the owner's bot is the connector's, or the reverse
        _ordinary(tmp_path, account="", chat=CHAT) if sender == "owner" else _place(tmp_path, account=B)
    origin = _turn(tmp_path, sent_from, route=(account, CHAT, ""))
    places.edit_place(tmp_path, sent_from.place_id, "add_note", author=places.AUTHOR_OWNER, text="edited since")
    ids, (note,) = _record(tmp_path, _receipt(account, origin))
    assert ids == [sent_from.place_id] and note["text"] == NOTE.format(sent_from.label) and not asked
    assert note["transport"]["receipt"]["account_id"] == account


def test_an_origin_from_an_earlier_connection_is_not_applied(tmp_path):
    mine, _anyone = _place(tmp_path), _ordinary(tmp_path, account="", chat=CHAT)  # a place of the chat naming no bot
    earlier = _turn(tmp_path, mine)
    places.edit_place(tmp_path, mine.place_id, "disconnect", author=places.AUTHOR_OWNER)
    again = _place(tmp_path)  # the same identity, a new connection
    assert again.place_id == mine.place_id and again.connected_revision > mine.revision
    assert _record(tmp_path, _receipt(origin=earlier))[0] == []


def test_an_origin_from_the_current_connection_is_applied(tmp_path):
    mine = _place(tmp_path)
    places.edit_place(tmp_path, mine.place_id, "disconnect", author=places.AUTHOR_OWNER)
    again = _place(tmp_path)
    assert _record(tmp_path, _receipt(origin=_turn(tmp_path, again)))[0] == [again.place_id]


@pytest.mark.parametrize("case", ["place disconnected", "route of another chat", "place not covering the route",
                                  "another transport", "no stored task", "no admitted place"])
def test_an_origin_that_does_not_prove_a_place_is_not_applied(tmp_path, case):
    mine, _anyone = _place(tmp_path), _ordinary(tmp_path, account="", chat=CHAT)  # the latter covers the chat, too
    other = _ordinary(tmp_path, chat=ELSEWHERE)
    origin = (_turn(tmp_path, other) if case == "route of another chat"
              else _turn(tmp_path, other, route=(A, CHAT, "")) if case == "place not covering the route"
              else _turn(tmp_path, mine, transport="slack-bridge") if case == "another transport"
              else {"kind": "tool", "task_id": "never-stored"} if case == "no stored task"
              else _turn(tmp_path, mine, admitted=False) if case == "no admitted place"
              else _turn(tmp_path, mine))
    if case == "place disconnected":  # the row is gone; the place left covers the chat and proves nothing
        places.edit_place(tmp_path, mine.place_id, "disconnect", author=places.AUTHOR_OWNER)
    assert _record(tmp_path, _receipt(origin=origin))[0] == []


@pytest.mark.parametrize("receipt, covered", [((B, CHAT, ""), True), ((B, ELSEWHERE, ""), False),
                                              ((B, CHAT, "9"), False), (("999", CHAT, ""), False)])
def test_a_wide_place_ties_a_receipt_only_to_a_task_of_its_own_route(tmp_path, receipt, covered):
    """A place of no bot, chat or topic covers all of them; the task's route says which one it was admitted for."""
    anywhere = _ordinary(tmp_path, account="", chat="")
    account, chat, thread = receipt
    origin = _turn(tmp_path, anywhere, route=(B, CHAT, ""))
    notes = _record(tmp_path, _receipt(account, origin, chat=chat, thread=thread))[0]
    assert notes == ([anywhere.place_id] if covered else [])


def test_a_duplicate_receipt_changes_nothing(tmp_path):
    mine, _plain = _place(tmp_path), _ordinary(tmp_path, account="", chat=CHAT)
    payload = _receipt(origin=_turn(tmp_path, mine))
    assert _record(tmp_path, payload)[0] == [mine.place_id]
    again = PresenceDeliveryRecorder(tmp_path).record("telegram-bot", payload)
    assert again["duplicate"] is True
    assert len(_rows(tmp_path, "presence_delivery")) == 1 and len(_rows(tmp_path, "place_notice")) == 1


def test_equally_specific_places_do_not_hide_the_proven_one(tmp_path):
    first = _ordinary(tmp_path, chat=CHAT)
    _ordinary(tmp_path, chat=CHAT, contribution="extra", label="Second group")
    assert _record(tmp_path, _receipt(B, _turn(tmp_path, first)))[0] == [first.place_id]


def test_another_connector_keeps_its_lookup_by_the_destination_a_legacy_binding_kept(tmp_path, monkeypatch):
    legacy = places.Place(place_id="legacy", connector="mail", provider="legacy", label="Legacy binding",
                          state=places.STATE_PAUSED, destination={"transport": "email", "account_id": B,
                                                                  "conversation_id": CHAT, "thread_id": ""})
    monkeypatch.setattr(places, "places_of_connector", lambda _root, name: [legacy] if name == "mail" else [])
    record_late_delivery(tmp_path, "mail", _receipt(B, provider="email"), "receipt-1")
    note, = _rows(tmp_path, "place_notice")
    assert note["place_id"] == "legacy" and note["text"] == NOTE.format("Legacy binding")
