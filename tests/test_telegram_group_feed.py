"""Your own bot (A) in groups: the bridge's feed and relay for the telegram-bot connector (#4091).

The feed (skills/telegram/lib/telegram_group_feed.py) is the shipped code over real core places in a temporary
data folder; the provider is a recording stand-in for this bridge's client. Nothing leaves the machine.
"""
from __future__ import annotations

import asyncio
import contextlib
import gc
import hashlib
import json
import sqlite3
import sys
import time
import types
import weakref

import pytest

from ouroboros import places
from tests.test_telegram_proxy import _PACKAGE, _load

A = "123456"
CHAT = "-1001500"
TOKEN = f"{A}:SECRET-PART"


@pytest.fixture(autouse=True)
def _one_process_of_readers():
    """Each test is its own process for the one-reader map (telegram_owner_bot): a poller run claims its bot."""
    from ouroboros import telegram_owner_bot
    with telegram_owner_bot._holders_lock:
        saved = dict(telegram_owner_bot._holders)
        telegram_owner_bot._holders.clear()
    yield
    with telegram_owner_bot._holders_lock:
        telegram_owner_bot._holders.clear()
        telegram_owner_bot._holders.update(saved)


@pytest.fixture()
def lib():
    _load()
    return (sys.modules[f"{_PACKAGE}.lib.telegram_group_feed"], sys.modules[f"{_PACKAGE}.lib.telegram_api"],
            sys.modules[f"{_PACKAGE}.lib.telegram_poll_binding"])


class _Client:
    """This bridge's client as the relay uses it: its bot, binding check, raw request and download."""

    def __init__(self, api, bot=A, answer=(200, {"ok": True, "result": {"message_id": 7}}), failure=None):
        self.api, self.bot_id, self.token, self.answer, self.failure = api, bot, f"{bot}:SECRET-PART", answer, failure
        self.calls, self.downloads, self.changed = [], [], False

    def check_binding(self):
        if self.changed:
            raise self.api.TelegramBindingChanged()

    async def raw_request(self, method, *, data=None, files=None, timeout=30):
        self.check_binding()
        self.calls.append((method, dict(data or {}), {key: (name, handle.read()) for key, (name, handle)
                                                       in (files or {}).items()}))
        if self.failure is not None:
            raise self.failure
        return self.answer

    async def _download_bytes(self, file_path, limit=None):
        self.downloads.append((file_path, limit))
        return b"file-bytes"


def _bridge(feed, tmp_path, client, *, verified=(A, "my_own_bot"), bound=True):
    root = tmp_path / "data"
    root.mkdir(exist_ok=True)
    holder = {"verified": verified, "bound": bound}
    holder["capable"] = True
    bridge = feed.FeedBridge(tmp_path / "skill-state", root, client_factory=lambda: client,
                             verified=lambda: holder["verified"], owner_bound=lambda: holder["bound"],
                             capable=lambda: holder["capable"])
    return bridge, root, holder


def _place(root, *, account=A, chat=CHAT, contribution="owner_bot", state=places.STATE_ACTIVE):
    return places.connect_place(root, connector="telegram-bot", provider="telegram", label="Group",
                                account_id=account, conversation_id=chat, contribution_id=contribution,
                                scope="conversation", state=state)


def _message(text="hello", *, chat=CHAT, chat_type="supergroup", message_id=10, update_id=100, **extra):
    return {"update_id": update_id, "message": {"message_id": message_id, "date": 1700000000,
                                                "chat": {"id": int(chat), "type": chat_type, "title": "Book club"},
                                                "from": {"id": 77, "first_name": "Mira"}, "text": text, **extra}}


def _run(coroutine):
    return asyncio.run(coroutine)


def _consent(bridge):
    bridge.set_consent(True, A)


def test_only_group_and_supergroup_updates_enter_the_feed(lib):
    feed, _api, _poll = lib
    assert feed.group_chat(_message(chat_type="group"))[0] == "message"
    assert feed.group_chat({"update_id": 1, "edited_message": _message()["message"]})[0] == "edited_message"
    assert feed.group_chat({"update_id": 1, "my_chat_member": {"chat": {"id": -5, "type": "group"}}})
    private = {"update_id": 2, "message": {"message_id": 1, "chat": {"id": 42, "type": "private"}, "text": "/panic"}}
    channel = {"update_id": 3, "channel_post": {"message_id": 1, "chat": {"id": -100, "type": "channel"}}}
    callback = {"update_id": 4, "callback_query": {"id": "x", "message": {"chat": {"id": -5, "type": "group"}}}}
    assert feed.group_chat(private) is None and feed.group_chat(channel) is None and feed.group_chat(callback) is None
    assert feed.group_chat({"update_id": 5, "message": {"chat": {"id": -100, "type": "channel"}}}) is None


def test_without_consent_a_group_update_is_dropped_and_nothing_is_kept(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _place(root)
    assert bridge.publish(_message(), A) == "off"
    assert not (tmp_path / "skill-state" / "group_feed.sqlite3").exists()
    assert bridge.facts()["enabled"] is False and bridge.facts()["reason"] == "feed_off"


def test_consent_names_the_verified_bot_once_and_needs_a_bound_owner(lib, tmp_path):
    feed, api, _poll = lib
    bridge, _root, holder = _bridge(feed, tmp_path, _Client(api))
    with pytest.raises(feed.FeedRefusal) as wrong:
        bridge.set_consent(True, "999")
    assert wrong.value.code == "account_changed" and not bridge.enabled_for(A)
    holder["bound"] = False
    with pytest.raises(feed.FeedRefusal) as unbound:
        bridge.set_consent(True, A)
    assert unbound.value.code == "owner_not_bound"
    holder["bound"], holder["capable"] = True, False
    with pytest.raises(feed.FeedRefusal) as nobody:  # no connector would read the groups: nothing is kept
        bridge.set_consent(True, A)
    assert nobody.value.code == "connector_unavailable" and bridge.status()["capable"] is False
    holder["capable"] = True
    bridge.set_consent(True, A)
    assert bridge.enabled_for(A) and not bridge.enabled_for("654321")
    facts = bridge.facts()
    assert facts["identity"] == {"sends_as": {"id": A, "label": "@my_own_bot", "kind": "bot"}}
    assert facts["invite_url"] == "https://t.me/my_own_bot?startgroup=true" and facts["enabled"] is True
    holder["verified"] = ("654321", "another_bot")  # the token now names another bot: the consent does not carry
    assert bridge.facts()["enabled"] is False and bridge.facts()["account_id"] == "654321"
    assert bridge.consent_account_id() == A  # the owner's own door still shows the consent saved for A
    bridge.set_consent(False, A)  # Off for the consent shown, while the token names another bot
    holder["verified"] = (A, "my_own_bot")
    assert not bridge.enabled_for(A) and bridge.consent_account_id() == ""


def test_off_withdraws_only_the_consent_the_owner_saw_without_asking_telegram(lib, tmp_path):
    feed, api, _poll = lib
    bridge, _root, holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    holder["verified"] = None  # Telegram is away: the bridge confirms no bot now
    with pytest.raises(feed.FeedRefusal) as other:
        bridge.set_consent(False, "654321")  # the owner saw another bot's consent: this one stays
    assert other.value.code == "account_changed" and bridge.consent_account_id() == A
    holder["verified"] = (A, "my_own_bot")
    assert bridge.enabled_for(A)
    holder["verified"] = None
    bridge.set_consent(False, A)
    assert bridge.consent_account_id() == "" and not bridge.enabled_for(A)
    bridge.set_consent(False, A)  # already off: nothing to withdraw, nothing refused
    assert bridge.consent_account_id() == ""


def test_off_is_never_claimed_when_the_local_store_cannot_take_it(lib, tmp_path):
    feed, api, _poll = lib
    bridge, _root, holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    holder["verified"] = None  # Telegram is away, which never matters for Off
    real_use, real_write = bridge._use, bridge._write_consent

    def locked(_work, **_kwargs):
        raise sqlite3.OperationalError("database is locked")
    bridge._use = locked  # the one serialization cannot be taken
    with pytest.raises(sqlite3.OperationalError):
        bridge.set_consent(False, A)
    assert bridge.consent_account_id() == A  # nothing claimed: the consent stands as it was
    bridge._use = real_use

    def unwritable(_enabled, _account):
        raise OSError("read-only medium")
    bridge._write_consent = unwritable
    with pytest.raises(OSError):
        bridge.set_consent(False, A)
    assert bridge.consent_account_id() == A
    bridge._write_consent = real_write
    bridge.set_consent(False, A)  # the medium takes it again: Off is saved, still without Telegram
    saved = json.loads((tmp_path / "skill-state" / "group_feed.json").read_text(encoding="utf-8"))
    assert saved["enabled"] is False and bridge.consent_account_id() == ""


def test_a_new_consent_is_published_only_once_the_old_membership_is_gone(lib, tmp_path):
    feed, api, _poll = lib
    bridge, _root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    bridge.publish(_message(update_id=5), A)
    assert bridge.member_of(A, CHAT)
    bridge.set_consent(False, A)
    real_write, seen = bridge._write_consent, []

    def watching(enabled, account):
        if enabled:  # what any other reader of the store finds when the consent appears
            seen.append(bridge.member_of(A, CHAT))
        real_write(enabled, account)
    bridge._write_consent = watching
    _consent(bridge)
    assert seen == [False] and bridge.enabled_for(A)
    bridge._write_consent = real_write
    bridge.publish(_message(update_id=6, message_id=11), A)  # new evidence of this period
    assert bridge.connect_allowed(A, CHAT)
    bridge.set_consent(False, A)

    def unwritable(enabled, account):
        if enabled:
            raise OSError("read-only medium")
        real_write(enabled, account)
    bridge._write_consent = unwritable
    with pytest.raises(OSError):
        _consent(bridge)
    assert bridge.consent_account_id() == "" and not bridge.member_of(A, CHAT)  # not published; old evidence gone


def test_what_waited_through_an_off_and_on_again_keeps_and_confirms_nothing(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    _place(root)
    bridge.publish(_message(update_id=1, message_id=1), A)
    [first] = bridge.read(A, 0)
    earlier = bridge._held()
    real = bridge._transaction

    def off_and_on_meanwhile(db, work):  # the owner's Off and a quick On of the same bot while this write waited
        bridge._write_consent(False, "")
        bridge._write_consent(True, A)
        bridge._transaction = real
        return real(db, work)
    bridge._transaction = off_and_on_meanwhile
    assert bridge.publish(_message(update_id=2, message_id=2), A) == "off"  # taken under the earlier consent
    assert bridge.enabled_for(A) and bridge._held() != earlier
    assert bridge.read(A, first["feed_id"] + 1, held=earlier) == []  # let in under the earlier one: confirms nothing
    assert [item["feed_id"] for item in bridge.read(A, 0)] == [first["feed_id"]]  # it waits on, read under the new
    assert bridge.publish(_message(update_id=3, message_id=3), A) == "stored"


def test_the_owner_door_reads_the_saved_consent_in_three_states(lib, tmp_path):
    feed, api, _poll = lib
    bridge, _root, _holder = _bridge(feed, tmp_path, _Client(api))
    assert bridge.consent_account_id() == ""  # never saved: Off
    _consent(bridge)
    assert bridge.consent_account_id() == A and bridge.status()["consent_account_id"] == A
    assert "consent_account_id" not in bridge.facts()  # never on the connector's identity door
    path = tmp_path / "skill-state" / "group_feed.json"
    path.write_text("{half a record", encoding="utf-8")
    assert bridge.consent_account_id() is None and not bridge.enabled_for(A)  # unusable: unknown, and no access
    bridge.set_consent(False, A)  # Off replaces bytes in which no consent stands
    assert bridge.consent_account_id() == ""
    path.unlink()
    path.mkdir()  # unreadable now: unknown, never Off
    assert bridge.consent_account_id() is None and not bridge.enabled_for(A)
    with pytest.raises(feed.FeedRefusal) as unknown:
        bridge.set_consent(False, A)
    assert unknown.value.code == "consent_unknown"


def test_the_native_door_answers_a_refused_off_with_the_local_status(lib, tmp_path, monkeypatch):
    _feed, _api_mod, _poll = lib
    wiring = sys.modules[f"{_PACKAGE}.lib.telegram_group_wiring"]
    api = _Lifecycle(tmp_path)
    bridge = wiring.feed(api)
    (tmp_path / "group_feed.json").write_text(json.dumps({"enabled": True, "account_id": A}), encoding="utf-8")
    monkeypatch.setattr(wiring, "_status", lambda _api, feed_bridge: {"consent_account_id":
                                                                      feed_bridge.consent_account_id()})

    class _Post:
        method = "POST"

        def __init__(self, body):
            self.body = body

        async def json(self):
            return self.body

    route = wiring.make_route(api)
    refused = _run(route(_Post({"enabled": False, "expected_account_id": "654321"})))
    assert refused.status_code == 409
    assert json.loads(refused.body) == {"ok": False, "code": "account_changed",
                                        "error": "Groups are on for another bot now; nothing was turned off.",
                                        "consent_account_id": A}
    real_write = bridge._write_consent

    def unwritable(_enabled, _account):
        raise OSError("read-only medium")
    bridge._write_consent = unwritable
    failed = _run(route(_Post({"enabled": False, "expected_account_id": A})))
    assert failed.status_code == 503 and json.loads(failed.body) == {
        "ok": False, "code": "unwritable", "error": "The choice could not be saved.", "consent_account_id": A}
    bridge._write_consent = real_write
    done = _run(route(_Post({"enabled": False, "expected_account_id": A})))
    assert done.status_code == 200 and json.loads(done.body) == {"ok": True, "consent_account_id": ""}
    assert not bridge.enabled_for(A)
    assert _run(route(_Post({"enabled": False, "expected_account_id": 123456}))).status_code == 400


def test_before_an_exact_place_only_a_sighting_is_kept(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    _place(root, contribution="place")  # the connector's own declaration for account A: not A's group place
    _place(root, account="", chat=CHAT, contribution="place")  # nor a blank account's
    update = _message("private words", photo=[{"file_id": "photo-1", "file_unique_id": "u"}])
    assert bridge.publish(update, A) == "sighting"
    [item] = bridge.read(A, 0)
    assert set(item) == {"feed_id", "sighting"}
    text = json.dumps(item)
    assert "private words" not in text and "Mira" not in text and "photo-1" not in text
    assert item["sighting"]["chat"] == {"id": CHAT, "type": "supergroup", "title": "Book club"}
    assert item["sighting"]["membership"] == "present"


def test_an_exact_owner_bot_place_keeps_the_whole_update_with_its_place_and_topic(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    place = _place(root)
    update = _message("in a topic", message_thread_id=9, is_topic_message=True,
                      document={"file_id": "doc-1", "file_name": "a.pdf"})
    assert bridge.publish(update, A) == "stored"
    [item] = bridge.read(A, 0)
    assert item["update"] == update
    assert item["place"] == {"place_id": place.place_id, "connected_revision": place.connected_revision,
                             "thread_id": "9"}
    assert bridge.offered(A, place.place_id, place.connected_revision, "9", "doc-1")
    assert not bridge.offered(A, place.place_id, place.connected_revision, "", "doc-1")


def test_confirmed_items_leave_and_their_replay_is_a_duplicate(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    _place(root)
    assert bridge.publish(_message(update_id=1, message_id=1), A) == "stored"
    assert bridge.publish(_message(update_id=1, message_id=1), A) == "duplicate"
    [first] = bridge.read(A, 0)
    assert bridge.read(A, first["feed_id"] + 1) == []
    assert bridge.publish(_message(update_id=1, message_id=1), A) == "duplicate"  # after the ACK too
    assert bridge.publish(_message(update_id=2, message_id=2), A) == "stored"


def test_a_busy_spool_is_a_counted_gap_and_never_a_wait(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)  # its transaction created the store
    _place(root)
    blocker = sqlite3.connect(tmp_path / "skill-state" / "group_feed.sqlite3", isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        started = time.monotonic()
        assert bridge.publish(_message(), A) == "gap"
        assert time.monotonic() - started < 2.0
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    status = bridge.status()
    assert status["write_gaps"] == 1 and status["last_write_gap_at"]


def _membership(update_id, status):
    return {"update_id": update_id, "my_chat_member": {"chat": {"id": int(CHAT), "type": "supergroup"}, "date": 1,
                                                       "new_chat_member": {"status": status, "user": {"id": int(A)}}}}


def test_membership_follows_arrival_and_a_replay_never_moves_it_back(lib, tmp_path):
    feed, api, _poll = lib
    bridge, _root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    first = _message(update_id=5)
    assert bridge.publish(first, A) == "sighting"
    assert bridge.connect_allowed(A, CHAT)
    assert bridge.publish(_membership(7, "left"), A) == "sighting"
    assert not bridge.connect_allowed(A, CHAT)
    assert bridge.publish(first, A) == "duplicate"  # the same update again (a restart before its offset was saved)
    assert not bridge.connect_allowed(A, CHAT)
    # After a quiet week Telegram may number from anywhere: a rejoin with a lower id is still the newest evidence.
    assert bridge.publish(_membership(3, "member"), A) == "sighting"
    assert bridge.connect_allowed(A, CHAT)
    channelish = "-1009"
    assert not bridge.connect_allowed(A, channelish)  # never seen as a group
    assert not bridge.connect_allowed("654321", CHAT)


def test_a_new_consent_period_needs_new_membership_evidence(lib, tmp_path):
    feed, api, _poll = lib
    bridge, _root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    bridge.publish(_message(update_id=5), A)
    assert bridge.connect_allowed(A, CHAT)
    bridge.set_consent(False, A)
    assert bridge.publish(_membership(6, "left"), A) == "off"  # unseen while groups were off
    _consent(bridge)
    assert not bridge.connect_allowed(A, CHAT)  # what was seen before may have changed: it is not used
    _consent(bridge)  # Allow again in the same period keeps what it saw
    assert bridge.publish(_message(update_id=8, message_id=12), A) == "sighting"
    assert bridge.connect_allowed(A, CHAT)
    _consent(bridge)
    assert bridge.connect_allowed(A, CHAT)


def test_discovery_keeps_each_groups_newest_sighting_and_its_replays_stay_duplicates(lib, tmp_path):
    feed, api, _poll = lib
    bridge, _root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    words = [_message(f"words {n}", update_id=10 + n, message_id=20 + n) for n in range(3)]
    for update in words:
        assert bridge.publish(update, A) == "sighting"
    other = _message("elsewhere", chat="-1009999", update_id=30, message_id=40)
    assert bridge.publish(other, A) == "sighting"
    items = bridge.read(A, 0)
    assert [item["sighting"]["update_id"] for item in items] == [12, 30]  # one waiting sighting per group
    assert bridge.publish(words[0], A) == "duplicate" and bridge.publish(words[2], A) == "duplicate"
    assert bridge.connect_allowed(A, CHAT) and bridge.status()["pending"] == 2


def test_file_offer_lasts_for_the_connection_beyond_the_consumer_offset(lib, tmp_path):
    feed, api, _poll = lib
    client = _Client(api, answer=(200, {"ok": True, "result": {"file_path": "documents/book.pdf"}}))
    bridge, root, _holder = _bridge(feed, tmp_path, client)
    _consent(bridge)
    place = _place(root)
    assert bridge.publish(_message(document={"file_id": "doc-1"}), A) == "stored"

    def expire_at(moment):
        bridge._use(lambda db: bridge._transaction(db, lambda: bridge._expire(db, moment)))
    week = 8 * 86400
    expire_at(time.time() + week)  # the item still waits for the connector: its file and topic stay
    assert bridge.offered(A, place.place_id, place.connected_revision, "", "doc-1")
    assert bridge._message_thread(A, CHAT, 10) == ""
    [item] = bridge.read(A, 0)
    assert bridge.read(A, item["feed_id"] + 1) == []  # consumed
    expire_at(time.time() + week)
    assert bridge.offered(A, place.place_id, place.connected_revision, "", "doc-1")
    assert bridge._message_thread(A, CHAT, 10) is None
    request = {"account_id": A, "place_id": place.place_id, "connected_revision": place.connected_revision}
    status, answer, _ = _run(bridge.handle("call", {**request, "method": "getFile", "params": {"file_id": "doc-1"}}))
    assert status == 200
    assert _run(bridge.handle("file", {**request, "file_path": answer["result"]["file_path"]})) == (
        200, b"file-bytes", "application/octet-stream")


def test_a_file_offered_again_stays_offered_and_bookkeeping_waits_only_for_its_own_bot(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    place = _place(root)
    assert bridge.publish(_message(document={"file_id": "doc-1"}, update_id=1, message_id=1), A) == "stored"
    week = 7 * 86400

    def expire_at(moment):
        bridge._use(lambda db: bridge._transaction(db, lambda: bridge._expire(db, moment)))
    bridge._use(lambda db: db.execute("UPDATE offers SET offered_at=offered_at-?", (week,)))  # first offered a week ago
    assert bridge.publish(_message(document={"file_id": "doc-1"}, update_id=2, message_id=2), A) == "stored"
    items = bridge.read(A, 0)
    assert bridge.read(A, items[-1]["feed_id"] + 1) == []  # both consumed
    expire_at(time.time() + 2 * 86400)
    assert bridge.offered(A, place.place_id, place.connected_revision, "", "doc-1")  # offered again: from then on
    bridge._use(lambda db: db.execute(  # another bot's item from long ago, still waiting for its connector
        "INSERT INTO feed(account_id,dedupe,body,received_at) VALUES('654321','m:-5:1:0','{}',?)",
        (time.time() - 3 * week,)))
    places.edit_place(root, place.place_id, "disconnect", author=places.AUTHOR_OWNER)
    expire_at(time.time() + week + 86400)
    assert not bridge.offered(A, place.place_id, place.connected_revision, "", "doc-1")  # not held by another bot's
    assert bridge._message_thread(A, CHAT, 2) is None


def test_a_place_store_that_cannot_be_read_now_is_no_disconnect(lib, tmp_path, monkeypatch):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    place = _place(root)
    assert bridge.publish(_message(update_id=1, message_id=1), A) == "stored"

    def unreadable(*_args, **_kwargs):
        raise places.PlaceError("places_unreadable", "places")
    monkeypatch.setattr(places, "place_records", unreadable)
    places_file = places.places_path(root)
    places_file.write_bytes(places_file.read_bytes() + b" ")  # changed on disk: read again, and it fails now
    assert bridge.publish(_message(update_id=2, message_id=2), A) == "stored"  # the last view stands
    assert [item.get("place", {}).get("place_id") for item in bridge.read(A, 0)] == [place.place_id] * 2
    monkeypatch.setattr(places, "get_place", unreadable)
    status, body, _ = _run(bridge.handle("file", {"account_id": A, "place_id": place.place_id,
                                                  "connected_revision": place.connected_revision, "file_path": "x"}))
    assert status == 403 and body["relay_refusal"] == "feed_unavailable" and body["parameters"] == {"retry_after": 5}


def test_a_batch_keeps_nothing_once_its_consent_changed_and_nothing_without_a_bound_owner(lib, tmp_path):
    feed, api, _poll = lib
    wiring = sys.modules[f"{_PACKAGE}.lib.telegram_group_wiring"]
    bridge, root, holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    _place(root)
    period = bridge.intake(A)  # read once when the batch is received
    assert wiring.allowed_updates(period) == ALLOWED
    bridge._write_consent(False, "")
    bridge._write_consent(True, A)  # Off and On again while the batch was on its way
    assert bridge.publish(_message(update_id=1, message_id=1), A, period) == "off"
    assert bridge.publish(_message(update_id=2, message_id=2), A, bridge.intake(A)) == "stored"
    holder["bound"] = False  # the consent stays saved, but nobody is bound to this bridge now
    assert bridge.intake(A) is None and wiring.allowed_updates(None) is None
    assert bridge.publish(_message(update_id=3, message_id=3), A) == "off"
    assert bridge.read(A, 0) == []  # no handover while the owner is unbound
    holder["bound"] = True
    assert [item["update"]["update_id"] for item in bridge.read(A, 0)] == [2]


def test_a_long_poll_confirms_nothing_once_its_consent_changed_and_waits_without_reading(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    _place(root)
    bridge.publish(_message(update_id=1, message_id=1), A)
    [first] = bridge.read(A, 0)
    real_holding, real_use, uses = bridge._holding, bridge._use, []

    def let_in_then_off_and_on(request):
        let_in = real_holding(request)
        bridge._holding = real_holding
        bridge._write_consent(False, "")
        bridge._write_consent(True, A)  # a new consent while this poll waits
        return let_in
    bridge._holding = let_in_then_off_and_on
    bridge._use = lambda work, **kwargs: uses.append(1) or real_use(work, **kwargs)
    status, body, _ = _run(bridge.handle("updates", {"account_id": A, "offset": str(first["feed_id"] + 1),
                                                     "timeout": "1"}))
    assert status == 200 and body["result"] == [] and len(uses) == 1  # one read; its idle second touched no disk
    bridge._use = real_use
    assert [item["feed_id"] for item in bridge.read(A, 0)] == [first["feed_id"]]  # the old offset stayed unconfirmed


def test_identity_with_groups_names_this_periods_groups_for_the_same_observation(lib, tmp_path):
    feed, api, _poll = lib
    bridge, _root, holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    bridge.publish(_membership(1, "member"), A)
    bridge.publish(_message("elsewhere", chat="-1009999", update_id=2, message_id=5), A)
    bridge.publish(_message("gone", chat="-1007777", update_id=3, message_id=6), A)
    bridge.publish({"update_id": 4, "my_chat_member": {"chat": {"id": -1007777, "type": "supergroup"}, "date": 1,
                                                       "new_chat_member": {"status": "left", "user": {"id": int(A)}}}}, A)
    status, body, _ = _run(bridge.handle("identity", {"groups": "1"}))
    assert status == 200 and body["account_id"] == A and body["group_ids"] == ["-1001500", "-1009999"]
    assert "group_ids" not in _run(bridge.handle("identity", {}))[1]  # never read without groups=1
    real_use = bridge._use

    def rebound_meanwhile(work, **kwargs):
        holder["verified"] = ("654321", "another_bot")  # a settings rebind during the index read
        return real_use(work, **kwargs)
    bridge._use = rebound_meanwhile
    assert bridge.facts(groups=True)["group_ids"] is None  # the header's bot and the ids could differ: unknown
    bridge._use = real_use
    holder["verified"] = (A, "my_own_bot")
    holder["bound"] = False
    assert bridge.facts(groups=True)["group_ids"] == []  # no bound owner: no ids
    holder["bound"] = True
    bridge.set_consent(False, A)
    assert bridge.facts(groups=True)["group_ids"] == []  # Off: no consent period, no ids
    holder["verified"] = None
    assert bridge.facts(groups=True)["group_ids"] is None  # no confirmed bot: unknown


def test_an_idle_read_takes_no_write_transaction(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    _place(root)
    bridge.publish(_message(), A)
    [item] = bridge.read(A, 0)
    writes = []
    real = bridge._transaction
    bridge._transaction = lambda db, work: writes.append(work) or real(db, work)
    assert bridge.read(A, item["feed_id"]) == [item]  # nothing below the offset: read only
    assert writes == []
    assert bridge.read(A, item["feed_id"] + 1) == []  # the confirmation is the one write
    assert len(writes) == 1
    assert bridge.read(A, item["feed_id"] + 1) == [] and len(writes) == 1


def test_reads_after_off_or_another_bot_confirm_and_show_nothing(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    _place(root)
    bridge.publish(_message(), A)
    bridge.set_consent(False, A)
    status, body, _ = _run(bridge.handle("updates", {"account_id": A, "offset": "99", "timeout": "0"}))
    assert status == 403 and body["relay_refusal"] == "feed_off"
    bridge.set_consent(True, A)
    holder["verified"] = ("654321", "another_bot")
    status, body, _ = _run(bridge.handle("updates", {"account_id": A, "offset": "99", "timeout": "0"}))
    assert body["relay_refusal"] == "account_mismatch"
    holder["verified"] = (A, "my_own_bot")
    status, body, _ = _run(bridge.handle("updates", {"account_id": A, "offset": "0", "timeout": "0"}))
    assert status == 200 and len(body["result"]) == 1  # kept, never confirmed by the refused reads


def _send(bridge, method="sendMessage", params=None, place=None, *, thread="", operation="call", files=None):
    request = {"account_id": A, "method": method, "params": params if params is not None else
               {"chat_id": CHAT, "text": "hi"}, "thread_id": thread,
               "place_id": place.place_id if place else "", "connected_revision": place.connected_revision if place else 0}
    if files is not None:
        request["files"] = files
    return _run(bridge.handle(operation, request))


def test_relay_refuses_unlisted_methods_arguments_and_nested_destinations(lib, tmp_path):
    feed, api, _poll = lib
    client = _Client(api)
    bridge, root, _holder = _bridge(feed, tmp_path, client)
    _consent(bridge)
    bridge.publish(_message(), A)
    place = _place(root)
    for method, params, code in [
        ("getUpdates", {"offset": 0}, "method_not_allowed"),
        ("sendMessage", {"chat_id": CHAT, "text": "x", "disable_notification": True}, "argument_not_allowed"),
        ("sendMessage", {"chat_id": CHAT, "text": "x", "reply_parameters": {"message_id": 1, "chat_id": "-42"}},
         "argument_not_allowed"),
        ("sendMessage", {"chat_id": "@public_group", "text": "x"}, "argument_not_allowed"),
        ("sendPhoto", {"chat_id": CHAT, "photo": "https://example.invalid/a.png"}, "method_not_allowed"),
    ]:
        status, body, _ = _send(bridge, method, params, place)
        assert (status, body["relay_refusal"]) == (403, code), method
    status, body, _ = _send(bridge, "sendMediaGroup", {"chat_id": CHAT, "media": json.dumps(
        [{"type": "photo", "media": "https://example.invalid/a.png"}, {"type": "photo", "media": "attach://f1"}])},
        place, operation="upload", files=[])
    assert body["relay_refusal"] == "argument_not_allowed"
    assert client.calls == []


def test_relay_sends_only_to_the_exact_active_place_at_its_revision_and_topic(lib, tmp_path):
    feed, api, _poll = lib
    client = _Client(api)
    bridge, root, _holder = _bridge(feed, tmp_path, client)
    _consent(bridge)
    bridge.publish(_message(), A)
    place = _place(root)
    stale = type(place)(**{**place.__dict__, "connected_revision": place.connected_revision + 5})
    assert _send(bridge, place=stale)[1]["relay_refusal"] == "binding_changed"
    assert _send(bridge, params={"chat_id": CHAT, "text": "x"}, place=place, thread="9")[1]["relay_refusal"] == (
        "argument_not_allowed")  # a topic's speech must name that topic, never fall to General
    status, body, _ = _send(bridge, params={"chat_id": CHAT, "text": "x", "message_thread_id": 9}, place=place,
                            thread="9")
    assert status == 200 and body == {"ok": True, "result": {"message_id": 7}}
    places.edit_place(root, place.place_id, "set_state", author=places.AUTHOR_OWNER, state=places.STATE_SILENT)
    assert _send(bridge, place=place)[1]["relay_refusal"] == "place_not_active"
    places.edit_place(root, place.place_id, "disconnect", author=places.AUTHOR_OWNER)
    assert _send(bridge, place=place)[1]["relay_refusal"] == "binding_changed"
    again = _place(root)
    assert again.place_id == place.place_id and again.connected_revision != place.connected_revision
    assert _send(bridge, place=place)[1]["relay_refusal"] == "binding_changed"  # the old connection stays closed
    assert [call[0] for call in client.calls] == ["sendMessage"]


def test_actions_keep_their_messages_topic_and_moderation_needs_the_whole_group(lib, tmp_path):
    feed, api, _poll = lib
    reply = {"ok": True, "result": {"message_id": 70, "message_thread_id": 9, "is_topic_message": True,
                                    "chat": {"id": int(CHAT), "type": "supergroup"}}}
    client = _Client(api, answer=(200, reply))
    bridge, root, _holder = _bridge(feed, tmp_path, client)
    _consent(bridge)
    bridge.publish(_message("in topic five", message_id=50, update_id=1, message_thread_id=5, is_topic_message=True), A)
    place = _place(root)  # a chat-wide place: it holds every topic of its group
    status, _body, _ = _send(bridge, params={"chat_id": CHAT, "text": "hi", "message_thread_id": 9}, place=place,
                             thread="9")
    assert status == 200 and bridge._message_thread(A, CHAT, 70) == "9"  # the bot's own reply: its topic is known
    topic = types.SimpleNamespace(thread_id="9")  # a place of topic 9 only
    bridge._in_scope(A, CHAT, topic, "editMessageText", {"message_id": 70})
    bridge._in_scope(A, CHAT, topic, "setMessageReaction", {"message_id": 70})
    bridge._in_scope(A, CHAT, topic, "sendMessage", {"reply_parameters": {"message_id": 70}})
    for method, params in [("editMessageText", {"message_id": 50}), ("deleteMessage", {"message_id": 51}),
                           ("banChatMember", {"user_id": 77}), ("restrictChatMember", {"user_id": 77}),
                           ("sendMessage", {"reply_parameters": json.dumps({"message_id": 50})})]:
        with pytest.raises(feed.FeedRefusal) as refused:
            bridge._in_scope(A, CHAT, topic, method, params)
        assert refused.value.code == "argument_not_allowed", method
    whole = types.SimpleNamespace(thread_id="")
    bridge._in_scope(A, CHAT, whole, "deleteMessage", {"message_id": 51})
    status, _body, _ = _send(bridge, "banChatMember", {"chat_id": CHAT, "user_id": 77}, place)
    assert status == 200 and [call[0] for call in client.calls] == ["sendMessage", "banChatMember"]
    unwritable = tmp_path / "not-a-folder"
    unwritable.write_text("x", encoding="utf-8")
    broken = feed.FeedBridge(unwritable, root, client_factory=lambda: client, verified=lambda: (A, "my_own_bot"),
                             owner_bound=lambda: True)
    broken._remember_sent(A, CHAT, reply)  # after a send nothing here raises: its answer stays Telegram's


def _reaction(update_id, message_id):
    return {"update_id": update_id, "message_reaction": {
        "chat": {"id": int(CHAT), "type": "supergroup"}, "message_id": message_id, "date": 1, "user": {"id": 77},
        "old_reaction": [], "new_reaction": [{"type": "emoji", "emoji": "\U0001F44D"}]}}


def test_a_reaction_reaches_the_place_of_the_topic_its_message_is_in(lib, tmp_path):
    feed, api, _poll = lib
    bridge, root, _holder = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    _place(root)  # a chat-wide place: a reaction's item names the topic its message is known to be in
    bridge.publish(_message("in a topic", message_id=60, update_id=1, message_thread_id=9, is_topic_message=True), A)
    assert bridge.publish(_reaction(2, 60), A) == "stored"
    assert [item["place"]["thread_id"] for item in bridge.read(A, 0)] == ["9", "9"]
    topic_place = places.Place(place_id="topic-9", connector="telegram-bot", provider="telegram", account_id=A,
                               conversation_id=CHAT, thread_id="9", contribution_id="owner_bot", scope="conversation",
                               connected_revision=3)
    bridge._current_places = lambda: [topic_place]  # a place of topic 9 alone
    assert bridge.publish(_reaction(3, 60), A) == "stored"  # its message is in topic 9: that topic's place
    assert bridge.read(A, 0)[-1]["place"] == {"place_id": "topic-9", "connected_revision": 3, "thread_id": "9"}
    assert bridge.publish(_reaction(4, 61), A) == "sighting"  # a message never seen: General, no place of it


def test_relay_checks_again_after_the_files_are_read(lib, tmp_path):
    feed, api, _poll = lib
    client = _Client(api)
    bridge, root, _holder = _bridge(feed, tmp_path, client)
    _consent(bridge)
    bridge.publish(_message(), A)
    place = _place(root)
    own = root / "state" / "skills" / "telegram-bot" / "outbox_files"
    own.mkdir(parents=True)
    data = b"%PDF-1.4 synthetic"
    (own / "a.pdf").write_bytes(data)
    rows = [{"field": "document", "path": str(own / "a.pdf"), "name": "a.pdf", "size": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}]
    real_files = bridge._files

    def silent_meanwhile(*args):
        opened = real_files(*args)
        places.edit_place(root, place.place_id, "set_state", author=places.AUTHOR_OWNER, state=places.STATE_SILENT)
        return opened
    bridge._files = silent_meanwhile
    status, body, _ = _send(bridge, "sendDocument", {"chat_id": CHAT, "caption": "c"}, place, operation="upload",
                            files=rows)
    assert body["relay_refusal"] == "place_not_active" and client.calls == []


def test_a_send_let_in_before_an_off_or_an_off_and_on_again_sends_nothing(lib, tmp_path):
    feed, api, _poll = lib
    client = _Client(api)
    bridge, root, _holder = _bridge(feed, tmp_path, client)
    _consent(bridge)
    bridge.publish(_message(), A)
    place = _place(root)
    real_bound = bridge._bound

    def changing(*steps):
        calls = []

        def bound(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:  # the owner's choices land while the call waits for its last checks
                for enabled in steps:
                    bridge._write_consent(enabled, A)
            return real_bound(*args, **kwargs)
        return bound
    bridge._bound = changing(False, True)  # Off and a quick On of the same bot
    status, body, _ = _send(bridge, place=place)
    assert (status, body["relay_refusal"]) == (403, "feed_unavailable") and client.calls == []
    bridge._bound = real_bound
    assert _send(bridge, place=place)[0] == 200  # asked again, under the new consent
    bridge._bound = changing(False)  # Off alone
    status, body, _ = _send(bridge, place=place)
    assert (status, body["relay_refusal"]) == (403, "feed_off") and [call[0] for call in client.calls] == ["sendMessage"]


def test_upload_reads_only_the_connectors_own_unchanged_files(lib, tmp_path):
    feed, api, _poll = lib
    client = _Client(api)
    bridge, root, _holder = _bridge(feed, tmp_path, client)
    _consent(bridge)
    bridge.publish(_message(), A)
    place = _place(root)
    own = root / "state" / "skills" / "telegram-bot" / "outbox_files"
    own.mkdir(parents=True)
    data = b"\x89PNG synthetic"
    (own / "p.png").write_bytes(data)
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(data)
    good = {"field": "photo", "path": str(own / "p.png"), "name": "p.png", "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}
    params = {"chat_id": CHAT, "caption": "c"}
    assert _send(bridge, "sendPhoto", params, place, operation="upload",
                 files=[{**good, "path": str(outside)}])[1]["relay_refusal"] == "argument_not_allowed"
    assert _send(bridge, "sendPhoto", params, place, operation="upload",
                 files=[{**good, "sha256": "0" * 64}])[1]["relay_refusal"] == "file_changed"
    status, body, _ = _send(bridge, "sendPhoto", params, place, operation="upload", files=[good])
    assert status == 200
    [(method, sent, files)] = client.calls
    assert method == "sendPhoto" and sent == {"chat_id": CHAT, "caption": "c"} and files == {"photo": ("p.png", data)}


def test_provider_outcomes_pass_through_truthfully(lib, tmp_path):
    feed, api, _poll = lib
    cases = [
        ((400, {"ok": False, "error_code": 400, "description": f"Bad Request for {TOKEN}"}), None, 400),
        ((200, None), None, 502),
        ((200, {"result": {}}), None, 502),
        (None, api.TelegramTransportError("refused", unsent=True), 503),
        (None, api.TelegramTransportError("reset", unsent=False), 502),
        (None, OSError("a file read failed mid-upload"), 502),  # the request may have left: unknown, never "not sent"
    ]
    for answer, failure, expected in cases:
        client = _Client(api, answer=answer, failure=failure)
        bridge, root, _holder = _bridge(feed, tmp_path, client)
        _consent(bridge)
        bridge.publish(_message(), A)
        place = _place(root)
        status, body, _ = _send(bridge, place=place)
        assert status == expected
        if expected == 400:
            assert TOKEN not in json.dumps(body) and body["error_code"] == 400
        if expected == 503:
            assert body == {"relay_outcome": "not_sent"}
        if expected == 502:
            assert body == {"relay_outcome": "unknown"} and "ok" not in body
    client = _Client(api)
    bridge, root, _holder = _bridge(feed, tmp_path, client)
    place = _place(root)
    client.changed = True  # the token changed before the request: nothing was sent
    status, body, _ = _send(bridge, place=place)
    assert status == 403 and body["relay_refusal"] == "bridge_not_ready"


def test_file_reads_need_an_offered_file_of_this_connection_and_topic(lib, tmp_path):
    feed, api, _poll = lib
    client = _Client(api, answer=(200, {"ok": True, "result": {"file_id": "doc-1", "file_path": "documents/a.pdf"}}))
    bridge, root, _holder = _bridge(feed, tmp_path, client)
    _consent(bridge)
    place = _place(root)
    bridge.publish(_message(document={"file_id": "doc-1"}), A)
    request = {"account_id": A, "place_id": place.place_id, "connected_revision": place.connected_revision,
               "thread_id": ""}
    refused = _run(bridge.handle("call", {**request, "method": "getFile", "params": {"file_id": "doc-2"}}))
    assert refused[1]["relay_refusal"] == "file_not_offered"
    status, body, _ = _run(bridge.handle("call", {**request, "method": "getFile", "params": {"file_id": "doc-1"}}))
    reference = body["result"]["file_path"]  # opaque and URL-safe: the provider's path stays here
    assert status == 200 and reference.startswith("feed/") and "documents" not in reference
    assert body["result"]["file_id"] == "doc-1"
    assert _run(bridge.handle("file", {**request, "file_path": "documents/a.pdf"}))[0] == 404  # never a raw path
    assert _run(bridge.handle("file", {**request, "file_path": reference, "thread_id": "9"}))[0] == 404  # other topic
    status, content, media = _run(bridge.handle("file", {**request, "file_path": reference}))
    assert (status, content, media) == (200, b"file-bytes", "application/octet-stream")
    assert client.downloads == [("documents/a.pdf", 20 * 1024 * 1024)]  # the connector's limit, not the bridge's
    places.edit_place(root, place.place_id, "disconnect", author=places.AUTHOR_OWNER)
    _place(root)
    assert _run(bridge.handle("file", {**request, "file_path": reference}))[1]["relay_refusal"] == "binding_changed"


def test_the_http_doors_carry_a_query_revision_and_a_local_server_s_absolute_path_to_the_bytes(lib, tmp_path,
                                                                                                monkeypatch):
    """Through the Host doors: a JSON getFile, then a GET whose revision is query text, read from a --local store
    by the absolute path Telegram named (a client that strips its leading slash never sees it)."""
    from starlette.testclient import TestClient

    from ouroboros import telegram_owner_bot
    from ouroboros.gateway import host_service, host_sign_in
    from tests.test_telegram_api_base_loopback import _BotServer

    assert host_sign_in._authenticated is host_service._authenticated
    feed, api, _poll = lib
    store = tmp_path / "bot-api-store"
    (store / A / "documents").mkdir(parents=True)
    absolute = store / A / "documents" / "report.pdf"
    absolute.write_bytes(b"%PDF-1.4 the group's document")

    class _LocalServer(_BotServer):
        def respond(self, http_method, path, form):
            if path.endswith("/getFile"):
                self.calls.append(("getFile", form))
                return 200, json.dumps({"ok": True, "result": {"file_id": form["file_id"], "file_size": 29,
                                                                "file_path": str(absolute)}}).encode(), "application/json"
            return super().respond(http_method, path, form)

    async def authenticated(_ctx, token, permission=""):
        return "telegram-bot", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    monkeypatch.setattr("ouroboros.gateway.host_telegram_feed.official_connector", lambda *_a, **_k: True)

    async def scenario():
        async with _LocalServer() as server:
            native = api.TelegramClient(TOKEN, api_base_url=server.origin, files_dir=str(store))
            bridge, root, _holder = _bridge(feed, tmp_path, native)
            _consent(bridge)
            place = _place(root)
            bridge.publish(_message(document={"file_id": "doc-local"}), A)
            withdraw = telegram_owner_bot.register_bridge(bridge)
            try:
                with TestClient(host_service.create_host_service_app(root)) as http:
                    head = {"x-skill-token": "t"}
                    # The Host's own loop runs in the client's thread; this loop keeps serving the fake Bot API.
                    got = (await asyncio.to_thread(http.post, "/telegram/feed/call", headers=head, json={
                        "account_id": A, "place_id": place.place_id, "connected_revision": place.connected_revision,
                        "thread_id": "", "method": "getFile", "params": {"file_id": "doc-local"}})).json()
                    assert "result" in got, got
                    reference = got["result"]["file_path"]
                    query = {"account_id": A, "place_id": place.place_id, "thread_id": "",
                             "connected_revision": str(place.connected_revision), "file_path": reference.lstrip("/")}
                    answer = await asyncio.to_thread(http.get, "/telegram/feed/file", headers=head, params=query)
                    padded = await asyncio.to_thread(http.get, "/telegram/feed/file", headers=head,
                                                     params={**query, "connected_revision": f"0{place.connected_revision}"})
                    return got, answer, padded, server
            finally:
                withdraw()
    got, answer, padded, server = asyncio.run(scenario())
    assert got["ok"] is True and str(absolute) not in json.dumps(got)
    assert answer.status_code == 200 and answer.content == b"%PDF-1.4 the group's document"
    assert padded.status_code == 403 and padded.json()["relay_refusal"] == "binding_changed"  # not plain decimal
    assert [method for method, _form in server.calls].count("getFile") == 1


class _Cursor:
    def __init__(self):
        self.offset, self.saved = 0, []

    def is_replay(self, _update):
        return False

    def held_target(self, _update):
        return None

    def retire_target(self, _key):
        pass

    def remember_early(self, _update):
        pass

    def defer_group(self, _update, _period):
        return False

    def group_period(self, _update, period):
        return period

    def save(self, offset, handled):
        self.offset = offset
        self.saved.append((offset, list(handled)))


class _Binding:
    def current(self, _generation):
        return True


def test_a_group_update_bypasses_the_owner_authorizer_and_waits_in_the_early_phase(lib):
    _feed, _api, poll = lib
    seen, authorized = [], []

    async def authorize(_api, _client, update, _pinned, _lang):
        authorized.append(update["update_id"])
        return None

    async def on_group(update):
        seen.append(update["update_id"])

    group_panic = _message("/panic", update_id=11)
    callback = {"update_id": 12, "callback_query": {"id": "q", "from": {"id": 77},
                                                    "message": {"chat": {"id": int(CHAT), "type": "supergroup"}}}}
    cursor = _Cursor()
    _run(poll.handle_poll_batch(None, None, [_message(update_id=10), group_panic, callback], cursor, _Binding(), 1,
                                "42", 20, "full_access", "en", authorize=authorize,
                                is_panic=lambda update: "/panic" in json.dumps(update), on_message=None,
                                on_callback=None, on_group=on_group))
    assert seen == [10, 11] and authorized == [12] and cursor.offset == 13
    seen.clear()
    cursor = _Cursor()
    _run(poll.handle_poll_batch(None, None, [_message(update_id=20), group_panic], cursor, _Binding(), 1, "42", 0,
                                "full_access", "en", authorize=authorize, is_panic=lambda update: True,
                                on_message=None, on_callback=None, on_group=on_group))
    assert seen == [] and cursor.offset == 0  # over the limit: left for the next poll, never taken as Panic


def test_the_real_poller_feeds_groups_beside_the_owner_with_the_group_subscription(tmp_path, monkeypatch):
    from tests.test_telegram_api_base_loopback import _BotServer, _host_inbox, _owner_message, _state, _until
    plugin, _api, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)
    wiring = sys.modules[f"{_PACKAGE}.lib.telegram_group_wiring"]

    async def scenario(consent: bool):
        async with _BotServer() as server:
            (tmp_path / str(consent)).mkdir()
            state, api = _state(tmp_path / str(consent), TELEGRAM_API_BASE=server.origin)
            if consent:
                (state / "group_feed.json").write_text(json.dumps({"enabled": True, "account_id": A}), "utf-8")
            place = _place(state)  # the bridge's data root in this stand-in is its state folder
            server.push(_message("for the group", message_id=50))
            server.push(_message("elsewhere", chat="-1009999", message_id=51))
            server.push({"message": {"message_id": 52, "date": int(time.time()) + 5, "text": "hi bot",
                                     "chat": {"id": 900, "type": "private"}, "from": {"id": 900}}})
            server.push(_owner_message("owner words"))
            poller = asyncio.create_task(plugin._poller(api))
            bridge = wiring.feed(api)
            try:
                await _until(lambda: len(received) == 1 and len(server.methods("getUpdates")) >= 2)
                items = bridge.read(A, 0)
                if consent:
                    assert bridge.status()["intake_ready"], (bridge.status(), api.logs)
                    assert len(items) == 2, (bridge.status(), api.logs)
            finally:
                poller.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poller
            return server, items, place
    server, items, place = asyncio.run(scenario(True))
    assert [item["text"] for item in received] == ["owner words"]
    assert json.loads(server.methods("getUpdates")[0]["allowed_updates"]) == ALLOWED
    assert [item.get("update", {}).get("message", {}).get("text") for item in items] == ["for the group", None]
    assert items[0]["place"]["place_id"] == place.place_id and items[1]["sighting"]["chat"]["id"] == "-1009999"
    assert "hi bot" not in json.dumps(items)
    received.clear()
    server, items, _place_off = asyncio.run(scenario(False))
    assert items == []
    assert [item["text"] for item in received] == ["owner words"]
    assert "allowed_updates" not in server.methods("getUpdates")[0]  # Telegram's remembered selection, as before
    assert not (tmp_path / "False" / "state" / "group_feed.sqlite3").exists()


ALLOWED = ["message", "edited_message", "callback_query", "my_chat_member", "message_reaction",
           "message_reaction_count"]



class _Lifecycle:
    """A loaded bridge's API as the wiring uses it: state, routes and unload callbacks."""

    def __init__(self, state):
        self.state, self.unloads, self.routes = state, [], []

    def get_state_dir(self):
        return str(self.state)

    def get_runtime_info(self):
        return {"data_dir": str(self.state)}

    def log(self, *_args, **_fields):
        pass

    def register_route(self, path, handler, methods):
        self.routes.append(path)

    def on_unload(self, callback):
        self.unloads.append(callback)


class _Slotted(_Lifecycle):
    """A stand-in without weak references: pinned while loaded, released at its unload all the same."""

    __slots__ = ("state", "unloads", "routes")


def test_an_unloaded_bridge_leaves_no_feed_verified_bot_or_registration_behind(lib, tmp_path):
    feed_mod, api_mod, _poll = lib
    wiring = sys.modules[f"{_PACKAGE}.lib.telegram_group_wiring"]
    from ouroboros.telegram_owner_bot import current_bridge

    api = _Lifecycle(tmp_path)
    wiring.install(api, lambda current: (_Client(api_mod), 1))
    bridge = wiring.feed(api)
    assert wiring.feed(api) is bridge and api.routes == ["group_feed"]
    wiring.note_verified(api, 1, A, {"id": int(A), "username": "my_own_bot"})

    async def polling():
        assert current_bridge() is bridge  # registered only while the poller runs
    asyncio.run(wiring.while_polling(api, polling()))
    assert current_bridge() is None
    bridge.withdraw = wiring.register_running(api, bridge)  # a poller still running at unload
    alive = weakref.ref(api)
    for callback in api.unloads:
        callback()
    assert current_bridge() is None and wiring._held_feed(api) is None and wiring._noted(api) is None
    del api
    bridge = None
    gc.collect()
    assert alive() is None  # nothing here kept the unloaded API

    kept = wiring.feed(_Lifecycle(tmp_path))  # without unload hooks: its own callbacks do not pin the API either
    gc.collect()
    assert kept.verified() is None and kept.owner_bound() is False

    slotted = _Slotted(tmp_path)
    wiring.install(slotted, lambda current: (_Client(api_mod), 1))
    wiring.note_verified(slotted, 1, A, {"id": int(A)})
    assert wiring._held_feed(slotted) is not None and wiring._noted(slotted) is not None
    for callback in slotted.unloads:
        callback()
    assert wiring._held_feed(slotted) is None and wiring._noted(slotted) is None
    assert id(slotted) not in wiring._PINNED_FEEDS and id(slotted) not in wiring._PINNED
    reloaded = _Lifecycle(tmp_path)
    assert wiring.feed(reloaded) is not kept  # a reload builds its own feed
