"""The owner's private Panic never waits behind group disk work in its batch, and the groups follow next (#4091).

``handle_poll_batch`` with the shipped ``PollCursor`` and Panic test, over two deliveries of one batch as Telegram
redelivers updates whose offset was not confirmed. The owner authorizer is a stand-in that only compares ids, as the
bridge's own does once its owner is bound. No network, no Host, no real Panic: the message handler only records.
"""
from __future__ import annotations

import asyncio
import sys

from tests.test_telegram_proxy import _PACKAGE, _load

OWNER, STRANGER, GROUP, BOT = 42, 900, -1001500, "123456"


def _group(uid, mid):
    return {"update_id": uid, "message": {"message_id": mid, "date": 1, "text": "group words",
                                          "chat": {"id": GROUP, "type": "supergroup"}, "from": {"id": 77}}}


def _private(uid, mid, text, sender=OWNER):
    return {"update_id": uid, "message": {"message_id": mid, "date": 1, "text": text,
                                          "chat": {"id": sender, "type": "private"}, "from": {"id": sender}}}


class _Binding:
    def current(self, _generation):
        return True


class _Api:
    def log(self, *_args, **_kwargs):
        pass


class _Bridge:
    """What the poller hands its batches: the real cursor on disk, and what each handler saw."""

    def __init__(self, tmp_path):
        plugin, _api, _notifier = _load()
        self.poll = sys.modules[f"{_PACKAGE}.lib.telegram_poll_binding"]
        self.is_panic = plugin._panic_update
        self.path = tmp_path / "poll_offset.json"
        self.spooled, self.handled = [], []

    def cursor(self):
        """The cursor as a (re)started poller reads it from its file."""
        return self.poll.PollCursor(self.path, self.poll.CLOUD_BASE, BOT)

    def deliver(self, updates, cursor=None):
        cursor = cursor or self.cursor()

        async def on_group(update):
            self.spooled.append(update["update_id"])

        async def authorize(_api, _client, update, pinned, _lang):
            message = update.get("message") or {}
            chat, sender = (message.get("chat") or {}).get("id"), (message.get("from") or {}).get("id")
            return pinned if pinned and str(chat) == pinned and str(sender) == pinned else None

        async def on_message(_api, _client, update, update_id, _mode, _lang, reply_target=None):
            self.handled.append(update_id)

        async def on_callback(*_args, **_kwargs):
            return "en"

        asyncio.run(self.poll.handle_poll_batch(
            _Api(), object(), updates, cursor, _Binding(), 1, str(OWNER), 20, "full_access", "en",
            authorize=authorize, is_panic=self.is_panic, on_message=on_message, on_callback=on_callback,
            on_group=on_group))
        return cursor


def test_an_owner_panic_passes_the_groups_ahead_of_it_and_they_follow_on_the_next_delivery(tmp_path):
    bridge = _Bridge(tmp_path)
    batch = [_group(101, 1), _private(102, 9, "/panic"), _group(103, 2)]
    first = bridge.deliver(batch)
    assert bridge.handled == [102] and bridge.spooled == []  # no group disk work before the Panic
    assert first.offset == 0  # nothing confirmed past the waiting groups
    second = bridge.deliver(batch)  # Telegram delivers the same batch again (a restarted poller reads the file)
    assert bridge.handled == [102]  # the control is not repeated
    assert bridge.spooled == [101, 103] and second.offset == 104  # the groups had their turn, in order
    third = bridge.deliver([_group(104, 3)])
    assert bridge.spooled == [101, 103, 104] and third.offset == 105


def test_private_words_before_the_panic_keep_their_order_and_run_once(tmp_path):
    bridge = _Bridge(tmp_path)
    batch = [_group(201, 1), _private(202, 10, "do this first"), _private(203, 11, "/panic"), _group(204, 2)]
    bridge.deliver(batch)
    assert bridge.handled == [202, 203] and bridge.spooled == []
    cursor = bridge.deliver(batch)
    assert bridge.handled == [202, 203] and bridge.spooled == [201, 204] and cursor.offset == 205


def test_a_panic_from_someone_else_holds_nothing(tmp_path):
    bridge = _Bridge(tmp_path)
    cursor = bridge.deliver([_group(301, 1), _private(302, 12, "/panic", sender=STRANGER), _group(303, 2)])
    assert bridge.spooled == [301, 303] and bridge.handled == [] and cursor.offset == 304


def test_without_a_group_before_it_a_batch_keeps_its_usual_order(tmp_path):
    bridge = _Bridge(tmp_path)
    cursor = bridge.deliver([_private(401, 13, "words"), _private(402, 14, "/panic"), _group(403, 3)])
    assert bridge.handled == [401, 402] and bridge.spooled == [403] and cursor.offset == 404


def test_a_group_only_batch_yields_to_the_next_poll_and_keeps_its_original_period(tmp_path):
    bridge = _Bridge(tmp_path)
    period = {"enabled": True, "account_id": BOT, "at": "first"}
    seen = []

    async def deliver(cursor, batch):
        async def on_group(update):
            seen.append((update["update_id"], cursor.group_period(update, period)))

        await bridge.poll.handle_poll_batch(
            _Api(), object(), batch, cursor, _Binding(), 1, str(OWNER), 20, "full_access", "en",
            authorize=None, is_panic=bridge.is_panic, on_message=None, on_callback=None,
            on_group=on_group, group_period=period)

    batch = [_group(i, i) for i in range(11, 17)]
    asyncio.run(deliver(bridge.cursor(), batch))
    cursor = bridge.cursor()
    assert cursor.offset == 15 and [uid for uid, _ in seen] == [11, 12, 13, 14]
    assert [cursor.group_period(item, None) for item in batch[-2:]] == [period, period]
    asyncio.run(deliver(cursor, batch[-2:]))
    assert [uid for uid, _ in seen] == list(range(11, 17)) and cursor.offset == 17


def test_a_card_reply_past_a_frozen_group_keeps_its_found_addressee_when_the_host_refuses_it(tmp_path):
    bridge = _Bridge(tmp_path)
    found, reads = [], []

    def deliver(batch, refuse):
        async def on_group(update):
            bridge.spooled.append(update["update_id"])

        async def authorize(_api, _client, update, pinned, _lang):
            message = update.get("message") or {}
            sender = (message.get("from") or {}).get("id")
            return pinned if str(sender) == pinned else None

        async def on_message(_api, _client, update, update_id, _mode, _lang, reply_target=None):
            if update_id == 503:  # the owner's reply to a card: its addressee is looked up once
                found.append(reply_target(lambda: reads.append(update_id) or {"card": f"read {len(reads)}"}))
                if refuse:
                    raise bridge.poll.InjectNotAccepted("not recorded")
            bridge.handled.append(update_id)

        async def on_callback(*_args, **_kwargs):
            return "en"

        asyncio.run(bridge.poll.handle_poll_batch(
            _Api(), object(), batch, bridge.cursor(), _Binding(), 1, str(OWNER), 20, "full_access", "en",
            authorize=authorize, is_panic=bridge.is_panic, on_message=on_message, on_callback=on_callback,
            on_group=on_group))

    batch = [_group(501, 1), _private(502, 15, "/panic"), _private(503, 16, "about that card")]
    deliver(batch, refuse=True)  # the Panic passes the group; the reply after it is taken, then refused
    assert bridge.handled == [502] and bridge.spooled == []
    deliver(batch, refuse=False)  # delivered again: the same addressee, never looked up a second time
    assert found == [{"card": "read 1"}, {"card": "read 1"}] and reads == [503]
    assert bridge.handled == [502, 503] and bridge.spooled == [501]
