"""Normal transition and media receipts for the bundled group intake (no Telegram or model calls)."""
import asyncio
import sys
import threading
from types import SimpleNamespace

import pytest

from ouroboros import places, presence_runner, telegram_owner_bot
from ouroboros.gateway.host_service_places import _capture_with_group_membership
from tests.test_telegram_group_feed import A, ALLOWED, CHAT, _Client, _bridge, _consent, _membership, _message, _place, _send
from tests.test_telegram_group_feed import _one_process_of_readers as _one_process_of_readers
from tests.test_telegram_group_feed import lib as lib
from tests.test_telegram_proxy import _PACKAGE


def test_allow_waits_for_the_same_reader_to_establish_the_subscription(lib, tmp_path):
    feed, api, poll = lib
    bridge, root, _ = _bridge(feed, tmp_path, _Client(api))
    wiring = sys.modules[f"{_PACKAGE}.lib.telegram_group_wiring"]
    _place(root)
    old, fresh = _message("off batch"), _message("group open", update_id=101, message_id=11)
    requests = []

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def get_updates(offset, **options):
            requests.append((offset, options))
            if len(requests) == 1:
                started.set()
                await release.wait()
                return [old]
            return [] if len(requests) == 2 else [fresh]

        client = SimpleNamespace(bot_id=A, get_updates=get_updates)
        state = SimpleNamespace(log=lambda *_a: None)
        cursor = poll.PollCursor(tmp_path / "cursor.json", poll.CLOUD_BASE, A)
        waiting = asyncio.create_task(wiring.get_updates(bridge, client, 0))
        await started.wait()
        _consent(bridge)
        assert bridge.status()["enabled"] and not bridge.status()["intake_ready"]
        release.set()
        updates, period = await waiting
        assert period is None and updates == [old]
        # As in the ordinary poll path, an already observed Off batch keeps its original period on redelivery.
        cursor.defer_group(old, period)
        await wiring.on_group(state, bridge, client, period, cursor=cursor)(old)
        assert not bridge.status()["intake_ready"]
        assert (await wiring.get_updates(bridge, client, 100))[0] == []
        assert bridge.status()["intake_ready"]
        _, period = await wiring.get_updates(bridge, client, 100)
        await wiring.on_group(state, bridge, client, period, cursor=cursor)(old)
        await wiring.on_group(state, bridge, client, period, cursor=cursor)(fresh)
        [item] = bridge.read(A, 0)
        assert item["update"] == fresh
        bridge.set_consent(False, A)
        assert not bridge.status()["intake_ready"]
        _consent(bridge)
        assert not bridge.status()["intake_ready"]

    asyncio.run(scenario())
    assert requests == [(0, {}), (100, {"allowed_updates": ALLOWED, "poll_timeout": 0}),
                        (100, {"allowed_updates": ALLOWED})]


def test_sighting_and_full_event_share_identity_but_two_edits_are_distinct(lib, tmp_path):
    feed, api, _ = lib
    bridge, root, _ = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    update = _message("before connect")
    assert bridge.publish(update, A) == "sighting"
    _place(root)
    assert bridge.publish(update, A) == "duplicate"  # deliberately still pending
    [previous] = bridge.read(A, 0)
    assert bridge.read(A, previous["feed_id"] + 1) == []
    assert bridge.publish(update, A) == "duplicate"  # same event already confirmed
    for uid, words in [(101, "first edit"), (102, "second edit")]:
        edit = {"update_id": uid, "edited_message": {**update["message"], "text": words, "edit_date": 1700000001}}
        assert bridge.publish(edit, A) == "stored"
        assert bridge.publish(edit, A) == "duplicate"
    assert [item["update"]["edited_message"]["text"] for item in bridge.read(A, 0)] == ["first edit", "second edit"]


def test_membership_restriction_and_migration_use_telegram_evidence(lib, tmp_path):
    feed, api, _ = lib
    bridge, _, _ = _bridge(feed, tmp_path, _Client(api))
    _consent(bridge)
    bridge.publish(_membership(99, "member"), A)
    restricted = _membership(100, "restricted")
    restricted["my_chat_member"]["new_chat_member"]["is_member"] = False
    bridge.publish(restricted, A)
    assert bridge.facts(groups=True)["group_ids"] == []
    other = "-1009999"
    bridge.publish(_message(chat=other, update_id=101, message_id=11, migrate_from_chat_id=int(CHAT)), A)
    assert bridge.facts(groups=True)["group_ids"] == [other] and not bridge.connect_allowed(A, CHAT)


class _QuietClient(_Client):
    def __init__(self, api):
        super().__init__(api)
        self.member = None  # no available observation from the provider yet

    async def raw_request(self, method, *, data=None, files=None, timeout=30):
        answer = await super().raw_request(method, data=data, files=files, timeout=timeout)
        if method == "getChatMember":
            return ((200, {"ok": True, "result": {"user": {"id": int(A)}, "status": self.member}})
                    if self.member else (503, None))
        return answer


def test_quiet_group_sends_again_without_a_new_message_after_off_on(lib, tmp_path):
    feed, api, _ = lib
    client = _QuietClient(api)
    bridge, root, _ = _bridge(feed, tmp_path, client)
    _consent(bridge)
    bridge.publish(_message(), A)
    place = _place(root)
    bridge.set_consent(False, A)
    _consent(bridge)
    assert bridge.facts(groups=True)["group_ids"] == []
    assert _send(bridge, place=place)[1]["relay_refusal"] == "feed_unavailable"
    client.member = "member"
    assert _send(bridge, place=place)[0] == 200
    assert _send(bridge, place=place)[0] == 200
    assert [(method, fields) for method, fields, _ in client.calls if method == "getChatMember"] == [
        ("getChatMember", {"chat_id": CHAT, "user_id": A}), ("getChatMember", {"chat_id": CHAT, "user_id": A})]
    assert bridge.facts(groups=True)["group_ids"] == [CHAT]


def test_quiet_group_pending_arrival_keeps_the_exact_silent_binding(lib, tmp_path, monkeypatch):
    feed, api, _ = lib
    client = _QuietClient(api)
    client.member = "member"
    bridge, root, _ = _bridge(feed, tmp_path, client)
    _consent(bridge)
    place = _place(root, state=places.STATE_SILENT)
    bridge.publish(_message(), A)
    bridge.set_consent(False, A)
    _consent(bridge)
    monkeypatch.setattr(telegram_owner_bot, "current_bridge", lambda: bridge)
    from ouroboros import presence_events
    monkeypatch.setattr(presence_events, "settled_resend", lambda *_a: None)
    monkeypatch.setattr(presence_runner, "capture_presence_event", lambda admission, *_a, **_k: admission.place)
    ctx = SimpleNamespace(data_dir=root, _auth_capacity=threading.BoundedSemaphore(2))
    event = SimpleNamespace(provider="telegram", account_id=A, conversation_id=CHAT, thread_id="")
    result = asyncio.run(_capture_with_group_membership(ctx, SimpleNamespace(place=place), event, (), "t", "i", 0))
    assert result == (place, None) and places.get_place(root, place.place_id).state == places.STATE_SILENT
    assert [row[0] for row in client.calls] == ["getChatMember"]


@pytest.mark.parametrize("read_timeout", [True, False])
def test_file_bytes_outcome_is_typed_only_after_getfile(lib, tmp_path, read_timeout):
    feed, api, _ = lib
    client = _Client(api, answer=(200, {"ok": True, "result": {"file_path": "documents/readme.txt"}}))
    bridge, root, _ = _bridge(feed, tmp_path, client)
    _consent(bridge)
    place = _place(root)
    bridge.publish(_message(document={"file_id": "doc", "file_name": "readme.txt"}), A)
    _, body, _ = _send(bridge, "getFile", {"file_id": "doc"}, place)

    async def read_bytes(_path, limit=None):
        assert limit and _path == "documents/readme.txt"
        if read_timeout:
            raise api.TelegramFileReadTimeout("file read")
        return b"ordinary file"

    client._download_bytes = read_bytes
    request = {"account_id": A, "place_id": place.place_id, "connected_revision": place.connected_revision,
               "file_path": body["result"]["file_path"]}
    result = asyncio.run(bridge.handle("file", request))
    assert result == ((504, {"relay_outcome": "file_read_timeout"}, "") if read_timeout
                      else (200, b"ordinary file", "application/octet-stream"))
