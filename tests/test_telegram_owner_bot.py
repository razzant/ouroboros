"""Core's rules for your own Telegram bot (A) in groups (#4091): the owner_bot projection, its exact binding,
Host ingestion isolation and the Host doors the telegram-bot connector calls. Real places in a temporary data
folder; the bridge is a stand-in registered as the bundled bridge registers its feed. Nothing leaves the machine.
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from ouroboros import place_guard, places
from ouroboros import telegram_owner_bot as owner
from ouroboros.gateway import host_service
from ouroboros.place_projection import connect_declared_place
from ouroboros.places_migration import matching_declarations
from ouroboros.presence_admission import PresenceAdmissionError, admit_presence_turn

A, B, CHAT = "123456", "777000", "-1001500"
TARGET = {"provider": "telegram", "account_id": A, "conversation_id": CHAT, "thread_id": ""}
BASE_ROW = {"skill": "telegram-bot", "id": "place", "key": "telegram-bot:place", "provides": "place",
            "label": "Telegram", "destination": "vendor:Telegram", "scope": "account", "wakes": ["mention"],
            "facts": ["mention", {"id": "signal.words", "label": "Words you chose"}],
            "source": {"url": "/api/extensions/telegram-bot/status", "refresh_s": 15}}


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
def owner_bot(monkeypatch):
    monkeypatch.setattr(owner, "owner_bridge_bot_id", lambda *_args, **_kwargs: A)
    monkeypatch.setattr(owner, "official_connector", lambda *_args, **_kwargs: True)
    return A


def _place(root, *, account=A, chat=CHAT, contribution=owner.OWNER_BOT_CONTRIBUTION, state=places.STATE_ACTIVE):
    return places.connect_place(root, connector="telegram-bot", provider="telegram", label="Group",
                                account_id=account, conversation_id=chat, contribution_id=contribution,
                                scope="conversation", state=state)


def test_owner_bot_is_projected_only_from_the_live_connector_offering_it(monkeypatch, tmp_path, owner_bot):
    monkeypatch.setattr(owner, "_live_status_route", lambda: False)
    assert owner.with_owner_bot([BASE_ROW], tmp_path) == [BASE_ROW]  # an older connector: B alone
    monkeypatch.setattr(owner, "_live_status_route", lambda: True)
    rows = owner.with_owner_bot([BASE_ROW], tmp_path)
    assert rows[0] == BASE_ROW and len(rows) == 2
    projected = rows[1]
    assert projected["id"] == "owner_bot" and projected["key"] == "telegram-bot:owner_bot"
    assert projected["connect_mode"] == "exact" and projected["scope"] == "conversation"
    assert projected["source"] == {"url": "/api/extensions/telegram-bot/owner_bot_status", "refresh_s": 15}
    assert projected["facts"] == BASE_ROW["facts"] and projected["wakes"] == BASE_ROW["wakes"]
    monkeypatch.setattr(owner, "official_connector", lambda *_args, **_kwargs: False)
    assert owner.with_owner_bot([BASE_ROW], tmp_path) == [BASE_ROW]  # a fork or an edited copy: B alone
    monkeypatch.setattr(owner, "official_connector", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(owner, "owner_bridge_bot_id", lambda *_args, **_kwargs: "")
    assert owner.with_owner_bot([BASE_ROW], tmp_path) == [BASE_ROW]  # no bridge bot here: nothing to offer


STOCK = {"id": "ouroboros", "kind": "hub", "name": "Ouroboros", "url": "https://github.com/razzant/OuroborosHub",
         "part": "official", "ref": "main", "commit": "", "path": ""}
DIGEST = "sha256:" + "a" * 64


@pytest.mark.parametrize("source, official, digest, token_hash, expected", [
    (STOCK, True, DIGEST, DIGEST, True),
    (STOCK, True, DIGEST, None, True),  # the projection: no token to compare
    (STOCK, True, DIGEST, "sha256:" + "b" * 64, False),  # the token was issued for other bytes
    (STOCK, True, "sha256:" + "c" * 64, DIGEST, False),  # edited since its Official landing
    (STOCK, False, DIGEST, DIGEST, False),  # never endorsed as Official
    ({**STOCK, "ref": "candidate"}, True, DIGEST, DIGEST, False),  # another branch of the hub
    ({**STOCK, "url": "https://github.com/someone/OuroborosHub"}, True, DIGEST, DIGEST, False),  # another hub
    ({**STOCK, "path": "skills"}, True, DIGEST, DIGEST, False),  # another folder of it
    ({**STOCK, "part": "community"}, True, DIGEST, DIGEST, False),
])
def test_official_is_the_stock_hub_s_exact_release_from_local_records(monkeypatch, tmp_path, source, official, digest,
                                                                      token_hash, expected):
    import json

    from ouroboros.skill_install_record import install_record_path
    loaded = type("Loaded", (), {"content_hash": DIGEST})()
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_a, **_k: loaded)
    monkeypatch.setattr("ouroboros.deployment_extensions.admission_refusal", lambda _loaded: "")
    path = install_record_path(tmp_path, "telegram-bot")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "basis": "trusted", "entry": "telegram-bot", "source": source,
                                "official": official, "release": {"version": "0.14.0", "digest": digest}}),
                    encoding="utf-8")
    assert owner.official_connector(tmp_path, content_hash=token_hash) is expected
    path.unlink()
    assert owner.official_connector(tmp_path, content_hash=token_hash) is False  # no record: unknown, never Official
    monkeypatch.setattr("ouroboros.deployment_extensions.admission_refusal", lambda _loaded: "disabled")
    assert owner.official_connector(tmp_path) is False


def test_a_projected_declaration_never_vouches_for_an_unnamed_legacy_place(tmp_path):
    projected = {**BASE_ROW, "id": "owner_bot", "projected_from": "place"}
    legacy = places.Place(place_id="p", connector="telegram-bot", provider="telegram")
    assert matching_declarations(legacy, [BASE_ROW, projected]) == [BASE_ROW]
    mine = places.Place(place_id="q", connector="telegram-bot", provider="telegram", contribution_id="owner_bot")
    assert matching_declarations(mine, [BASE_ROW, projected]) == [projected]
    assert owner.declaring_place(mine).contribution_id == "place"  # its facts are the connector's own row's


class _Bridge:
    def __init__(self, allowed):
        self.allowed, self.asked = allowed, []

    def connect_allowed(self, account, chat):
        self.asked.append((account, chat))
        return self.allowed

    async def handle(self, operation, request):
        if operation == "file":
            return 200, b"bytes", "application/octet-stream"
        return 200, {"operation": operation, "request": dict(request)}, ""


def test_the_exact_writer_connects_only_a_group_the_running_bridge_confirms(monkeypatch, tmp_path, owner_bot):
    projected = {**BASE_ROW, "id": "owner_bot", "projected_from": "place", "connect_mode": "exact",
                 "scope": "conversation"}
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [BASE_ROW, projected])
    with pytest.raises(places.PlaceError):  # no bridge running here
        connect_declared_place(tmp_path, "telegram-bot", account_id=A, conversation_id=CHAT, contribution_id="owner_bot")
    bridge = _Bridge(False)
    withdraw = owner.register_bridge(bridge)
    try:
        with pytest.raises(places.PlaceError):
            connect_declared_place(tmp_path, "telegram-bot", account_id=A, conversation_id=CHAT,
                                   contribution_id="owner_bot")
        bridge.allowed = True
        place = connect_declared_place(tmp_path, "telegram-bot", account_id=A, conversation_id=CHAT,
                                       label="Book club", contribution_id="owner_bot")
    finally:
        withdraw()
    assert bridge.asked[-1] == (A, CHAT)
    assert (place.contribution_id, place.account_id, place.conversation_id, place.state) == (
        "owner_bot", A, CHAT, places.STATE_SILENT)
    plain = connect_declared_place(tmp_path, "telegram-bot", account_id=B, conversation_id="-200")
    assert plain.contribution_id == "place"  # without a contribution: the connector's own declaration, as before
    assert owner.current_bridge() is None


def test_an_owner_bot_binding_is_exact_current_and_never_falls_back(tmp_path, owner_bot):
    _place(tmp_path, account="", contribution="place")  # a blank-account place of the same chat
    _place(tmp_path, account=A, chat="", contribution="place")  # a whole-account place of A
    answer = place_guard.outbox_binding_facts(tmp_path, "telegram-bot", TARGET, contribution_id="owner_bot")
    assert answer["allowed"] is False and answer["code"] == "place_not_connected"
    default = place_guard.outbox_binding_facts(tmp_path, "telegram-bot", TARGET)
    assert default["allowed"] is False and default["code"] == "place_not_connected"  # A never binds through B
    place = _place(tmp_path)
    bound = place_guard.outbox_binding_facts(tmp_path, "telegram-bot", TARGET, contribution_id="owner_bot")
    assert bound["allowed"] is True and bound["place_id"] == place.place_id
    assert bound["binding"]["connected_revision"] == place.connected_revision
    again = place_guard.outbox_binding_facts(tmp_path, "telegram-bot", TARGET, binding=bound["binding"],
                                             contribution_id="owner_bot")
    assert again["allowed"] is True
    places.edit_place(tmp_path, place.place_id, "set_state", author=places.AUTHOR_OWNER, state=places.STATE_SILENT)
    held = place_guard.outbox_binding_facts(tmp_path, "telegram-bot", TARGET, binding=bound["binding"],
                                            contribution_id="owner_bot")
    assert held["allowed"] is False and held["state"] == "silent"
    places.edit_place(tmp_path, place.place_id, "disconnect", author=places.AUTHOR_OWNER)
    _place(tmp_path)  # the same identity connected again: a new connection
    stale = place_guard.outbox_binding_facts(tmp_path, "telegram-bot", TARGET, binding=bound["binding"],
                                             contribution_id="owner_bot")
    assert stale["allowed"] is False and stale["code"] == "place_disconnected"
    with pytest.raises(ValueError):
        place_guard.outbox_binding_facts(tmp_path, "telegram-bot", TARGET, contribution_id="something_else")


def _admit(root, binding_id="", account=A, revision=None, thread=""):
    return admit_presence_turn(drive_root=root, authenticated_transport_skill="telegram-bot", binding_id=binding_id,
                               provider="telegram", account_id=account, conversation_id=CHAT, thread_id=thread,
                               live_event=True, persist_import=False, connected_revision=revision)


def test_host_ingestion_admits_your_bot_only_through_its_named_exact_place(tmp_path, owner_bot):
    wildcard = _place(tmp_path, account="", contribution="place")
    with pytest.raises(PresenceAdmissionError) as refused:
        _admit(tmp_path)  # the resolver would pick the blank-account place: never for A
    assert refused.value.code == "place_not_connected"
    with pytest.raises(PresenceAdmissionError):
        _admit(tmp_path, binding_id=wildcard.place_id)
    assert _admit(tmp_path, account=B).place.place_id == wildcard.place_id  # B keeps its rules
    place = _place(tmp_path)
    with pytest.raises(PresenceAdmissionError):
        _admit(tmp_path)  # found, not named
    admitted = _admit(tmp_path, binding_id=place.place_id, revision=place.connected_revision, thread="9")
    assert admitted.place.place_id == place.place_id and admitted.connected_revision == place.connected_revision
    with pytest.raises(PresenceAdmissionError) as moved:
        _admit(tmp_path, binding_id=place.place_id, revision=place.connected_revision + 1)
    assert moved.value.code == "presence_origin_mismatch"
    with pytest.raises(PresenceAdmissionError):
        _admit(tmp_path, binding_id=place.place_id, revision=-1)  # an owner_bot event names its connection


@pytest.fixture()
def doors(tmp_path, monkeypatch):
    from ouroboros.gateway import host_sign_in

    assert host_sign_in._authenticated is host_service._authenticated
    who = {"skill": "telegram-bot", "official": True, "checked": []}

    async def authenticated(_ctx, token, permission=""):
        if not token:
            raise host_service.HostServiceAuthError("missing skill token")
        assert permission in {"", "presence"}  # /identity asks none; every feed door asks presence
        who.setdefault("asked", []).append(permission)
        return who["skill"], {"content_hash": "sha256:issued"}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)

    def official(_root, *, content_hash=None):
        who["checked"].append(content_hash)
        return who["official"]
    monkeypatch.setattr("ouroboros.gateway.host_telegram_feed.official_connector", official)
    with TestClient(host_service.create_host_service_app(tmp_path)) as client:
        yield client, who


def test_the_feed_doors_serve_only_an_official_release_and_the_poller_any_admitted_one(doors):
    client, who = doors
    headers = {"x-skill-token": "t"}
    who["official"] = False
    for method, path in [("GET", "/telegram/feed/identity"), ("GET", "/telegram/feed/updates"),
                         ("POST", "/telegram/feed/call"), ("POST", "/telegram/feed/upload"),
                         ("GET", "/telegram/feed/file")]:
        response = client.request(method, path, headers=headers, json={} if method == "POST" else None)
        assert response.status_code == 403 and response.json()["relay_refusal"] == "connector_mismatch", path
    assert set(who["checked"]) == {"sha256:issued"}  # the check reads the package the token was issued for
    poller = client.post("/telegram/feed/poller", headers=headers, json={"bot_id": "x1"})
    assert poller.status_code == 400 and poller.json()["code"] == "bot_id_invalid"  # reached: it keeps A from B


def test_the_doors_serve_only_the_telegram_bot_connector_and_refuse_without_the_bridge(doors):
    client, who = doors
    headers = {"x-skill-token": "t"}
    assert client.get("/telegram/feed/updates?account_id=1").json()["relay_refusal"] == "auth_blocked"
    who["skill"] = "another-skill"
    assert client.get("/telegram/feed/updates", headers=headers).json()["relay_refusal"] == "connector_mismatch"
    who["skill"] = "telegram-bot"
    for method, path in [("GET", "/telegram/feed/updates"), ("POST", "/telegram/feed/call"),
                         ("POST", "/telegram/feed/upload"), ("GET", "/telegram/feed/file")]:
        response = client.request(method, path, headers=headers, json={} if method == "POST" else None)
        body = response.json()
        assert response.status_code == 403 and body["ok"] is False and body["error_code"] == 403
        assert body["relay_refusal"] == "bridge_not_ready" and body["parameters"] == {"retry_after": 5}
    assert set(who["asked"]) == {"presence"}
    assert client.get("/identity", headers=headers).json()["telegram_group_feed"] == 1


def test_arbitration_is_core_s_own_answer_in_either_start_order(doors, monkeypatch):
    client, _who = doors
    headers = {"x-skill-token": "t"}
    monkeypatch.setattr(owner, "owner_bridge_bot_id", lambda *_args, **_kwargs: A)
    monkeypatch.setattr("ouroboros.gateway.host_telegram_feed.owner_bridge_bot_id",
                        lambda *_args, **_kwargs: A)
    body = client.get(f"/telegram/feed/identity?bot_id={A}", headers=headers).json()
    assert body["same_as_owner_bridge"] is True and body["reason"] == "bridge_not_ready"  # no poller yet: still true
    assert client.get(f"/telegram/feed/identity?bot_id={B}", headers=headers).json()["same_as_owner_bridge"] is False
    assert client.get("/telegram/feed/identity?bot_id=x", headers=headers).json()["same_as_owner_bridge"] is None
    assert client.get("/telegram/feed/identity?groups=1", headers=headers).json()["group_ids"] is None  # no bridge
    assert "group_ids" not in client.get("/telegram/feed/identity", headers=headers).json()
    monkeypatch.setattr("ouroboros.gateway.host_telegram_feed.owner_bridge_bot_id",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("settings unreadable")))
    assert client.get(f"/telegram/feed/identity?bot_id={A}", headers=headers).json()["same_as_owner_bridge"] is None


def test_the_doors_pass_requests_to_the_registered_bridge(doors):
    client, _who = doors
    headers = {"x-skill-token": "t"}
    withdraw = owner.register_bridge(_Bridge(True))
    try:
        call = client.post("/telegram/feed/call", headers=headers, json={"account_id": A, "method": "sendMessage"})
        assert call.json() == {"operation": "call", "request": {"account_id": A, "method": "sendMessage"}}
        updates = client.get("/telegram/feed/updates?account_id=1&offset=2", headers=headers)
        assert updates.json()["request"] == {"account_id": "1", "offset": "2"}
        identity = client.get("/telegram/feed/identity?groups=1", headers=headers).json()
        assert identity["request"] == {"groups": "1"} and identity["group_ids"] is None  # asked, unknown by this one
        assert client.get("/telegram/feed/identity", headers=headers).json()["request"] == {}
        file = client.get("/telegram/feed/file?file_path=x", headers=headers)
        assert file.content == b"bytes" and file.headers["content-type"] == "application/octet-stream"
        assert client.post("/telegram/feed/call", headers=headers, content=b"[1]").json()["relay_refusal"] == (
            "argument_not_allowed")
    finally:
        withdraw()
    assert client.get("/telegram/feed/updates", headers=headers).json()["relay_refusal"] == "bridge_not_ready"


def test_a_former_personal_bridge_place_never_serves_an_ordinary_lookup(tmp_path, monkeypatch):
    place = _place(tmp_path)  # stored as your own bot A's group
    targets = {"account_id": [A], "conversation_id": [CHAT]}
    monkeypatch.setattr(owner, "owner_bridge_bot_id", lambda *_args, **_kwargs: "")  # A's token is no longer the bridge's
    assert places.resolve_place(tmp_path, connector="telegram-bot", provider="telegram", account_id=A,
                                conversation_id=CHAT) is None
    assert places.place_for_target(tmp_path, connector="telegram-bot", targets=targets,
                                   conversation="conversation_id")[0] is None
    assert places.outbox_place(tmp_path, connector="telegram-bot", provider="telegram", account_id=A,
                               conversation_id=CHAT) is None
    checked = place_guard.outbox_binding_facts(tmp_path, "telegram-bot", TARGET)  # the same bot, now an ordinary B
    assert checked["place_id"] != place.place_id and checked["allowed"] is True and checked["state"] == "unbound"
    monkeypatch.setattr(owner, "owner_bridge_bot_id", lambda *_args, **_kwargs: A)  # A is the bridge's again
    assert places.place_for_target(tmp_path, connector="telegram-bot", targets=targets, conversation="conversation_id",
                                   owner_bot=owner.owner_bot_of(tmp_path, "telegram-bot"))[0] == place
    assert owner.owner_bot_of(tmp_path, "slack") == ""  # other connectors read nothing


def test_your_own_bots_events_are_captured_only_from_an_official_release(tmp_path, monkeypatch):
    import asyncio
    import threading
    from types import SimpleNamespace

    from ouroboros.gateway.host_service_places import _unofficial_owner_bot

    asked = []

    def official(_root, *, content_hash=None):
        asked.append(content_hash)
        return asked[-1] == "sha256:official"
    monkeypatch.setattr(owner, "official_connector", official)
    ctx = SimpleNamespace(data_dir=tmp_path, _auth_capacity=threading.BoundedSemaphore(2))
    mine = SimpleNamespace(place=_place(tmp_path))
    separate = SimpleNamespace(place=_place(tmp_path, account=B, chat="-200", contribution="place"))
    assert asyncio.run(_unofficial_owner_bot(ctx, mine, {"content_hash": "sha256:fork"})) is True
    assert asyncio.run(_unofficial_owner_bot(ctx, mine, {"content_hash": "sha256:official"})) is False
    assert asyncio.run(_unofficial_owner_bot(ctx, separate, {"content_hash": "sha256:fork"})) is False  # B as before
    assert asked == ["sha256:fork", "sha256:official"]  # the bytes its token was issued for; B's place read nothing


def test_waiting_group_arrivals_use_their_own_bot_connection_or_the_ordinary_path(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import presence_runner
    from ouroboros.gateway.host_service_places import _capture_connector_arrival

    calls = []
    period = {"account_id": A, "enabled": True, "at": "current"}
    bridge = SimpleNamespace(intake=lambda account: period if account == A else None,
                             connect_allowed=lambda account, chat: calls.append((account, chat)) or True)
    monkeypatch.setattr(owner, "current_bridge", lambda: bridge)
    monkeypatch.setattr(presence_runner, "capture_presence_event", lambda admission, *_a, **_k: admission.place.place_id)
    ctx = SimpleNamespace(data_dir=tmp_path)
    for place in (_place(tmp_path), _place(tmp_path, account=B, chat="-200", contribution="place")):
        admission = SimpleNamespace(place=place)
        event = SimpleNamespace(provider="telegram", account_id=place.account_id,
                                conversation_id=place.conversation_id, thread_id="")
        assert _capture_connector_arrival(ctx, admission, event, (), "test-turn", "test-input", 0) == (place.place_id, None)
    assert calls == [(A, CHAT)]
