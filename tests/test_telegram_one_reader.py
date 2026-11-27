"""One Telegram bot, one local reader (#4091 phase 2): the bundled bridge and the telegram-bot connector's own bot
are given a bot once per process, before either may contact Telegram with it, and keep it until restart.

The bridge is the shipped poller against the loopback fake Bot API; the connector's side is the shipped
PluginAPI settings read. Nothing leaves the machine.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import threading

import pytest

from ouroboros import extension_loader
from ouroboros import telegram_owner_bot as owner
from tests.test_telegram_proxy import _PACKAGE, _TOKEN, _load

BOT = _TOKEN.split(":", 1)[0]  # 123456, the stand-in bridge's bot
OTHER = "654321:OTHER-SECRET"


@pytest.fixture(autouse=True)
def fresh_process():
    with owner._holders_lock:
        saved = dict(owner._holders)
        owner._holders.clear()
    yield
    with owner._holders_lock:
        owner._holders.clear()
        owner._holders.update(saved)


def _connector_read(token: str, root=None) -> dict:
    """The connector's runtime reading its token through the real PluginAPI (an authorized read) over ``root``."""
    impl = extension_loader.PluginAPIImpl(extension_loader._PluginAPIConfig(
        skill_name="telegram-bot", permissions=["read_settings"], env_allowlist=["TELEGRAM_PUBLIC_BOT_TOKEN"],
        state_dir=__import__("pathlib").Path("/tmp"), settings_reader=lambda: {"TELEGRAM_PUBLIC_BOT_TOKEN": token},
        granted_keys=["TELEGRAM_PUBLIC_BOT_TOKEN"], drive_root=root))
    return impl.get_settings(["TELEGRAM_PUBLIC_BOT_TOKEN"])


def _bridge_state(root, *, state="ready", reason="", bot=BOT, owner="42"):
    """The bridge's own persisted facts (``bridge_status.json`` and its ``TELEGRAM_CHAT_ID``) in ``root``."""
    folder = root / "state" / "skills" / "telegram"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "bridge_status.json").write_text(json.dumps({"state": state, "reason_code": reason, "bot": bot}),
                                               encoding="utf-8")
    (folder / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": owner}), encoding="utf-8")
    return folder


def test_whoever_is_given_a_bot_first_keeps_it_and_other_bots_are_untouched():
    assert _connector_read(f"{BOT}:CONNECTOR-SECRET") == {"TELEGRAM_PUBLIC_BOT_TOKEN": f"{BOT}:CONNECTOR-SECRET"}
    assert owner.bot_holder(BOT) == "telegram-bot"
    assert owner.claim_bot("telegram", BOT) == "telegram-bot"  # the bridge is refused: the connector keeps it
    assert owner.claim_bot("telegram", "654321") == "telegram"  # another bot is the bridge's
    assert _connector_read(OTHER) == {}  # and is never handed to the connector after that
    assert _connector_read(f"{BOT}:ROTATED") == {"TELEGRAM_PUBLIC_BOT_TOKEN": f"{BOT}:ROTATED"}  # its own, rotated
    assert "SECRET" not in json.dumps(owner._holders)  # ids only


def test_a_bot_the_bridge_holds_is_never_handed_to_the_connector_whatever_its_settings_say():
    assert owner.claim_bot("telegram", BOT) == "telegram"
    assert _connector_read(f"{BOT}:ANOTHER-SPELLING") == {}  # read as not saved: no deleteWebhook, no getUpdates
    impl_state = extension_loader.PluginAPIImpl(extension_loader._PluginAPIConfig(
        skill_name="telegram-bot", permissions=["read_settings"], env_allowlist=["TELEGRAM_PUBLIC_BOT_TOKEN"],
        state_dir=__import__("pathlib").Path("/tmp"),
        settings_reader=lambda: {"TELEGRAM_PUBLIC_BOT_TOKEN": f"{BOT}:X"},
        granted_keys=["TELEGRAM_PUBLIC_BOT_TOKEN"], plugin_api_generation="2.7"))
    state = impl_state.get_settings_state(["TELEGRAM_PUBLIC_BOT_TOKEN"])
    assert state["values"] == {} and state["missing"] == ["TELEGRAM_PUBLIC_BOT_TOKEN"]
    other_skill = extension_loader.PluginAPIImpl(extension_loader._PluginAPIConfig(
        skill_name="another", permissions=["read_settings"], env_allowlist=["TELEGRAM_PUBLIC_BOT_TOKEN"],
        state_dir=__import__("pathlib").Path("/tmp"),
        settings_reader=lambda: {"TELEGRAM_PUBLIC_BOT_TOKEN": f"{BOT}:X"},
        granted_keys=["TELEGRAM_PUBLIC_BOT_TOKEN"]))
    assert other_skill.get_settings(["TELEGRAM_PUBLIC_BOT_TOKEN"]) == {"TELEGRAM_PUBLIC_BOT_TOKEN": f"{BOT}:X"}
    assert owner.bot_holder(BOT) == "telegram"


def test_a_claim_and_a_read_racing_for_one_bot_leave_it_to_exactly_one_reader():
    for _ in range(200):
        with owner._holders_lock:
            owner._holders.clear()
        start, results = threading.Barrier(2), {}

        def bridge():
            start.wait()
            results["bridge"] = owner.claim_bot("telegram", BOT) == "telegram"

        def connector():
            start.wait()
            results["connector"] = bool(_connector_read(f"{BOT}:CONNECTOR"))

        threads = [threading.Thread(target=bridge), threading.Thread(target=connector)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results["bridge"] != results["connector"]


def _scenario(tmp_path, monkeypatch, connector_first: bool):
    from tests.test_telegram_api_base_loopback import _BotServer, _host_inbox, _owner_message, _state, _until
    plugin, _api, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)

    async def run():
        async with _BotServer() as server:
            state, api = _state(tmp_path, TELEGRAM_API_BASE=server.origin)
            server.push(_owner_message("owner words"))
            if connector_first:  # the connector's runtime read its token first: it is the reader
                assert _connector_read(f"{BOT}:CONNECTOR-SECRET")
            poller = asyncio.create_task(plugin._poller(api))
            try:
                if connector_first:
                    await asyncio.sleep(1.5)  # a back-off of re-checks, never a request
                else:
                    await _until(lambda: len(received) == 1 and len(server.methods("getUpdates")) >= 1)
                    # The bridge's long poll is out (the fake answers each poll at once; the next is in flight):
                    # the connector's read of the same bot, by another spelling, hands it nothing.
                    assert _connector_read(f"{BOT}:CONNECTOR-SECRET") == {}
                    await _until(lambda: len(server.methods("getUpdates")) >= 2)
            finally:
                poller.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poller
            return server, state
    return asyncio.run(run()) + (received,)


def test_the_bridge_never_contacts_telegram_with_a_bot_the_connector_holds(tmp_path, monkeypatch):
    server, state, received = _scenario(tmp_path, monkeypatch, connector_first=True)
    assert server.calls == [] and received == []  # no getMe, no setMyCommands, no getUpdates
    status = json.loads((state / "bridge_status.json").read_text(encoding="utf-8"))
    assert (status["state"], status["reason_code"]) == ("degraded", "telegram_bot_in_use")
    assert owner.bot_holder(BOT) == "telegram-bot"
    wiring = sys.modules[f"{_PACKAGE}.lib.telegram_group_wiring"]
    assert wiring.TelegramBotInUse("x", status_code=409).transient


def test_a_bot_the_bridge_reads_stays_its_own_while_the_connector_asks_for_it(tmp_path, monkeypatch):
    server, _state_dir, received = _scenario(tmp_path, monkeypatch, connector_first=False)
    assert [item["text"] for item in received] == ["owner words"]
    assert owner.bot_holder(BOT) == "telegram"
    assert {method for method, _form in server.calls} >= {"getMe", "getUpdates"}


def test_identity_answers_from_settings_the_one_decider_in_every_process(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from ouroboros.gateway import host_service, host_sign_in

    assert host_sign_in._authenticated is host_service._authenticated
    async def authenticated(_ctx, token, permission=""):
        return "telegram-bot", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    monkeypatch.setattr("ouroboros.gateway.host_telegram_feed.owner_bridge_bot_id", lambda *_a, **_k: BOT)
    monkeypatch.setattr("ouroboros.gateway.host_telegram_feed.official_connector", lambda *_a, **_k: True)
    monkeypatch.setattr(owner, "owner_bridge_bot_id", lambda *_a, **_k: BOT)  # the same Settings for every reader
    with TestClient(host_service.create_host_service_app(tmp_path)) as client:
        ask = lambda bot: client.get(f"/telegram/feed/identity?bot_id={bot}",  # noqa: E731
                                     headers={"x-skill-token": "t"}).json()
        assert ask(BOT)["same_as_owner_bridge"] is True  # the owner's configured bot is the bridge's, from Settings
        assert _connector_read(f"{BOT}:CONNECTOR") == {}  # A is never handed to the connector, whichever reads first
        assert ask(BOT)["same_as_owner_bridge"] is True  # a read does not move it
        assert owner.claim_bot("telegram", BOT) == "telegram"
        assert _connector_read(OTHER) == {"TELEGRAM_PUBLIC_BOT_TOKEN": OTHER}  # a distinct bot is the connector's
        assert ask("654321")["same_as_owner_bridge"] is False


@pytest.fixture()
def asking_connector(monkeypatch, tmp_path):
    """A connector offering owner_bot_status: it asks before polling; its saved, granted token names bot BOT."""
    from starlette.testclient import TestClient

    from ouroboros.gateway import host_service, host_sign_in

    assert host_sign_in._authenticated is host_service._authenticated
    monkeypatch.setattr(owner, "_live_status_route", lambda: True)
    saved = {"TELEGRAM_PUBLIC_BOT_TOKEN": f"{BOT}:CONNECTOR-SECRET", "readable": True}

    def load_settings(strict=False, **_kwargs):
        if saved.get("unreadable"):
            raise OSError("settings unreadable")
        return {"TELEGRAM_PUBLIC_BOT_TOKEN": saved["TELEGRAM_PUBLIC_BOT_TOKEN"]}
    monkeypatch.setattr("ouroboros.config.load_settings", load_settings)
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_a, **_k: object())
    monkeypatch.setattr("ouroboros.skill_loader.grant_status_for_skill", lambda *_a, **_k: {
        "requested_keys": ["TELEGRAM_PUBLIC_BOT_TOKEN"],
        "granted_keys": ["TELEGRAM_PUBLIC_BOT_TOKEN"] if saved["readable"] else []})

    async def authenticated(_ctx, token, permission=""):
        return "telegram-bot", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    with TestClient(host_service.create_host_service_app(tmp_path)) as client:
        def ask(bot_id):
            response = client.post("/telegram/feed/poller", headers={"x-skill-token": "t"}, json={"bot_id": bot_id})
            return response.status_code, response.json()
        yield ask, saved, client


def test_an_asking_connector_s_reads_claim_nothing_and_its_poll_asks_first(asking_connector):
    ask, _saved, client = asking_connector
    assert client.get("/identity", headers={"x-skill-token": "t"}).json()["telegram_group_poll_claim"] == 1
    assert _connector_read(f"{BOT}:CONNECTOR-SECRET")  # a status or Widget read: nothing claimed
    assert owner.bot_holder(BOT) == ""
    assert owner.claim_bot("telegram", BOT) == "telegram"  # so the bridge may still take it
    assert _connector_read(f"{BOT}:CONNECTOR-SECRET")  # a passive read still sees its saved token
    assert ask(BOT) == (200, {"ok": True, "owned": False, "holder": "telegram"})  # held before Telegram


def test_the_connector_s_ask_keeps_its_bot_from_the_bridge_and_binds_its_saved_token(asking_connector):
    ask, saved, _client = asking_connector
    assert ask(BOT) == (200, {"ok": True, "owned": True})
    assert ask(BOT) == (200, {"ok": True, "owned": True})  # asking again before each poll keeps it
    assert owner.claim_bot("telegram", BOT) == "telegram-bot"
    assert ask("654321") == (409, {"ok": False, "code": "account_changed"})  # not its saved bot
    saved["readable"] = False
    assert ask(BOT) == (409, {"ok": False, "code": "account_changed"})  # its token is not granted now
    saved["unreadable"] = True
    assert ask(BOT) == (503, {"ok": False, "code": "settings_unknown"})
    assert ask("12x")[0] == 400


def test_a_connector_switching_bots_keeps_the_first_until_restart():
    assert _connector_read(f"{BOT}:FIRST")  # an older connector polled BOT
    assert _connector_read(OTHER)  # its owner saved another bot since: its last poll of BOT may still be out
    assert owner.claim_bot("telegram", BOT) == "telegram-bot" and owner.bot_holder("654321") == "telegram-bot"
    with owner._holders_lock:
        owner._holders.clear()  # a restart: the one boundary that frees both
    assert owner.claim_bot("telegram", BOT) == "telegram"


def test_while_the_bridge_s_long_poll_is_held_open_the_connector_is_refused_both_ways(tmp_path, monkeypatch):
    """A sustained loopback getUpdates: the bridge's request stays open while the connector asks for its bot."""
    import time
    import urllib.parse

    from tests.test_telegram_api_base_loopback import _BotServer, _state, _until
    plugin, _api, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    monkeypatch.setattr(owner, "_live_status_route", lambda: False)

    class _HoldingServer(_BotServer):
        """getUpdates answers only when released: the request is in flight until then."""

        def __init__(self):
            super().__init__()
            self.open_polls, self.released = 0, asyncio.Event()

        async def _serve(self, reader, writer):
            head = await reader.readuntil(b"\r\n\r\n")
            first, *lines = head.decode("latin-1").split("\r\n")
            headers = dict(line.lower().split(": ", 1) for line in lines if ": " in line)
            body = await reader.readexactly(int(headers.get("content-length", 0)))
            http_method, target, _version = first.split(" ")
            form = {key: values[-1] for key, values in urllib.parse.parse_qs(body.decode()).items()}
            path = urllib.parse.urlsplit(target).path
            if path.endswith("/getUpdates"):
                self.open_polls += 1
                await self.released.wait()
                self.open_polls -= 1
            status, payload, payload_type = self.respond(http_method, path, form)
            writer.write(f"HTTP/1.1 {status} X\r\nContent-Type: {payload_type}\r\nContent-Length: {len(payload)}\r\n"
                         "Connection: close\r\n\r\n".encode() + payload)
            await writer.drain()
            writer.close()

    async def run():
        async with _HoldingServer() as server:
            _state_dir, api = _state(tmp_path, TELEGRAM_API_BASE=server.origin)
            poller = asyncio.create_task(plugin._poller(api))
            try:
                await _until(lambda: server.open_polls == 1)
                started = time.monotonic()
                assert _connector_read(f"{BOT}:CONNECTOR") == {}  # an older connector: no token handed
                monkeypatch.setattr(owner, "_live_status_route", lambda: True)
                assert _connector_read(f"{BOT}:CONNECTOR")  # an asking connector's passive read sees its token
                assert owner.claim_bot("telegram-bot", BOT) == "telegram"  # and its ask is refused: still held
                assert server.open_polls == 1 and time.monotonic() - started < 5  # the bridge's poll still open
                server.released.set()
            finally:
                poller.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poller
            return server
    server = asyncio.run(run())
    assert [method for method, _form in server.calls].count("getUpdates") >= 1
    assert owner.bot_holder(BOT) == "telegram"


@pytest.fixture()
def bridge_on(monkeypatch):
    """The bundled bridge may run here with its saved token naming bot BOT."""
    monkeypatch.setattr(owner, "owner_bridge_bot_id", lambda *_a, **_k: BOT)


def test_a_bridge_that_worked_for_its_owner_keeps_its_bot_from_a_duplicate_token_in_either_order(tmp_path, bridge_on,
                                                                                               monkeypatch):
    _bridge_state(tmp_path)
    assert _connector_read(f"{BOT}:DUPLICATE", tmp_path) == {}  # the connector read first, e.g. after a restart
    assert owner.bot_holder(BOT) == ""  # decided from Settings: the read records no holder for the owner's bot
    assert owner.claim_bot("telegram", BOT) == "telegram"
    monkeypatch.setattr("ouroboros.config.load_settings", lambda strict=False, **_k: {
        "TELEGRAM_PUBLIC_BOT_TOKEN": f"{BOT}:DUPLICATE"})
    assert owner.connector_token_held(tmp_path) is True  # the typed fact the bridge's group_feed status shows
    assert _connector_read(OTHER, tmp_path) == {"TELEGRAM_PUBLIC_BOT_TOKEN": OTHER}  # a distinct bot: untouched
    with owner._holders_lock:
        owner._holders.clear()
    monkeypatch.setattr(owner, "_live_status_route", lambda: True)
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_a, **_k: object())
    monkeypatch.setattr("ouroboros.skill_loader.grant_status_for_skill", lambda *_a, **_k: {
        "requested_keys": ["TELEGRAM_PUBLIC_BOT_TOKEN"], "granted_keys": ["TELEGRAM_PUBLIC_BOT_TOKEN"]})
    assert owner.claim_connector_poll(tmp_path, BOT) == (200, {"ok": True, "owned": False, "holder": "telegram"})


@pytest.mark.parametrize("facts", [
    {"state": "connecting"},  # never confirmed by getMe
    {"state": "degraded", "reason": "telegram_startup_deferred"},
    {"state": "degraded", "reason": "telegram_bot_in_use"},  # it gave the bot up
    {"state": "degraded", "reason": "telegram_conflict"},
    {"state": "error", "reason": "telegram_rejected"},
    {"bot": "777001"},  # ready with another bot
    {"owner": ""},  # never bound to its owner
    {"owner": "0"},
])
def test_the_owner_s_bot_is_the_bridge_s_whatever_the_health_record_says(tmp_path, bridge_on, facts):
    _bridge_state(tmp_path, **facts)
    # A is decided from Settings, not a health record: a transient deferral or conflict never gives it to the connector.
    assert _connector_read(f"{BOT}:DUPLICATE", tmp_path) == {}
    assert owner.claim_bot("telegram", BOT) == "telegram"


def test_the_owner_s_bot_is_never_handed_even_to_a_connector_that_asks_first(tmp_path, bridge_on, monkeypatch):
    monkeypatch.setattr(owner, "_live_status_route", lambda: True)  # a connector offering owner_bot_status
    assert _connector_read(f"{BOT}:DUPLICATE", tmp_path) == {}  # it reaches your bot through the relay only
    assert _connector_read(OTHER, tmp_path) == {"TELEGRAM_PUBLIC_BOT_TOKEN": OTHER}  # its own bot B: read freely
    assert owner.bot_holder("654321") == ""  # and claimed by nothing: it asks before polling


def test_settings_that_cannot_be_read_hold_the_connector_s_token_back(tmp_path, monkeypatch):
    def unreadable(*_args, strict=False, **_kwargs):
        if strict:
            raise OSError("settings unreadable")
        return ""
    monkeypatch.setattr(owner, "owner_bridge_bot_id", unreadable)
    assert _connector_read(OTHER, tmp_path) == {}  # whose bot it is is unknown: not handed until a readable answer
    monkeypatch.setattr(owner, "owner_bridge_bot_id", lambda *_a, **_k: "")
    assert _connector_read(OTHER, tmp_path) == {"TELEGRAM_PUBLIC_BOT_TOKEN": OTHER}


def test_an_off_bridge_hands_its_bot_to_no_one_and_an_unreadable_settings_is_unknown(tmp_path, monkeypatch):
    _bridge_state(tmp_path, state="degraded", reason="telegram_unreachable")
    monkeypatch.setattr(owner, "owner_bridge_bot_id", lambda *_a, **_k: BOT)
    monkeypatch.setattr("ouroboros.config.load_settings", lambda strict=False, **_k: {
        "TELEGRAM_PUBLIC_BOT_TOKEN": f"{BOT}:DUPLICATE"})
    assert owner.connector_token_held(tmp_path) is True  # a deferred bridge still keeps its bot from the connector

    def unreadable(strict=False, **_kwargs):
        raise OSError("settings unreadable")
    monkeypatch.setattr("ouroboros.config.load_settings", unreadable)
    assert owner.connector_token_held(tmp_path) is None  # unknown, never "not held"
    monkeypatch.setattr("ouroboros.config.load_settings", lambda strict=False, **_k: {})
    monkeypatch.setattr(owner, "owner_bridge_bot_id", lambda *_a, **_k: "")  # Off, or its token names no bot
    assert _connector_read(f"{BOT}:DUPLICATE", tmp_path)  # not the owner's bot now: the connector may read it


def test_after_a_restart_the_duplicate_token_never_takes_the_working_bridge_s_bot(tmp_path, monkeypatch, bridge_on):
    """The real poller first: it records ready for its owner. A restart, then the connector reads first."""
    from tests.test_telegram_api_base_loopback import _BotServer, _host_inbox, _owner_message, _until
    from tests.test_telegram_proxy import _Api
    plugin, _api, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)
    state_dir = tmp_path / "state" / "skills" / "telegram"
    state_dir.mkdir(parents=True)

    async def run(server, expect):
        (state_dir / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42",
                                                             "TELEGRAM_API_BASE": server.origin}), encoding="utf-8")
        api = _Api(state_dir, {"TELEGRAM_BOT_TOKEN": _TOKEN})
        server.push(_owner_message(expect))
        poller = asyncio.create_task(plugin._poller(api))
        try:
            await _until(lambda: expect in [item["text"] for item in received])
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller

    async def scenario():
        async with _BotServer() as server:
            await run(server, "first run")
            assert json.loads((state_dir / "bridge_status.json").read_text(encoding="utf-8"))["state"] == "ready"
            with owner._holders_lock:
                owner._holders.clear()  # an ordinary restart
            assert _connector_read(f"{BOT}:DUPLICATE", tmp_path) == {}  # the connector's first read is refused
            await run(server, "after restart")  # the owner's bridge still reads its bot
    asyncio.run(scenario())
    assert [item["text"] for item in received] == ["first run", "after restart"]
    assert owner.bot_holder(BOT) == "telegram"


def test_a_script_or_companion_environment_asks_whose_bot_it_is_in_its_callers_own_root(tmp_path, monkeypatch):
    from ouroboros.tools import skill_exec

    roots = []

    def owner_of(root, **_kwargs):
        roots.append(str(root))
        return BOT if str(root) == str(tmp_path / "mine") else ""
    monkeypatch.setattr(owner, "owner_bridge_bot_id", owner_of)
    settings = {"TELEGRAM_PUBLIC_BOT_TOKEN": _TOKEN}
    env = skill_exec._scrub_env(["TELEGRAM_PUBLIC_BOT_TOKEN"], tmp_path / "state", "telegram-bot",
                                granted_keys=["TELEGRAM_PUBLIC_BOT_TOKEN"], settings=settings,
                                drive_root=tmp_path / "mine")
    assert "TELEGRAM_PUBLIC_BOT_TOKEN" not in env  # the bridge's bot in that root: never handed to the connector
    assert roots and set(roots) == {str(tmp_path / "mine")}
