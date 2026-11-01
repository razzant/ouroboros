"""TELEGRAM_API_BASE end to end: the real skill code against Bot API servers on loopback.

Each fake server is a plain HTTP origin on 127.0.0.1 that answers the Bot API methods this
skill uses and numbers updates the way tdlib/telegram-bot-api's queue does (an offset more
than ten past its tail or far before its head is answered from its oldest update). HTTPX,
the poller, the bridge handlers, the typing cue, the notifier, the menu manager, the
companion's owner check and the settings form are the shipped code; only the server and the
host's /chat/inject receiver are substituted. Nothing leaves the machine.
"""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import sys
import time
import urllib.parse
from pathlib import Path

import httpx
import pytest

from tests.test_telegram_proxy import _Api, _TOKEN, _bridge_events, _load
from tests.test_telegram_settings import _RouteRequest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "telegram" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import companion  # noqa: E402
import telegram_menu  # noqa: E402
from telegram_settings import merge_settings  # noqa: E402

_CLOUD = "https://api.telegram.org"
_MESSAGE_IDS = itertools.count(100)


class _BotServer:
    """One Bot API server: its own update queue, file store answers and call log."""

    def __init__(self, *, first_update_id: int = 1, files: dict | None = None, serve_files: bool = True):
        self.calls: list[tuple[str, dict]] = []
        self.queue: list[tuple[int, dict]] = []
        self.next_id = first_update_id
        self.files = dict(files or {})  # file_id -> (file_path, bytes)
        self.serve_files = serve_files
        self.button = {"type": "commands"}
        self.port = 0
        self._server = None

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def push(self, update: dict) -> int:
        update_id = self.next_id
        self.next_id += 1
        self.queue.append((update_id, {"update_id": update_id, **update}))
        return update_id

    def methods(self, name: str) -> list[dict]:
        return [form for method, form in self.calls if method == name]

    def _get_updates(self, offset: int) -> list[dict]:
        if not self.queue:
            return []
        head, tail = self.queue[0][0], self.next_id
        start = offset if 0 < offset <= tail + 10 and offset >= head - 100000 else head
        self.queue = [item for item in self.queue if item[0] >= start]
        return [update for _id, update in self.queue]

    def respond(self, http_method: str, path: str, form: dict) -> tuple[int, bytes, str]:
        file_prefix = f"/file/bot{_TOKEN}/"
        if http_method == "GET" and path.startswith(file_prefix):
            wanted = urllib.parse.unquote(path[len(file_prefix):])
            self.calls.append(("file", {"path": wanted}))
            for file_path, content in self.files.values():
                if self.serve_files and wanted == file_path:
                    return 200, content, "application/octet-stream"
            return 404, json.dumps({"ok": False, "error_code": 404, "description": "Not Found"}).encode(), "application/json"
        prefix = f"/bot{_TOKEN}/"
        if not path.startswith(prefix):
            return 404, b'{"ok": false, "error_code": 404, "description": "Not Found"}', "application/json"
        method = path[len(prefix):]
        self.calls.append((method, form))
        result: object = {"message_id": 555}
        if method == "getMe":
            result = {"id": 1, "is_bot": True, "username": "loopback_bot"}
        elif method in ("setMyCommands", "sendChatAction", "answerCallbackQuery"):
            result = True
        elif method == "getUpdates":
            result = self._get_updates(int(form.get("offset") or 0))
        elif method == "getFile":
            file_path, _content = self.files[form["file_id"]]
            result = {"file_id": form["file_id"], "file_path": file_path}
        elif method == "getChat":
            result = {"id": 42, "type": "private"}
        elif method == "getChatMenuButton":
            result = self.button
        elif method == "setChatMenuButton":
            self.button = form["menu_button"]
            result = True
        return 200, json.dumps({"ok": True, "result": result}).encode(), "application/json"

    async def _serve(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            first, *lines = head.decode("latin-1").split("\r\n")
            headers = dict(line.lower().split(": ", 1) for line in lines if ": " in line)
            body = await reader.readexactly(int(headers.get("content-length", 0)))
            http_method, target, _version = first.split(" ")
            form: dict = {}
            content_type = headers.get("content-type", "")
            if content_type.startswith("application/x-www-form-urlencoded"):
                form = {key: values[-1] for key, values in urllib.parse.parse_qs(body.decode()).items()}
            elif content_type.startswith("application/json"):
                form = json.loads(body or b"{}")
            status, payload, payload_type = self.respond(http_method, urllib.parse.urlsplit(target).path, form)
            writer.write(f"HTTP/1.1 {status} X\r\nContent-Type: {payload_type}\r\nContent-Length: {len(payload)}\r\n"
                         "Connection: close\r\n\r\n".encode() + payload)
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_exc):
        self._server.close()
        await self._server.wait_closed()


class _RefusingServer(_BotServer):
    """A reachable origin that refuses this token for good."""

    def respond(self, http_method, path, form):
        self.calls.append((path.rsplit("/", 1)[-1], form))
        return 401, b'{"ok": false, "error_code": 401, "description": "Unauthorized"}', "application/json"


async def _until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


def _recording_clients(monkeypatch, *, transport=None) -> list[str]:
    """Every HTTPX client the skill builds records its request URLs (and may get a transport)."""
    urls: list[str] = []
    real_client = httpx.AsyncClient

    async def record(request):
        urls.append(str(request.url))

    def factory(**kwargs):
        if transport is not None:
            kwargs["transport"] = transport
        return real_client(event_hooks={"request": [record]}, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return urls


def _host_inbox(monkeypatch, plugin) -> list[dict]:
    """The host's /chat/inject receiver: keeps each payload and the parked file bytes."""
    received: list[dict] = []

    async def host_post(_api, path, payload):
        if path == "/chat/inject":
            files = [Path(item["path"]).read_bytes() for item in payload.get("attachments") or []]
            received.append({**payload, "attachment_bytes": files})
        return 200, {}

    monkeypatch.setattr(plugin, "_host_post", host_post)
    return received


def _owner_message(text: str = "", **fields) -> dict:
    return {"message": {"message_id": next(_MESSAGE_IDS), "date": int(time.time()) + 5,
                        "chat": {"id": 42, "type": "private"}, "from": {"id": 42, "first_name": "Owner"},
                        **({"text": text} if text else {}), **fields}}


def _state(tmp_path: Path, **settings) -> tuple[Path, _Api]:
    state = tmp_path / "state"
    state.mkdir()
    (state / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", **settings}), encoding="utf-8")
    return state, _Api(state, {"TELEGRAM_BOT_TOKEN": _TOKEN})


async def _run_companion_owner_check(monkeypatch, state: Path) -> list[int]:
    """companion.run() up to its verified owner: the menu it builds installs and restores."""
    (state / "runtime_config.json").write_text(json.dumps({
        "schema": 2, "core_port": 8765, "owner_chat_id": 42,
        "button_text": "Ouroboros", "tunnel": "cloudflare_quick",
    }), encoding="utf-8")
    original_wait = companion._wait_for_owner
    reached: list[int] = []

    async def verified_owner(*args):
        owner, menu = await original_wait(*args)
        reached.append(owner)
        assert await menu.install("https://loopback.trycloudflare.com/")
        assert await menu.restore()
        raise companion.ShutdownRequested("Loopback Bot API qualification complete.")

    monkeypatch.setenv("OUROBOROS_SKILL_STATE_DIR", str(state))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", _TOKEN)
    monkeypatch.setattr(companion, "require_isolated_process_group", lambda: None)
    monkeypatch.setattr(companion, "start_parent_lifeline", lambda *_args: (None, None))
    monkeypatch.setattr(companion, "_wait_for_owner", verified_owner)
    assert await companion.run() == 0
    return reached


async def _every_send(plugin, notifier, api, state: Path) -> None:
    """Bridge handlers, the typing refresher, the notifier and the standalone menu manager."""
    for factory, event in _bridge_events():
        await getattr(plugin, factory)(api)(event)
    await plugin._typing_cue(api)._send(42)
    assert await notifier._push_notification(api, 42, "done", trust_env=False) == ("sent", None)
    client = plugin._telegram_client(api)
    await client.send_audio(42, b"audio", "sample.mp3")
    await client.answer_callback_query("cb-1")
    manager = telegram_menu.TelegramMenuManager(_TOKEN, 42, state)
    assert await manager.verify_private_owner() == "@loopback_bot"
    assert await manager.install("https://loopback.trycloudflare.com/")
    assert await manager.restore()


_SEND_METHODS = {"sendMessage", "sendChatAction", "sendPhoto", "sendVideo", "sendDocument", "sendAudio",
                 "editMessageText", "answerCallbackQuery", "getChat", "getMe", "getChatMenuButton",
                 "setChatMenuButton"}


def test_without_a_saved_base_every_call_stays_on_the_cloud_origin(tmp_path, monkeypatch):
    plugin, _telegram_api, notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    store_bytes = b"cloud file bytes"
    bot = _BotServer(files={"doc-1": ("documents/file_0.txt", store_bytes)})

    async def handler(request: httpx.Request) -> httpx.Response:
        form = {key: values[-1] for key, values in urllib.parse.parse_qs(request.content.decode(errors="ignore")).items()}
        if request.headers.get("content-type", "").startswith("application/json"):
            form = json.loads(request.content or b"{}")
        status, payload, payload_type = bot.respond(request.method, request.url.path, form)
        return httpx.Response(status, content=payload, headers={"content-type": payload_type})

    urls = _recording_clients(monkeypatch, transport=httpx.MockTransport(handler))
    state, api = _state(tmp_path)

    async def scenario():
        client, server, *_rest = await plugin._start_poller(api)
        assert server == (_CLOUD, None)
        assert await client.get_updates(0) == []
        assert await client.download_file("doc-1") == store_bytes
        await _every_send(plugin, notifier, api, state)

    asyncio.run(scenario())
    assert {method for method, _form in bot.calls} >= _SEND_METHODS | {"setMyCommands", "getUpdates", "getFile", "file"}
    assert urls and all(url.startswith((f"{_CLOUD}/bot{_TOKEN}/", f"{_CLOUD}/file/bot{_TOKEN}/")) for url in urls)
    assert not [row for row in api.logs if row[0] in ("error", "warning")]


@pytest.mark.parametrize("local_mode", [True, False])
def test_every_consumer_reaches_a_saved_server_and_reads_its_store(tmp_path, monkeypatch, local_mode):
    """`--local` servers answer getFile with an absolute store path and no HTTP file;
    ordinary ones serve relative paths over HTTP. Both reach the owner's message, and
    both keep the download cap."""
    plugin, telegram_api, notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    store = tmp_path / "tba-store"
    (store / "documents").mkdir(parents=True)
    content = b"owner document bytes"
    (store / "documents" / "file_0.txt").write_bytes(content)
    file_path = str(store / "documents" / "file_0.txt") if local_mode else "documents/file_0.txt"
    urls = _recording_clients(monkeypatch)
    received = _host_inbox(monkeypatch, plugin)

    async def scenario():
        async with _BotServer(first_update_id=4242, files={"doc-1": (file_path, content)},
                              serve_files=not local_mode) as bot:
            state, api = _state(tmp_path, TELEGRAM_API_BASE=bot.origin, TELEGRAM_API_FILES_DIR=str(store))
            bot.push(_owner_message(caption="see file", document={
                "file_id": "doc-1", "file_name": "notes.txt", "mime_type": "text/plain", "file_size": len(content)}))
            poller = asyncio.create_task(plugin._poller(api))
            try:
                await _until(lambda: received and len(bot.methods("getUpdates")) >= 2)
            finally:
                poller.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poller
            await _every_send(plugin, notifier, api, state)
            reached = await _run_companion_owner_check(monkeypatch, state)
            monkeypatch.setattr(telegram_api, "_MAX_TELEGRAM_DOWNLOAD_BYTES", len(content) - 1)
            with pytest.raises(RuntimeError, match="10 MiB download limit"):
                await plugin._telegram_client(api).download_file("doc-1")
            return bot, api, state, reached

    bot, api, state, reached = asyncio.run(scenario())
    assert reached == [42]
    assert [item["attachment_bytes"] for item in received] == [[content]]
    assert received[0]["text"] == "see file"
    assert {method for method, _form in bot.calls} >= _SEND_METHODS | {"setMyCommands", "getUpdates", "getFile", "file"}
    assert urls and all(url.startswith(f"{bot.origin}/") for url in urls)
    assert bot.button == {"type": "commands"}
    assert json.loads((state / "poll_offset.json").read_text(encoding="utf-8"))["offset"] == 4243
    assert not [row for row in api.logs if row[0] in ("error", "warning")]


def test_a_save_moves_polling_sends_and_the_menu_to_the_new_server(tmp_path, monkeypatch):
    """A Save reaches every call at once and polling after its current long poll. The new
    server is read from its own oldest update, never with the old server's offset: each
    update is handled once and confirmed there, never replayed."""
    plugin, _telegram_api, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)

    async def scenario():
        async with _BotServer(first_update_id=900000) as old, _BotServer(first_update_id=7) as new:
            state, api = _state(tmp_path, TELEGRAM_API_BASE=old.origin)
            old.push(_owner_message("hello from the old server"))
            shared = httpx.AsyncClient(timeout=12, follow_redirects=False, trust_env=False)
            menu = telegram_menu.TelegramMenuManager(_TOKEN, 42, state, client=shared)
            poller = asyncio.create_task(plugin._poller(api))
            try:
                await _until(lambda: received and len(old.methods("getUpdates")) >= 2)
                assert await menu.current() == {"type": "commands"}
                assert len(old.methods("getChatMenuButton")) == 1
                new.push(_owner_message("/help"))
                reply = await plugin._make_settings_save(api)(
                    _RouteRequest({"TELEGRAM_API_BASE": new.origin + "/"}, host="127.0.0.1"))
                assert reply.status_code == 200
                assert f"Bot API server: {new.origin}." in json.loads(reply.body)["message"]
                await plugin._make_outbound(api)({"text": "sent after Save", "transport": {}})
                assert [form["text"] for form in new.methods("sendMessage")] == ["sent after Save"]
                assert await menu.current() == {"type": "commands"}
                assert len(new.methods("getChatMenuButton")) == 1
                await _until(lambda: len(new.methods("getUpdates")) >= 4)
                old_polls = len(old.methods("getUpdates"))
                await _until(lambda: len(new.methods("getUpdates")) >= 8)
                assert len(old.methods("getUpdates")) == old_polls
            finally:
                poller.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poller
                await shared.aclose()
            return old, new, api, state

    old, new, api, state = asyncio.run(scenario())
    assert [item["text"] for item in received] == ["hello from the old server"]
    help_replies = [form for form in new.methods("sendMessage") if form["text"] != "sent after Save"]
    assert len(help_replies) == 1
    assert [form["offset"] for form in new.methods("getUpdates")][:2] == ["0", "8"]
    assert new.methods("getMe") and new.methods("setMyCommands")
    assert not [form for form in old.methods("sendMessage")]
    assert json.loads((state / "poll_offset.json").read_text(encoding="utf-8"))["offset"] == 8
    assert ("info", "Telegram Bot API settings changed; polling follows the saved server.") in api.logs
    assert not [row for row in api.logs if row[0] in ("error", "warning")]


class _CompanionStatus:
    def __init__(self) -> None:
        self.transitions: list[tuple[str, str]] = []

    @property
    def state(self) -> str:
        return self.transitions[-1][0] if self.transitions else "starting"

    @property
    def reason_code(self) -> str:
        return self.transitions[-1][1] if self.transitions else ""

    def transition(self, state: str, _message: str, *, reason_code: str, **_kwargs) -> None:
        self.transitions.append((state, reason_code))


def test_the_companion_owner_check_reaches_a_server_saved_after_a_refusal(tmp_path, monkeypatch):
    plugin, _telegram_api, _notifier = _load()

    async def scenario():
        async with _RefusingServer() as refusing, _BotServer() as corrected:
            state, api = _state(tmp_path, TELEGRAM_API_BASE=refusing.origin)
            status = _CompanionStatus()

            async def tick(_delay, _stop, **_kwargs):
                if status.reason_code == "telegram_owner_rejected" and not corrected.calls:
                    reply = await plugin._make_settings_save(api)(
                        _RouteRequest({"TELEGRAM_API_BASE": corrected.origin}, host="127.0.0.1"))
                    assert reply.status_code == 200

            monkeypatch.setattr(companion, "_wait_or_stop", tick)
            async with httpx.AsyncClient(timeout=12, follow_redirects=False, trust_env=False) as shared:
                owner, menu = await companion._wait_for_owner(
                    companion.TelegramSettingsObserver(state, 8765), _TOKEN, state, shared, status, asyncio.Event())
                assert owner == 42 and await menu.install("https://loopback.trycloudflare.com/")
            return refusing, corrected, status

    refusing, corrected, status = asyncio.run(scenario())
    assert [method for method, _form in refusing.calls] == ["getMe"]
    assert [method for method, _form in corrected.calls][:2] == ["getMe", "getChat"]
    assert corrected.button["type"] == "web_app"
    assert status.transitions == [("blocked", "telegram_owner_rejected")]


class _Tunnel:
    returncode = None

    def __init__(self, *_args) -> None:
        pass

    async def start(self) -> None:
        return None

    async def wait_url(self, timeout_sec: float) -> str:
        return "https://loopback.trycloudflare.com/"

    async def wait(self) -> int:
        await asyncio.Event().wait()
        return 0

    async def stop(self) -> None:
        return None


@pytest.mark.parametrize("save_before_wait", [False, True])
def test_a_mini_app_blocked_by_a_refusing_server_resumes_on_a_saved_server(tmp_path, monkeypatch, save_before_wait):
    # The real menu sync and blocked wait: an invalid (unknown) or unchanged server keeps the
    # block, and the first changed server saved in Settings installs the button there.
    plugin, _telegram_api, _notifier = _load()

    async def scenario():
        async with _RefusingServer() as refusing, _BotServer() as corrected:
            state, api = _state(tmp_path, TELEGRAM_API_BASE=refusing.origin)
            settings = state / "settings.json"
            saved = settings.read_text(encoding="utf-8")
            status = _CompanionStatus()
            ticks: list[int] = []
            if save_before_wait:
                # A corrected Save can land before the old server's failure is handled.
                respond = refusing.respond
                def reject_after_save(*args):
                    merge_settings(state, {"TELEGRAM_API_BASE": corrected.origin})
                    return respond(*args)
                refusing.respond = reject_after_save

            async def tick(_delay, _stop, **_kwargs):
                ticks.append(len(refusing.calls))
                assert len(ticks) < 10, "still blocked after another server was saved"
                if save_before_wait:
                    return
                if len(ticks) in (1, 3):
                    settings.write_text(saved.replace(refusing.origin, "ftp://not-a-server"), encoding="utf-8")
                elif len(ticks) == 2:
                    settings.write_text(saved, encoding="utf-8")
                elif len(ticks) == 4:
                    settings.write_text(saved, encoding="utf-8")
                    reply = await plugin._make_settings_save(api)(
                        _RouteRequest({"TELEGRAM_API_BASE": corrected.origin}, host="127.0.0.1"))
                    assert reply.status_code == 200

            async def verified(*_args, **_kwargs):
                return None

            async def monitored(*_args, **_kwargs):
                raise companion.SidecarStoppedError("menu installed on the saved server")

            monkeypatch.setattr(companion, "_wait_or_stop", tick)
            monkeypatch.setattr(companion, "QuickTunnel", _Tunnel)
            monkeypatch.setattr(companion, "_wait_for_public_verification", verified)
            monkeypatch.setattr(companion, "_monitor_generation", monitored)
            sidecar = type("Sidecar", (), {"set_public_url": lambda self, _url: None,
                                           "clear_public_url": lambda self: None})()
            server_task = asyncio.create_task(asyncio.Event().wait())
            async with httpx.AsyncClient(timeout=12, follow_redirects=False, trust_env=False) as shared:
                context = companion._GenerationContext(
                    menu=telegram_menu.TelegramMenuManager(_TOKEN, 42, state, client=shared),
                    settings_observer=companion.TelegramSettingsObserver(state, 8765), owner=42,
                    button_text="Ouroboros", core_port=8765, status=status,
                    stop_event=asyncio.Event(), server_task=server_task)
                try:
                    with pytest.raises(companion.SidecarStoppedError, match="saved server"):
                        await companion._run_tunnel_generations(state / "cloudflared", state, 45678, sidecar, context)
                finally:
                    server_task.cancel()
                    await asyncio.gather(server_task, return_exceptions=True)
            return refusing, corrected, status, ticks

    refusing, corrected, status, ticks = asyncio.run(scenario())
    assert [method for method, _form in refusing.calls] == ["getChatMenuButton"]
    assert len(ticks) == (1 if save_before_wait else 4)  # unknown/unchanged readings keep it blocked
    assert corrected.button["type"] == "web_app"
    assert ("blocked", "telegram_rejected") in status.transitions
    assert status.transitions[-1] == ("ready", "healthy")


@pytest.mark.parametrize("reason", ["MENU_BLOCKED", "SETTINGS_BLOCKED"])
def test_another_saved_server_does_not_end_other_blocks(tmp_path, monkeypatch, reason):
    state, _api = _state(tmp_path, TELEGRAM_API_BASE="http://127.0.0.1:8081")
    settings = state / "settings.json"
    blocked = getattr(companion._GenerationResult, reason)
    ticks: list[int] = []

    class _DriftedMenu:
        async def check_snapshot_owned(self):
            raise companion.TelegramMenuConflictError("still changed outside the skill")

    async def tick(_delay, _stop, **_kwargs):
        ticks.append(1)
        values = json.loads(settings.read_text(encoding="utf-8"))
        values["TELEGRAM_API_BASE"] = f"http://127.0.0.1:{8081 + len(ticks)}"
        if reason == "SETTINGS_BLOCKED":
            values["TELEGRAM_CHAT_ID"] = "0"
        if len(ticks) == 3:
            values["TELEGRAM_MINIAPP_ENABLED"] = "off"
        settings.write_text(json.dumps(values), encoding="utf-8")

    monkeypatch.setattr(companion, "_wait_or_stop", tick)

    async def scenario():
        observer = companion.TelegramSettingsObserver(state, 8765)
        if reason == "SETTINGS_BLOCKED":
            observer.owner_chat_id = lambda: 42  # the owner stays; only exposure is unsafe
        return await companion._wait_blocked_state(blocked, _DriftedMenu(), observer, 42, _CompanionStatus(),
                                                   asyncio.Event(), None)

    assert asyncio.run(scenario()) is companion._GenerationResult.DISABLED
    assert len(ticks) == 3


@pytest.mark.parametrize("invalid", [False, True])
def test_unreadable_menu_settings_recover_through_the_companion_owner_check(tmp_path, monkeypatch, invalid):
    _load()
    read_base = telegram_menu.telegram_api_base
    reads = []

    def unavailable_once(state):
        reads.append(1)
        if len(reads) == 1:
            error = ValueError if invalid else telegram_menu.TelegramSettingsError
            raise error("unavailable test setting")
        return read_base(state)

    monkeypatch.setattr(telegram_menu, "telegram_api_base", unavailable_once)

    async def scenario():
        async with _BotServer() as server:
            state, _api = _state(tmp_path, TELEGRAM_API_BASE=server.origin)
            status = _CompanionStatus()

            async def tick(_delay, _stop, **_kwargs):
                assert status.state == "degraded"
                assert status.reason_code == "telegram_owner_unobservable"
                assert not server.calls

            monkeypatch.setattr(companion, "_wait_or_stop", tick)
            async with httpx.AsyncClient(timeout=12, follow_redirects=False, trust_env=False) as shared:
                owner, _menu = await companion._wait_for_owner(
                    companion.TelegramSettingsObserver(state, 8765), _TOKEN, state, shared, status, asyncio.Event())
            assert owner == 42
            assert [method for method, _form in server.calls] == ["getMe", "getChat"]
            assert status.transitions == [("degraded", "telegram_owner_unobservable")]

    asyncio.run(scenario())
    assert len(reads) == 3
