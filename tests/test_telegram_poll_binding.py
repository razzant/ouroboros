"""The poller's binding to one Bot API server: its cursor, and its stop after a refusal.

The servers are the loopback fakes of test_telegram_api_base_loopback (or the same fake behind
an in-memory transport when two URLs must reach one backend, or the cloud must be modelled).
The poller, the Save handler and, for the refusal cases, the host's real supervised-task
runner are the shipped code. Nothing leaves the machine.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sys
import threading
import time
import types
import urllib.parse

import httpx
import pytest

from ouroboros import extension_plugin_api
from ouroboros.extension_registry_state import _StagedSupervisedTask
from tests.test_telegram_api_base_loopback import (
    _BotServer,
    _RefusingServer,
    _host_inbox,
    _owner_message,
    _recording_clients,
    _state,
    _until,
)
from tests.test_telegram_proxy import _PACKAGE, _TOKEN, _load
from tests.test_telegram_settings import _RouteRequest

_CLOUD = "https://api.telegram.org"


class _CloudServer(_BotServer):
    """The cloud's documented getUpdates: every update below the offset is confirmed and gone."""

    def _get_updates(self, offset: int) -> list[dict]:
        self.queue = [item for item in self.queue if item[0] >= offset]
        return [update for _id, update in self.queue]


class _ConflictingServer(_BotServer):
    """Another poller holds this token for the first few getUpdates, then stops."""

    def __init__(self, conflicts: int, **kwargs):
        super().__init__(**kwargs)
        self.conflicts = conflicts

    def respond(self, http_method, path, form):
        if path.endswith("/getUpdates") and self.conflicts:
            self.conflicts -= 1
            self.calls.append(("getUpdates", form))
            return 409, b'{"ok": false, "error_code": 409, "description": "Conflict"}', "application/json"
        return super().respond(http_method, path, form)


def _in_memory(monkeypatch, servers: dict[str, _BotServer]) -> list[str]:
    """Route each host to its fake server in memory; any other host is unreachable."""
    hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(f"{request.url.scheme}://{request.url.host}:{request.url.port or 443}")
        server = servers.get(request.url.host)
        if server is None:
            raise httpx.ConnectError("unreachable", request=request)
        form = {key: values[-1] for key, values in urllib.parse.parse_qs(request.content.decode(errors="ignore")).items()}
        if request.headers.get("content-type", "").startswith("application/json"):
            form = json.loads(request.content or b"{}")
        status, payload, payload_type = server.respond(request.method, request.url.path, form)
        return httpx.Response(status, content=payload, headers={"content-type": payload_type})

    _recording_clients(monkeypatch, transport=httpx.MockTransport(handler))
    return hosts


async def _poll(plugin, api, condition) -> None:
    poller = asyncio.create_task(plugin._poller(api))
    try:
        await _until(condition)
    finally:
        poller.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poller


def _cursor(state) -> dict:
    return json.loads((state / "poll_offset.json").read_text(encoding="utf-8"))


def _texts(received) -> list[str]:
    return [item["text"] for item in received]


def test_a_new_server_numbering_into_the_old_range_delivers_every_unseen_update_once(tmp_path, monkeypatch):
    plugin, _api_mod, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)

    async def scenario():
        async with _BotServer(first_update_id=2000000) as old, _BotServer(first_update_id=2000005) as new:
            state, api = _state(tmp_path, TELEGRAM_API_BASE=old.origin)
            for number in range(8):
                old.push(_owner_message(f"old {number}"))
            poller = asyncio.create_task(plugin._poller(api))
            try:
                await _until(lambda: len(received) == 8 and len(old.methods("getUpdates")) >= 2)
                for number in range(3):
                    new.push(_owner_message(f"new {number}"))
                reply = await plugin._make_settings_save(api)(
                    _RouteRequest({"TELEGRAM_API_BASE": new.origin}, host="127.0.0.1"))
                assert reply.status_code == 200
                await _until(lambda: len(received) == 11 and len(new.methods("getUpdates")) >= 2)
            finally:
                poller.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poller
            return old, new, state

    old, new, state = asyncio.run(scenario())
    assert _texts(received) == [f"old {n}" for n in range(8)] + [f"new {n}" for n in range(3)]
    assert old.methods("getUpdates")[-1]["offset"] == "2000008"
    assert new.methods("getUpdates")[0]["offset"] == "0"
    assert _cursor(state)["offset"] == 2000008 and _cursor(state)["base"] == new.origin


def test_a_return_to_the_cloud_reads_its_lower_numbers_without_contacting_the_old_server(tmp_path, monkeypatch):
    plugin, _api_mod, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)
    cloud = _CloudServer(first_update_id=8)
    hosts = _in_memory(monkeypatch, {"api.telegram.org": cloud})
    state, api = _state(tmp_path)
    old = "http://127.0.0.1:8081"
    (state / "poll_offset.json").write_text(json.dumps({"offset": 900001, "base": old}), encoding="utf-8")
    cloud.push(_owner_message("back on the cloud"))

    asyncio.run(_poll(plugin, api, lambda: received and len(cloud.methods("getUpdates")) >= 2))

    assert _texts(received) == ["back on the cloud"]
    assert [form["offset"] for form in cloud.methods("getUpdates")][:2] == ["0", "9"]
    assert all(host == "https://api.telegram.org:443" for host in hosts)
    assert _cursor(state)["base"] == _CLOUD and _cursor(state)["offset"] == 9


def test_a_second_spelling_of_the_same_server_does_not_handle_the_last_batch_again(tmp_path, monkeypatch):
    plugin, _api_mod, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)
    backend = _BotServer(first_update_id=500)
    _in_memory(monkeypatch, {"127.0.0.1": backend, "localhost": backend})
    handled = _owner_message("/help")
    backend.push(handled)  # handled through 127.0.0.1, never confirmed there
    backend.push(_owner_message("fresh"))
    state, api = _state(tmp_path, TELEGRAM_API_BASE="http://localhost:8081")
    (state / "poll_offset.json").write_text(json.dumps({
        "offset": 501, "base": "http://127.0.0.1:8081",
        "handled": {"http://127.0.0.1:8081": [f"m:42:{handled['message']['message_id']}"]},
    }), encoding="utf-8")

    asyncio.run(_poll(plugin, api, lambda: received and len(backend.methods("getUpdates")) >= 2))

    assert _texts(received) == ["fresh"]
    assert not backend.methods("sendMessage")  # no second /help reply
    assert [form["offset"] for form in backend.methods("getUpdates")][:2] == ["0", "502"]


def test_an_unchanged_cloud_restart_keeps_a_legacy_offset(tmp_path, monkeypatch):
    plugin, _api_mod, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)
    cloud = _CloudServer(first_update_id=4243)
    _in_memory(monkeypatch, {"api.telegram.org": cloud})
    state, api = _state(tmp_path)
    (state / "poll_offset.json").write_text(json.dumps({"offset": 4243}), encoding="utf-8")
    cloud.push(_owner_message("still here"))

    asyncio.run(_poll(plugin, api, lambda: received and len(cloud.methods("getUpdates")) >= 2))

    assert _texts(received) == ["still here"]
    assert cloud.methods("getUpdates")[0]["offset"] == "4243"
    assert _cursor(state)["offset"] == 4244 and _cursor(state)["base"] == _CLOUD
    assert _cursor(state)["bot"] == "123456"


class _AnyTokenCloud(_CloudServer):
    """The cloud fake answering every token of the bot as its own (the queue is the bot's)."""

    def respond(self, http_method, path, form):
        path = re.sub(r"^/(file/)?bot[^/]+/", lambda match: f"/{match.group(1) or ''}bot{_TOKEN}/", path)
        return super().respond(http_method, path, form)


def test_another_bot_on_the_same_server_reads_its_own_queue_from_the_start(tmp_path, monkeypatch):
    # The cursor was the bot 999999's; the token is now the bot 123456's, whose IDs are lower
    # and whose owner chat shares the chat ID, so its message IDs can equal the old bot's.
    plugin, _api_mod, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)
    cloud = _AnyTokenCloud(first_update_id=10)
    _in_memory(monkeypatch, {"api.telegram.org": cloud})
    state, api = _state(tmp_path)
    first = _owner_message("first for the new bot")
    cloud.push(first)
    cloud.push(_owner_message("second for the new bot"))
    (state / "poll_offset.json").write_text(json.dumps({
        "offset": 5000, "base": _CLOUD, "bot": "999999",
        "handled": {_CLOUD: [f"m:42:{first['message']['message_id']}"]},
    }), encoding="utf-8")

    asyncio.run(_poll(plugin, api, lambda: len(received) == 2 and len(cloud.methods("getUpdates")) >= 2))

    assert _texts(received) == ["first for the new bot", "second for the new bot"]
    assert [form["offset"] for form in cloud.methods("getUpdates")][:2] == ["0", "12"]
    assert _cursor(state)["bot"] == "123456" and _cursor(state)["offset"] == 12
    assert list(_cursor(state)["handled"]) == [_CLOUD]


def test_a_rotated_token_of_the_same_bot_keeps_its_position_and_no_secret_is_saved(tmp_path, monkeypatch):
    plugin, _api_mod, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)
    cloud = _AnyTokenCloud(first_update_id=4243)
    _in_memory(monkeypatch, {"api.telegram.org": cloud})
    state, api = _state(tmp_path)
    api.granted["TELEGRAM_BOT_TOKEN"] = "123456:ROTATED-SECRET"
    (state / "poll_offset.json").write_text(json.dumps({"offset": 4243, "base": _CLOUD, "bot": "123456"}),
                                            encoding="utf-8")
    cloud.push(_owner_message("after rotation"))

    asyncio.run(_poll(plugin, api, lambda: received and len(cloud.methods("getUpdates")) >= 2))

    assert _texts(received) == ["after rotation"]
    assert cloud.methods("getUpdates")[0]["offset"] == "4243"
    saved = (state / "poll_offset.json").read_text(encoding="utf-8")
    assert _cursor(state)["bot"] == "123456" and "SECRET" not in saved


def test_a_server_that_lost_its_queue_is_read_past_one_poll_of_updates(tmp_path, monkeypatch):
    # The same base answers from an oldest update far below the saved offset (its queue was
    # reset). Each returned update is confirmed, so the next poll moves past this one's limit
    # instead of being answered the same head again.
    plugin, _api_mod, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    received = _host_inbox(monkeypatch, plugin)

    async def scenario():
        async with _BotServer(first_update_id=1) as server:
            state, api = _state(tmp_path, TELEGRAM_API_BASE=server.origin, TELEGRAM_MAX_UPDATES_PER_POLL="2")
            (state / "poll_offset.json").write_text(
                json.dumps({"offset": 900001, "base": server.origin, "bot": "123456"}), encoding="utf-8")
            for number in range(5):
                server.push(_owner_message(f"message {number}"))
            await _poll(plugin, api, lambda: len(received) == 5 and len(server.methods("getUpdates")) >= 4)
            return server, state

    server, state = asyncio.run(scenario())
    assert _texts(received) == [f"message {number}" for number in range(5)]
    assert [form["offset"] for form in server.methods("getUpdates")][:4] == ["900001", "3", "5", "6"]
    assert _cursor(state)["offset"] == 6


@contextlib.contextmanager
def _host_loop(monkeypatch):
    """The server's shared event loop on its own thread, as the host runs extension tasks."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(extension_plugin_api, "is_server_process", lambda: True)
    monkeypatch.setattr(extension_plugin_api, "get_global_event_bus", lambda: types.SimpleNamespace(_loop=loop))
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def _supervise(plugin, api):
    """The host's real supervised-task runner, with the poller's restart budget and a fast backoff."""
    spec = _StagedSupervisedTask(name="poller", factory=plugin._make_poller(api), restart_policy="on_failure",
                                 max_restarts=10, backoff_seconds=0.01)
    host = types.SimpleNamespace(_skill="telegram", _disclose_model_capable_dispatch=lambda *_args: None)
    return extension_plugin_api.PluginAPIImpl._start_supervised_task(host, spec)


def _wait(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def _save(plugin, api, base: str) -> None:
    reply = asyncio.run(plugin._make_settings_save(api)(_RouteRequest({"TELEGRAM_API_BASE": base}, host="127.0.0.1")))
    assert reply.status_code == 200


def _status(state) -> dict:
    path = state / "bridge_status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


_UNREADABLE = "Telegram settings could not be read while polling waits; still waiting."


def _refusal_then_correction(tmp_path, monkeypatch, *, refused_at_startup: bool, read_failure: bool = False):
    plugin, _api_mod, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    monkeypatch.setattr(sys.modules[f"{_PACKAGE}.lib.telegram_poll_binding"], "_PARK_CHECK_SEC", 0.02)
    received = _host_inbox(monkeypatch, plugin)
    with _host_loop(monkeypatch) as loop:
        servers = [_BotServer(first_update_id=10), _RefusingServer(), _BotServer(first_update_id=70)]
        for server in servers:
            asyncio.run_coroutine_threadsafe(server.__aenter__(), loop).result(5)
        first, refusing, corrected = servers
        state, api = _state(tmp_path, TELEGRAM_API_BASE=(refusing if refused_at_startup else first).origin)
        future = None
        try:
            if not refused_at_startup:
                first.push(_owner_message("before"))
            future = _supervise(plugin, api)
            if not refused_at_startup:
                _wait(lambda: _texts(received) == ["before"])
                _save(plugin, api, refusing.origin)
            _wait(lambda: refusing.calls and _status(state).get("state") == "error")
            time.sleep(0.5)  # ten restarts with this backoff take well under this
            assert len(refusing.calls) == 1 and not future.done()
            if read_failure:
                settings = state / "settings.json"
                saved = settings.read_text(encoding="utf-8")
                settings.write_text("{", encoding="utf-8")
                _wait(lambda: ("warning", _UNREADABLE) in api.logs)
                time.sleep(0.1)
                settings.write_text(saved, encoding="utf-8")  # readable again and unchanged
                time.sleep(0.1)
                assert len(refusing.calls) == 1 and not future.done()
                assert api.logs.count(("warning", _UNREADABLE)) == 1
            corrected.push(_owner_message("after"))
            _save(plugin, api, corrected.origin)
            _wait(lambda: "after" in _texts(received) and len(corrected.methods("getUpdates")) >= 2)
        finally:
            if future is not None:
                future.cancel()
            for server in servers:
                asyncio.run_coroutine_threadsafe(server.__aexit__(None, None, None), loop).result(5)
    assert [method for method, _form in refusing.calls] == ["getMe"]
    assert _status(state) == {"state": "ready", "reason_code": ""}
    assert ("info", "Telegram settings changed after a rejection; polling resumes.") in api.logs
    return api


def test_a_refused_new_server_parks_polling_until_a_corrected_save(tmp_path, monkeypatch):
    api = _refusal_then_correction(tmp_path, monkeypatch, refused_at_startup=False)
    assert any(level == "error" and message.startswith("Telegram token validation failed")
               for level, message in api.logs)


def test_a_refusal_at_startup_parks_polling_until_a_corrected_save(tmp_path, monkeypatch):
    _refusal_then_correction(tmp_path, monkeypatch, refused_at_startup=True)


def test_a_settings_read_failure_while_parked_asks_nothing_and_resumes_nothing(tmp_path, monkeypatch):
    _refusal_then_correction(tmp_path, monkeypatch, refused_at_startup=False, read_failure=True)


def _parked(tmp_path, monkeypatch):
    _load()
    binding = sys.modules[f"{_PACKAGE}.lib.telegram_poll_binding"]
    monkeypatch.setattr(binding, "_PARK_CHECK_SEC", 0.01)
    state, api = _state(tmp_path, TELEGRAM_API_BASE="http://127.0.0.1:8081")
    return binding, state / "settings.json", api


def test_a_failed_read_is_no_change_and_only_a_changed_valid_read_ends_the_wait(tmp_path, monkeypatch):
    binding, settings, api = _parked(tmp_path, monkeypatch)
    fingerprint = binding.server_fingerprint(api)
    assert fingerprint is not None
    saved = settings.read_text(encoding="utf-8")

    async def scenario():
        parked = asyncio.create_task(binding.park_until_settings_change(api, fingerprint))
        for content in ("{", saved, "{", saved):  # failing and unchanged reads, in turn
            settings.write_text(content, encoding="utf-8")
            await asyncio.sleep(0.05)
            assert not parked.done()
        assert api.logs.count(("warning", _UNREADABLE)) == 1
        settings.write_text(saved.replace("8081", "8082"), encoding="utf-8")
        await asyncio.wait_for(parked, 5)

    asyncio.run(scenario())
    assert ("info", "Telegram settings changed after a rejection; polling resumes.") in api.logs


def test_a_wait_begun_without_a_reading_ends_only_on_a_successful_one(tmp_path, monkeypatch):
    binding, settings, api = _parked(tmp_path, monkeypatch)
    saved = settings.read_text(encoding="utf-8")
    settings.write_text("{", encoding="utf-8")
    assert binding.server_fingerprint(api) is None

    async def scenario():
        parked = asyncio.create_task(binding.park_until_settings_change(api, None))
        await asyncio.sleep(0.05)
        assert not parked.done() and api.logs.count(("warning", _UNREADABLE)) == 1
        settings.write_text(saved, encoding="utf-8")
        await asyncio.wait_for(parked, 5)

    asyncio.run(scenario())


@pytest.mark.parametrize("unavailable", [{}, {"TELEGRAM_BOT_TOKEN": ""}, {"TELEGRAM_BOT_TOKEN": None}])
def test_an_unavailable_token_is_no_reading_and_only_a_changed_token_ends_the_wait(tmp_path, monkeypatch, unavailable):
    # A closing host answers {} for the token: that says nothing about the saved token.
    binding, _settings, api = _parked(tmp_path, monkeypatch)
    fingerprint = binding.server_fingerprint(api)
    granted = dict(api.granted)

    async def scenario():
        parked = asyncio.create_task(binding.park_until_settings_change(api, fingerprint))
        api.granted = dict(unavailable)
        assert binding.server_fingerprint(api) is None
        await asyncio.sleep(0.05)
        assert not parked.done()
        api.granted = granted  # available again and unchanged
        await asyncio.sleep(0.05)
        assert not parked.done()
        api.granted = {"TELEGRAM_BOT_TOKEN": "123456:CORRECTED"}
        await asyncio.wait_for(parked, 5)

    asyncio.run(scenario())
    assert api.logs.count(("warning", _UNREADABLE)) == 1


@pytest.mark.parametrize("readable", [True, False])
def test_a_parked_wait_stays_cancellable(tmp_path, monkeypatch, readable):
    binding, settings, api = _parked(tmp_path, monkeypatch)
    fingerprint = binding.server_fingerprint(api)
    if not readable:
        settings.write_text("{", encoding="utf-8")

    async def scenario():
        parked = asyncio.create_task(binding.park_until_settings_change(api, fingerprint))
        await asyncio.sleep(0.05)
        parked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parked

    asyncio.run(scenario())
    assert not any("resumes" in message for _level, message in api.logs)


@pytest.mark.parametrize("conflicts", [1, 15])  # 15 outlasts the supervisor's ten restarts
def test_a_competing_poll_is_retried_without_a_save_or_the_restart_budget(tmp_path, monkeypatch, conflicts):
    plugin, _api_mod, _notifier = _load()
    monkeypatch.setattr(plugin, "_HONOR_ENV_PROXIES", False)
    monkeypatch.setattr(plugin, "TELEGRAM_RETRY_INITIAL_SEC", 0.01)
    monkeypatch.setattr(plugin, "next_telegram_retry_delay", lambda delay: delay)
    received = _host_inbox(monkeypatch, plugin)
    written: list[tuple[str, str]] = []
    save_status = plugin._save_bridge_status
    monkeypatch.setattr(plugin, "_save_bridge_status",
                        lambda api, state, reason_code="": (written.append((state, reason_code)),
                                                            save_status(api, state, reason_code))[-1])
    server = _ConflictingServer(conflicts, first_update_id=30)
    with _host_loop(monkeypatch) as loop:
        asyncio.run_coroutine_threadsafe(server.__aenter__(), loop).result(5)
        state, api = _state(tmp_path, TELEGRAM_API_BASE=server.origin)
        server.push(_owner_message("after the conflict"))
        future = None
        try:
            future = _supervise(plugin, api)
            _wait(lambda: _texts(received) == ["after the conflict"]
                  and len(server.methods("getUpdates")) >= conflicts + 2)
            assert not future.done()
        finally:
            if future is not None:
                future.cancel()
            asyncio.run_coroutine_threadsafe(server.__aexit__(None, None, None), loop).result(5)
    assert len(server.methods("getMe")) == 1  # one run of the poller: nothing restarted it
    # Degraded while the token is held, ready after the first successful poll: one write each.
    assert written == [("ready", ""), ("degraded", "telegram_conflict"), ("ready", "")]
    assert _status(state) == {"state": "ready", "reason_code": ""}
    warnings = [message for level, message in api.logs if level == "warning" and "degraded" in message]
    assert warnings == ["Telegram poller degraded (TelegramRequestRejected): Telegram API getUpdates returned HTTP 409."]
    assert ("info", "Telegram poller recovered after TelegramRequestRejected; polling resumed.") in api.logs
    assert not any(level == "error" for level, _message in api.logs)
