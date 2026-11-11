"""Share… in the Mini App (skills/telegram lib/telegram_share.py): the bot prepares exactly the words the owner's
Mini App posts as one Telegram message for Telegram's own chat picker, for the bound owner only.

Hermetic: a loopback Bot API server stands in for Telegram (TELEGRAM_API_BASE), the real sidecar app forwards the
owner's session to an in-process ASGI app hosting the real route and the bridge's own client, and nothing leaves
127.0.0.1. No Telegram client runs here, so the picker itself (shareMessage and its events) is not exercised."""
from __future__ import annotations

import asyncio
import contextlib
import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from tests.test_telegram_api_base_loopback import _BotServer, _state
from tests.test_telegram_miniapp_sidecar import BOT_TOKEN, OWNER_ID, PUBLIC_ORIGIN, PUBLIC_URL, _authenticate, sidecar_module
from tests.test_telegram_proxy import _Api, _TOKEN, _load

ROUTE = "/api/extensions/telegram/miniapp/prepare-share"
MARK = {"X-Ouroboros-Telegram-MiniApp": "1"}
WORDS = "Stay by the river.\n\n1. Praça do Comércio — 9:30\n\n```\n  npm test\n\n```\nשלום · 😀 https://example.org/a?b=1"


class _ShareBot(_BotServer):
    """A Bot API server that answers savePreparedInlineMessage with `answer` (HTTP `status`)."""

    def __init__(self, answer=None, status=200):
        super().__init__()
        self.answer = answer or {"ok": True, "result": {"id": "prepared-1", "expiration_date": 1_900_000_000}}
        self.status = status

    def respond(self, http_method, path, form):
        if path == f"/bot{_TOKEN}/savePreparedInlineMessage":
            self.calls.append(("savePreparedInlineMessage", form))
            return self.status, json.dumps(self.answer).encode(), "application/json"
        return super().respond(http_method, path, form)


class _RegisteringApi(_Api):
    """The PluginAPI stand-in that keeps what ``plugin.register`` registers and starts nothing."""

    def __init__(self, state_dir, granted):
        super().__init__(state_dir, granted)
        self.routes = {}

    def register_route(self, name, handler, methods):
        self.routes[name] = (tuple(methods), handler)

    def register_supervised_task(self, *_args, **_kwargs):
        pass

    def subscribe_event(self, *_args, **_kwargs):
        pass

    def register_settings_section(self, *_args, **_kwargs):
        pass


def _api(state_dir, token=_TOKEN):
    return _RegisteringApi(state_dir, {"TELEGRAM_BOT_TOKEN": token} if token else {})


def _route_app(plugin, api, factory=None):
    """The route as the skill registers it; `factory` swaps only the bridge client for one check."""
    if factory is None:
        plugin.register(api)
        methods, handler = api.routes["miniapp/prepare-share"]
        assert methods == ("POST",)
    else:
        handler = plugin.make_prepare_share(api, factory)
    return Starlette(routes=[Route(ROUTE, handler, methods=["POST"])])


async def _post(app, *, client=("127.0.0.1", 123), headers=MARK, **kwargs):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=client), base_url="http://127.0.0.1") as http:
        return await http.post(ROUTE, headers=headers, **kwargs)


@pytest.fixture
def plugin(monkeypatch):
    loaded, _api_module, _notifier = _load()
    monkeypatch.setattr(loaded, "_HONOR_ENV_PROXIES", False)
    monkeypatch.setattr(loaded, "register_miniapp", lambda _api: False)  # no companion process here
    return loaded


def test_the_owner_message_is_prepared_exactly_as_shown_through_the_configured_server(tmp_path, plugin):
    async def scenario():
        async with _ShareBot() as bot:
            state_dir, _plain = _state(tmp_path, TELEGRAM_API_BASE=bot.origin)
            reply = await _post(_route_app(plugin, _api(state_dir)), json={"text": WORDS})
            assert reply.status_code == 200, reply.text
            assert reply.json() == {"id": "prepared-1", "expiration_date": 1_900_000_000}
            [form] = bot.methods("savePreparedInlineMessage")
            assert form["user_id"] == "42", "the bound owner, never a user named by the page"
            assert (form["allow_user_chats"], form["allow_group_chats"], form["allow_channel_chats"], form["allow_bot_chats"]) \
                == ("true", "true", "true", "false")
            result = json.loads(form["result"])
            assert result["type"] == "article" and len(result["id"]) == 32
            assert result["title"] == "Stay by the river."
            assert result["input_message_content"] == {"message_text": WORDS, "link_preview_options": {"is_disabled": True}}
            assert [method for method, _form in bot.calls] == ["savePreparedInlineMessage"], "preparing sends nothing"
    asyncio.run(scenario())


def test_exactly_telegrams_bound_in_utf16_units_is_taken_and_one_more_is_not(tmp_path, plugin):
    async def scenario():
        async with _ShareBot() as bot:
            state_dir, _plain = _state(tmp_path, TELEGRAM_API_BASE=bot.origin)
            app = _route_app(plugin, _api(state_dir))
            assert (await _post(app, json={"text": "😀" * 2048})).status_code == 200  # 4096 UTF-16 units
            over = await _post(app, json={"text": "😀" * 2048 + "a"})
            assert over.status_code == 413 and over.json()["reason_code"] == "text_too_long"
            huge = await _post(app, content=json.dumps({"text": "a" * 70_000}).encode(), headers={**MARK, "Content-Type": "application/json"})
            assert huge.status_code == 413
            assert len(bot.methods("savePreparedInlineMessage")) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("body,code", [
    ({"text": WORDS, "user_id": 7}, "invalid_request"),
    ({"text": WORDS, "chat_id": 7}, "invalid_request"),
    ({"message": WORDS}, "invalid_request"),
    ({"text": 5}, "invalid_request"),
    ([WORDS], "invalid_request"),
    ({"text": " \n\t"}, "empty_text"),
])
def test_only_the_words_are_accepted(tmp_path, plugin, body, code):
    async def scenario():
        async with _ShareBot() as bot:
            state_dir, _plain = _state(tmp_path, TELEGRAM_API_BASE=bot.origin)
            app = _route_app(plugin, _api(state_dir))
            reply = await _post(app, json=body)
            assert reply.status_code == 400 and reply.json()["reason_code"] == code
            garbled = await _post(app, content=b"{not json", headers={**MARK, "Content-Type": "application/json"})
            assert garbled.status_code == 400 and garbled.json()["reason_code"] == "invalid_request"
            assert not bot.calls
    asyncio.run(scenario())


@pytest.mark.parametrize("headers,client", [
    ({}, ("127.0.0.1", 123)),                                           # the desktop page: no Mini App here
    ({"X-Ouroboros-Telegram-MiniApp": "spoofed"}, ("127.0.0.1", 123)),  # a mark the sidecar never writes
    (MARK, ("203.0.113.8", 4444)),                                      # a mark from another computer
])
def test_only_what_the_sidecar_forwards_reaches_telegram(tmp_path, plugin, headers, client):
    async def scenario():
        async with _ShareBot() as bot:
            state_dir, _plain = _state(tmp_path, TELEGRAM_API_BASE=bot.origin)
            reply = await _post(_route_app(plugin, _api(state_dir)), headers=headers, client=client, json={"text": WORDS})
            assert reply.status_code == 403 and reply.json()["reason_code"] == "miniapp_only"
            assert not bot.calls
    asyncio.run(scenario())


def test_no_owner_no_token_a_changed_bot_and_telegrams_refusals_are_said_without_secrets(tmp_path, plugin):
    async def scenario():
        async with _ShareBot() as bot:
            state_dir, _plain = _state(tmp_path, TELEGRAM_API_BASE=bot.origin)
            api = _api(state_dir)
            (state_dir / "settings.json").write_text(json.dumps({"TELEGRAM_API_BASE": bot.origin}), encoding="utf-8")
            unbound = await _post(_route_app(plugin, api), json={"text": WORDS})
            assert unbound.status_code == 409 and unbound.json()["reason_code"] == "owner_not_bound"
            (state_dir / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42", "TELEGRAM_API_BASE": bot.origin}),
                                                     encoding="utf-8")
            tokenless = await _post(_route_app(plugin, _api(state_dir, token="")), json={"text": WORDS})
            assert tokenless.status_code == 503 and tokenless.json()["reason_code"] == "token_missing"

            def replaced_meanwhile(target):
                client = plugin._bridge_client(target)
                target.granted["TELEGRAM_BOT_TOKEN"] = "654321:ANOTHER-BOT"
                plugin.binding_for(target).read_now(target)
                return client
            changed = await _post(_route_app(plugin, api, replaced_meanwhile), json={"text": WORDS})
            assert changed.status_code == 409 and changed.json()["reason_code"] == "bot_changed"
            assert not bot.calls, "a client of the replaced token asks Telegram nothing"
        async with _ShareBot({"ok": False, "error_code": 400, "description": f"Bad Request: bot{_TOKEN} {WORDS}"}, 400) as bot:
            state_dir, _plain = _state(tmp_path / "refused", TELEGRAM_API_BASE=bot.origin)
            refused = await _post(_route_app(plugin, _api(state_dir)), json={"text": WORDS})
            assert refused.status_code == 502 and refused.json()["reason_code"] == "telegram_rejected"
            assert _TOKEN not in refused.text and "Bad Request" not in refused.text and "river" not in refused.text
        async with _ShareBot({"ok": True, "result": {"expiration_date": 1}}) as bot:
            state_dir, _plain = _state(tmp_path / "empty", TELEGRAM_API_BASE=bot.origin)
            empty = await _post(_route_app(plugin, _api(state_dir)), json={"text": WORDS})
            assert empty.status_code == 502 and empty.json()["reason_code"] == "invalid_answer"
    (tmp_path / "refused").mkdir()
    (tmp_path / "empty").mkdir()
    asyncio.run(scenario())


def test_the_skill_proxy_carries_the_request_and_its_failure_is_said(tmp_path, plugin):
    seen: list[bytes] = []

    async def refuse(reader, writer):
        seen.append(await reader.readline())
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()

    async def scenario():
        proxy = await asyncio.start_server(refuse, "127.0.0.1", 0)
        port = proxy.sockets[0].getsockname()[1]
        try:
            async with _ShareBot() as bot:
                state_dir, _plain = _state(tmp_path, TELEGRAM_API_BASE=bot.origin, TELEGRAM_PROXY=f"http://127.0.0.1:{port}")
                reply = await _post(_route_app(plugin, _api(state_dir)), json={"text": WORDS})
                assert reply.status_code == 504 and reply.json()["reason_code"] == "telegram_unreachable"
                assert _TOKEN not in reply.text
                assert not bot.calls, "with a proxy saved, the server is reached only through it"
        finally:
            proxy.close()
            await proxy.wait_closed()
        assert seen and b"savePreparedInlineMessage" in seen[0]
    asyncio.run(scenario())


def test_the_real_sidecar_forwards_only_the_owners_session_from_its_origin(tmp_path, plugin):
    async def scenario():
        now = 1_750_000_000.0
        async with _ShareBot() as bot:
            state_dir, _plain = _state(tmp_path, TELEGRAM_CHAT_ID=str(OWNER_ID), TELEGRAM_API_BASE=bot.origin)
            upstream = httpx.AsyncClient(transport=httpx.ASGITransport(app=_route_app(plugin, _api(state_dir))))
            gateway = sidecar_module.TelegramProxySidecar(
                BOT_TOKEN, OWNER_ID, 8765, seams=sidecar_module.SidecarSeams(http_client=upstream, clock=lambda: now))
            gateway.set_public_url(PUBLIC_URL)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url=PUBLIC_ORIGIN) as browser:
                anonymous = await browser.post(ROUTE, headers={"Origin": PUBLIC_ORIGIN}, json={"text": WORDS})
                assert anonymous.status_code == 401
                assert (await _authenticate(browser, auth_date=int(now))).status_code == 204
                elsewhere = await browser.post(ROUTE, headers={"Origin": "https://attacker.example"}, json={"text": WORDS})
                assert elsewhere.status_code == 403
                assert not bot.calls
                shared = await browser.post(ROUTE, headers={"Origin": PUBLIC_ORIGIN, "X-Ouroboros-Telegram-MiniApp": "spoofed"},
                                            json={"text": WORDS})
                assert shared.status_code == 200, shared.text
                assert shared.json() == {"id": "prepared-1", "expiration_date": 1_900_000_000}
            await upstream.aclose()
            [form] = bot.methods("savePreparedInlineMessage")
            assert form["user_id"] == str(OWNER_ID)
            assert json.loads(form["result"])["input_message_content"]["message_text"] == WORDS
    asyncio.run(scenario())
