from __future__ import annotations

import asyncio
import hashlib
import hmac
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import httpx
import pytest

SIDECAR_PATH = Path(__file__).parents[1] / "skills" / "telegram" / "scripts" / "sidecar.py"
SPEC = importlib.util.spec_from_file_location("telegram_miniapp_poc_sidecar", SIDECAR_PATH)
assert SPEC and SPEC.loader
sidecar_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sidecar_module
SPEC.loader.exec_module(sidecar_module)

BOT_TOKEN = "example"
OWNER_ID = 12345
PUBLIC_URL = "https://quiet-river-42.trycloudflare.com/"
PUBLIC_ORIGIN = PUBLIC_URL.rstrip("/")


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def signed_init_data(
    *,
    user_id: int = OWNER_ID,
    auth_date: int,
    bot_token: str = BOT_TOKEN,
    is_bot: bool = False,
    extra: dict[str, str] | None = None,
) -> str:
    user: dict[str, Any] = {"id": user_id, "first_name": "Owner"}
    if is_bot:
        user["is_bot"] = True
    values = {
        "auth_date": str(auth_date),
        "query_id": "AAE-test-query",
        "user": json.dumps(user, separators=(",", ":"), ensure_ascii=False),
    }
    if extra:
        values.update(extra)
    check_string = "\n".join(f"{key}={value}" for key, value in sorted(values.items()))
    secret = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    values["hash"] = hmac.new(secret, check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    return urlencode(values)


def test_init_data_hmac_freshness_owner_and_bot_binding() -> None:
    now = 1_750_000_000
    valid = signed_init_data(auth_date=now, extra={"signature": "telegram-signature"})
    assert sidecar_module.validate_telegram_init_data(
        valid,
        bot_token=BOT_TOKEN,
        owner_user_id=OWNER_ID,
        now=now,
    ) == OWNER_ID

    invalid_values = [
        valid + "&start_param=tampered",
        valid + "&user=%7B%22id%22%3A12345%7D",
        signed_init_data(user_id=OWNER_ID + 1, auth_date=now),
        signed_init_data(user_id=OWNER_ID, auth_date=now, is_bot=True),
        signed_init_data(user_id=OWNER_ID, auth_date=now - 301),
        signed_init_data(user_id=OWNER_ID, auth_date=now + 31),
    ]
    for value in invalid_values:
        with pytest.raises(sidecar_module.TelegramInitDataError):
            sidecar_module.validate_telegram_init_data(
                value,
                bot_token=BOT_TOKEN,
                owner_user_id=OWNER_ID,
                now=now,
            )


def test_public_url_is_exactly_a_single_label_quick_tunnel() -> None:
    gateway = sidecar_module.TelegramProxySidecar(BOT_TOKEN, OWNER_ID, 8765)
    assert gateway.set_public_url("https://Quiet-River-42.TryCloudflare.com") == PUBLIC_URL
    assert gateway.public_host == "quiet-river-42.trycloudflare.com"
    assert gateway.app is gateway

    for value in (
        "http://quiet-river-42.trycloudflare.com/",
        "https://trycloudflare.com/",
        "https://nested.quiet-river-42.trycloudflare.com/",
        "https://quiet-river-42.trycloudflare.com.attacker.test/",
        "https://quiet-river-42.trycloudflare.com/path",
        "https://quiet-river-42.trycloudflare.com/?query=1",
        "https://localhost.run/",
        "https://user@quiet-river-42.trycloudflare.com/",
    ):
        with pytest.raises(ValueError):
            gateway.set_public_url(value)


async def _authenticate(
    browser: httpx.AsyncClient,
    *,
    auth_date: int,
    origin: str = PUBLIC_ORIGIN,
) -> httpx.Response:
    return await browser.post(
        sidecar_module.AUTH_PATH,
        headers={"Origin": origin},
        json={"init_data": signed_init_data(auth_date=auth_date)},
    )


def test_bootstrap_default_deny_auth_cookie_and_raw_http_proxy() -> None:
    async def scenario() -> None:
        now = [1_750_000_000.0]
        captured: list[dict[str, Any]] = []

        async def upstream_handler(request: httpx.Request) -> httpx.Response:
            captured.append(
                {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "body": await request.aread(),
                }
            )
            return httpx.Response(
                201,
                headers=[
                    ("Content-Type", "application/octet-stream"),
                    ("Cache-Control", "public, max-age=3600"),
                    ("Expires", "Thu, 01 Jan 2099 00:00:00 GMT"),
                    ("Set-Cookie", "core-cookie=must-not-escape; Path=/"),
                    ("Access-Control-Allow-Origin", "*"),
                    ("Connection", "x-hop"),
                    ("X-Hop", "must-not-escape"),
                    ("Location", "http://127.0.0.1:8765/next"),
                    ("X-Upstream", "preserved"),
                ],
                stream=_ChunkStream(b"raw-upstream-", b"body"),
                request=request,
            )

        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler))
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN,
            OWNER_ID,
            8765,
            seams=sidecar_module.SidecarSeams(
                http_client=upstream_client,
                clock=lambda: now[0],
            ),
        )
        gateway.set_public_url(PUBLIC_URL)

        transport = httpx.ASGITransport(app=gateway)
        async with httpx.AsyncClient(transport=transport, base_url=PUBLIC_ORIGIN) as browser:
            root = await browser.get("/")
            assert root.status_code == 200
            assert root.headers[sidecar_module.GATEWAY_MARKER_HEADER] == "1"
            assert "telegram-web-app.js" in root.text
            assert "frame-ancestors" not in root.headers["content-security-policy"]
            assert "x-frame-options" not in root.headers

            bootstrap_js = await browser.get(sidecar_module.BOOTSTRAP_JS_PATH)
            assert bootstrap_js.status_code == 200
            assert "initData" in bootstrap_js.text

            protected = [
                await browser.get("/static/app.js"),
                await browser.get("/api/settings"),
                await browser.get("/api/files/download?path=x"),
                await browser.post("/api/chat/upload", headers={"Origin": PUBLIC_ORIGIN}, content=b"x"),
                await browser.get("/ws"),
                await browser.get(sidecar_module.AUTH_PATH),
            ]
            assert [response.status_code for response in protected] == [401] * len(protected)
            assert not captured

            auth = await _authenticate(browser, auth_date=int(now[0]))
            assert auth.status_code == 204
            set_cookie = auth.headers["set-cookie"]
            lowered_cookie = set_cookie.lower()
            assert sidecar_module.SESSION_COOKIE in set_cookie
            assert "secure" in lowered_cookie
            assert "httponly" in lowered_cookie
            assert "samesite=strict" in lowered_cookie
            assert "path=/" in lowered_cookie

            session_check = await browser.get(sidecar_module.SESSION_PATH)
            assert session_check.status_code == 204

            raw_token = browser.cookies.get(sidecar_module.SESSION_COOKIE)
            assert raw_token
            stored_keys = list(gateway._sessions)
            assert len(stored_keys) == 1
            assert isinstance(stored_keys[0], bytes) and len(stored_keys[0]) == 32
            assert raw_token.encode("ascii") != stored_keys[0]

            proxied = await browser.post(
                "/api/chat/upload?project=a%2Fb",
                headers={
                    "Origin": PUBLIC_ORIGIN,
                    "Authorization": "Bearer example",
                    "X-Ouroboros-Password": "example",
                    "X-Forwarded-For": "203.0.113.8",
                    "CF-Connecting-IP": "203.0.113.8",
                    "Connection": "x-remove",
                    "X-Remove": "must-not-pass",
                    sidecar_module.MINIAPP_REQUEST_MARKER_HEADER: "spoofed",
                    "X-Widget": "preserved",
                    "Content-Type": "application/octet-stream",
                },
                content=b"streamed-upload",
            )
            assert proxied.status_code == 201
            assert proxied.content == b"raw-upstream-body"
            assert proxied.headers["location"] == PUBLIC_ORIGIN + "/next"
            assert proxied.headers["x-upstream"] == "preserved"
            assert proxied.headers["cache-control"] == "private, no-store"
            assert proxied.headers["pragma"] == "no-cache"
            assert proxied.headers["expires"] == "0"
            assert "set-cookie" not in proxied.headers
            assert "access-control-allow-origin" not in proxied.headers
            assert "x-hop" not in proxied.headers

            assert len(captured) == 1
            forwarded = captured[0]
            assert forwarded["method"] == "POST"
            assert forwarded["url"] == "http://127.0.0.1:8765/api/chat/upload?project=a%2Fb"
            assert forwarded["body"] == b"streamed-upload"
            assert forwarded["headers"]["host"] == "127.0.0.1:8765"
            assert forwarded["headers"]["x-widget"] == "preserved"
            assert forwarded["headers"][sidecar_module.MINIAPP_REQUEST_MARKER_HEADER.lower()] == "1"
            for blocked in (
                "authorization",
                "content-length",
                "cookie",
                "origin",
                "x-ouroboros-password",
                "x-forwarded-for",
                "cf-connecting-ip",
                "x-remove",
            ):
                assert blocked not in forwarded["headers"]

            method_count = len(captured)
            arbitrary = await browser.request("BREW", "/api/state", headers={"Origin": PUBLIC_ORIGIN})
            assert arbitrary.status_code == 405
            assert len(captured) == method_count

        await upstream_client.aclose()

    asyncio.run(scenario())


def test_auth_rate_limit_is_bounded_and_counted() -> None:
    async def scenario() -> None:
        now = 1_750_000_000
        rate_now = [100.0]
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN,
            OWNER_ID,
            8765,
            limits=sidecar_module.SidecarLimits(
                auth_global_limit=3,
                auth_client_limit=2,
                auth_rate_window_sec=10,
            ),
            seams=sidecar_module.SidecarSeams(
                clock=lambda: now,
                rate_clock=lambda: rate_now[0],
            ),
        )
        gateway.set_public_url(PUBLIC_URL)
        transport = httpx.ASGITransport(app=gateway)
        async with httpx.AsyncClient(transport=transport, base_url=PUBLIC_ORIGIN) as browser:
            headers = {"Origin": PUBLIC_ORIGIN, "CF-Connecting-IP": "203.0.113.8"}
            payload = {"init_data": signed_init_data(auth_date=now)}
            assert (await browser.post(sidecar_module.AUTH_PATH, headers=headers, json=payload)).status_code == 204
            assert (await browser.post(sidecar_module.AUTH_PATH, headers=headers, json=payload)).status_code == 204
            assert (await browser.post(sidecar_module.AUTH_PATH, headers=headers, json=payload)).status_code == 429
            diagnostics = gateway.diagnostics()
            assert diagnostics["auth_success"] == 2
            assert diagnostics["auth_rate_limited"] == 1
            assert set(diagnostics) == {
                "auth_success",
                "auth_rejected",
                "auth_rate_limited",
                "auth_busy",
                "active_sessions",
                "active_websockets",
            }
            rate_now[0] += 11
            assert (await browser.post(sidecar_module.AUTH_PATH, headers=headers, json=payload)).status_code == 204
        await gateway.aclose()

    asyncio.run(scenario())


def test_host_origin_auth_fail_closed_and_never_echoes_launch_data() -> None:
    async def scenario() -> None:
        now = 1_750_000_000
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN,
            OWNER_ID,
            8765,
            seams=sidecar_module.SidecarSeams(clock=lambda: now),
        )
        gateway.set_public_url(PUBLIC_URL)
        transport = httpx.ASGITransport(app=gateway)
        async with httpx.AsyncClient(transport=transport, base_url=PUBLIC_ORIGIN) as browser:
            wrong_host = await browser.get("/", headers={"Host": "attacker.example"})
            assert wrong_host.status_code == 421

            raw = signed_init_data(auth_date=now)
            wrong_origin = await browser.post(
                sidecar_module.AUTH_PATH,
                headers={"Origin": "https://attacker.example"},
                json={"init_data": raw},
            )
            assert wrong_origin.status_code == 403

            tampered = raw + "&start_param=secret-marker"
            rejected = await browser.post(
                sidecar_module.AUTH_PATH,
                headers={"Origin": PUBLIC_ORIGIN},
                json={"init_data": tampered},
            )
            assert rejected.status_code == 401
            assert "secret-marker" not in rejected.text
            assert BOT_TOKEN not in rejected.text

            auth = await _authenticate(browser, auth_date=now)
            assert auth.status_code == 204
            no_origin = await browser.post("/api/settings", json={})
            foreign_origin = await browser.post(
                "/api/settings",
                headers={"Origin": "https://attacker.example"},
                json={},
            )
            foreign_get = await browser.get(
                "/api/settings",
                headers={"Origin": "https://attacker.example"},
            )
            assert [no_origin.status_code, foreign_origin.status_code, foreign_get.status_code] == [403, 403, 403]

        await gateway.aclose()

    asyncio.run(scenario())


def test_fixed_session_expiry_and_public_url_rotation_invalidate_cookie() -> None:
    async def scenario() -> None:
        now = [1_750_000_000.0]

        async def upstream_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_ChunkStream(b"ok"), request=request)

        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler))
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN,
            OWNER_ID,
            8765,
            limits=sidecar_module.SidecarLimits(session_ttl_sec=2),
            seams=sidecar_module.SidecarSeams(
                http_client=upstream_client,
                clock=lambda: now[0],
            ),
        )
        gateway.set_public_url(PUBLIC_URL)
        transport = httpx.ASGITransport(app=gateway)
        async with httpx.AsyncClient(transport=transport, base_url=PUBLIC_ORIGIN) as browser:
            assert (await _authenticate(browser, auth_date=int(now[0]))).status_code == 204
            assert (await browser.get("/api/state")).status_code == 200
            gateway.set_public_url(PUBLIC_URL)
            assert (await browser.get("/api/state")).status_code == 200

            now[0] += 3
            assert (await browser.get("/api/state")).status_code == 401
            expired_root = await browser.get("/")
            assert expired_root.headers[sidecar_module.GATEWAY_MARKER_HEADER] == "1"

            now[0] += 1
            assert (await _authenticate(browser, auth_date=int(now[0]))).status_code == 204
            old_token = browser.cookies.get(sidecar_module.SESSION_COOKIE)
            gateway.set_public_url("https://fresh-tunnel.trycloudflare.com/")

        new_transport = httpx.ASGITransport(app=gateway)
        async with httpx.AsyncClient(
            transport=new_transport,
            base_url="https://fresh-tunnel.trycloudflare.com",
            headers={"Cookie": f"{sidecar_module.SESSION_COOKIE}={old_token}"},
        ) as new_browser:
            assert (await new_browser.get("/api/state")).status_code == 401
            assert (await new_browser.get("/")).status_code == 200

        await upstream_client.aclose()

    asyncio.run(scenario())


def test_session_expiry_uses_monotonic_time_across_wall_clock_rollback() -> None:
    async def scenario() -> None:
        wall = [1_750_000_000.0]
        monotonic = [100.0]

        async def upstream_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_ChunkStream(b"ok"), request=request)

        upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler))
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN,
            OWNER_ID,
            8765,
            limits=sidecar_module.SidecarLimits(session_ttl_sec=2),
            seams=sidecar_module.SidecarSeams(
                http_client=upstream_client,
                clock=lambda: wall[0],
                session_clock=lambda: monotonic[0],
            ),
        )
        gateway.set_public_url(PUBLIC_URL)
        transport = httpx.ASGITransport(app=gateway)
        async with httpx.AsyncClient(transport=transport, base_url=PUBLIC_ORIGIN) as browser:
            assert (await _authenticate(browser, auth_date=int(wall[0]))).status_code == 204
            assert (await browser.get("/api/state")).status_code == 200
            wall[0] -= 3600
            monotonic[0] += 3
            assert (await browser.get("/api/state")).status_code == 401
        await upstream_client.aclose()

    asyncio.run(scenario())


def test_owner_reset_revokes_existing_session_on_next_request() -> None:
    async def scenario() -> None:
        now = 1_750_000_000
        current_owner = [OWNER_ID]
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN,
            OWNER_ID,
            8765,
            seams=sidecar_module.SidecarSeams(
                clock=lambda: now,
                exposure_guard=lambda: current_owner[0] == OWNER_ID,
            ),
        )
        gateway.set_public_url(PUBLIC_URL)
        transport = httpx.ASGITransport(app=gateway)
        async with httpx.AsyncClient(transport=transport, base_url=PUBLIC_ORIGIN) as browser:
            assert (await _authenticate(browser, auth_date=now)).status_code == 204
            assert (await browser.get(sidecar_module.SESSION_PATH)).status_code == 204
            current_owner[0] = OWNER_ID + 1
            assert (await browser.get(sidecar_module.SESSION_PATH)).status_code == 503
            assert gateway.public_url is None
            assert gateway.diagnostics()["active_sessions"] == 0
        await gateway.aclose()

    asyncio.run(scenario())


class _FakeUpstreamWebSocket:
    def __init__(self, *, echo_after_client: bool = False) -> None:
        self.echo_after_client = echo_after_client
        self.sent: list[str | bytes] = []
        self.entered = False
        self.exited = False
        self._client_message = asyncio.Event()
        self._yielded = False
        self._never = asyncio.Event()

    async def __aenter__(self) -> "_FakeUpstreamWebSocket":
        self.entered = True
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.exited = True

    async def send(self, value: str | bytes) -> None:
        self.sent.append(value)
        self._client_message.set()

    def __aiter__(self) -> "_FakeUpstreamWebSocket":
        return self

    async def __anext__(self) -> str:
        if not self.echo_after_client:
            await self._never.wait()
            raise StopAsyncIteration
        if self._yielded:
            raise StopAsyncIteration
        await self._client_message.wait()
        self._yielded = True
        return "from-core"


class _FakeConnector:
    def __init__(self, upstream: _FakeUpstreamWebSocket) -> None:
        self.upstream = upstream
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> _FakeUpstreamWebSocket:
        self.calls.append((url, kwargs))
        return self.upstream


class _ProxyAwareConnector:
    def __init__(self, upstream: _FakeUpstreamWebSocket) -> None:
        self.upstream = upstream
        self.proxy_values: list[Any] = []

    def __call__(self, url: str, *, proxy: Any = "auto", **_kwargs: Any) -> _FakeUpstreamWebSocket:
        assert url.startswith("ws://127.0.0.1:8765/")
        self.proxy_values.append(proxy)
        return self.upstream


async def _invoke_websocket(
    gateway: Any,
    *,
    cookie: str | None,
    origin: str = PUBLIC_ORIGIN,
    path: str = "/ws",
    query: bytes = b"",
    client_messages: list[dict[str, Any]] | None = None,
    after_accept: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    await queue.put({"type": "websocket.connect"})
    for message in client_messages or []:
        await queue.put(message)
    sent: list[dict[str, Any]] = []
    headers = [(b"host", b"quiet-river-42.trycloudflare.com"), (b"origin", origin.encode("ascii"))]
    if cookie is not None:
        headers.append((b"cookie", f"{sidecar_module.SESSION_COOKIE}={cookie}".encode("ascii")))
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "wss",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query,
        "headers": headers,
        "client": ("203.0.113.8", 12345),
        "server": ("quiet-river-42.trycloudflare.com", 443),
        "subprotocols": [],
    }

    async def receive() -> dict[str, Any]:
        return await queue.get()

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if message.get("type") == "websocket.accept" and after_accept is not None:
            after_accept()

    await gateway(scope, receive, send)
    return sent


def test_websocket_requires_owner_origin_relays_and_is_bounded() -> None:
    async def scenario() -> None:
        upstream = _FakeUpstreamWebSocket(echo_after_client=True)
        connector = _FakeConnector(upstream)
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN,
            OWNER_ID,
            8765,
            limits=sidecar_module.SidecarLimits(
                session_ttl_sec=30,
                max_websockets=1,
            ),
            seams=sidecar_module.SidecarSeams(websocket_connect=connector),
        )
        gateway.set_public_url(PUBLIC_URL)
        token, _session = gateway._issue_session()

        missing = await _invoke_websocket(gateway, cookie=None)
        foreign = await _invoke_websocket(gateway, cookie=token, origin="https://attacker.example")
        unknown = await _invoke_websocket(gateway, cookie=token, path="/other")
        assert missing[-1] == {"type": "websocket.close", "code": 4401, "reason": ""}
        assert foreign[-1] == {"type": "websocket.close", "code": 4403, "reason": ""}
        assert unknown[-1] == {"type": "websocket.close", "code": 4404, "reason": ""}
        assert not connector.calls

        relayed = await _invoke_websocket(
            gateway,
            cookie=token,
            query=b"thread=1",
            client_messages=[{"type": "websocket.receive", "text": "from-client"}],
        )
        assert connector.calls[0][0] == "ws://127.0.0.1:8765/ws?thread=1"
        assert "proxy" not in connector.calls[0][1]
        assert upstream.sent == ["from-client"]
        assert {"type": "websocket.accept", "subprotocol": None, "headers": []} in relayed
        assert {"type": "websocket.send", "text": "from-core"} in relayed
        assert upstream.exited is True

        assert gateway._claim_websocket() is True
        assert gateway._claim_websocket() is False
        gateway._release_websocket()
        await gateway.aclose()

    asyncio.run(scenario())


def test_open_websocket_stops_forwarding_after_owner_revocation() -> None:
    async def scenario() -> None:
        owner_current = [True]
        upstream = _FakeUpstreamWebSocket()
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN,
            OWNER_ID,
            8765,
            seams=sidecar_module.SidecarSeams(
                websocket_connect=_FakeConnector(upstream),
                exposure_guard=lambda: owner_current[0],
            ),
        )
        gateway.set_public_url(PUBLIC_URL)
        token, _session = gateway._issue_session()
        messages = await _invoke_websocket(
            gateway,
            cookie=token,
            client_messages=[{"type": "websocket.receive", "text": "must-not-forward"}],
            after_accept=lambda: owner_current.__setitem__(0, False),
        )

        assert upstream.sent == []
        assert messages[-1]["type"] == "websocket.close"
        assert messages[-1]["code"] in {1000, 4401}
        assert gateway.public_url is None
        assert gateway.diagnostics()["active_sessions"] == 0
        await gateway.aclose()

    asyncio.run(scenario())


def test_websocket_closes_at_fixed_session_expiry() -> None:
    async def scenario() -> None:
        upstream = _FakeUpstreamWebSocket()
        connector = _FakeConnector(upstream)
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN,
            OWNER_ID,
            8765,
            limits=sidecar_module.SidecarLimits(session_ttl_sec=1),
            seams=sidecar_module.SidecarSeams(websocket_connect=connector),
        )
        gateway.set_public_url(PUBLIC_URL)
        token, _session = gateway._issue_session()
        started = time.monotonic()
        messages = await _invoke_websocket(gateway, cookie=token)
        elapsed = time.monotonic() - started
        assert elapsed < 2.5
        assert {"type": "websocket.accept", "subprotocol": None, "headers": []} in messages
        assert messages[-1] == {"type": "websocket.close", "code": 4401, "reason": ""}
        assert upstream.exited is True
        await gateway.aclose()

    asyncio.run(scenario())


def test_websocket_disables_proxy_when_connector_supports_it() -> None:
    async def scenario() -> None:
        upstream = _FakeUpstreamWebSocket(echo_after_client=True)
        connector = _ProxyAwareConnector(upstream)
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN,
            OWNER_ID,
            8765,
            seams=sidecar_module.SidecarSeams(websocket_connect=connector),
        )
        gateway.set_public_url(PUBLIC_URL)
        token, _session = gateway._issue_session()
        await _invoke_websocket(
            gateway,
            cookie=token,
            client_messages=[{"type": "websocket.receive", "text": "ping"}],
        )
        assert connector.proxy_values == [None]
        await gateway.aclose()

    asyncio.run(scenario())


def test_proxy_start_send_failure_closes_upstream_once() -> None:
    async def run():
        closed = []
        async def body():
            yield b"module"
        async def close():
            closed.append(True)
        async def send(_message):
            raise OSError("downstream disconnected before headers")
        response = sidecar_module._ProxyStreamResponse(200, [], body(), close)
        with pytest.raises(OSError):
            await response({}, None, send)
        assert closed == [True]
    asyncio.run(run())


def test_real_pool_queues_module_burst_and_cancels_waiters() -> None:
    """Healthy finite responses may outlive the former three-second pool bound."""
    async def run():
        release = asyncio.Event()
        saturated = asyncio.Event()
        handlers = set()
        accepted = 0
        async def handle(reader, writer):
            nonlocal accepted
            task = asyncio.current_task()
            handlers.add(task)
            try:
                await reader.readuntil(b"\r\n\r\n")
                accepted += 1
                if accepted == 32:
                    saturated.set()
                await release.wait()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(task)
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        gateway = sidecar_module.TelegramProxySidecar(BOT_TOKEN, OWNER_ID, port)
        client = await gateway._get_http_client()
        jobs = []
        try:
            url = f"http://127.0.0.1:{port}/static/module.js"
            jobs = [asyncio.create_task(client.get(url)) for _ in range(40)]
            await asyncio.wait_for(saturated.wait(), 5)
            # The first 32 requests own all slots. One additional waiter can be
            # cancelled without retiring the client or replaying any request.
            cancelled = asyncio.create_task(client.get(url))
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            await asyncio.sleep(3.2)
            assert all(not job.done() for job in jobs), "finite burst expired during pool acquisition"
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(*jobs), 5)
            assert all(response.status_code == 200 and response.content == b"ok" for response in responses)
            assert accepted == 40
        finally:
            release.set()
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            await gateway.aclose()
            server.close()
            await server.wait_closed()
            if handlers:
                await asyncio.gather(*list(handlers), return_exceptions=True)
    asyncio.run(run())
