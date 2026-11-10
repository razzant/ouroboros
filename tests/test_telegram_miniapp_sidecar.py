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
import websockets
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

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


@pytest.mark.parametrize("error, expected_code", [
    (ConnectionClosedError(None, Close(1009, "frame too large")), 1009),
    (ConnectionClosedError(Close(1000, "peer close"), Close(1009, "frame too large"), False), 1009),
    (ConnectionClosedError(Close(1011, "upstream failed"), Close(1011, "upstream failed"), True), 1011),
    (ConnectionClosedError(Close(1005, ""), None), 1011),
    (ConnectionClosedError(None, None), 1011),
    (RuntimeError("private upstream detail"), 1011),
])
@pytest.mark.parametrize("direction", ["receive", "send"])
def test_websocket_does_not_report_upstream_failure_as_normal(error, expected_code, direction):
    async def scenario():
        class FailingUpstream(_FakeUpstreamWebSocket):
            async def __anext__(self):
                if direction != "receive":
                    return await super().__anext__()
                raise error

            async def send(self, value):
                if direction != "send":
                    return await super().send(value)
                raise error

        upstream = FailingUpstream()
        gateway = sidecar_module.TelegramProxySidecar(
            BOT_TOKEN, OWNER_ID, 8765,
            seams=sidecar_module.SidecarSeams(websocket_connect=_FakeConnector(upstream)),
        )
        gateway.set_public_url(PUBLIC_URL)
        token, _session = gateway._issue_session()
        try:
            messages = await asyncio.wait_for(_invoke_websocket(
                gateway, cookie=token,
                client_messages=[{"type": "websocket.receive", "text": "from-client"}] if direction == "send" else [],
            ), 5)
            assert messages[-1] == {"type": "websocket.close", "code": expected_code, "reason": ""}
            assert upstream.exited
            assert gateway.diagnostics()["active_websockets"] == 0
        finally:
            await gateway.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("size_delta", [0, 1])
def test_real_upstream_size_boundary_preserves_close_cause(size_delta):
    async def scenario():
        payload = "x" * (sidecar_module._MAX_WS_MESSAGE_BYTES + size_delta)
        closed = asyncio.Event()
        observed_codes = []

        async def core(upstream):
            try:
                await upstream.send(payload)
                if not size_delta:
                    await upstream.send("following event")
                    await upstream.close()
                else:
                    await upstream.wait_closed()
                observed_codes.append(upstream.close_code)
            finally:
                closed.set()

        async with websockets.serve(core, "127.0.0.1", 0, compression=None) as server:
            port = server.sockets[0].getsockname()[1]
            gateway = sidecar_module.TelegramProxySidecar(BOT_TOKEN, OWNER_ID, port)
            gateway.set_public_url(PUBLIC_URL)
            token, _session = gateway._issue_session()
            try:
                messages = await asyncio.wait_for(_invoke_websocket(gateway, cookie=token), 10)
                await asyncio.wait_for(closed.wait(), 5)
                delivered = [row.get("text") for row in messages if row["type"] == "websocket.send"]
                assert delivered == ([] if size_delta else [payload, "following event"])
                expected = 1009 if size_delta else 1000
                assert messages[-1] == {"type": "websocket.close", "code": expected, "reason": ""}
                assert observed_codes == [expected]
                assert gateway.diagnostics()["active_websockets"] == 0
            finally:
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
    async def run() -> None:
        closed: list[bool] = []

        async def body():
            yield b"module"

        async def close() -> None:
            closed.append(True)

        async def send(_message: dict[str, Any]) -> None:
            raise OSError("downstream disconnected before headers")

        async def receive() -> dict[str, Any]:  # an ASGI server that raises at the headers reports nothing here
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        response = sidecar_module._ProxyStreamResponse(200, [], body(), close)
        with pytest.raises(OSError):
            await response({}, receive, send)
        assert closed == [True]

    asyncio.run(run())


class _LoopbackCore:
    """A real loopback HTTP/1.1 core: it counts the connections it holds open and answers one request per connection
    once `release` is set (at once when `held` is False), then closes it; `keep_open` leaves the closing to the client,
    so a response the client never closes keeps its socket."""

    def __init__(self, *, held: bool, keep_open: bool = False) -> None:
        self.release = asyncio.Event()
        self.keep_open = keep_open
        if not held:
            self.release.set()
        self.open = self.peak = self.accepted = 0
        self.paths: list[str] = []
        self.handlers: set[asyncio.Task] = set()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self.handlers.add(task)
        self.open += 1
        self.peak = max(self.peak, self.open)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            self.accepted += 1
            self.paths.append(head.split(b" ", 2)[1].decode("ascii"))
            await self.release.wait()
            closing = b"" if self.keep_open else b"Connection: close\r\n"
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/javascript\r\nContent-Length: 2\r\n" + closing + b"\r\nok")
            await writer.drain()
            if self.keep_open:
                await reader.read()  # until the client lets the connection go
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.open -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            self.handlers.discard(task)

    async def settle(self, accepted: int) -> None:
        while self.accepted < accepted:
            await asyncio.sleep(0.01)


async def _owner_session(gateway: Any) -> tuple[httpx.AsyncClient, str]:
    browser = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url=PUBLIC_ORIGIN)
    assert (await _authenticate(browser, auth_date=int(time.time()))).status_code == 204
    token = browser.cookies.get(sidecar_module.SESSION_COOKIE)
    assert token
    return browser, token


async def _stop(server: asyncio.AbstractServer, core: _LoopbackCore, gateway: Any, *jobs: asyncio.Task) -> None:
    core.release.set()
    for job in jobs:
        if not job.done():
            job.cancel()
    await asyncio.gather(*jobs, return_exceptions=True)
    await gateway.aclose()
    server.close()
    await server.wait_closed()
    if core.handlers:
        await asyncio.wait_for(asyncio.gather(*list(core.handlers), return_exceptions=True), 5)


def test_sidecar_route_queues_a_finite_burst_past_its_32_sockets_and_a_cancelled_wait_never_reaches_the_core() -> None:
    """The real pool behind the real route: the former 3 s acquisition bound turned this burst's tail into 502s."""
    async def scenario() -> None:
        core = _LoopbackCore(held=True)
        server = await asyncio.start_server(core.handle, "127.0.0.1", 0)
        gateway = sidecar_module.TelegramProxySidecar(BOT_TOKEN, OWNER_ID, server.sockets[0].getsockname()[1])
        gateway.set_public_url(PUBLIC_URL)
        browser, _token = await _owner_session(gateway)
        jobs: list[asyncio.Task] = []
        try:
            jobs = [asyncio.create_task(browser.get(f"/static/modules/m{index}.js")) for index in range(40)]
            await asyncio.wait_for(core.settle(32), 5)
            waiter = asyncio.create_task(browser.get("/static/modules/cancelled.js"))
            await asyncio.sleep(0.1)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            await asyncio.sleep(3.2)
            assert (core.accepted, core.peak) == (32, 32), "the 32 sockets bound the burst"
            assert not any(job.done() for job in jobs), "a finite burst expired while it waited for a socket"
            core.release.set()
            responses = await asyncio.wait_for(asyncio.gather(*jobs), 10)
            assert [response.status_code for response in responses] == [200] * 40
            assert all(response.content == b"ok" for response in responses)
            assert core.accepted == 40 and core.peak <= 32
            assert "/static/modules/cancelled.js" not in core.paths, "a cancelled wait never reaches the core"
        finally:
            await browser.aclose()
            await _stop(server, core, gateway, *jobs)

    asyncio.run(scenario())


def test_a_phone_gone_before_the_headers_releases_its_socket_so_an_unbounded_wait_cannot_starve() -> None:
    """With no acquisition deadline a leaked upstream response would hold its socket for good: 34 deliveries that fail
    at the headers are followed by an ordinary request that must still get through."""
    async def scenario() -> None:
        core = _LoopbackCore(held=False, keep_open=True)
        server = await asyncio.start_server(core.handle, "127.0.0.1", 0)
        gateway = sidecar_module.TelegramProxySidecar(BOT_TOKEN, OWNER_ID, server.sockets[0].getsockname()[1])
        gateway.set_public_url(PUBLIC_URL)
        browser, token = await _owner_session(gateway)
        try:
            def receiver() -> Any:
                # As an ASGI server does: the (empty) body once, then nothing until the client goes.
                sent = []

                async def receive() -> dict[str, Any]:
                    if not sent:
                        sent.append(True)
                        return {"type": "http.request", "body": b"", "more_body": False}
                    await asyncio.Event().wait()
                    raise AssertionError("unreachable")
                return receive

            async def gone(message: dict[str, Any]) -> None:
                if message["type"] == "http.response.start":
                    raise OSError("the phone left before the headers")

            for index in range(34):
                path = f"/static/modules/gone{index}.js"
                scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
                         "scheme": "https", "path": path, "raw_path": path.encode(), "query_string": b"",
                         "headers": [(b"host", b"quiet-river-42.trycloudflare.com"),
                                     (b"cookie", f"{sidecar_module.SESSION_COOKIE}={token}".encode())],
                         "client": ("127.0.0.1", 50000 + index), "server": ("127.0.0.1", 443)}
                with pytest.raises(OSError):
                    await asyncio.wait_for(gateway(scope, receiver(), gone), 5)
            assert core.accepted == 34
            ordinary = await asyncio.wait_for(browser.get("/static/modules/after.js"), 5)
            assert (ordinary.status_code, ordinary.content) == (200, b"ok")
            assert core.peak <= 32
        finally:
            await browser.aclose()
            await _stop(server, core, gateway)

    asyncio.run(scenario())


def test_closing_the_gateway_settles_every_request_it_holds() -> None:
    """Observed shutdown, not a cancellation claim: requests on held sockets answer 502 at once; queued ones are not
    dropped by the close, they go on to fresh sockets and settle when the core answers."""
    async def scenario() -> None:
        core = _LoopbackCore(held=True)
        server = await asyncio.start_server(core.handle, "127.0.0.1", 0)
        gateway = sidecar_module.TelegramProxySidecar(BOT_TOKEN, OWNER_ID, server.sockets[0].getsockname()[1])
        gateway.set_public_url(PUBLIC_URL)
        browser, _token = await _owner_session(gateway)
        held: list[asyncio.Task] = []
        queued: list[asyncio.Task] = []
        try:
            held = [asyncio.create_task(browser.get(f"/static/modules/held{index}.js")) for index in range(32)]
            await asyncio.wait_for(core.settle(32), 5)
            queued = [asyncio.create_task(browser.get(f"/static/modules/queued{index}.js")) for index in range(4)]
            await asyncio.sleep(0.2)
            await asyncio.wait_for(gateway.aclose(), 5)
            answered = await asyncio.wait_for(asyncio.gather(*held), 5)
            assert {(response.status_code, response.text) for response in answered} == {(502, "Ouroboros is unavailable.")}
            await asyncio.wait_for(core.settle(36), 5)
            core.release.set()
            later = await asyncio.wait_for(asyncio.gather(*queued), 5)
            assert [response.status_code for response in later] == [200] * 4
        finally:
            await browser.aclose()
            await _stop(server, core, gateway, *held, *queued)

    asyncio.run(scenario())


class _StreamingCore:
    """A loopback core: `/stream…` answers with an endless chunked body, as a stream widget's EventSource stays open;
    any other path answers `ok` at once. It counts the connections it holds and the paths it was asked for."""

    def __init__(self) -> None:
        self.open = 0
        self.paths: list[str] = []

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.open += 1
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            path = head.split(b" ", 2)[1].decode("ascii")
            self.paths.append(path)
            if path.startswith("/stream"):
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                while not reader.at_eof():
                    writer.write(b"c\r\ndata: tick\n\n\r\n")
                    await writer.drain()
                    await asyncio.sleep(0.05)
            else:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.open -= 1
            writer.close()


def test_under_the_shipped_uvicorn_streams_whose_phones_left_give_back_every_socket() -> None:
    """The shipped starter (companion.start_sidecar: uvicorn) over real loopback TCP. uvicorn reports a gone client only
    to receive() (ASGI 2.3: its send() then returns quietly) and never cancels the request. 32 endless upstream streams
    take every socket and a 33rd waits; all 33 phones leave, half by RST and half by FIN. Each stream must stop and give
    its socket back, the queued one ends at its body read without an ASGI error, and a later request gets a socket.
    Without the receive() race the core kept all 32 streams and the later request starved."""
    import logging
    import types

    scripts = str(SIDECAR_PATH.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import companion

    async def until(check: Callable[[], bool]) -> None:
        while not check():
            await asyncio.sleep(0.02)

    async def scenario() -> None:
        core = _StreamingCore()
        server = await asyncio.start_server(core.handle, "127.0.0.1", 0)
        gateway = companion.TelegramProxySidecar(BOT_TOKEN, OWNER_ID, server.sockets[0].getsockname()[1])
        gateway.set_public_url(PUBLIC_URL)
        seen: list[str] = []

        async def observed(scope: dict[str, Any], receive: Any, send: Any) -> None:  # records, changes nothing
            async def tap() -> dict[str, Any]:
                message = await receive()
                seen.append(message["type"])
                return message
            await gateway.app(scope, tap, send)

        uv, uv_task, port = await companion.start_sidecar(types.SimpleNamespace(app=observed))
        errors: list[logging.LogRecord] = []
        handler = logging.Handler(level=logging.ERROR)
        handler.emit = errors.append  # type: ignore[method-assign]
        logging.getLogger("uvicorn.error").addHandler(handler)
        host = PUBLIC_URL.split("//", 1)[1].rstrip("/")
        phones: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", headers={"Host": host}) as http:
                auth = await http.post(sidecar_module.AUTH_PATH, headers={"Origin": PUBLIC_ORIGIN},
                                       json={"init_data": signed_init_data(auth_date=int(time.time()))})
                assert auth.status_code == 204
                cookie = auth.headers["set-cookie"].split(";", 1)[0]

                async def phone(path: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.write(f"GET {path} HTTP/1.1\r\nHost: {host}\r\nCookie: {cookie}\r\n\r\n".encode())
                    await writer.drain()
                    return reader, writer

                phones = [await phone(f"/stream/{index}") for index in range(32)]
                for reader, _writer in phones:
                    assert b" 200 " in await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
                phones.append(await phone("/stream/queued"))
                await asyncio.sleep(0.3)
                assert core.open == 32 and "/stream/queued" not in core.paths, "live phones keep their streams"
                for index, (_reader, writer) in enumerate(phones):
                    (writer.transport.abort if index % 2 == 0 else writer.close)()
                await asyncio.wait_for(until(lambda: core.open == 0), 5)
                await asyncio.wait_for(until(lambda: seen.count("http.disconnect") >= 33), 5)
                later = await asyncio.wait_for(http.get("/ok", headers={"Cookie": cookie}), 5)
                assert (later.status_code, later.text) == (200, "ok")
                assert errors == [], [record.getMessage() for record in errors]
        finally:
            logging.getLogger("uvicorn.error").removeHandler(handler)
            for _reader, writer in phones:
                writer.close()
            uv.should_exit = True
            await asyncio.wait_for(uv_task, 10)
            await gateway.aclose()
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
