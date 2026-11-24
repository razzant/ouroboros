"""Session-guarded HTTP forwarding, header sanitization, and streaming for the Telegram Mini App sidecar."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Iterable

from starlette.requests import Request
from starlette.responses import Response

MINIAPP_REQUEST_MARKER_HEADER = "X-Ouroboros-Telegram-MiniApp"
_MINIAPP_REQUEST_MARKER = MINIAPP_REQUEST_MARKER_HEADER.lower().encode("ascii")

_HOP_BY_HOP_HEADERS = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}
_REQUEST_BLOCKED_HEADERS = _HOP_BY_HOP_HEADERS | {
    b"authorization",
    b"content-length",
    b"cookie",
    b"expect",
    b"forwarded",
    b"host",
    b"origin",
    b"referer",
    b"true-client-ip",
    b"x-client-ip",
    _MINIAPP_REQUEST_MARKER,
    b"x-ouroboros-password",
    b"x-real-ip",
}
_REQUEST_BLOCKED_PREFIXES = (
    b"cf-",
    b"proxy-",
    b"sec-websocket-",
    b"x-envoy-",
    b"x-forwarded-",
    b"x-original-",
    b"x-real-",
    b"x-rewrite-",
)
_RESPONSE_BLOCKED_HEADERS = _HOP_BY_HOP_HEADERS | {
    b"cache-control",
    b"cdn-cache-control",
    b"clear-site-data",
    b"cloudflare-cdn-cache-control",
    b"date",
    b"expires",
    b"pragma",
    b"refresh",
    b"server",
    b"set-cookie",
    b"surrogate-control",
}
_BUILTIN_STATIC_CACHE = b"private, no-cache, must-revalidate"


def header_values(scope: dict[str, Any], name: bytes) -> list[str]:
    return [value.decode("latin-1") for key, value in scope.get("headers", []) if key.lower() == name]


def _connection_tokens(headers: Iterable[tuple[bytes, bytes]]) -> set[bytes]:
    tokens: set[bytes] = set()
    for name, value in headers:
        if name.lower() != b"connection":
            continue
        for item in value.lower().split(b","):
            token = item.strip()
            if token:
                tokens.add(token)
    return tokens


def request_headers(scope: dict[str, Any]) -> list[tuple[bytes, bytes]]:
    raw_headers = list(scope.get("headers", []))
    blocked = _REQUEST_BLOCKED_HEADERS | _connection_tokens(raw_headers)
    output: list[tuple[bytes, bytes]] = []
    for raw_name, value in raw_headers:
        name = raw_name.lower()
        if name in blocked or any(name.startswith(prefix) for prefix in _REQUEST_BLOCKED_PREFIXES):
            continue
        output.append((name, value))
    if not any(name == b"accept-encoding" for name, _ in output):
        # The body is relayed raw: the phone's own choice reaches the core, never httpx's default.
        output.append((b"accept-encoding", b"identity"))
    output.append((_MINIAPP_REQUEST_MARKER, b"1"))
    return output


def response_headers(
    headers: Iterable[tuple[bytes, bytes]],
    *,
    core_origin: str,
    public_origin: str,
    revalidated_static: bool = False,
) -> list[tuple[bytes, bytes]]:
    """Every answer is ``no-store`` except an owner's GET of a builtin ``/static`` file marked ``_BUILTIN_STATIC_CACHE``."""
    raw_headers = list(headers)
    blocked = _RESPONSE_BLOCKED_HEADERS | _connection_tokens(raw_headers)
    keep = revalidated_static and (b"cache-control", _BUILTIN_STATIC_CACHE) in ((n.lower(), v) for n, v in raw_headers)
    output: list[tuple[bytes, bytes]] = []
    for raw_name, raw_value in raw_headers:
        name = raw_name.lower()
        if name in blocked or name.startswith(b"access-control-"):
            continue
        value = raw_value
        if name == b"location":
            try:
                location = raw_value.decode("latin-1")
            except UnicodeError:
                continue
            if location == core_origin or location.startswith(core_origin + "/"):
                location = public_origin + location[len(core_origin) :].lstrip("/")
                value = location.encode("latin-1")
        output.append((name, value))
    if keep:
        return [*output, (b"cache-control", _BUILTIN_STATIC_CACHE), (b"cdn-cache-control", b"no-store")]
    return [*output, (b"cache-control", b"private, no-store"), (b"pragma", b"no-cache"), (b"expires", b"0")]


def plain_response(status_code: int, text: str, *, headers: dict[str, str] | None = None) -> Response:
    response_headers_map = {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    }
    if headers:
        response_headers_map.update(headers)
    return Response(text, status_code=status_code, media_type="text/plain", headers=response_headers_map)


async def bounded_body(request: Request, limit: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise ValueError("request body is too large")
    return bytes(body)


class ProxyStreamResponse:
    """Minimal ASGI streaming response that aborts and closes upstream immediately on session revocation."""

    def __init__(
        self,
        status_code: int,
        headers: list[tuple[bytes, bytes]],
        body: AsyncIterator[bytes],
        close: Callable[[], Awaitable[None]],
        *,
        is_valid: Callable[[], bool] | None = None,
        wait_invalid: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body = body
        self.close = close
        self._is_valid = is_valid
        self._wait_invalid = wait_invalid

    async def __call__(self, _scope: dict[str, Any], receive: Any, send: Any) -> None:
        if self._is_valid is not None and not self._is_valid():
            await self.close()
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"cache-control", b"no-store"),
                ],
            })
            await send({"type": "http.response.body", "body": b"Telegram owner session expired.", "more_body": False})
            return
        delivery = asyncio.ensure_future(self._deliver(send))
        gone = asyncio.ensure_future(self._client_gone(receive))
        tasks = [delivery, gone]
        guard = asyncio.ensure_future(self._wait_invalid()) if self._wait_invalid is not None else None
        if guard is not None:
            tasks.append(guard)
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.close()
        if not delivery.cancelled() and delivery.exception() is not None:
            raise delivery.exception()

    async def _deliver(self, send: Any) -> None:
        if self._is_valid is not None and not self._is_valid():
            return
        await send({"type": "http.response.start", "status": self.status_code, "headers": self.headers})
        async for chunk in self.body:
            if self._is_valid is not None and not self._is_valid():
                return
            if chunk:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
        if self._is_valid is not None and not self._is_valid():
            return
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    @staticmethod
    async def _client_gone(receive: Any) -> None:
        while (await receive())["type"] != "http.disconnect":
            pass


async def send_with_session_guard(
    send_coro: Awaitable[Any],
    *,
    is_valid: Callable[[], bool],
    wait_invalid: Callable[[], Awaitable[None]],
) -> tuple[Any, bool]:
    """Await ``send_coro`` while racing session expiry/revocation and cleanly cancelling on caller abort."""
    if not is_valid():
        close = getattr(send_coro, "close", None)
        if callable(close):
            close()
        return None, True
    send_task = asyncio.ensure_future(send_coro)
    guard_task = asyncio.ensure_future(wait_invalid())
    returned = False
    try:
        done, _pending = await asyncio.wait((send_task, guard_task), return_when=asyncio.FIRST_COMPLETED)
        if guard_task in done or not is_valid():
            return None, True
        response = send_task.result()
        if not is_valid():
            if hasattr(response, "aclose"):
                try:
                    await response.aclose()
                except Exception:
                    pass
            return None, True
        returned = True
        return response, False
    finally:
        if not guard_task.done():
            guard_task.cancel()
        if not returned:
            if not send_task.done():
                send_task.cancel()
            else:
                try:
                    leaked = send_task.result()
                except BaseException:
                    leaked = None
                if leaked is not None and hasattr(leaked, "aclose"):
                    try:
                        await leaked.aclose()
                    except Exception:
                        pass
            await asyncio.gather(send_task, guard_task, return_exceptions=True)
        else:
            await asyncio.gather(guard_task, return_exceptions=True)


__all__ = [
    "MINIAPP_REQUEST_MARKER_HEADER",
    "ProxyStreamResponse",
    "bounded_body",
    "header_values",
    "plain_response",
    "request_headers",
    "response_headers",
    "send_with_session_guard",
]
