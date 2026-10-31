"""Authenticated Telegram Mini App gateway for the existing Ouroboros SPA.

The public tunnel must terminate here, never at the loopback-trusted Ouroboros
gateway.  This process validates Telegram launch data, issues a short-lived
opaque owner session, and only then proxies HTTP and WebSocket traffic to the
unchanged local SPA.

The module intentionally has no logging.  In particular, Telegram initData,
the bot token, and session cookies must never enter process logs.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import ipaddress
import json
import re
import secrets
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from http.cookies import CookieError, SimpleCookie
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable
from urllib.parse import parse_qsl, urlsplit

import httpx
import websockets
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response
from starlette.websockets import WebSocket

AUTH_PATH = "/__telegram/auth"
BOOTSTRAP_JS_PATH = "/__telegram/bootstrap.js"
SESSION_PATH = "/__telegram/session"
RESERVED_PREFIX = "/__telegram/"
SESSION_COOKIE = "__Host-ouroboros-telegram"
GATEWAY_MARKER_HEADER = "X-Ouroboros-Telegram-Gateway"
MINIAPP_REQUEST_MARKER_HEADER = "X-Ouroboros-Telegram-MiniApp"
_MINIAPP_REQUEST_MARKER = MINIAPP_REQUEST_MARKER_HEADER.lower().encode("ascii")

_MAX_AUTH_BODY_BYTES = 16 * 1024
_MAX_INIT_DATA_BYTES = 12 * 1024
_MAX_RAW_TARGET_BYTES = 32 * 1024
_MAX_WS_MESSAGE_BYTES = 8 * 1024 * 1024
_SESSION_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_TRYCLOUDFLARE_SUFFIX = ".trycloudflare.com"
_ALLOWED_HTTP_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}

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

_BOOTSTRAP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <title>Ouroboros</title>
  <style>
    html{color-scheme:dark;--bg-primary:#0f0f11;--text-secondary:rgba(255,255,255,.58)}
    html[data-theme="light"]{color-scheme:light;--bg-primary:#ffffff;--text-secondary:#626977}
    html,body{height:100%;margin:0;background:var(--bg-primary);color:var(--text-secondary);
      font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
    main{height:100%;display:grid;place-items:center;text-align:center;padding:24px;box-sizing:border-box}
    #status{max-width:28rem}
  </style>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <script src="/__telegram/bootstrap.js"></script>
</head>
<body><main><div id="status" role="status">Connecting to Ouroboros…</div></main></body>
</html>
"""

# Runs in <head>, after Telegram's script and before the first paint: the interface's room and ink
# (web/ui.css) for the Mini App's saved Light or Dark (web/theme.js keeps it under this key), else
# Telegram's own light or dark (the interface's System here), else the device's. Then, once the
# document is parsed, the launch.
_BOOTSTRAP_JS = r"""(() => {
  const webApp = window.Telegram && window.Telegram.WebApp;
  let saved = '';
  try { saved = localStorage.getItem('ouroboros.theme') || ''; } catch (_) { /* blocked: Telegram's own */ }
  const scheme = webApp && webApp.colorScheme;
  document.documentElement.dataset.theme = saved === 'light' || saved === 'dark' ? saved
    : scheme === 'light' || scheme === 'dark' ? scheme
    : window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  const say = (text) => { const status = document.getElementById('status'); if (status) status.textContent = text; };
  const fail = () => say('Open this app from its Telegram bot menu.');
  const start = async () => {
    try {
      if (!webApp || !webApp.initData) { fail(); return; }
      webApp.ready();
      webApp.expand();
      const response = await fetch('/__telegram/auth', {
        method: 'POST',
        credentials: 'same-origin',
        cache: 'no-store',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({init_data: webApp.initData}),
      });
      if (!response.ok) { fail(); return; }
      const session = await fetch('/__telegram/session', {
        method: 'GET', credentials: 'same-origin', cache: 'no-store'
      });
      if (!session.ok) {
        say('This Telegram web client cannot store the secure owner session. Use the native Telegram app.');
        return;
      }
      window.location.replace('/');
    } catch (_) { fail(); }
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
  else start();
})();
"""


class TelegramInitDataError(ValueError):
    """Telegram launch data failed closed validation."""


class _Session:
    __slots__ = ("expires_at", "generation")

    def __init__(self, expires_at: float, generation: int) -> None:
        self.expires_at = expires_at
        self.generation = generation


@dataclass(frozen=True)
class SidecarLimits:
    """Bounded public-gateway limits kept together at the lifecycle seam."""

    session_ttl_sec: int = 3600
    init_data_max_age_sec: int = 300
    future_skew_sec: int = 30
    max_sessions: int = 8
    max_websockets: int = 8
    max_auth_concurrency: int = 4
    auth_global_limit: int = 60
    auth_client_limit: int = 12
    auth_rate_window_sec: int = 60
    max_auth_clients: int = 128


@dataclass(frozen=True)
class SidecarSeams:
    """Injectable I/O and clocks used by production wiring and tests."""

    http_client: httpx.AsyncClient | None = None
    websocket_connect: Callable[..., Any] | None = None
    clock: Callable[[], float] | None = None
    session_clock: Callable[[], float] | None = None
    rate_clock: Callable[[], float] | None = None
    exposure_guard: Callable[[], bool] | None = None


class _ProxyStreamResponse:
    """Minimal ASGI response that preserves a streamed upstream body verbatim."""

    def __init__(
        self,
        status_code: int,
        headers: list[tuple[bytes, bytes]],
        body: AsyncIterator[bytes],
        close: Callable[[], Awaitable[None]],
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.body = body
        self.close = close

    async def __call__(self, _scope: dict[str, Any], _receive: Any, send: Any) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": self.headers,
            }
        )
        try:
            async for chunk in self.body:
                if chunk:
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            await self.close()


def _data_check_string(items: Iterable[tuple[str, str]]) -> str:
    return "\n".join(f"{key}={value}" for key, value in sorted(items))


def validate_telegram_init_data(
    raw_init_data: str,
    *,
    bot_token: str,
    owner_user_id: int,
    now: float | None = None,
    max_age_sec: int = 300,
    future_skew_sec: int = 30,
) -> int:
    """Validate Telegram initData and return the bound owner user id.

    This implements Telegram's bot-token HMAC construction.  Every decoded
    field except ``hash`` participates in the signed data-check string.
    """

    if not isinstance(raw_init_data, str):
        raise TelegramInitDataError("invalid launch data")
    if not isinstance(bot_token, str) or not bot_token:
        raise TelegramInitDataError("gateway is not configured")
    if isinstance(owner_user_id, bool) or not isinstance(owner_user_id, int) or owner_user_id <= 0:
        raise TelegramInitDataError("gateway is not configured")
    if max_age_sec <= 0 or future_skew_sec < 0:
        raise TelegramInitDataError("gateway is not configured")
    try:
        encoded_size = len(raw_init_data.encode("utf-8", "strict"))
    except UnicodeError as exc:
        raise TelegramInitDataError("invalid launch data") from exc
    if not raw_init_data or encoded_size > _MAX_INIT_DATA_BYTES:
        raise TelegramInitDataError("invalid launch data")

    try:
        pairs = parse_qsl(
            raw_init_data,
            keep_blank_values=True,
            strict_parsing=True,
            encoding="utf-8",
            errors="strict",
            max_num_fields=64,
        )
    except (UnicodeError, ValueError) as exc:
        raise TelegramInitDataError("invalid launch data") from exc

    values: dict[str, str] = {}
    for key, value in pairs:
        if not key or key in values or "\n" in key or "\r" in key:
            raise TelegramInitDataError("invalid launch data")
        values[key] = value

    supplied_hash = values.get("hash", "")
    if len(supplied_hash) != 64:
        raise TelegramInitDataError("invalid launch data")
    try:
        bytes.fromhex(supplied_hash)
    except ValueError as exc:
        raise TelegramInitDataError("invalid launch data") from exc

    signed_pairs = [(key, value) for key, value in values.items() if key != "hash"]
    secret_key = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    expected_hash = hmac.new(
        secret_key,
        _data_check_string(signed_pairs).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, supplied_hash.lower()):
        raise TelegramInitDataError("invalid launch data")

    auth_date_raw = values.get("auth_date", "")
    if not auth_date_raw or any(char < "0" or char > "9" for char in auth_date_raw):
        raise TelegramInitDataError("invalid launch data")
    auth_date = int(auth_date_raw)
    observed_now = time.time() if now is None else float(now)
    if auth_date > observed_now + future_skew_sec or observed_now - auth_date > max_age_sec:
        raise TelegramInitDataError("expired launch data")

    try:
        user = json.loads(values.get("user", ""))
    except (TypeError, ValueError) as exc:
        raise TelegramInitDataError("invalid launch data") from exc
    if not isinstance(user, dict):
        raise TelegramInitDataError("invalid launch data")
    user_id = user.get("id")
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise TelegramInitDataError("invalid launch data")
    if user.get("is_bot") is True:
        raise TelegramInitDataError("invalid launch data")
    if not hmac.compare_digest(str(user_id), str(owner_user_id)):
        raise TelegramInitDataError("wrong Telegram owner")
    return user_id


def _normalise_public_url(value: str) -> tuple[str, str]:
    try:
        parsed = urlsplit(str(value or "").strip())
        port = parsed.port
    except ValueError as exc:
        raise ValueError("public URL is invalid") from exc
    host = str(parsed.hostname or "").lower()
    if (
        parsed.scheme.lower() != "https"
        or not host
        or parsed.username
        or parsed.password
        or port not in (None, 443)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("public URL must be an HTTPS origin root")
    try:
        host.encode("ascii", "strict")
    except UnicodeError as exc:
        raise ValueError("public URL host must be ASCII") from exc
    labels = host.split(".")
    if len(host) > 253 or any(not _DNS_LABEL_RE.fullmatch(label) for label in labels):
        raise ValueError("public URL host is invalid")
    tunnel_label = host[: -len(_TRYCLOUDFLARE_SUFFIX)] if host.endswith(_TRYCLOUDFLARE_SUFFIX) else ""
    if not tunnel_label or "." in tunnel_label or not _DNS_LABEL_RE.fullmatch(tunnel_label):
        raise ValueError("public URL must be a Cloudflare Quick Tunnel origin")
    return f"https://{host}/", host


def _header_values(scope: dict[str, Any], name: bytes) -> list[str]:
    return [
        value.decode("latin-1")
        for key, value in scope.get("headers", [])
        if key.lower() == name
    ]


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


def _request_headers(scope: dict[str, Any]) -> list[tuple[bytes, bytes]]:
    raw_headers = list(scope.get("headers", []))
    blocked = _REQUEST_BLOCKED_HEADERS | _connection_tokens(raw_headers)
    output: list[tuple[bytes, bytes]] = []
    for raw_name, value in raw_headers:
        name = raw_name.lower()
        if name in blocked or any(name.startswith(prefix) for prefix in _REQUEST_BLOCKED_PREFIXES):
            continue
        output.append((name, value))
    output.append((_MINIAPP_REQUEST_MARKER, b"1"))
    return output


def _response_headers(
    headers: Iterable[tuple[bytes, bytes]],
    *,
    core_origin: str,
    public_origin: str,
) -> list[tuple[bytes, bytes]]:
    raw_headers = list(headers)
    blocked = _RESPONSE_BLOCKED_HEADERS | _connection_tokens(raw_headers)
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
    output.extend(
        [
            (b"cache-control", b"private, no-store"),
            (b"pragma", b"no-cache"),
            (b"expires", b"0"),
        ]
    )
    return output


def _plain_response(status_code: int, text: str, *, headers: dict[str, str] | None = None) -> Response:
    response_headers = {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    }
    if headers:
        response_headers.update(headers)
    return Response(text, status_code=status_code, media_type="text/plain", headers=response_headers)


async def _bounded_body(request: Request, limit: int) -> bytes:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise ValueError("request body is too large")
    return bytes(body)


class TelegramProxySidecar:
    """ASGI auth boundary and reverse proxy for the existing Ouroboros SPA."""

    def __init__(
        self,
        bot_token: str,
        owner_user_id: int,
        core_port: int,
        *,
        limits: SidecarLimits | None = None,
        seams: SidecarSeams | None = None,
    ) -> None:
        limits = limits or SidecarLimits()
        seams = seams or SidecarSeams()
        if not isinstance(bot_token, str) or not bot_token:
            raise ValueError("bot token is required")
        if isinstance(owner_user_id, bool) or not isinstance(owner_user_id, int) or owner_user_id <= 0:
            raise ValueError("owner user id must be a positive integer")
        if isinstance(core_port, bool) or not isinstance(core_port, int) or not 1 <= core_port <= 65535:
            raise ValueError("core port is invalid")
        if not isinstance(limits.session_ttl_sec, int) or not 1 <= limits.session_ttl_sec <= 24 * 60 * 60:
            raise ValueError("session TTL must be between 1 second and 24 hours")
        if not isinstance(limits.init_data_max_age_sec, int) or not 1 <= limits.init_data_max_age_sec <= 3600:
            raise ValueError("initData maximum age is invalid")
        if not isinstance(limits.future_skew_sec, int) or not 0 <= limits.future_skew_sec <= 300:
            raise ValueError("future clock skew is invalid")
        if not isinstance(limits.max_sessions, int) or not 1 <= limits.max_sessions <= 64:
            raise ValueError("maximum session count is invalid")
        if not isinstance(limits.max_websockets, int) or not 1 <= limits.max_websockets <= 32:
            raise ValueError("maximum WebSocket count is invalid")
        for value, minimum, maximum, label in (
            (limits.max_auth_concurrency, 1, 32, "auth concurrency"),
            (limits.auth_global_limit, 1, 10_000, "global auth limit"),
            (limits.auth_client_limit, 1, 1_000, "per-client auth limit"),
            (limits.auth_rate_window_sec, 1, 3_600, "auth rate window"),
            (limits.max_auth_clients, 1, 1_024, "auth client table"),
        ):
            if not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{label} is invalid")

        self._bot_token = bot_token
        self._owner_user_id = owner_user_id
        self._core_port = core_port
        self._core_origin = f"http://127.0.0.1:{core_port}"
        self._core_url = httpx.URL(self._core_origin)
        self._session_ttl_sec = limits.session_ttl_sec
        self._init_data_max_age_sec = limits.init_data_max_age_sec
        self._future_skew_sec = limits.future_skew_sec
        self._max_sessions = limits.max_sessions
        self._max_websockets = limits.max_websockets
        self._max_auth_concurrency = limits.max_auth_concurrency
        self._auth_global_limit = limits.auth_global_limit
        self._auth_client_limit = limits.auth_client_limit
        self._auth_rate_window_sec = limits.auth_rate_window_sec
        self._max_auth_clients = limits.max_auth_clients
        self._wall_clock = seams.clock or time.time
        # Telegram freshness needs Unix wall time; session expiry must resist a
        # host clock rollback.  Tests that inject only ``clock`` retain the old
        # single-clock convenience, while production defaults to monotonic.
        self._session_clock = seams.session_clock or (
            seams.clock if seams.clock is not None else time.monotonic
        )
        self._rate_clock = seams.rate_clock or time.monotonic
        self._websocket_connect = seams.websocket_connect or websockets.connect
        self._exposure_guard = seams.exposure_guard
        self._http_client = seams.http_client
        self._owns_http_client = seams.http_client is None
        self._closed = False

        self._state_lock = threading.RLock()
        self._public_url: str | None = None
        self._public_host: str | None = None
        self._generation = 0
        self._sessions: OrderedDict[bytes, _Session] = OrderedDict()
        self._active_websockets = 0
        self._active_auth = 0
        self._auth_global: deque[float] = deque()
        self._auth_clients: OrderedDict[str, deque[float]] = OrderedDict()
        self._auth_counters = {
            "auth_success": 0,
            "auth_rejected": 0,
            "auth_rate_limited": 0,
            "auth_busy": 0,
        }

    @property
    def app(self) -> "TelegramProxySidecar":
        return self

    @property
    def public_url(self) -> str | None:
        with self._state_lock:
            return self._public_url

    @property
    def public_host(self) -> str | None:
        with self._state_lock:
            return self._public_host

    def set_public_url(self, url: str) -> str:
        normalised, host = _normalise_public_url(url)
        with self._state_lock:
            if normalised != self._public_url:
                self._public_url = normalised
                self._public_host = host
                self._generation += 1
                self._sessions.clear()
        return normalised

    def clear_public_url(self) -> None:
        with self._state_lock:
            self._public_url = None
            self._public_host = None
            self._generation += 1
            self._sessions.clear()

    def _exposure_is_authorized(self) -> bool:
        guard = self._exposure_guard
        if guard is None:
            return True
        try:
            authorized = guard() is True
        except Exception:
            authorized = False
        if authorized:
            return True
        self.clear_public_url()
        return False

    def _public_target(self) -> tuple[str, str] | None:
        with self._state_lock:
            if self._public_url is None or self._public_host is None:
                return None
            return self._public_url, self._public_host

    def _issue_session(self) -> tuple[str, _Session]:
        with self._state_lock:
            now = self._session_clock()
            self._prune_sessions_locked(now)
            while True:
                token = secrets.token_urlsafe(32)
                digest = hashlib.sha256(token.encode("ascii")).digest()
                if digest not in self._sessions:
                    break
            session = _Session(now + self._session_ttl_sec, self._generation)
            self._sessions[digest] = session
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
            return token, session

    def _prune_sessions_locked(self, now: float) -> None:
        expired = [
            token
            for token, session in self._sessions.items()
            if session.expires_at <= now or session.generation != self._generation
        ]
        for token in expired:
            self._sessions.pop(token, None)

    def _lookup_session(self, token: str) -> _Session | None:
        if not _SESSION_TOKEN_RE.fullmatch(token):
            return None
        digest = hashlib.sha256(token.encode("ascii")).digest()
        with self._state_lock:
            self._prune_sessions_locked(self._session_clock())
            session = self._sessions.get(digest)
            if session is not None:
                self._sessions.move_to_end(digest)
            return session

    def _claim_websocket(self) -> bool:
        with self._state_lock:
            if self._active_websockets >= self._max_websockets:
                return False
            self._active_websockets += 1
            return True

    def _release_websocket(self) -> None:
        with self._state_lock:
            self._active_websockets = max(0, self._active_websockets - 1)

    def diagnostics(self) -> dict[str, int]:
        with self._state_lock:
            return {
                **self._auth_counters,
                "active_sessions": len(self._sessions),
                "active_websockets": self._active_websockets,
            }

    @staticmethod
    def _auth_client_key(scope: dict[str, Any]) -> str:
        forwarded = _header_values(scope, b"cf-connecting-ip")
        raw = forwarded[0] if len(forwarded) == 1 else ""
        if not raw:
            client = scope.get("client")
            if isinstance(client, (tuple, list)) and client:
                raw = str(client[0] or "")
        try:
            return ipaddress.ip_address(raw.strip()).compressed
        except ValueError:
            return "unknown"

    @staticmethod
    def _prune_rate_bucket(bucket: deque[float], cutoff: float) -> None:
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def _claim_auth(self, scope: dict[str, Any]) -> int | None:
        now = self._rate_clock()
        cutoff = now - self._auth_rate_window_sec
        key = self._auth_client_key(scope)
        with self._state_lock:
            if self._active_auth >= self._max_auth_concurrency:
                self._auth_counters["auth_busy"] += 1
                return 503
            self._prune_rate_bucket(self._auth_global, cutoff)
            bucket = self._auth_clients.get(key)
            if bucket is None:
                bucket = deque()
                self._auth_clients[key] = bucket
            else:
                self._auth_clients.move_to_end(key)
            self._prune_rate_bucket(bucket, cutoff)
            while len(self._auth_clients) > self._max_auth_clients:
                self._auth_clients.popitem(last=False)
            if len(self._auth_global) >= self._auth_global_limit or len(bucket) >= self._auth_client_limit:
                self._auth_counters["auth_rate_limited"] += 1
                return 429
            self._auth_global.append(now)
            bucket.append(now)
            self._active_auth += 1
            return None

    def _release_auth(self) -> None:
        with self._state_lock:
            self._active_auth = max(0, self._active_auth - 1)

    @staticmethod
    def _cookie_token(scope: dict[str, Any]) -> str | None:
        cookie_headers = _header_values(scope, b"cookie")
        if len(cookie_headers) != 1:
            return None
        raw = cookie_headers[0]
        named_values = []
        for part in raw.split(";"):
            name, separator, value = part.partition("=")
            if separator and name.strip() == SESSION_COOKIE:
                named_values.append(value.strip())
        if len(named_values) != 1:
            return None
        try:
            parsed = SimpleCookie()
            parsed.load(raw)
        except CookieError:
            return None
        morsel = parsed.get(SESSION_COOKIE)
        if morsel is None or morsel.value != named_values[0]:
            return None
        return morsel.value

    def _request_session(self, scope: dict[str, Any]) -> tuple[str, _Session] | None:
        token = self._cookie_token(scope)
        if token is None:
            return None
        session = self._lookup_session(token)
        if session is None:
            return None
        return token, session

    def _validate_public_request(self, scope: dict[str, Any], *, require_origin: bool) -> int | None:
        if not self._exposure_is_authorized():
            return 503
        target = self._public_target()
        if target is None:
            return 503
        public_url, public_host = target
        host_values = _header_values(scope, b"host")
        if len(host_values) != 1 or host_values[0].lower() != public_host:
            return 421
        origin_values = _header_values(scope, b"origin")
        if len(origin_values) > 1:
            return 403
        expected_origin = public_url[:-1]
        if require_origin:
            if len(origin_values) != 1 or origin_values[0] != expected_origin:
                return 403
        elif origin_values and origin_values[0] != expected_origin:
            return 403
        return None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
                timeout=httpx.Timeout(connect=3.0, read=None, write=None, pool=3.0),
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
            )
        return self._http_client

    async def _authenticate(self, scope: dict[str, Any], receive: Any) -> Response:
        claim_error = self._claim_auth(scope)
        if claim_error is not None:
            return _plain_response(
                claim_error,
                "Authentication is busy." if claim_error == 503 else "Too many authentication attempts.",
            )
        try:
            content_types = _header_values(scope, b"content-type")
            if len(content_types) != 1 or content_types[0].split(";", 1)[0].strip().lower() != "application/json":
                with self._state_lock:
                    self._auth_counters["auth_rejected"] += 1
                return _plain_response(415, "Unsupported request.")
            request = Request(scope, receive=receive)
            try:
                body = await asyncio.wait_for(
                    _bounded_body(request, _MAX_AUTH_BODY_BYTES),
                    timeout=3.0,
                )
                payload = json.loads(body)
            except asyncio.TimeoutError:
                with self._state_lock:
                    self._auth_counters["auth_rejected"] += 1
                return _plain_response(408, "Authentication request timed out.")
            except (UnicodeError, ValueError, TypeError):
                with self._state_lock:
                    self._auth_counters["auth_rejected"] += 1
                return _plain_response(400, "Invalid request.")
            if not isinstance(payload, dict) or set(payload) != {"init_data"}:
                with self._state_lock:
                    self._auth_counters["auth_rejected"] += 1
                return _plain_response(400, "Invalid request.")
            raw_init_data = payload.get("init_data")
            if not self._exposure_is_authorized():
                return _plain_response(503, "Gateway is not ready.")
            try:
                validate_telegram_init_data(
                    raw_init_data,
                    bot_token=self._bot_token,
                    owner_user_id=self._owner_user_id,
                    now=self._wall_clock(),
                    max_age_sec=self._init_data_max_age_sec,
                    future_skew_sec=self._future_skew_sec,
                )
            except TelegramInitDataError:
                with self._state_lock:
                    self._auth_counters["auth_rejected"] += 1
                return _plain_response(401, "Invalid Telegram launch.")

            token, _session = self._issue_session()
            with self._state_lock:
                self._auth_counters["auth_success"] += 1
            response = Response(status_code=204, headers={"Cache-Control": "no-store"})
            response.set_cookie(
                SESSION_COOKIE,
                token,
                max_age=self._session_ttl_sec,
                path="/",
                secure=True,
                httponly=True,
                samesite="strict",
            )
            return response
        finally:
            self._release_auth()

    @staticmethod
    def _bootstrap_response() -> HTMLResponse:
        return HTMLResponse(
            _BOOTSTRAP_HTML,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'none'; script-src 'self' https://telegram.org; "
                    "style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; object-src 'none'"
                ),
                GATEWAY_MARKER_HEADER: "1",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @staticmethod
    def _bootstrap_js_response() -> Response:
        return Response(
            _BOOTSTRAP_JS,
            media_type="application/javascript",
            headers={
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    def _upstream_url(self, scope: dict[str, Any]) -> httpx.URL:
        raw_path = scope.get("raw_path") or str(scope.get("path") or "/").encode("ascii", "strict")
        query = scope.get("query_string", b"")
        if (
            not isinstance(raw_path, bytes)
            or not isinstance(query, bytes)
            or not raw_path.startswith(b"/")
            or len(raw_path) + len(query) > _MAX_RAW_TARGET_BYTES
            or any(char in raw_path + query for char in (0, 10, 13))
        ):
            raise ValueError("invalid request target")
        combined = raw_path + (b"?" + query if query else b"")
        return self._core_url.copy_with(raw_path=combined)

    async def _proxy_http(self, scope: dict[str, Any], receive: Any) -> Any:
        try:
            upstream_url = self._upstream_url(scope)
        except (UnicodeError, ValueError):
            return _plain_response(400, "Invalid request target.")

        request = Request(scope, receive=receive)
        client = await self._get_http_client()
        upstream_request = client.build_request(
            str(scope.get("method") or "GET").upper(),
            upstream_url,
            headers=_request_headers(scope),
            content=request.stream(),
        )
        try:
            upstream = await client.send(upstream_request, stream=True)
        except (httpx.HTTPError, OSError):
            return _plain_response(502, "Ouroboros is unavailable.")

        public_target = self._public_target()
        if public_target is None:
            await upstream.aclose()
            return _plain_response(503, "Gateway is not ready.")
        public_origin = public_target[0]
        return _ProxyStreamResponse(
            upstream.status_code,
            _response_headers(
                upstream.headers.raw,
                core_origin=self._core_origin,
                public_origin=public_origin,
            ),
            upstream.aiter_raw(),
            upstream.aclose,
        )

    async def _handle_http(self, scope: dict[str, Any], receive: Any) -> Any:
        method = str(scope.get("method") or "GET").upper()
        require_origin = method not in {"GET", "HEAD"}
        validation_error = self._validate_public_request(scope, require_origin=require_origin)
        if validation_error is not None:
            messages = {
                403: "Origin rejected.",
                421: "Host rejected.",
                503: "Gateway is not ready.",
            }
            return _plain_response(validation_error, messages[validation_error])

        path = str(scope.get("path") or "/")
        if path == AUTH_PATH and method == "POST":
            return await self._authenticate(scope, receive)
        if path == BOOTSTRAP_JS_PATH and method == "GET":
            return self._bootstrap_js_response()

        session = self._request_session(scope)
        if path == SESSION_PATH and method == "GET":
            if session is None:
                return _plain_response(401, "Secure owner session unavailable.")
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        if path == "/" and method == "GET" and session is None:
            return self._bootstrap_response()
        if session is None:
            return _plain_response(401, "Telegram owner session required.")
        if path.startswith(RESERVED_PREFIX):
            return _plain_response(404, "Not found.")
        if method not in _ALLOWED_HTTP_METHODS:
            return _plain_response(
                405,
                "Method not allowed.",
                headers={"Allow": "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS"},
            )
        return await self._proxy_http(scope, receive)

    async def _client_to_upstream(self, client: WebSocket, upstream: Any) -> None:
        while True:
            message = await client.receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                return
            if message_type != "websocket.receive":
                continue
            text = message.get("text")
            data = message.get("bytes")
            if not self._exposure_is_authorized():
                return
            if text is not None:
                if len(text.encode("utf-8")) > _MAX_WS_MESSAGE_BYTES:
                    return
                await upstream.send(text)
            elif data is not None:
                if len(data) > _MAX_WS_MESSAGE_BYTES:
                    return
                await upstream.send(data)

    async def _upstream_to_client(self, upstream: Any, client: WebSocket) -> None:
        async for message in upstream:
            if not self._exposure_is_authorized():
                return
            if isinstance(message, str):
                await client.send_text(message)
            else:
                await client.send_bytes(bytes(message))

    async def _session_guard(self, token: str, expires_at: float) -> None:
        while True:
            now = self._session_clock()
            if (
                not self._exposure_is_authorized()
                or now >= expires_at
                or self._lookup_session(token) is None
            ):
                return
            await asyncio.sleep(min(1.0, max(0.01, expires_at - now)))

    async def _handle_websocket(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        client = WebSocket(scope, receive=receive, send=send)
        validation_error = self._validate_public_request(scope, require_origin=True)
        if validation_error is not None:
            await client.close(code=1013 if validation_error == 503 else 4403)
            return
        session_pair = self._request_session(scope)
        if session_pair is None:
            await client.close(code=4401)
            return
        if str(scope.get("path") or "") != "/ws":
            await client.close(code=4404)
            return
        if scope.get("subprotocols"):
            await client.close(code=4403)
            return
        try:
            upstream_url = self._upstream_url(scope)
        except (UnicodeError, ValueError):
            await client.close(code=4400)
            return
        ws_url = "ws://127.0.0.1:" + str(self._core_port) + str(upstream_url.raw_path, "ascii")
        token, session = session_pair
        if not self._claim_websocket():
            await client.close(code=4429)
            return

        try:
            connect_kwargs: dict[str, Any] = {
                "compression": None,
                "open_timeout": 3,
                "close_timeout": 3,
                "ping_interval": 20,
                "ping_timeout": 20,
                "max_size": _MAX_WS_MESSAGE_BYTES,
                "max_queue": 16,
            }
            try:
                parameters = inspect.signature(self._websocket_connect).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "proxy" in parameters:
                # Automatic proxy support and this opt-out arrived together in
                # websockets 15. Older releases had no automatic proxy and may
                # leak an unknown kwarg into the socket layer.
                connect_kwargs["proxy"] = None
            connector = self._websocket_connect(ws_url, **connect_kwargs)
            async with connector as upstream:
                await client.accept()
                from_client = asyncio.create_task(self._client_to_upstream(client, upstream))
                from_upstream = asyncio.create_task(self._upstream_to_client(upstream, client))
                guard = asyncio.create_task(self._session_guard(token, session.expires_at))
                tasks = {from_client, from_upstream, guard}
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                expired = guard in done
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                try:
                    await client.close(code=4401 if expired else 1000)
                except Exception:
                    pass
        except Exception:
            try:
                await client.close(code=1013)
            except Exception:
                pass
        finally:
            self._release_websocket()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        with self._state_lock:
            self._sessions.clear()

    async def _handle_lifespan(self, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type == "http":
            response = await self._handle_http(scope, receive)
            await response(scope, receive, send)
        elif scope_type == "websocket":
            await self._handle_websocket(scope, receive, send)
        elif scope_type == "lifespan":
            await self._handle_lifespan(receive, send)
        else:
            raise RuntimeError("unsupported ASGI scope")


__all__ = [
    "AUTH_PATH",
    "BOOTSTRAP_JS_PATH",
    "GATEWAY_MARKER_HEADER",
    "MINIAPP_REQUEST_MARKER_HEADER",
    "SESSION_COOKIE",
    "SESSION_PATH",
    "SidecarLimits",
    "SidecarSeams",
    "TelegramInitDataError",
    "TelegramProxySidecar",
    "validate_telegram_init_data",
]
