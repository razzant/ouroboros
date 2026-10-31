"""Which browser requests reach the core: the local API's browser boundary.

A page on another site that the owner happens to open can make the browser send requests here
(form posts, ``no-cors`` fetches, images, WebSockets) even though it cannot read the answers.
The browser's own ``Sec-Fetch-Site`` and ``Origin`` headers, which no page script can set, tell
those apart from the core's own pages when the browser sends them. Native clients (the CLI, the
launcher, sidecars, skills, Android's native side) send neither header and pass. ``NetworkAuthGate``
applies the boundary to every HTTP request and WebSocket handshake before any handler runs, on
loopback too.

A limit, not a guarantee: browsers send Fetch Metadata only to potentially trustworthy origins
(HTTPS, ``localhost``, loopback addresses) and omit ``Origin`` on simple GET/HEAD loads. When the
core is reached through an ordinary plain-HTTP hostname, another site's GET/HEAD therefore looks
like a native client's and runs its handler (its answer stays unreadable to that page).

Only passive files stay reachable from any page: the interface and setup pages, their static
files, the login form, the widget frame bridge script, and the reviewed module and declared files of
extensions. Hosts are not
restricted (an accepted residual): a page whose own hostname resolves to this machine (DNS
rebinding) is same-origin here, and when it resolves to loopback the network password does not
apply either.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable

from starlette.routing import compile_path

log = logging.getLogger(__name__)

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]

CROSS_ORIGIN_CODE = "cross_origin_request"
JSON_CONTENT_TYPE_CODE = "json_content_type_required"
_DEFAULT_PORTS = {"http": "80", "https": "443"}
_ORIGIN = re.compile(r"(https?)://([^/?#@\s]+)")
_PASSIVE_EXACT = frozenset({"/", "/onboarding", "/auth/login", "/api/widget-bridge.js"})
_PASSIVE_PREFIXES = ("/static/", "/onboarding/static/", "/auth/", "/api/widget-assets/")
# Exactly the two static extension-file routes (gateway/router.py): module/{entry:path} and
# ui/{revision}/{path:path}. A shorter shape such as ui/action is a registered dynamic route, not a
# file, so it must stay behind the boundary; only these two reach a passive file handler.
_EXTENSION_FILES = re.compile(r"/api/extensions/[^/]+/(?:module/|ui/[^/]+/)")
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_JSON_MEDIA = re.compile(r"application/(?:[\w.+-]+\+)?json")
_refused_seen: set[tuple[str, str, str]] = set()


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1").strip()
    return None


def _host_port(scheme: str, netloc: str) -> str:
    netloc = netloc.lower()
    host, sep, port = netloc.rpartition(":")
    if sep and "]" not in port and port == _DEFAULT_PORTS.get(scheme):
        return host
    return netloc


def _own_origin(scope: Scope) -> str:
    """``scheme://host[:port]`` of the URL this server sees, as Starlette derives it (no X-Forwarded-Host)."""
    scheme = {"ws": "http", "wss": "https"}.get(scope.get("scheme") or "http", scope.get("scheme") or "http")
    netloc = _header(scope, b"host")
    if not netloc:
        server = scope.get("server") or ("", None)
        netloc = f"{server[0]}:{server[1]}" if server[1] is not None else str(server[0])
    return f"{scheme}://{_host_port(scheme, netloc)}"


def same_origin_browser(scope: Scope) -> bool:
    """A request from one of this server's own pages, or from no browser at all.

    The browser's ``Sec-Fetch-Site`` decides when present (``same-origin``, or ``none`` for a
    navigation the owner typed), even behind a proxy that rewrites Host or terminates TLS.
    Without it ``Origin`` must name the URL this server sees. ``Origin: null`` (an opaque
    frame or a sandboxed page) and a malformed Origin never pass; a request with neither
    header is taken for a native client's (see the module note on plain-HTTP hostnames)."""
    origin = _header(scope, b"origin")
    site = _header(scope, b"sec-fetch-site")
    match = _ORIGIN.fullmatch(origin.lower()) if origin is not None else None
    if origin is not None and match is None:
        return False
    if site is not None:
        return site.lower() in {"same-origin", "none"}
    if match is None:
        return True
    scheme = match.group(1)
    return f"{scheme}://{_host_port(scheme, match.group(2))}" == _own_origin(scope)


def passive_file(scope: Scope) -> bool:
    """A GET/HEAD of a file any page may load: pages, static bundles, the login form, extension files."""
    if scope.get("type") != "http" or scope.get("method") not in {"GET", "HEAD"}:
        return False
    path = scope.get("path") or "/"
    if path == "/auth/logout":
        return False
    return path in _PASSIVE_EXACT or path.startswith(_PASSIVE_PREFIXES) or bool(_EXTENSION_FILES.match(path))


def browser_request_refused(scope: Scope) -> bool:
    """Whether the boundary refuses this HTTP request or WebSocket handshake (logged once per origin and path)."""
    if passive_file(scope) or same_origin_browser(scope):
        return False
    key = (_header(scope, b"origin") or "", _header(scope, b"sec-fetch-site") or "", scope.get("path") or "")
    if key not in _refused_seen and len(_refused_seen) < 256:
        _refused_seen.add(key)
        log.warning("browser_boundary refused %s %s origin=%s sec-fetch-site=%s",
                    scope.get("method") or "WEBSOCKET", key[2], key[0] or "-", key[1] or "-")
    return True


def _non_json_routes() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """``(method, path)`` of each endpoint whose published request body is not JSON (``endpoint_index``:
    uploads, forms, extension routes' ``any``); ``ANY`` stands for every method."""
    from ouroboros.gateway.endpoint_index import AUTH_ENDPOINTS, HTTP_SHAPES

    routes = {tuple(token.split(" ", 1)) for token, shape in {**HTTP_SHAPES, **AUTH_ENDPOINTS}.items()
              if set(shape.request.split("|")) & {"multipart", "form", "any", "text", "file"}}
    return tuple((method, compile_path(path)[0]) for method, path in sorted(routes))


_NON_JSON: tuple[tuple[str, re.Pattern[str]], ...] | None = None


def _json_contract(scope: Scope) -> bool:
    """A host API body route: an ``/api`` method and path whose published body is JSON."""
    global _NON_JSON
    path, method = scope.get("path") or "/", scope.get("method")
    if method not in _BODY_METHODS or not path.startswith("/api/"):
        return False
    if _NON_JSON is None:
        _NON_JSON = _non_json_routes()
    return not any(declared in {method, "ANY"} and pattern.match(path) for declared, pattern in _NON_JSON)


async def json_body_refused(scope: Scope, receive: Receive) -> tuple[bool, Receive]:
    """Whether a JSON contract got a non-empty body without a JSON Content-Type, and the ``receive`` to pass on.

    An empty body needs no type (body-less POSTs, a proxy's empty chunked stream). An unknown
    length is decided by the first body chunk, which is replayed to the handler unchanged."""
    content_type = (_header(scope, b"content-type") or "").split(";", 1)[0].strip().lower()
    if not _json_contract(scope) or _JSON_MEDIA.fullmatch(content_type):
        return False, receive
    length = _header(scope, b"content-length")
    if length is not None:
        return length.strip() not in {"", "0"}, receive
    held: list[dict[str, Any]] = []
    while True:
        message = await receive()
        held.append(message)
        if message.get("type") != "http.request" or message.get("body") or not message.get("more_body"):
            break

    async def replay() -> dict[str, Any]:
        return held.pop(0) if held else await receive()

    return bool(held[-1].get("body")) and held[-1].get("type") == "http.request", replay
