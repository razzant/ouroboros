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
like a native client's and runs its handler (its answer stays unreadable to that page). Chromium
sends no Fetch Metadata with a WebSocket handshake at all, so behind a proxy the core's own socket
passes only when the browser's Host and HTTPS scheme reach this server (DEPLOYMENT).

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
from typing import Any
from urllib.parse import quote

log = logging.getLogger(__name__)

Scope = dict[str, Any]

CROSS_ORIGIN_CODE = "cross_origin_request"
_DEFAULT_PORTS = {"http": "80", "https": "443"}
_ORIGIN = re.compile(r"(https?)://([^/?#@\s]+)")
_PRINTABLE = re.compile(r"[\x21-\x7e]{1,300}")  # a logged value: visible ASCII, bounded
_PASSIVE_EXACT = frozenset({"/", "/onboarding", "/auth/login", "/api/widget-bridge.js"})
_PASSIVE_PREFIXES = ("/static/", "/onboarding/static/", "/auth/", "/api/widget-assets/")
# Exactly the two static extension-file routes (gateway/router.py): module/{entry:path} and
# ui/{revision}/{path:path}. A shorter shape such as ui/action is a registered dynamic route, not a
# file, so it must stay behind the boundary; only these two reach a passive file handler.
_EXTENSION_FILES = re.compile(r"/api/extensions/[^/]+/(?:module/|ui/[^/]+/)")
_refused_seen: set[tuple[str, str, str, str]] = set()


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
    Without it (a Chromium WebSocket handshake included) ``Origin`` must name the URL this
    server sees. ``Origin: null`` (an opaque frame or a sandboxed page) and a malformed Origin
    never pass; a request with neither header is taken for a native client's (see the module
    note on plain-HTTP hostnames)."""
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


def _logged_origin(origin: str | None) -> str:
    """An origin as compared, for the log: ``scheme://host[:port]``, ``null`` or ``-``; anything else
    (credentials, a path or query, odd bytes, an overlong name) is ``malformed``, never its raw text."""
    if origin is None:
        return "-"
    match = _ORIGIN.fullmatch(origin.lower())
    if match is None:
        return "null" if origin.lower() == "null" else "malformed"
    shown = f"{match.group(1)}://{_host_port(match.group(1), match.group(2))}"
    return shown if _PRINTABLE.fullmatch(shown) else "malformed"


def browser_request_refused(scope: Scope) -> bool:
    """Whether the boundary refuses this HTTP request or WebSocket handshake.

    Logged once per received origin, the origin this server sees (``expected``, so a corrected Host or
    scheme shows), Fetch Metadata site and path: sanitized and bounded, never a raw header."""
    if passive_file(scope) or same_origin_browser(scope):
        return False
    site = (_header(scope, b"sec-fetch-site") or "-").lower()
    path = quote(scope.get("path") or "/", safe="/:@!$&'()*+,;=-._~")
    key = (_logged_origin(_header(scope, b"origin")), _logged_origin(_own_origin(scope)),
           site if _PRINTABLE.fullmatch(site) and len(site) <= 32 else "malformed",
           path if len(path) <= 200 else f"{path[:200]}…")
    if key not in _refused_seen and len(_refused_seen) < 256:
        _refused_seen.add(key)
        log.warning("browser_boundary refused %s %s origin=%s expected=%s sec-fetch-site=%s",
                    scope.get("method") or "WEBSOCKET", key[3], key[0], key[1], key[2])
    return True
