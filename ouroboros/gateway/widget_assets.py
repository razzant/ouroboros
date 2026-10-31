"""Per-mount, temporary read capabilities: a publication's declared browser files, or GET/HEAD of a
framed widget's own skill routes.

Only NetworkAuthGate admits the two read prefixes without normal network authentication.
Controls stay on /api/widgets/asset-leases, outside every child fetch-bridge prefix.
The gate owns one ephemeral lease table; restart drops it. A file lease never executes a
plugin handler or discovers files at read time. A route lease (``scope: "routes"``, for
classic module and route-iframe mounts) runs only its own skill's GET/HEAD routes, without
the owner's cookies or credentials; ending the lease ends its open responses the way a client
disconnect would; expiry is noticed when such a response next sends or at another lease operation, not by
a timer. The mount's parent may re-establish a route lease under the token its frame already holds (the
table is ephemeral: a core restart or a missed renewal must not cost the running document); a token ended
by Stop (revoke), logout or a password change stays ended in this process, within the last MAX_ENDED such
tokens. Neither kind authenticates any other API.
"""
from __future__ import annotations

import asyncio
import contextlib
import pathlib
import re
import secrets
import threading
import time
import weakref
from dataclasses import dataclass
from urllib.parse import quote, unquote, urljoin, urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ouroboros.extension_liveness import widget_publication_ready
from ouroboros.extension_registry_state import live_widget_files, live_widget_route_generation
from ouroboros.extension_widget_assets import WIDGET_ASSET_TYPES, widget_asset_path
from ouroboros.gateway._helpers import request_drive_root

ASSET_PREFIX = "/api/widget-assets/"
RESOURCE_PREFIX = "/api/widget-resources/"
LEASE_TTL_SECONDS = 120
LEASE_RENEW_SECONDS = 40
MAX_LEASES = 512
MAX_ENDED = 4096
GONE = "gone"
_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Access-Control-Allow-Origin": "null", "Vary": "Origin", "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "sandbox; default-src 'none'"}


def redact_asset_links(text: str) -> str:
    return re.sub(r"(/api/widget-(?:assets|resources)/)[A-Za-z0-9_-]+", r"\1[redacted]", text)


@dataclass(frozen=True)
class _Lease:
    owner: str
    skill: str
    tab_id: str
    generation: str
    drive_root: pathlib.Path
    expires: float
    kind: str = "files"


@dataclass
class _MintAdmission:
    cancelled: bool = False


class WidgetAssetLeases:
    """Ephemeral mount lifecycle; immutable bytes remain owned by the live bundle."""

    def __init__(self):
        self._lock = threading.RLock()
        self._leases: dict[str, _Lease] = {}
        # Open route-lease responses by token; ending the lease ends them (no timer or poller).
        self._streams: dict[str, set] = {}
        # Tokens explicitly ended in this process (revoke, logout, password change): never re-established.
        self._ended: dict[str, None] = {}
        # Only in-flight controls retain admissions. Logout also cancels mints
        # queued for a worker; this is not a persistent session revocation list.
        self._admissions: weakref.WeakValueDictionary[str, _MintAdmission] = weakref.WeakValueDictionary()
        self._password = None

    def admit(self, owner: str) -> _MintAdmission:
        with self._lock:
            admission = self._admissions.get(owner)
            if admission is None:
                admission = _MintAdmission()
                self._admissions[owner] = admission
            return admission

    def password_changed(self, password: str) -> None:
        with self._lock:
            if self._password != password:
                for admission in self._admissions.values():
                    admission.cancelled = True
                self._admissions.clear()
                for token in list(self._leases):
                    self._end(token, final=True)
                self._password = password

    def revoke_owner(self, owner: str) -> None:
        with self._lock:
            admission = self._admissions.pop(owner, None)
            if admission is not None:
                admission.cancelled = True
            for token in [key for key, lease in self._leases.items() if lease.owner == owner]:
                self._end(token, final=True)

    def _end(self, token: str, final: bool = False) -> None:
        self._leases.pop(token, None)
        for stream in self._streams.pop(token, ()):
            stream.cancel()
        if final:
            self._ended[token] = None
            while len(self._ended) > MAX_ENDED:
                del self._ended[next(iter(self._ended))]

    def _prune(self) -> None:
        now = time.monotonic()
        for token in [key for key, lease in self._leases.items() if lease.expires <= now]:
            self._end(token)

    def _files(self, token: str, lease: _Lease):
        if lease.kind == "files" and widget_publication_ready(lease.skill, lease.generation, lease.drive_root):
            files = live_widget_files(lease.skill, lease.tab_id, lease.generation)
            if files is not None:
                return files
        self._end(token)
        return None

    def _routes(self, token: str, lease: _Lease) -> bool:
        """The tab is still a live framed tab of its skill. A first mint also requires the mount's own
        publication; a held or re-established lease (any ``token``) follows the tab's live one."""
        generation = live_widget_route_generation(lease.skill, lease.tab_id)
        if (lease.kind == "routes" and generation and (token or generation == lease.generation)
                and widget_publication_ready(lease.skill, generation, lease.drive_root)):
            return True
        self._end(token)
        return False

    def _ready(self, token: str, lease: _Lease) -> bool:
        return self._routes(token, lease) if lease.kind == "routes" else self._files(token, lease) is not None

    def open(self, owner: str, skill: str, tab_id: str, generation: str, drive_root: pathlib.Path,
             epoch: str, admission: _MintAdmission, kind: str = "files", token: str | None = None):
        """A new lease, or with ``token`` a route lease re-established under the URL its frame already uses:
        the same owner's live lease of that tab is renewed, a lost one is recreated for the tab's current
        publication, and one ended explicitly in this process answers ``GONE``."""
        with self._lock:
            if epoch != self._password or admission.cancelled:
                return None
            self._prune()
            lease = _Lease(owner, skill, tab_id, generation, drive_root, time.monotonic() + LEASE_TTL_SECONDS, kind)
            if token is not None:
                if token in self._ended:
                    return GONE
                held = self._leases.get(token)
                if held is not None and (held.owner, held.skill, held.tab_id, held.kind) != (owner, skill, tab_id, kind):
                    return None
                if not self._routes(token if held is not None else "-", lease):
                    return None
            elif not self._ready("", lease):
                return None
            if token is None or token not in self._leases:
                if len(self._leases) >= MAX_LEASES:
                    return None
                token = token or secrets.token_urlsafe(32)
            self._leases[token] = lease
            return self._description(token, kind)

    @staticmethod
    def _description(token: str, kind: str) -> dict:
        prefix = RESOURCE_PREFIX if kind == "routes" else ASSET_PREFIX
        return {"token": token, "base_url": f"{prefix}{token}/",
                "expires_in": LEASE_TTL_SECONDS, "renew_after": LEASE_RENEW_SECONDS}

    def renew(self, owner: str, token: str, epoch: str):
        with self._lock:
            if epoch != self._password:
                return None
            self._prune()
            lease = self._leases.get(token)
            if lease is None or lease.owner != owner or not self._ready(token, lease):
                return None
            self._leases[token] = _Lease(lease.owner, lease.skill, lease.tab_id, lease.generation,
                                         lease.drive_root, time.monotonic() + LEASE_TTL_SECONDS, lease.kind)
            return self._description(token, lease.kind)

    def revoke(self, owner: str, token: str) -> None:
        """Stop ends a well-formed token for good, held or not: a restart or a prune may already have dropped
        it while a re-establishment is admitted, which must then find it ended. Another owner's live lease is
        left alone."""
        if not _TOKEN.fullmatch(token):
            return
        with self._lock:
            lease = self._leases.get(token)
            if lease is None or lease.owner == owner:
                self._end(token, final=True)

    def read(self, token: str, path: str):
        with self._lock:
            self._prune()
            lease = self._leases.get(token)
            if lease is None or lease.kind != "files":
                return None
            files = self._files(token, lease)
            return files.get(path) if files is not None else None

    def route_skill(self, token: str, stream) -> str | None:
        """The skill a route lease reads, with ``stream`` attached so that ending the lease ends it."""
        with self._lock:
            self._prune()
            lease = self._leases.get(token)
            if lease is None or lease.kind != "routes" or not self._routes(token, lease):
                return None
            self._streams.setdefault(token, set()).add(stream)
            return lease.skill

    def detach(self, token: str, stream) -> None:
        with self._lock:
            streams = self._streams.get(token)
            if streams is not None:
                streams.discard(stream)
                if not streams:
                    del self._streams[token]

    def live(self, token: str) -> bool:
        """Lock-free check for every response message: not ended, revoked or expired."""
        lease = self._leases.get(token)
        return lease is not None and lease.kind == "routes" and lease.expires > time.monotonic()


async def widget_asset_response(request: Request, leases: WidgetAssetLeases) -> Response:
    """Reserved prefix: no fallthrough, cookie fallback, redirects or write methods."""
    token, _, path = request.scope["path"][len(ASSET_PREFIX):].partition("/")
    # Uvicorn's access writer holds this scope. Never log the bearer path/query,
    # including errors, and do not put it into a login redirect's next parameter.
    request.scope["path"] = ASSET_PREFIX + "[redacted]/" + path
    request.scope["raw_path"] = request.scope["path"].encode("utf-8")
    request.scope["query_string"] = b""
    status, message = 404, "widget asset unavailable"
    if request.method not in {"GET", "HEAD"}:
        status, message = 405, "widget assets are read-only"
    elif not _TOKEN.fullmatch(token) or not widget_asset_path(path):
        status, message = 400, "invalid widget asset path"
    else:
        content = await asyncio.to_thread(leases.read, token, path)
        if content is not None:
            headers = {**_HEADERS, "Content-Length": str(len(content))}
            return Response(b"" if request.method == "HEAD" else content,
                            media_type=WIDGET_ASSET_TYPES[pathlib.PurePosixPath(path).suffix], headers=headers)
    return JSONResponse({"error": message}, status_code=status, headers=_HEADERS)


# GET names under the extension prefix that the host serves itself (router.py); a route lease never reaches them.
_HOST_OWNED = frozenset({"manifest", "module", "ui", "settings_section"})
_RESOURCE_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"}
_DROPPED_REQUEST_HEADERS = frozenset({b"cookie", b"authorization", b"proxy-authorization"})
_SET_RESPONSE_HEADERS = ((b"cache-control", b"no-store"), (b"referrer-policy", b"no-referrer"))


class _GrantStream:
    """One response under a route lease. Ending it looks to the handler like its client leaving;
    ``cancel`` is called by the lease's end (any thread) and, as supervised work, by skill unload."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.ended = asyncio.Event()

    def cancel(self) -> None:
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self.ended.set)


def _own_route(rest: str) -> bool:
    """A route path of the own skill: no empty, dot or host-owned segment; a registered route may end in "/"."""
    parts = rest.split("/")
    if len(parts) > 1 and parts[-1] == "":
        parts.pop()
    return (bool(rest) and "\\" not in rest and "\0" not in rest and parts[0] not in _HOST_OWNED
            and not any(part in {"", ".", ".."} for part in parts))


def _granted_location(value: str, own_origin: str, route_url: str, skill_prefix: str, grant_base: str):
    """Where a redirect from a granted response may lead: the own skill's routes become grant URLs, another
    origin stays as it is, and anything else on this origin (another skill, the core) is ``None``: refused."""
    from ouroboros.browser_boundary import _host_port

    target = urlsplit(urljoin(route_url, value))
    scheme = target.scheme.lower()
    # The same origin however it is spelled: `http://host:80` is `http://host` (the browser_boundary rule).
    if f"{scheme}://{_host_port(scheme, target.netloc)}" != own_origin.lower():
        return value
    rest = target.path[len(skill_prefix):] if target.path.startswith(skill_prefix) else ""
    if not _own_route(unquote(rest)) or any(unquote(part) in {".", ".."} for part in rest.split("/")):
        return None
    return grant_base + rest + (f"?{target.query}" if target.query else "") + (f"#{target.fragment}" if target.fragment else "")


async def widget_resource_response(scope, receive, send, leases: WidgetAssetLeases, app) -> None:
    """Reserved prefix: GET/HEAD of the leased widget's own skill route, dispatched by the ordinary router.

    No cookie or credential header reaches the handler. Its own-prefix redirects are rewritten under the
    grant; one to another skill or the core is refused. Revocation, expiry, logout, password change and
    skill unload end an open response through its ``receive``/``send``, as a client disconnect would."""
    from ouroboros.browser_boundary import _own_origin
    from ouroboros.contracts.plugin_api import ExtensionRegistrationError
    from ouroboros.extension_registry_state import extension_work_scope, list_routes

    token, _, rest = scope["path"][len(RESOURCE_PREFIX):].partition("/")
    query = scope.get("query_string") or b""
    # The access writer holds this scope: never log the bearer path or the query.
    scope["path"] = RESOURCE_PREFIX + "[redacted]/" + rest
    scope["raw_path"] = scope["path"].encode("utf-8")
    scope["query_string"] = b""

    async def refuse(status: int, message: str) -> None:
        await JSONResponse({"error": message}, status_code=status, headers=_RESOURCE_HEADERS)(scope, receive, send)

    if scope.get("method") not in {"GET", "HEAD"}:
        return await refuse(405, "widget resources are read-only")
    if not _TOKEN.fullmatch(token) or not _own_route(rest):
        return await refuse(400, "invalid widget resource path")
    stream = _GrantStream()
    skill = await asyncio.to_thread(leases.route_skill, token, stream)
    if skill is None:
        return await refuse(404, "widget resource unavailable")
    skill_prefix = f"/api/extensions/{skill}/"
    route_path = skill_prefix + rest
    own_origin = _own_origin(scope)
    grant_base = f"{RESOURCE_PREFIX}{token}/"
    child = {**scope, "path": route_path, "raw_path": quote(route_path).encode("ascii"), "query_string": query,
             "headers": [(k, v) for k, v in scope.get("headers") or () if k.lower() not in _DROPPED_REQUEST_HEADERS]}
    sent = {"start": False, "complete": False, "length": False, "refused": False}

    async def guarded_receive():
        if stream.ended.is_set():
            return {"type": "http.disconnect"}
        message = asyncio.ensure_future(receive())
        ended = asyncio.ensure_future(stream.ended.wait())
        try:
            await asyncio.wait({message, ended}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            ended.cancel()
            if not message.done():
                message.cancel()
        return message.result() if not message.cancelled() else {"type": "http.disconnect"}

    async def guarded_send(message):
        if sent["refused"]:
            return
        if stream.ended.is_set() or not leases.live(token):
            stream.ended.set()
            raise OSError("widget resource lease ended")
        if message["type"] == "http.response.start":
            headers = []
            for key, value in message.get("headers") or ():
                name = key.lower()
                if name in {b"cache-control", b"referrer-policy"}:
                    continue
                if name == b"location":
                    location = _granted_location(value.decode("latin-1"), own_origin, own_origin + route_path,
                                                 skill_prefix, grant_base)
                    if location is None:
                        sent["refused"] = sent["start"] = sent["complete"] = True
                        return await refuse(403, "a widget resource redirect must stay under its own skill")
                    value = location.encode("latin-1")
                sent["length"] = sent["length"] or name == b"content-length"
                headers.append((key, value))
            message = {**message, "headers": [*headers, *_SET_RESPONSE_HEADERS]}
            sent["start"] = True
        elif message["type"] == "http.response.body" and not message.get("more_body", False):
            sent["complete"] = True
        await send(message)

    spec = list_routes().get(route_path)
    try:
        with extension_work_scope(spec, stream, call_work=False) if spec else contextlib.nullcontext():
            await app(child, guarded_receive, guarded_send)
    except ExtensionRegistrationError:
        if not sent["start"]:
            await refuse(404, "widget resource unavailable")
    except Exception:
        if not stream.ended.is_set():
            raise
    finally:
        leases.detach(token, stream)
    if stream.ended.is_set() and not sent["complete"]:
        # An ended chunked body closes cleanly; a declared length cannot, so the server drops the connection.
        if not sent["start"]:
            await refuse(404, "widget resource unavailable")
        elif not sent["length"]:
            with contextlib.suppress(Exception):
                await send({"type": "http.response.body", "body": b"", "more_body": False})


async def api_widget_bridge_script(request: Request) -> Response:
    """The frame bridge as one public classic script: a route page loads it to talk to its mount (its own CSP
    admits it by nonce or 'self'). Static host code with no authority; the frame proves itself to the parent."""
    from ouroboros.gateway._helpers import request_repo_dir

    path = request_repo_dir(request) / "web" / "modules" / "widget_frame_bridge.js"
    source = await asyncio.to_thread(path.read_bytes)
    return Response(source, media_type="application/javascript; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff",
                             "Access-Control-Allow-Origin": "*"})


def _from_parent_page(request: Request) -> bool:
    """Same-origin parent page, never the opaque frame or another site (``browser_boundary``)."""
    from ouroboros.browser_boundary import same_origin_browser

    return same_origin_browser(request.scope)


async def api_widget_asset_lease(request: Request) -> Response:
    """Parent-authenticated JSON controls; opaque and cross-site browser requests are refused."""
    if not _from_parent_page(request):
        return JSONResponse({"error": "widget lease controls require the parent origin"}, status_code=403)
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        return JSONResponse({"error": "expected JSON"}, status_code=400)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > 2048:
            return JSONResponse({"error": "lease request too large"}, status_code=413)
    import json

    try:
        body = json.loads(raw)
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected lease object"}, status_code=400)
    leases = request.scope["widget_asset_leases"]
    owner = request.scope["widget_asset_owner"]
    epoch = request.scope["widget_asset_epoch"]
    op = request.path_params.get("op")
    fields = ("skill", "tab_id", "generation") if op is None else ("token",)
    # ``scope: "routes"`` mints the distinct own-route lease; a file lease never gains it. With ``token`` the
    # parent re-establishes the route lease its running frame already uses.
    kind = body.pop("scope", "files") if op is None else "files"
    token = body.pop("token", None) if op is None and kind == "routes" else None
    if (kind not in {"files", "routes"} or set(body) != set(fields)
            or not all(isinstance(body[k], str) and 0 < len(body[k]) <= 128 for k in fields)
            or (token is not None and not (isinstance(token, str) and _TOKEN.fullmatch(token)))):
        return JSONResponse({"error": "invalid lease fields"}, status_code=400)
    if op is None:
        result = await asyncio.to_thread(leases.open, owner, body["skill"], body["tab_id"],
                                         body["generation"], request_drive_root(request), epoch,
                                         request.scope["widget_asset_admission"], kind, token)
        if result == GONE:
            return JSONResponse({"error": "this widget lease was ended"}, status_code=410, headers={"Cache-Control": "no-store"})
    elif op == "renew":
        result = await asyncio.to_thread(leases.renew, owner, body["token"], epoch)
    elif op == "revoke":
        leases.revoke(owner, body["token"])
        result = {"ok": True}
    else:
        return JSONResponse({"error": "unknown lease operation"}, status_code=404)
    return JSONResponse(result or {"error": "widget publication or lease unavailable"},
                        status_code=200 if result else 409, headers={"Cache-Control": "no-store"})
