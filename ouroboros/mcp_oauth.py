"""Host-owned sign-in for remote HTTP MCP servers (MCP authorization, revision 2026-07-28).

An HTTP/SSE server that carries no saved credential and answers ``401`` with a ``Bearer`` challenge
asks for the owner's sign-in. Its transport carries one stored bearer (``StoredBearer``): refreshed
when the recorded expiry passed or after a ``401``, under a cross-process lock with a re-read (the
listing runs in the server process, calls in workers, and public clients' refresh tokens rotate);
with no grant, a rejected refresh or a repeated ``401`` it refuses at once with ``MCPAuthRequired``,
and a ``403 insufficient_scope`` records the wider scope the next sign-in asks for, keeping the grant
for what it already allows. Nothing here opens a browser or starts a sign-in on its own; when to ask
the owner is the mind's judgment. A sign-in is two explicit steps any process may run: ``begin``
(discovery, the client, PKCE; the pending sign-in, with the issuer and endpoints it was made for, is
saved in the server's file and a later begin replaces it) and ``complete`` (the provider's return, from
the builtin return page or an address the owner pasted). One file per server holds everything:
``state/mcp/oauth/server-<id>.json``, mode 0600 in a 0700 directory, removed by sign-out and by
configuration cleanup; a draft of another address (Settings Test) never replaces it. Only public
helpers of the MCP SDK are used: its ``OAuthClientProvider`` runs the interactive flow inside a call
and forgets the expiry it learned.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import weakref
from typing import Any, Dict, Optional, Tuple

import httpx

from ouroboros import config
from ouroboros.runtime_limits import MCP_OAUTH_HTTP_TIMEOUT_SEC, MCP_OAUTH_LOCK_WAIT_SEC
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

RETURN_PATH = "/onboarding/static/provider-return.html"
SIGNED_IN, SIGN_IN_REQUIRED = "signed_in", "sign_in_required"
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class MCPAuthRequired(RuntimeError):
    """The server needs the owner's sign-in; the server ran nothing for this request."""

    code = "MCP_AUTH_REQUIRED"

    def __init__(self, server_id: str, reason: str, detail: str = "") -> None:
        self.server_id, self.reason, self.detail = server_id, reason, detail
        super().__init__(
            f"MCP_AUTH_REQUIRED: server {server_id!r} needs the owner's sign-in ({reason}"
            f"{': ' + detail if detail else ''}). Nothing ran there. The owner can press Sign in on its card in "
            f"Settings → Tools, or remote_mcp_sign_in(server={server_id!r}, action='begin') (enable_tools loads it) issues "
            "a sign-in link to give them.")


def applies(cfg: Any) -> bool:
    """An HTTP/SSE server without a saved credential: its ``401`` Bearer challenge means sign-in."""
    return cfg.transport in ("streamable_http", "sse") and not cfg.carries_credentials()


# -- the server's file -------------------------------------------------------------------------


def _path(server_id: str):
    return config.DATA_DIR / "state" / "mcp" / "oauth" / f"server-{server_id}.json"


def _load(cfg: Any) -> Dict[str, Any]:
    """The server's sign-in file; a file written for another address of the same id is not its grant."""
    doc = read_json_dict(_path(cfg.id)) or {}
    return doc if doc.get("url") == cfg.url else {}


_DRAFT: "contextvars.ContextVar[bool]" = contextvars.ContextVar("mcp_oauth_draft", default=False)


@contextlib.contextmanager
def drafting():
    """A Settings Test of an unsaved draft: it uses (and refreshes) the grant of its own address, and never
    writes over the file of another address, such as the saved server's grant beside an edited URL."""
    token = _DRAFT.set(True)
    try:
        yield
    finally:
        _DRAFT.reset(token)


def _save(cfg: Any, doc: Dict[str, Any]) -> None:
    path = _path(cfg.id)
    recorded = (read_json_dict(path) or {}).get("url")
    if _DRAFT.get() and recorded and recorded != cfg.url:
        return  # a draft never replaces another address's grant; a saved address always writes (Save prunes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path.parent, 0o700)  # a rewrite's temporary file is never readable by another account
    atomic_write_json(path, {**doc, "v": 1, "url": cfg.url})
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


# Per event loop, per server; a lock lives only while a coroutine holds or awaits it (a bound lock refers to
# its loop, so a strongly held one would keep every closed loop alive).
_LOOP_LOCKS: "weakref.WeakKeyDictionary[Any, weakref.WeakValueDictionary]" = weakref.WeakKeyDictionary()
_LOOP_LOCKS_GUARD = threading.Lock()


@contextlib.asynccontextmanager
async def _locked(cfg: Any):
    """One refresh or sign-in step at a time per server. Coroutines of one event loop (a session's GET and
    POST) queue on an asyncio lock, so a waiter never blocks the loop that runs the holder's await; the
    owner-aware file lock across threads and processes (``store.name_lock``) is then polled off it."""
    from ouroboros import store

    with _LOOP_LOCKS_GUARD:
        locks = _LOOP_LOCKS.setdefault(asyncio.get_running_loop(), weakref.WeakValueDictionary())
        lock = locks.get(cfg.id)
        if lock is None:
            lock = locks[cfg.id] = asyncio.Lock()
    async with lock:
        deadline = time.monotonic() + MCP_OAUTH_LOCK_WAIT_SEC
        while True:
            held = store.name_lock(config.DATA_DIR, f"mcp/oauth/server-{cfg.id}.lock", timeout_sec=0)
            try:
                held.__enter__()  # one attempt: a busy name never blocks this loop
                break
            except store.StoreBusy:
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.05)
        try:
            yield
        finally:
            held.__exit__(None, None, None)


def _required(cfg: Any, doc: Dict[str, Any], reason: str, detail: str = "", **facts: Any) -> MCPAuthRequired:
    """Record that sign-in is needed (with what the server said) and return the refusal to raise."""
    doc = {**doc, **facts, "sign_in": {"state": SIGN_IN_REQUIRED, "reason": reason, "since": utc_now_iso()}}
    if reason in ("refresh_rejected", "token_rejected", "expired"):
        doc.pop("tokens", None)  # a grant the provider no longer honours is not kept as if it were
    _save(cfg, doc)
    return MCPAuthRequired(cfg.id, reason, detail)


def status(cfg: Any) -> Optional[Dict[str, Any]]:
    """``{state, reason?, required_scope?, since, issuer, pending?}`` once the server asked for sign-in;
    else ``None``. A grant that needs a wider scope stays signed in with reason ``insufficient_scope``."""
    if not applies(cfg):
        return None
    doc = _load(cfg)
    if not doc:
        return None
    signed_in = bool((doc.get("tokens") or {}).get("access_token"))
    record = doc.get("sign_in") or {}
    out: Dict[str, Any] = {"state": SIGNED_IN if signed_in else SIGN_IN_REQUIRED, "since": record.get("since"),
                           "issuer": doc.get("issuer") or ""}
    if not signed_in:
        out["reason"] = record.get("reason") or "not_signed_in"
    elif doc.get("required_scope"):
        out.update(reason="insufficient_scope", required_scope=doc["required_scope"])
    if doc.get("pending"):
        out["pending"] = True
    return out


_SECRETS: Dict[str, Tuple[int, tuple]] = {}


def secret_values(server_id: str) -> tuple:
    """This server's live sign-in secrets, for the masking of every text from or about it."""
    path = _path(server_id)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return ()
    cached = _SECRETS.get(server_id)
    if cached and cached[0] == stamp:
        return cached[1]
    doc = read_json_dict(path) or {}
    pending = doc.get("pending") or {}
    values = ((doc.get("tokens") or {}).get("access_token"), (doc.get("tokens") or {}).get("refresh_token"),
              (doc.get("client") or {}).get("client_secret"), pending.get("code_verifier"),
              (pending.get("client") or {}).get("client_secret"))
    _SECRETS[server_id] = (stamp, tuple(value for value in values if value))
    return _SECRETS[server_id][1]


def prune(configs: list) -> None:
    """Configuration-event remover (beside ``mcp_catalog.prune_catalogs``): a removed server, or one whose
    address changed, loses its sign-in file. Its lock file is never touched here: the holder removes it on
    release, and a refresh still holding it must keep later steps waiting until its own write is done."""
    directory = config.DATA_DIR / "state" / "mcp" / "oauth"
    wanted = {cfg.id: cfg.url for cfg in configs}
    with contextlib.suppress(OSError):
        for path in directory.glob("server-*.json"):
            server_id = path.name[len("server-"):-len(".json")]
            if wanted.get(server_id) != (read_json_dict(path) or {}).get("url"):
                path.unlink(missing_ok=True)


# -- the authorization server -------------------------------------------------------------------


def _client() -> httpx.AsyncClient:
    """A short-lived client for the authorization server: never the transport's, never its headers."""
    return httpx.AsyncClient(timeout=MCP_OAUTH_HTTP_TIMEOUT_SEC, follow_redirects=False)


def _checked(url: str) -> str:
    """An endpoint the (untrusted) MCP server named: HTTPS or loopback, and the address policy of servers."""
    from ouroboros.mcp_client import _validate_url

    text = _validate_url(str(url))
    if urllib.parse.urlparse(text).scheme != "https" and (urllib.parse.urlparse(text).hostname or "") not in _LOOPBACK:
        raise ValueError(f"authorization endpoint {text!r} is not HTTPS")
    return text


def _oauth_error(response: httpx.Response) -> str:
    with contextlib.suppress(ValueError):
        body = response.json()
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])
    return f"HTTP {response.status_code}"


async def _token_request(http: httpx.AsyncClient, endpoint: str, client: Dict[str, Any],
                         data: Dict[str, str]) -> httpx.Response:
    """A token-endpoint (or revocation) POST with the client's own authentication method."""
    data, auth = {**data, "client_id": client.get("client_id", "")}, None
    if client.get("client_secret"):
        if client.get("token_endpoint_auth_method") == "client_secret_basic":
            auth = httpx.BasicAuth(urllib.parse.quote(client["client_id"], safe=""),
                                   urllib.parse.quote(client["client_secret"], safe=""))
        else:
            data["client_secret"] = client["client_secret"]
    return await http.post(_checked(endpoint), data=data, auth=auth, headers={"Accept": "application/json"})


def _tokens(token: Any, previous_refresh: Optional[str], previous_scope: Optional[str]) -> Dict[str, Any]:
    """Absolute expiry (the SDK keeps only a relative one); the previous refresh token and scope when a
    response omits them (RFC 6749 §5.1 and §6: then they are unchanged)."""
    return {"access_token": token.access_token, "token_type": token.token_type or "Bearer",
            "refresh_token": token.refresh_token or previous_refresh, "scope": token.scope or previous_scope,
            "expires_at": time.time() + token.expires_in if token.expires_in else None}


def _same_registration(one: Dict[str, Any], other: Dict[str, Any]) -> bool:
    """One client registration: same issuer, id and authentication (a new secret is another registration)."""
    keys = ("issuer", "client_id", "client_secret", "token_endpoint_auth_method")
    return bool(one) and bool(other) and all(one.get(key) == other.get(key) for key in keys)


def _fresh(tokens: Dict[str, Any], rejected: Optional[str]) -> bool:
    return bool(tokens.get("access_token")) and tokens["access_token"] != rejected and (
        tokens.get("expires_at") is None or time.time() < float(tokens["expires_at"]))


async def current_token(cfg: Any, rejected: Optional[str] = None) -> str:
    """A usable access token: the stored one, or one refresh under the lock (re-read first: another
    process may already hold a newer token). Raises ``MCPAuthRequired`` when sign-in is needed."""
    from mcp.shared.auth import OAuthToken

    async with _locked(cfg):
        doc = _load(cfg)
        tokens = doc.get("tokens") or {}
        if not tokens.get("access_token"):
            raise MCPAuthRequired(cfg.id, (doc.get("sign_in") or {}).get("reason") or "not_signed_in")
        if _fresh(tokens, rejected):
            return tokens["access_token"]
        if not tokens.get("refresh_token"):
            raise _required(cfg, doc, "token_rejected" if rejected else "expired")
        data = {"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"],
                **({"resource": doc["resource"]} if doc.get("resource") else {})}
        async with _client() as http:
            response = await _token_request(http, (doc.get("metadata") or {}).get("token_endpoint", ""),
                                            doc.get("client") or {}, data)
        if response.status_code in (400, 401):
            error = _oauth_error(response)
            if error == "invalid_request":  # a malformed request, not a refused grant: the grant stays
                response.raise_for_status()
            # invalid_grant (expired or revoked), invalid_client (the registration is gone: register again,
            # so a pending sign-in made with the same registration goes too)
            facts: Dict[str, Any] = {}
            if error == "invalid_client":
                facts["client"] = None
                if _same_registration((doc.get("pending") or {}).get("client") or {}, doc.get("client") or {}):
                    facts["pending"] = None
            raise _required(cfg, doc, "refresh_rejected", error, **facts)
        response.raise_for_status()  # an unreachable authorization server is a failure, not a lost sign-in
        doc["tokens"] = _tokens(OAuthToken.model_validate_json(response.content), tokens.get("refresh_token"),
                                tokens.get("scope"))
        _save(cfg, doc)
        return doc["tokens"]["access_token"]


def _challenge(response: httpx.Response) -> Dict[str, Any]:
    """The ``WWW-Authenticate: Bearer`` parameters of a 401/403; empty for another scheme or none."""
    from mcp.client.auth.utils import (extract_field_from_www_auth, extract_resource_metadata_from_www_auth,
                                       extract_scope_from_www_auth)

    if response.status_code not in (401, 403) or not response.headers.get("WWW-Authenticate", "").strip().lower(
            ).startswith("bearer"):
        return {}
    facts = {"resource_metadata": extract_resource_metadata_from_www_auth(response),
             "scope": extract_scope_from_www_auth(response), "error": extract_field_from_www_auth(response, "error")}
    return {key: value for key, value in facts.items() if value} or {"bearer": True}


class StoredBearer(httpx.Auth):
    """The transport's only OAuth part: the stored bearer on requests to the server's own origin; after a
    Bearer ``401`` one refresh and one retry; a ``403 insufficient_scope`` (first answer or retried one)
    records the scope it needs; otherwise ``self.refusal``. A bare ``401`` (no Bearer challenge) passes
    through as today's error. A refusal also ends the connection (``clients``), so a session waiting on
    another request (an SSE stream) fails at once instead of at its tool timeout."""

    def __init__(self, cfg: Any) -> None:
        self.cfg, self.refusal, self.clients = cfg, None, []

    def _own_challenge(self, response: httpx.Response) -> Dict[str, Any]:
        from ouroboros.mcp_headers import origin_of

        # The answering origin (a redirect may have left the server's): only the server's own challenge counts.
        return _challenge(response) if origin_of(response.request.url) == origin_of(self.cfg.url) else {}

    async def _refused(self, exc: MCPAuthRequired) -> MCPAuthRequired:
        exc.mcp_unsent_joined = True  # the server answered before running anything
        self.refusal = exc
        for client in list(self.clients):
            with contextlib.suppress(Exception):
                await client.aclose()
        return exc

    async def async_auth_flow(self, request: httpx.Request):
        from ouroboros.mcp_headers import origin_of

        own = origin_of(request.url) == origin_of(self.cfg.url)
        tokens = (_load(self.cfg).get("tokens") or {}) if own else {}
        token = tokens.get("access_token")
        if token and not _fresh(tokens, None):
            try:
                token = await current_token(self.cfg)
            except MCPAuthRequired as exc:
                raise await self._refused(exc) from None
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        response = yield request
        challenge = self._own_challenge(response)
        if response.status_code == 401 and challenge:
            try:
                async with _locked(self.cfg):
                    doc = _load(self.cfg)
                    if not (doc.get("tokens") or {}).get("access_token"):
                        raise _required(self.cfg, doc, (doc.get("sign_in") or {}).get("reason") or "not_signed_in",
                                        challenge=challenge)
                    _save(self.cfg, {**doc, "challenge": challenge})
                token = await current_token(self.cfg, rejected=token)
            except MCPAuthRequired as exc:
                raise await self._refused(exc) from None
            request.headers["Authorization"] = f"Bearer {token}"
            response = yield request
            challenge = self._own_challenge(response)
            if response.status_code == 401 and challenge:
                async with _locked(self.cfg):
                    exc = _required(self.cfg, _load(self.cfg), "token_rejected")
                raise await self._refused(exc) from None
        if response.status_code == 403 and challenge.get("error") == "insufficient_scope":
            async with _locked(self.cfg):
                doc = _load(self.cfg)
                wanted = _union((doc.get("tokens") or {}).get("scope"), doc.get("required_scope"), challenge.get("scope"))
                _save(self.cfg, {**doc, "challenge": challenge, "required_scope": wanted})
            raise await self._refused(MCPAuthRequired(self.cfg.id, "insufficient_scope", challenge.get("scope") or ""))


def _union(*scopes: Optional[str]) -> Optional[str]:
    words = [word for scope in scopes if scope for word in scope.split()]
    return " ".join(dict.fromkeys(words)) or None


# -- the two sign-in steps ----------------------------------------------------------------------


def _server_port() -> int:
    """The live main-server port (its port file), else the configured one."""
    from ouroboros.config import AGENT_SERVER_PORT, resolve_port_file

    with contextlib.suppress(OSError, ValueError):
        port = int(resolve_port_file().read_text(encoding="utf-8").strip())
        if port > 0:
            return port
    return int(os.environ.get("OUROBOROS_SERVER_PORT") or AGENT_SERVER_PORT)


def redirect_uri(return_base: str = "") -> str:
    """The builtin return page under ``return_base`` when that is loopback or HTTPS (an MCP rule), else
    under this server's loopback address; a browser elsewhere then hands its address back by paste."""
    base = (return_base or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if not (parsed.scheme == "https" or (parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK)):
        base = f"http://127.0.0.1:{_server_port()}"
    return base + RETURN_PATH


def _state(server_id: str, nonce: str) -> str:
    raw = json.dumps({"v": 1, "mcp": server_id, "nonce": nonce}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _envelope(state: str) -> Dict[str, Any]:
    try:
        envelope = json.loads(base64.urlsafe_b64decode(str(state) + "=" * (-len(str(state)) % 4)).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("not an MCP sign-in return") from exc
    if not isinstance(envelope, dict) or envelope.get("v") != 1 or not envelope.get("mcp"):
        raise ValueError("not an MCP sign-in return")
    return envelope


def server_of(state: str) -> str:
    """The server a returned ``state`` names (route syntax only: ``complete`` checks the nonce)."""
    return str(_envelope(state)["mcp"])


def parse_return(address: str) -> Dict[str, str]:
    """The provider's return parameters from the address its browser stopped on (a paste)."""
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(str(address or "").strip()).query))
    if not query.get("state") or not (query.get("code") or query.get("error")):
        raise ValueError("the address carries no complete sign-in return (state and code)")
    return {key: query[key] for key in ("state", "code", "error", "error_description", "iss") if key in query}


async def _discover(cfg: Any, doc: Dict[str, Any], http: httpx.AsyncClient):
    """RFC 9728 resource metadata (the challenge's address first), then the authorization server's metadata,
    whose issuer must be the one the resource named (without resource metadata, the server's origin);
    PKCE S256 must be offered. Returns ``(prm, metadata, raw, issuer)``: the raw document keeps what the SDK
    model normalizes or drops (the issuer exactly as published; ``authorization_response_iss_parameter_supported``)."""
    from mcp.client.auth.utils import (build_oauth_authorization_server_metadata_discovery_urls,
                                       build_protected_resource_metadata_discovery_urls, create_oauth_metadata_request,
                                       handle_auth_metadata_response, handle_protected_resource_response)

    prm = None
    for url in build_protected_resource_metadata_discovery_urls(
            (doc.get("challenge") or {}).get("resource_metadata"), cfg.url):
        response = await http.send(create_oauth_metadata_request(_checked(url)))
        if response.status_code == 429 or response.status_code >= 500:
            response.raise_for_status()  # a busy or failing server is not "no metadata": never another issuer
        prm = await handle_protected_resource_response(response)
        if prm:
            break
    named = str(prm.authorization_servers[0]) if prm and prm.authorization_servers else None
    metadata, raw = None, {}
    for url in build_oauth_authorization_server_metadata_discovery_urls(named, cfg.url):
        response = await http.send(create_oauth_metadata_request(_checked(url)))
        ok, metadata = await handle_auth_metadata_response(response)
        if metadata:
            with contextlib.suppress(ValueError):
                raw = response.json()
            raw = raw if isinstance(raw, dict) else {}
        if metadata or not ok:
            break
    if metadata is None:
        raise MCPAuthRequired(cfg.id, "no_oauth_metadata", "the server publishes no authorization metadata")
    parsed = urllib.parse.urlsplit(cfg.url)
    expected = named or f"{parsed.scheme}://{parsed.netloc}"
    if str(metadata.issuer).rstrip("/") != expected.rstrip("/"):
        raise MCPAuthRequired(cfg.id, "issuer_mismatch", f"{metadata.issuer} is not {expected}")
    if "S256" not in (metadata.code_challenge_methods_supported or []):
        raise MCPAuthRequired(cfg.id, "no_pkce", "the authorization server does not offer PKCE S256")
    issuer = raw.get("issuer") if isinstance(raw.get("issuer"), str) and raw.get("issuer") else str(metadata.issuer)
    return prm, metadata, raw, issuer


async def _client_for(cfg: Any, doc: Dict[str, Any], metadata: Any, issuer: str, redirect: str,
                      http: httpx.AsyncClient, client_id: str, client_secret: str) -> Dict[str, Any]:
    """The owner's own registration if given, else the stored (or pending) client of this issuer and return
    address, else dynamic registration (deprecated in 2026-07-28 but offered by most servers), else a typed
    refusal naming the address to register."""
    from mcp.client.auth.utils import handle_registration_response
    from mcp.shared.auth import OAuthClientMetadata

    if client_id:
        methods = metadata.token_endpoint_auth_methods_supported or []
        method = "none" if not client_secret else (
            "client_secret_basic" if "client_secret_post" not in methods and "client_secret_basic" in methods
            else "client_secret_post")
        return {"client_id": client_id, "client_secret": client_secret or None, "token_endpoint_auth_method": method,
                "redirect_uri": redirect, "issuer": issuer, "source": "owner"}
    for stored in (doc.get("client") or {}, (doc.get("pending") or {}).get("client") or {}):  # reuse, never re-register
        if stored.get("issuer") == issuer and (stored.get("source") == "owner" or stored.get("redirect_uri") == redirect):
            return stored
    if not metadata.registration_endpoint:
        raise MCPAuthRequired(cfg.id, "no_self_registration",
                              f"the provider registers no apps itself: register one there for the return address "
                              f"{redirect}, then enter its client ID (and secret) under Sign in on this server's card "
                              "in Settings → Tools")
    endpoint = _checked(str(metadata.registration_endpoint))
    body = OAuthClientMetadata(client_name="Ouroboros", redirect_uris=[redirect], token_endpoint_auth_method="none",
                               grant_types=["authorization_code", "refresh_token"], response_types=["code"]).model_dump(
        mode="json", exclude_none=True)
    body["application_type"] = "native" if urllib.parse.urlparse(redirect).hostname in _LOOPBACK else "web"
    response = await http.post(endpoint, json=body)
    if response.status_code not in (200, 201):
        raise MCPAuthRequired(cfg.id, "registration_rejected", f"{_oauth_error(response)} from {endpoint}")
    info = await handle_registration_response(response)
    return {"client_id": info.client_id, "client_secret": info.client_secret,
            "token_endpoint_auth_method": info.token_endpoint_auth_method or "none", "redirect_uri": redirect,
            "issuer": issuer, "source": "registered"}


async def begin(cfg: Any, *, return_base: str = "", client_id: str = "", client_secret: str = "") -> Dict[str, Any]:
    """Start one sign-in; returns ``{server, url, redirect_uri}`` — the provider page the owner opens."""
    from mcp.client.auth.oauth2 import PKCEParameters
    from mcp.client.auth.utils import get_client_metadata_scopes

    doc = _load(cfg)
    async with _client() as http:
        prm, metadata, raw, issuer = await _discover(cfg, doc, http)
        stored = doc.get("client") or {}
        redirect = stored["redirect_uri"] if stored.get("source") == "owner" and not client_id else redirect_uri(return_base)
        client = await _client_for(cfg, doc, metadata, issuer, redirect, http, client_id, client_secret)
    authorize = _checked(str(metadata.authorization_endpoint))
    pkce, nonce = PKCEParameters.generate(), secrets.token_urlsafe(24)
    # A step-up keeps what the grant already allows and adds what the server asked for (MCP step-up flow).
    scope = _union(get_client_metadata_scopes((doc.get("challenge") or {}).get("scope"), prm, metadata),
                   (doc.get("tokens") or {}).get("scope"), doc.get("required_scope"))
    resource = str(prm.resource) if prm and prm.resource else None  # RFC 8707 where the server published PRM
    params = {"response_type": "code", "client_id": client["client_id"], "redirect_uri": client["redirect_uri"],
              "state": _state(cfg.id, nonce), "code_challenge": pkce.code_challenge, "code_challenge_method": "S256",
              **({"scope": scope} if scope else {}), **({"resource": resource} if resource else {})}
    facts = metadata.model_dump(mode="json", exclude_none=True)
    kept = {key: facts[key] for key in ("authorization_endpoint", "token_endpoint", "registration_endpoint",
                                         "revocation_endpoint") if key in facts}
    kept["authorization_response_iss_parameter_supported"] = raw.get(
        "authorization_response_iss_parameter_supported") is True
    async with _locked(cfg):  # the current grant keeps its own issuer and endpoints until this sign-in completes
        doc = _load(cfg)  # re-read: a call may have refreshed meanwhile
        doc["pending"] = {"nonce": nonce, "code_verifier": pkce.code_verifier, "client": client, "scope": scope,
                          "issuer": issuer, "resource": resource, "metadata": kept, "started": utc_now_iso()}
        _save(cfg, doc)
    return {"server": cfg.id, "url": f"{authorize}?{urllib.parse.urlencode(params)}",
            "redirect_uri": client["redirect_uri"]}


async def complete(cfg: Any, *, state: str, code: str = "", iss: Optional[str] = None, error: str = "",
                   error_description: str = "") -> Dict[str, Any]:
    """Finish the pending sign-in from the provider's return; returns ``{ok, server, scope}``."""
    from mcp.shared.auth import OAuthToken

    nonce = _envelope(state).get("nonce", "")
    doc = _load(cfg)
    pending = doc.get("pending") or {}
    if not pending or not nonce or not secrets.compare_digest(str(pending.get("nonce", "")), str(nonce)):
        raise MCPAuthRequired(cfg.id, "sign_in_superseded", "this return belongs to an older or unknown sign-in")
    issuer, metadata = pending.get("issuer") or "", pending.get("metadata") or {}
    if iss is not None and iss != issuer:  # RFC 9207: exactly the issuer the verifier was made for
        raise MCPAuthRequired(cfg.id, "issuer_mismatch", f"{iss} is not {issuer}")
    if iss is None and metadata.get("authorization_response_iss_parameter_supported"):
        raise MCPAuthRequired(cfg.id, "issuer_missing", "the provider promised an iss parameter and sent none")
    if error:
        raise MCPAuthRequired(cfg.id, "denied", error_description or error)
    client = pending.get("client") or {}
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": client.get("redirect_uri", ""),
            "code_verifier": pending.get("code_verifier", ""),
            **({"resource": pending["resource"]} if pending.get("resource") else {})}
    async with _client() as http:
        response = await _token_request(http, metadata.get("token_endpoint", ""), client, data)
    if response.status_code != 200:
        raise MCPAuthRequired(cfg.id, "exchange_rejected", _oauth_error(response))
    token = OAuthToken.model_validate_json(response.content)
    async with _locked(cfg):
        doc = _load(cfg)
        if (doc.get("pending") or {}).get("nonce") != pending.get("nonce"):
            raise MCPAuthRequired(cfg.id, "sign_in_superseded", "a newer sign-in started meanwhile")
        for key in ("pending", "challenge", "required_scope"):
            doc.pop(key, None)
        doc.update(issuer=issuer, resource=pending.get("resource"), metadata=metadata, client=client,
                   tokens=_tokens(token, None, pending.get("scope")),
                   sign_in={"state": SIGNED_IN, "since": utc_now_iso()})
        _save(cfg, doc)
    return {"ok": True, "server": cfg.id, "scope": doc["tokens"]["scope"] or ""}


async def sign_out(cfg: Any) -> Dict[str, Any]:
    """Forget this installation's tokens (the client stays for the next sign-in) and ask the provider to
    revoke them where it publishes a revocation endpoint (RFC 7009); ``revoked`` says what it confirmed."""
    async with _locked(cfg):
        doc = _load(cfg)
        tokens = doc.pop("tokens", None) or {}
        doc.pop("pending", None)
        if doc:
            doc["sign_in"] = {"state": SIGN_IN_REQUIRED, "reason": "signed_out", "since": utc_now_iso()}
            _save(cfg, doc)
    endpoint, revoked = (doc.get("metadata") or {}).get("revocation_endpoint"), None
    if endpoint and tokens:
        revoked = True
        async with _client() as http:
            for kind in ("refresh_token", "access_token"):
                if tokens.get(kind):
                    try:
                        response = await _token_request(http, endpoint, doc.get("client") or {},
                                                         {"token": tokens[kind], "token_type_hint": kind})
                        revoked = revoked and response.status_code == 200
                    except (httpx.HTTPError, ValueError):
                        revoked = False
    return {"ok": True, "server": cfg.id, "revoked": revoked}
