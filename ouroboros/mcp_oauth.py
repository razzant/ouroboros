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
``state/mcp/oauth/server-<id>.json``, mode 0600 in a 0700 directory; sign-out drops its tokens and
configuration cleanup removes it; a draft of another address (Settings Test) never replaces it. The
OAuth client itself (the lock, refresh, the pending sign-in and its return, registration, revocation)
is ``oauth_engine.py``; this module keeps what is MCP's: the server's file and its address binding, the
draft rule, RFC 9728 resource metadata and the issuer it names, the choice of client, the transport's
Bearer handling and ``MCPAuthRequired``. Only public helpers of the MCP SDK are used: its
``OAuthClientProvider`` runs the interactive flow inside a call and forgets the expiry it learned.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import logging
import urllib.parse
from typing import Any, Dict, Optional

import httpx

from ouroboros import config, oauth_engine
from ouroboros.oauth_engine import SIGN_IN_REQUIRED, SIGNED_IN
from ouroboros.oauth_engine import RETURN_PATH, parse_return  # noqa: F401 -- MCP's route, tool and tests read a paste here
from ouroboros.utils import read_json_dict, utc_now_iso

log = logging.getLogger(__name__)


class MCPAuthRequired(oauth_engine.SignInRequired):
    """The server needs the owner's sign-in; the server ran nothing for this request."""

    code = "MCP_AUTH_REQUIRED"

    def __init__(self, server_id: str, reason: str, detail: str = "") -> None:
        self.server_id = server_id
        super().__init__(reason, detail, message=(
            f"MCP_AUTH_REQUIRED: server {server_id!r} needs the owner's sign-in ({reason}"
            f"{': ' + detail if detail else ''}). Nothing ran there. The owner can press Sign in on its card in "
            f"Extensions → MCP servers, or remote_mcp_sign_in(server={server_id!r}, action='begin') (enable_tools loads it) issues "
            "a sign-in link to give them."))


def applies(cfg: Any) -> bool:
    """An HTTP/SSE server without a saved credential: its ``401`` Bearer challenge means sign-in."""
    return cfg.transport in ("streamable_http", "sse") and not cfg.carries_credentials()


# -- the server's file -------------------------------------------------------------------------


def _path(server_id: str):
    return config.DATA_DIR / "state" / "mcp" / "oauth" / f"server-{server_id}.json"


_FORMAT = 2  # v1 writers moved an open requirement's start to each refusal


def _upgrade(doc: Dict[str, Any]) -> Dict[str, Any]:
    record = doc.get("sign_in") or {}
    if record.get("state") == SIGN_IN_REQUIRED:
        return {**doc, "sign_in": {**record, "since": None}}
    return doc


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


def _grant(cfg: Any) -> oauth_engine.Grant:
    """The server's file as the engine's grant: bound to the server's address (a file written for another
    address of the same id is not its grant), refused with ``MCPAuthRequired``, and written unless a draft
    would replace another address's grant (a saved address always writes: Save prunes)."""
    path = _path(cfg.id)

    def may_write(_doc: Dict[str, Any]) -> bool:
        recorded = (read_json_dict(path) or {}).get("url") if _DRAFT.get() else None
        return not recorded or recorded == cfg.url

    return oauth_engine.Grant(path=path, lock=f"mcp/oauth/server-{cfg.id}.lock", binding={"url": cfg.url},
                              refusal=functools.partial(MCPAuthRequired, cfg.id), may_write=may_write,
                              format_version=_FORMAT, upgrade=_upgrade)


def _load(cfg: Any) -> Dict[str, Any]:
    return oauth_engine.load(_grant(cfg))


def _save(cfg: Any, doc: Dict[str, Any]) -> None:
    oauth_engine.save(_grant(cfg), doc)


def _locked(cfg: Any):
    """One refresh or sign-in step at a time per server (``oauth_engine.locked``)."""
    return oauth_engine.locked(_grant(cfg))


def _required(cfg: Any, doc: Dict[str, Any], reason: str, detail: str = "", **facts: Any) -> MCPAuthRequired:
    """Record that sign-in is needed (with what the server said) and return the refusal to raise."""
    return oauth_engine.required(_grant(cfg), doc, reason, detail, **facts)


def status(cfg: Any) -> Optional[Dict[str, Any]]:
    """``{state, reason?, required_scope?, required_scope_since?, since, issuer, pending?}`` once the server asked
    for sign-in; else ``None``. A grant that needs a wider scope stays signed in with reason ``insufficient_scope``
    and the start of that gap (``None`` when a file written before kept none)."""
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
        out.update(reason="insufficient_scope", required_scope=doc["required_scope"],
                   required_scope_since=doc.get("required_scope_since"))
    if doc.get("pending"):
        out["pending"] = True
    return out


def secret_values(server_id: str) -> tuple:
    """This server's live sign-in secrets, for the masking of every text from or about it."""
    return oauth_engine.secret_values(_path(server_id))


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


async def current_token(cfg: Any, rejected: Optional[str] = None) -> str:
    """A usable access token for the server (``oauth_engine.current_token``: the stored one, or one refresh
    under the lock). Raises ``MCPAuthRequired`` when sign-in is needed."""
    return await oauth_engine.current_token(_grant(cfg), rejected)


# -- the transport ------------------------------------------------------------------------------


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
        if token and not oauth_engine.fresh(tokens, None):
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
                wanted = oauth_engine.union((doc.get("tokens") or {}).get("scope"), doc.get("required_scope"),
                                            challenge.get("scope"))
                since = doc.get("required_scope_since") if doc.get("required_scope") else utc_now_iso()
                _save(self.cfg, {**doc, "challenge": challenge, "required_scope": wanted, "required_scope_since": since})
            raise await self._refused(MCPAuthRequired(self.cfg.id, "insufficient_scope", challenge.get("scope") or ""))


# -- the two sign-in steps ----------------------------------------------------------------------


def redirect_uri(return_base: str = "") -> str:
    """The builtin return page under ``return_base`` when that is loopback or HTTPS (an MCP rule), else
    under this server's loopback address; a browser elsewhere then hands its address back by paste."""
    return oauth_engine.redirect_uri(return_base)


def server_of(state: str) -> str:
    """The server a returned ``state`` names (route syntax only: ``complete`` checks the nonce)."""
    try:
        return str(oauth_engine.envelope(state, "mcp")["mcp"])
    except ValueError as exc:
        raise ValueError("not an MCP sign-in return") from exc


async def _discover(cfg: Any, doc: Dict[str, Any], http: httpx.AsyncClient):
    """RFC 9728 resource metadata (the challenge's address first), then the authorization server's metadata,
    whose issuer must be the one the resource named (without resource metadata, the server's origin);
    PKCE S256 must be offered (``oauth_engine.discover_issuer``). Returns ``(prm, metadata, endpoints, issuer)``."""
    from mcp.client.auth.utils import (build_protected_resource_metadata_discovery_urls, create_oauth_metadata_request,
                                       handle_protected_resource_response)

    prm = None
    for url in build_protected_resource_metadata_discovery_urls(
            (doc.get("challenge") or {}).get("resource_metadata"), cfg.url):
        response = await http.send(create_oauth_metadata_request(oauth_engine.checked(url)))
        if response.status_code == 429 or response.status_code >= 500:
            response.raise_for_status()  # a busy or failing server is not "no metadata": never another issuer
        prm = await handle_protected_resource_response(response)
        if prm:
            break
    named = str(prm.authorization_servers[0]) if prm and prm.authorization_servers else None
    parsed = urllib.parse.urlsplit(cfg.url)
    metadata, endpoints, issuer = await oauth_engine.discover_issuer(
        named, cfg.url, named or f"{parsed.scheme}://{parsed.netloc}", http,
        refusal=functools.partial(MCPAuthRequired, cfg.id))
    return prm, metadata, endpoints, issuer


async def _client_for(cfg: Any, doc: Dict[str, Any], metadata: Any, issuer: str, redirect: str,
                      http: httpx.AsyncClient, client_id: str, client_secret: str) -> Dict[str, Any]:
    """The owner's own registration if given, else the stored (or pending) client of this issuer and return
    address, else dynamic registration (deprecated in 2026-07-28 but offered by most servers), else a typed
    refusal naming the address to register."""
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
                              "in Extensions → MCP servers")
    return await oauth_engine.register_client(metadata, redirect, http, issuer=issuer,
                                              refusal=functools.partial(MCPAuthRequired, cfg.id))


async def begin(cfg: Any, *, return_base: str = "", client_id: str = "", client_secret: str = "") -> Dict[str, Any]:
    """Start one sign-in; returns ``{server, url, redirect_uri}`` — the provider page the owner opens."""
    from mcp.client.auth.utils import get_client_metadata_scopes

    doc = _load(cfg)
    async with oauth_engine.http_client() as http:
        prm, metadata, endpoints, issuer = await _discover(cfg, doc, http)
        stored = doc.get("client") or {}
        redirect = stored["redirect_uri"] if stored.get("source") == "owner" and not client_id else redirect_uri(return_base)
        client = await _client_for(cfg, doc, metadata, issuer, redirect, http, client_id, client_secret)
    # A step-up keeps what the grant already allows and adds what the server asked for (MCP step-up flow).
    scope = oauth_engine.union(get_client_metadata_scopes((doc.get("challenge") or {}).get("scope"), prm, metadata),
                               (doc.get("tokens") or {}).get("scope"), doc.get("required_scope"))
    resource = str(prm.resource) if prm and prm.resource else None  # RFC 8707 where the server published PRM
    url = await oauth_engine.authorize(_grant(cfg), endpoints, client, scope,
                                       functools.partial(oauth_engine.state, "mcp", cfg.id), issuer=issuer,
                                       resource=resource)
    return {"server": cfg.id, "url": url, "redirect_uri": client["redirect_uri"]}


async def complete(cfg: Any, *, state: str, code: str = "", iss: Optional[str] = None, error: str = "",
                   error_description: str = "") -> Dict[str, Any]:
    """Finish the pending sign-in from the provider's return; returns ``{ok, server, scope}``."""
    server_of(state)  # an MCP server's return (route syntax): another recipient's is refused before any read
    doc = await oauth_engine.complete(_grant(cfg), {"state": state, "code": code, "iss": iss, "error": error,
                                                    "error_description": error_description})
    from ouroboros import mcp_catalog  # after the sign-in lock: the two locks never nest

    mcp_catalog.clear_sign_in_refusal(cfg)
    return {"ok": True, "server": cfg.id, "scope": doc["tokens"]["scope"] or ""}


async def sign_out(cfg: Any) -> Dict[str, Any]:
    """Forget this installation's tokens (the client stays for the next sign-in) and ask the provider to
    revoke them where it publishes a revocation endpoint (RFC 7009); ``revoked`` says what it confirmed."""
    async with _locked(cfg):
        doc = _load(cfg)
        tokens = doc.pop("tokens", None) or {}
        doc.pop("pending", None)
        if doc:
            doc["sign_in"] = oauth_engine.required_state(doc, "signed_out")
            _save(cfg, doc)
    revoked = await oauth_engine.revoke((doc.get("metadata") or {}).get("revocation_endpoint"),
                                        doc.get("client") or {}, tokens)
    return {"ok": True, "server": cfg.id, "revoked": revoked}
