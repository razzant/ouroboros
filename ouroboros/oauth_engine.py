"""The core's one OAuth client for the owner's sign-ins; remote MCP servers use it (``mcp_oauth.py``).

A consumer describes each sign-in document as a ``Grant``: its file (mode 0600 in a 0700 directory),
the ``store.name_lock`` name that serializes its steps across threads and processes, the fields that
bind the file to its owner (a file bound otherwise is not this grant), the consumer's refusal (a
``SignInRequired``) and its write rule. Over it the engine runs the provider-neutral part of a sign-in:
the issuer's metadata (RFC 8414, then OpenID Connect discovery) with PKCE S256 required, RFC 7591
registration, the pending sign-in (``authorize``; a later one replaces it), the provider's return
(``complete``: the pending nonce by ``secrets.compare_digest``, RFC 9207 ``iss``, the code exchange; a
pasted address through ``parse_return``), one refresh under the lock with a re-read (``current_token``:
public clients' refresh tokens rotate, and listings, calls and sign-in steps run in different processes)
and RFC 7009 revocation. The ``state`` a provider carries back is a routing envelope naming its one
recipient (``mcp``), never authority: only the pending nonce completes a sign-in.
Every endpoint passes the MCP server address policy (``mcp_client._validate_url``) and is HTTPS or
loopback. Nothing here opens a browser or starts a sign-in by itself; provider names, scopes and the
words the owner reads are the consumer's.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import os
import pathlib
import secrets
import threading
import time
import urllib.parse
import weakref
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import httpx

from ouroboros import config
from ouroboros.runtime_limits import MCP_OAUTH_HTTP_TIMEOUT_SEC, MCP_OAUTH_LOCK_WAIT_SEC
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

RETURN_PATH = "/onboarding/static/provider-return.html"
SIGNED_IN, SIGN_IN_REQUIRED = "signed_in", "sign_in_required"
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_KINDS = ("mcp",)  # the recipients a returned ``state`` may name: a remote MCP server's sign-in


class SignInRequired(RuntimeError):
    """The local sign-in step cannot continue; the provider may already have processed a request."""

    code = "SIGN_IN_REQUIRED"

    def __init__(self, reason: str, detail: str = "", *, message: str = "") -> None:
        self.reason, self.detail = reason, detail
        super().__init__(message or f"SIGN_IN_REQUIRED: {reason}{': ' + detail if detail else ''}")


@dataclasses.dataclass(frozen=True)
class Grant:
    """One sign-in document. ``lock`` names its ``store.name_lock`` under ``state/``; ``binding`` holds the
    fields every write stamps and every read requires; ``refusal(reason, detail)`` is the consumer's
    ``SignInRequired``; ``may_write(doc)`` is the consumer's rule for a document about to be saved.
    ``format_version`` is the version this consumer writes; ``upgrade`` transforms a read record of an older
    version in memory only (nothing is written until a save)."""

    path: pathlib.Path
    lock: str
    binding: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    refusal: Callable[..., SignInRequired] = SignInRequired
    may_write: Callable[[Dict[str, Any]], bool] = lambda _doc: True
    format_version: int = 1
    upgrade: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


def load(grant: Grant) -> Dict[str, Any]:
    """The grant's document; a file written for another binding is not this grant's."""
    doc = read_json_dict(grant.path) or {}
    if not all(doc.get(key) == value for key, value in grant.binding.items()):
        return {}
    if doc and grant.upgrade is not None and doc.get("v") != grant.format_version:
        return {**grant.upgrade(doc), "v": grant.format_version}
    return doc


def save(grant: Grant, doc: Dict[str, Any]) -> None:
    """Write the document atomically with ``v`` and the binding stamped, unless the grant's rule refuses it."""
    if not grant.may_write(doc):
        return
    grant.path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(grant.path.parent, 0o700)  # a rewrite's temporary file is never readable by another account
    atomic_write_json(grant.path, {**doc, "v": grant.format_version, **grant.binding})
    with contextlib.suppress(OSError):
        os.chmod(grant.path, 0o600)


# Per event loop, per grant; a lock lives only while a coroutine holds or awaits it (a bound lock refers to
# its loop, so a strongly held one would keep every closed loop alive).
_LOOP_LOCKS: "weakref.WeakKeyDictionary[Any, weakref.WeakValueDictionary]" = weakref.WeakKeyDictionary()
_LOOP_LOCKS_GUARD = threading.Lock()


@contextlib.asynccontextmanager
async def locked(grant: Grant):
    """One refresh or sign-in step at a time per grant. Coroutines of one event loop (a session's GET and
    POST) queue on an asyncio lock, so a waiter never blocks the loop that runs the holder's await; the
    owner-aware file lock across threads and processes (``store.name_lock``) is then polled off it."""
    from ouroboros import store

    with _LOOP_LOCKS_GUARD:
        locks = _LOOP_LOCKS.setdefault(asyncio.get_running_loop(), weakref.WeakValueDictionary())
        lock = locks.get(grant.lock)
        if lock is None:
            lock = locks[grant.lock] = asyncio.Lock()
    async with lock:
        deadline = time.monotonic() + MCP_OAUTH_LOCK_WAIT_SEC
        while True:
            held = store.name_lock(config.DATA_DIR, grant.lock, timeout_sec=0)
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


def required_state(doc: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Keep the known or unknown start of an open requirement; date a new one."""
    prior = doc.get("sign_in") or {}
    since = prior.get("since") if prior.get("state") == SIGN_IN_REQUIRED else utc_now_iso()
    return {"state": SIGN_IN_REQUIRED, "reason": reason, "since": since}


def required(grant: Grant, doc: Dict[str, Any], reason: str, detail: str = "", **facts: Any) -> SignInRequired:
    """Record that sign-in is needed (with what the provider or the resource said) and return the refusal
    to raise."""
    doc = {**doc, **facts, "sign_in": required_state(doc, reason)}
    if reason in ("refresh_rejected", "token_rejected", "expired"):
        doc.pop("tokens", None)  # a grant the provider no longer honours is not kept as if it were
    save(grant, doc)
    return grant.refusal(reason, detail)


_SECRETS: Dict[str, Tuple[int, tuple]] = {}


def secret_values(path: pathlib.Path) -> tuple:
    """One sign-in document's live secrets (tokens, client secrets, the pending PKCE verifier), for the
    masking of every text from or about it; reread when the file changes."""
    key = str(path)
    try:
        stamp = pathlib.Path(path).stat().st_mtime_ns
    except OSError:
        return ()
    cached = _SECRETS.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    doc = read_json_dict(pathlib.Path(path)) or {}
    pending = doc.get("pending") or {}
    values = ((doc.get("tokens") or {}).get("access_token"), (doc.get("tokens") or {}).get("refresh_token"),
              (doc.get("client") or {}).get("client_secret"), pending.get("code_verifier"),
              (pending.get("client") or {}).get("client_secret"))
    _SECRETS[key] = (stamp, tuple(value for value in values if value))
    return _SECRETS[key][1]


# -- the authorization server -------------------------------------------------------------------


def http_client() -> httpx.AsyncClient:
    """A short-lived client for the authorization server: never a resource transport's, never its headers."""
    return httpx.AsyncClient(timeout=MCP_OAUTH_HTTP_TIMEOUT_SEC, follow_redirects=False)


def checked(url: str) -> str:
    """An endpoint an (untrusted) server or document named: HTTPS or loopback, and the address policy of servers."""
    from ouroboros.mcp_client import _validate_url

    text = _validate_url(str(url))
    if urllib.parse.urlparse(text).scheme != "https" and (urllib.parse.urlparse(text).hostname or "") not in _LOOPBACK:
        raise ValueError(f"authorization endpoint {text!r} is not HTTPS")
    return text


def oauth_error(response: httpx.Response) -> str:
    """The OAuth ``error`` code of a refusal, else its HTTP status."""
    with contextlib.suppress(ValueError):
        body = response.json()
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])
    return f"HTTP {response.status_code}"


async def token_request(http: httpx.AsyncClient, endpoint: str, client: Dict[str, Any],
                        data: Dict[str, str]) -> httpx.Response:
    """A token-endpoint (or revocation) POST with the client's own authentication method."""
    data, auth = {**data, "client_id": client.get("client_id", "")}, None
    if client.get("client_secret"):
        if client.get("token_endpoint_auth_method") == "client_secret_basic":
            auth = httpx.BasicAuth(urllib.parse.quote(client["client_id"], safe=""),
                                   urllib.parse.quote(client["client_secret"], safe=""))
        else:
            data["client_secret"] = client["client_secret"]
    return await http.post(checked(endpoint), data=data, auth=auth, headers={"Accept": "application/json"})


def token_record(token: Any, previous_refresh: Optional[str], previous_scope: Optional[str]) -> Dict[str, Any]:
    """The stored tokens of an SDK ``OAuthToken``: an absolute expiry (the SDK keeps only a relative one); the
    previous refresh token and scope when a response omits them (RFC 6749 §5.1 and §6: then they are unchanged)."""
    return {"access_token": token.access_token, "token_type": token.token_type or "Bearer",
            "refresh_token": token.refresh_token or previous_refresh, "scope": token.scope or previous_scope,
            "expires_at": time.time() + token.expires_in if token.expires_in else None}


def same_registration(one: Dict[str, Any], other: Dict[str, Any]) -> bool:
    """One client registration: same issuer, id and authentication (a new secret is another registration)."""
    keys = ("issuer", "client_id", "client_secret", "token_endpoint_auth_method")
    return bool(one) and bool(other) and all(one.get(key) == other.get(key) for key in keys)


def fresh(tokens: Dict[str, Any], rejected: Optional[str]) -> bool:
    """A stored access token that is not ``rejected`` and has not expired."""
    return bool(tokens.get("access_token")) and tokens["access_token"] != rejected and (
        tokens.get("expires_at") is None or time.time() < float(tokens["expires_at"]))


async def current_token(grant: Grant, rejected: Optional[str] = None) -> str:
    """A usable access token: the stored one, or one refresh under the lock (re-read first: another process
    may already hold a newer token); ``resource`` (RFC 8707) is sent when the grant has one. Raises the
    grant's refusal when sign-in is needed; an unreachable provider fails and keeps the grant."""
    from mcp.shared.auth import OAuthToken

    async with locked(grant):
        doc = load(grant)
        tokens = doc.get("tokens") or {}
        if not tokens.get("access_token"):
            raise grant.refusal((doc.get("sign_in") or {}).get("reason") or "not_signed_in")
        if fresh(tokens, rejected):
            return tokens["access_token"]
        if not tokens.get("refresh_token"):
            raise required(grant, doc, "token_rejected" if rejected else "expired")
        data = {"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"],
                **({"resource": doc["resource"]} if doc.get("resource") else {})}
        async with http_client() as http:
            response = await token_request(http, (doc.get("metadata") or {}).get("token_endpoint", ""),
                                           doc.get("client") or {}, data)
        if response.status_code in (400, 401):
            error = oauth_error(response)
            if error == "invalid_request":  # a malformed request, not a refused grant: the grant stays
                response.raise_for_status()
            # invalid_grant (expired or revoked), invalid_client (the registration is gone: register again,
            # so a pending sign-in made with the same registration goes too)
            facts: Dict[str, Any] = {}
            if error == "invalid_client":
                facts["client"] = None
                if same_registration((doc.get("pending") or {}).get("client") or {}, doc.get("client") or {}):
                    facts["pending"] = None
            raise required(grant, doc, "refresh_rejected", error, **facts)
        response.raise_for_status()  # an unreachable authorization server is a failure, not a lost sign-in
        doc["tokens"] = token_record(OAuthToken.model_validate_json(response.content), tokens.get("refresh_token"),
                                     tokens.get("scope"))
        save(grant, doc)
        return doc["tokens"]["access_token"]


def union(*scopes: Optional[str]) -> Optional[str]:
    """Space-separated scopes joined in first-seen order without repeats; ``None`` when there are none."""
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
    """The builtin return page under ``return_base`` (the browser's origin with the deployment prefix) when
    that is loopback or HTTPS, else under this server's loopback address, where a browser elsewhere stops
    and the owner pastes the address."""
    base = (return_base or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if not (parsed.scheme == "https" or (parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK)):
        base = f"http://127.0.0.1:{_server_port()}"
    return base + RETURN_PATH


def state(kind: str, name: str, nonce: str) -> str:
    """The ``state`` a provider carries back: the routing envelope ``{v: 1, <kind>: <name>, nonce}``
    (unpadded base64url JSON) naming the return's one recipient."""
    if kind not in _KINDS or not name or not nonce:
        raise ValueError(f"no {kind!r} sign-in state for {name!r}")
    raw = json.dumps({"v": 1, kind: name, "nonce": nonce}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def envelope(state: str, kind: Optional[str] = None) -> Dict[str, Any]:
    """The routing envelope a returned ``state`` carries (route syntax only: ``complete`` checks the nonce):
    one that names ``kind``, or any known kind when none is given; anything else (another recipient's) is a
    ``ValueError``."""
    refused = "not a sign-in return" + (f" for {kind!r}" if kind else "")
    try:
        decoded = json.loads(base64.urlsafe_b64decode(str(state) + "=" * (-len(str(state)) % 4)).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(refused) from exc
    if not isinstance(decoded, dict) or kind not in (None, *_KINDS) or decoded.get("v") != 1:
        raise ValueError(refused)
    if not any(decoded.get(key) for key in ((kind,) if kind else _KINDS)):
        raise ValueError(refused)
    return decoded


def parse_return(address: str) -> Dict[str, str]:
    """The provider's return parameters from the address its browser stopped on (a paste)."""
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(str(address or "").strip()).query))
    if not query.get("state") or not (query.get("code") or query.get("error")):
        raise ValueError("the address carries no complete sign-in return (state and code)")
    return {key: query[key] for key in ("state", "code", "error", "error_description", "iss") if key in query}


async def discover_issuer(named_issuer: Optional[str], fallback_url: str, expected_issuer: str,
                          http: httpx.AsyncClient, *, refusal: Callable[..., SignInRequired] = SignInRequired):
    """The authorization server's metadata (RFC 8414, then OpenID Connect discovery) of ``named_issuer``,
    else of ``fallback_url``'s origin; its issuer must be ``expected_issuer`` and PKCE S256 must be offered.
    Returns ``(metadata, endpoints, issuer)``: the SDK model, the addresses a sign-in keeps (``authorize``)
    and the issuer exactly as published (the model normalizes it), from the raw document, which also says
    whether the provider promises an ``iss`` in every return."""
    from mcp.client.auth.utils import (build_oauth_authorization_server_metadata_discovery_urls,
                                       create_oauth_metadata_request, handle_auth_metadata_response)

    metadata, raw = None, {}
    for url in build_oauth_authorization_server_metadata_discovery_urls(named_issuer, fallback_url):
        response = await http.send(create_oauth_metadata_request(checked(url)))
        ok, metadata = await handle_auth_metadata_response(response)
        if metadata:
            with contextlib.suppress(ValueError):
                raw = response.json()
            raw = raw if isinstance(raw, dict) else {}
        if metadata or not ok:
            break
    if metadata is None:
        raise refusal("no_oauth_metadata", "the server publishes no authorization metadata")
    if str(metadata.issuer).rstrip("/") != expected_issuer.rstrip("/"):
        raise refusal("issuer_mismatch", f"{metadata.issuer} is not {expected_issuer}")
    if "S256" not in (metadata.code_challenge_methods_supported or []):
        raise refusal("no_pkce", "the authorization server does not offer PKCE S256")
    issuer = raw.get("issuer") if isinstance(raw.get("issuer"), str) and raw.get("issuer") else str(metadata.issuer)
    facts = metadata.model_dump(mode="json", exclude_none=True)
    endpoints = {key: facts[key] for key in ("authorization_endpoint", "token_endpoint", "registration_endpoint",
                                             "revocation_endpoint") if key in facts}
    endpoints["authorization_response_iss_parameter_supported"] = raw.get(
        "authorization_response_iss_parameter_supported") is True
    return metadata, endpoints, issuer


async def register_client(metadata: Any, redirect: str, http: httpx.AsyncClient, *, issuer: str,
                          refusal: Callable[..., SignInRequired] = SignInRequired) -> Dict[str, Any]:
    """RFC 7591 registration of a public client returning to ``redirect`` at the registration endpoint the
    issuer's metadata names (the caller has checked that it names one); returns the client record."""
    from mcp.client.auth.utils import handle_registration_response
    from mcp.shared.auth import OAuthClientMetadata

    endpoint = checked(str(metadata.registration_endpoint or ""))
    body = OAuthClientMetadata(client_name="Ouroboros", redirect_uris=[redirect], token_endpoint_auth_method="none",
                               grant_types=["authorization_code", "refresh_token"], response_types=["code"]).model_dump(
        mode="json", exclude_none=True)
    body["application_type"] = "native" if urllib.parse.urlparse(redirect).hostname in _LOOPBACK else "web"
    response = await http.post(endpoint, json=body)
    if response.status_code not in (200, 201):
        raise refusal("registration_rejected", f"{oauth_error(response)} from {endpoint}")
    info = await handle_registration_response(response)
    return {"client_id": info.client_id, "client_secret": info.client_secret,
            "token_endpoint_auth_method": info.token_endpoint_auth_method or "none", "redirect_uri": redirect,
            "issuer": issuer, "source": "registered"}


async def authorize(grant: Grant, endpoints: Mapping[str, Any], client: Dict[str, Any], scope: Optional[str],
                    state: Callable[[str], str], *, issuer: str, resource: Optional[str] = None) -> str:
    """Start one sign-in and return the provider's authorize address for the owner to open: PKCE S256 and a
    fresh nonce, for which ``state`` makes the routing ``state`` (``functools.partial(state, kind, name)``).
    ``resource`` (RFC 8707) is sent again at the exchange and at every refresh. The pending sign-in, with
    the issuer and ``endpoints`` it is made for, is saved in the grant under its lock (re-read first: a call
    may have refreshed meanwhile); a later one replaces it, and the grant's tokens, issuer and endpoints stay
    until it completes."""
    from mcp.client.auth.oauth2 import PKCEParameters

    address = checked(str(endpoints.get("authorization_endpoint") or ""))
    pkce, nonce = PKCEParameters.generate(), secrets.token_urlsafe(24)
    params = {"response_type": "code", "client_id": client["client_id"], "redirect_uri": client["redirect_uri"],
              "state": state(nonce), "code_challenge": pkce.code_challenge, "code_challenge_method": "S256",
              **({"scope": scope} if scope else {}), **({"resource": resource} if resource else {})}
    async with locked(grant):
        doc = load(grant)
        doc["pending"] = {"nonce": nonce, "code_verifier": pkce.code_verifier, "client": client, "scope": scope,
                          "issuer": issuer, "resource": resource, "metadata": dict(endpoints), "started": utc_now_iso()}
        save(grant, doc)
    return f"{address}?{urllib.parse.urlencode(params)}"


async def complete(grant: Grant, returned: Mapping[str, Any]) -> Dict[str, Any]:
    """Finish the grant's pending sign-in from the provider's return (``state`` and ``code`` or ``error``,
    ``iss`` when sent). The nonce must be the pending one and ``iss`` the issuer its verifier was made for
    (RFC 9207); then the code is exchanged and the signed-in grant replaces the pending sign-in. Returns the
    saved document. A refused return raises the grant's refusal and saves nothing."""
    from mcp.shared.auth import OAuthToken

    nonce = envelope(returned.get("state") or "").get("nonce", "")
    doc = load(grant)
    pending = doc.get("pending") or {}
    if not pending or not nonce or not secrets.compare_digest(str(pending.get("nonce", "")), str(nonce)):
        raise grant.refusal("sign_in_superseded", "this return belongs to an older or unknown sign-in")
    issuer, metadata, iss = pending.get("issuer") or "", pending.get("metadata") or {}, returned.get("iss")
    if iss is not None and iss != issuer:  # RFC 9207: exactly the issuer the verifier was made for
        raise grant.refusal("issuer_mismatch", f"{iss} is not {issuer}")
    if iss is None and metadata.get("authorization_response_iss_parameter_supported"):
        raise grant.refusal("issuer_missing", "the provider promised an iss parameter and sent none")
    if returned.get("error"):
        raise grant.refusal("denied", returned.get("error_description") or returned["error"])
    client = pending.get("client") or {}
    data = {"grant_type": "authorization_code", "code": returned.get("code") or "",
            "redirect_uri": client.get("redirect_uri", ""), "code_verifier": pending.get("code_verifier", ""),
            **({"resource": pending["resource"]} if pending.get("resource") else {})}
    async with http_client() as http:
        response = await token_request(http, metadata.get("token_endpoint", ""), client, data)
    if response.status_code != 200:
        raise grant.refusal("exchange_rejected", oauth_error(response))
    token = OAuthToken.model_validate_json(response.content)
    async with locked(grant):
        doc = load(grant)
        if (doc.get("pending") or {}).get("nonce") != pending.get("nonce"):
            raise grant.refusal("sign_in_superseded", "a newer sign-in started meanwhile")
        for key in ("pending", "challenge", "required_scope", "required_scope_since"):
            doc.pop(key, None)
        doc.update(issuer=issuer, resource=pending.get("resource"), metadata=metadata, client=client,
                   tokens=token_record(token, None, pending.get("scope")),
                   sign_in={"state": SIGNED_IN, "since": utc_now_iso()})
        save(grant, doc)
    return doc


async def revoke(endpoint: Optional[str], client: Dict[str, Any], tokens: Dict[str, Any]) -> Optional[bool]:
    """Ask the provider to revoke a grant's tokens with that grant's own client (RFC 7009), the refresh token
    first: ``True`` when it confirmed every one, ``False`` when it did not or could not be asked, ``None``
    when it publishes no revocation address or there is nothing to revoke."""
    if not endpoint or not tokens:
        return None
    revoked = True
    async with http_client() as http:
        for kind in ("refresh_token", "access_token"):
            if tokens.get(kind):
                try:
                    response = await token_request(http, endpoint, client,
                                                   {"token": tokens[kind], "token_type_hint": kind})
                    revoked = revoked and response.status_code == 200
                except (httpx.HTTPError, ValueError):
                    revoked = False
    return revoked
