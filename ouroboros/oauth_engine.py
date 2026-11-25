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
and RFC 7009 revocation. The ``state`` a provider carries back is a routing envelope of one kind
(``mcp``, or ``sign_in`` with its service), never authority: only the pending nonce completes a sign-in.
Every endpoint passes the MCP server address policy (``mcp_client._validate_url``) and is HTTPS or
loopback. Nothing here opens a browser or starts a sign-in by itself; provider names, scopes and the
words the owner reads are the consumer's.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import secrets
import threading
import time
import urllib.parse
import weakref
from typing import Any, Awaitable, Callable, Collection, Dict, Mapping, Optional, Tuple

import httpx

from ouroboros import config
from ouroboros.contracts.skill_manifest import canonical_skill_name
from ouroboros.runtime_limits import MCP_OAUTH_HTTP_TIMEOUT_SEC, MCP_OAUTH_LOCK_WAIT_SEC, OAUTH_MIN_VALID_MAX_SEC
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

RETURN_PATH = "/onboarding/static/provider-return.html"
SIGNED_IN, SIGN_IN_REQUIRED = "signed_in", "sign_in_required"
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# The recipients a returned ``state`` may name, with the further fields each needs: a remote MCP server's
# sign-in, or a skill's sign-in to a service it declares (``svc``).
_KINDS = ("mcp", "sign_in")
_RETURN_KINDS = (*_KINDS, "skill")  # the builtin page also routes skills' own returns
_RETURN_FIELDS = ("state", "code", "error", "error_description", "iss")
_OWN_PARAMS = frozenset({"response_type", "client_id", "redirect_uri", "state", "code_challenge",
                         "code_challenge_method", "scope", "resource"})  # authorize parameters set here only
# A request that failed with one of these never left this host: no connection reached the provider. Any other failure,
# a timeout after sending included, may have reached it.
NOT_SENT = (httpx.ConnectError, httpx.ConnectTimeout)


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
    data_dir: Optional[pathlib.Path] = None
    strict_read: bool = False


def load(grant: Grant) -> Dict[str, Any]:
    """A missing document is empty; a strict grant never overwrites unreadable custody."""
    if grant.strict_read:
        try:
            doc = json.loads(grant.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            raise grant.refusal("sign_in_storage_unavailable", "the saved sign-in could not be read") from None
        if not isinstance(doc, dict) or not all(doc.get(key) == value for key, value in grant.binding.items()):
            raise grant.refusal("sign_in_storage_unavailable", "the saved sign-in could not be identified")
    else:
        doc = read_json_dict(grant.path) or {}
        if not all(doc.get(key) == value for key, value in grant.binding.items()):
            return {}
    if doc and grant.upgrade is not None and doc.get("v") != grant.format_version:
        return {**grant.upgrade(doc), "v": grant.format_version}
    return doc


def save(grant: Grant, doc: Dict[str, Any]) -> bool:
    """Write atomically with ``v`` and the binding stamped; False means the grant's rule refused the write."""
    if not grant.may_write(doc):
        return False
    grant.path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(grant.path.parent, 0o700)  # a rewrite's temporary file is never readable by another account
    atomic_write_json(grant.path, {**doc, "v": grant.format_version, **grant.binding})
    with contextlib.suppress(OSError):
        os.chmod(grant.path, 0o600)
    return True


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

    root = grant.data_dir if grant.data_dir is not None else config.DATA_DIR
    key = (root, grant.lock)
    with _LOOP_LOCKS_GUARD:
        locks = _LOOP_LOCKS.setdefault(asyncio.get_running_loop(), weakref.WeakValueDictionary())
        lock = locks.get(key)
        if lock is None:
            lock = locks[key] = asyncio.Lock()
    async with lock:
        deadline = time.monotonic() + MCP_OAUTH_LOCK_WAIT_SEC
        while True:
            held = store.name_lock(root, grant.lock, timeout_sec=0)
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
    """One sign-in document's live secrets (tokens, client secrets, the pending PKCE verifier, a transfer's held tokens), for the
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
    exchanged = (pending.get("exchange") or {}).get("tokens") or {}
    values = ((doc.get("tokens") or {}).get("access_token"), (doc.get("tokens") or {}).get("refresh_token"),
              (doc.get("client") or {}).get("client_secret"), pending.get("code_verifier"),
              (pending.get("client") or {}).get("client_secret"), exchanged.get("access_token"),
              exchanged.get("refresh_token"))
    transfers = doc.get("handovers") if isinstance(doc.get("handovers"), dict) else {}  # a consumer's transfers in progress
    held = [(slot.get("attempt") or {}).get("tokens") or {} for slot in transfers.values() if isinstance(slot, dict)]
    values += tuple(tokens.get(kind) for tokens in held for kind in ("access_token", "refresh_token"))
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


def token_record(token: Any, previous_refresh: Optional[str], previous_scope: Optional[str], *,
                 sent_at: Optional[float] = None) -> Dict[str, Any]:
    """Remember the scope of this response; expiry is no later than the evidenced lifetime from send."""
    start = time.time() if sent_at is None else sent_at
    return {"access_token": token.access_token, "token_type": token.token_type or "Bearer",
            "refresh_token": token.refresh_token or previous_refresh,
            "scope": previous_scope if token.scope is None else token.scope,
            "expires_at": None if token.expires_in is None else start + token.expires_in}


def response_token(response: httpx.Response, previous_refresh: Optional[str], requested_scope: Optional[str], *,
                   sent_at: float, refusal: Callable[..., SignInRequired] = SignInRequired) -> Dict[str, Any]:
    from mcp.shared.auth import OAuthToken

    try:
        token = OAuthToken.model_validate_json(response.content)
        record = token_record(token, previous_refresh, requested_scope, sent_at=sent_at)
    except (ValueError, OverflowError):
        raise refusal("invalid_token_response", "the provider did not return a usable token record") from None
    if not record["access_token"]:
        raise refusal("invalid_token_response", "the provider did not return an access token")
    return record


def same_registration(one: Dict[str, Any], other: Dict[str, Any]) -> bool:
    keys = ("issuer", "client_id", "client_secret", "token_endpoint_auth_method", "provider_binding")
    return bool(one) and bool(other) and all(one.get(key) == other.get(key) for key in keys)


def registration_ref(client: Mapping[str, Any], return_base: str = "") -> str:
    """Names the saved registration an owner is shown: client ID, provider binding and return rule with the address that rule gives
    here. A hash of public facts, never a secret: editing any of them changes it, a label or a secret does not."""
    rule = client.get("return") or "loopback"
    try:
        address = redirect_uri(return_base, rule)
    except ValueError:
        address = ""  # an unreadable rule still binds its own text
    facts = [client.get("client_id", ""), {"issuer": client.get("issuer"), "endpoints": client.get("endpoints")}, rule, address]
    return _digest(json.dumps(facts, sort_keys=True, separators=(",", ":")))


def consent_ref(doc: Mapping[str, Any]) -> str:
    """Names the consent an account document records: the account, the sign-in or transfer that filed it, its client and its time. A
    token refresh keeps it; a new sign-in, a transfer or another client changes it. Derived from the account document's fields, never stored in it; an imported verified transfer's receipt keeps the value it filed (``service_sign_in._settle``)."""
    consent = doc.get("consent") or {}
    facts = [(doc.get("account") or {}).get("key"), doc.get("consent_nonce"), (doc.get("origin") or {}).get("generation"),
             consent.get("via"), consent.get("at"), (doc.get("client") or {}).get("client_id")]
    return _digest(json.dumps(facts))[:32]


@contextlib.contextmanager
def lock_all(grants: Collection[Grant]):
    """The cross-process locks of ``locked``, taken together and never waited for: a held one is ``sign_in_busy`` before any change,
    else every one is held to the end of the block."""
    from ouroboros import store

    with contextlib.ExitStack() as held:
        for grant in grants:
            try:
                held.enter_context(store.name_lock(grant.data_dir if grant.data_dir is not None else config.DATA_DIR, grant.lock, timeout_sec=0))
            except store.StoreBusy:
                raise SignInRequired("sign_in_busy", "a sign-in step holds its lock, or the lock could not be taken") from None
        yield


def fresh(tokens: Dict[str, Any], rejected: Optional[str], min_valid: float = 0) -> bool:
    """Unknown expiry permits ordinary use, not a promise of any positive remaining lifetime."""
    access, expires = tokens.get("access_token"), tokens.get("expires_at")
    if not access or access == rejected:
        return False
    if expires is None:
        return min_valid == 0
    try:
        return not isinstance(expires, bool) and math.isfinite(float(expires)) and time.time() + min_valid < float(expires)
    except (TypeError, ValueError, OverflowError):
        return False


async def current_token(grant: Grant, rejected: Optional[str] = None, *, min_valid: float = 0) -> str:
    return (await current_grant(grant, rejected, min_valid=min_valid))["tokens"]["access_token"]


async def refresh_request(doc: Dict[str, Any], *, scope: Optional[str] = None) -> Tuple[httpx.Response, float]:
    data = {"grant_type": "refresh_token", "refresh_token": doc["tokens"]["refresh_token"],
            **({"resource": doc["resource"]} if doc.get("resource") else {}),
            **({"scope": scope} if scope else {})}
    async with http_client() as http:
        sent_at = time.time()
        response = await token_request(http, (doc.get("metadata") or {}).get("token_endpoint", ""),
                                       doc.get("client") or {}, data)
    return response, sent_at


def _refresh_failure(grant: Grant, doc: Dict[str, Any], response: httpx.Response, *, consent_bound: bool) -> None:
    """A 400/401 refresh answer ends the grant, an empty body included, unless it is ``invalid_request`` (MCP's
    rule). A consent-bound service ends it only on ``invalid_grant``/``invalid_client``: its other answers do not
    prove the grant died, so the grant stays and the answer surfaces as the provider's error."""
    if response.status_code not in (400, 401):
        return
    reason = oauth_error(response)
    if reason == "invalid_request" or (consent_bound and reason not in ("invalid_grant", "invalid_client")):
        return
    facts: Dict[str, Any] = {"last_error": None} if "last_error" in doc else {}
    if reason == "invalid_client" and (doc.get("pending") or {}).get("client"):
        if same_registration(doc["pending"]["client"], doc.get("client") or {}):
            facts["pending"] = None
    if reason == "invalid_client" and not (doc.get("client") or {}).get("provider_binding"):
        facts["client"] = None  # MCP may need to register again; a service keeps its historical selection
    raise required(grant, doc, "refresh_rejected", reason, **facts)


async def current_grant(grant: Grant, rejected: Optional[str] = None, *, min_valid: float = 0,
                        required_scope: Optional[str] = None,
                        qualify: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
                        on_refresh: Optional[Callable[[Dict[str, Any], str], bool]] = None) -> Dict[str, Any]:
    """Read, qualify and optionally refresh once under the account lock. A consent-bound consumer
    supplies its grant check and minimum scopes; its refresh asks only for the already granted scope.
    ``on_refresh(doc, step)`` lets a consumer mark the document before a refresh leaves (``sent``: True saves it
    first; raising refuses that refresh), lift that mark when the request never left (``unsent``: a ``NOT_SENT``
    failure only) and record the provider's answer with tokens (``renewed``)."""
    try:
        min_valid = float(min_valid)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"min_valid is 0 to {OAUTH_MIN_VALID_MAX_SEC:g} seconds") from None
    if not 0 <= min_valid <= OAUTH_MIN_VALID_MAX_SEC:
        raise ValueError(f"min_valid is 0 to {OAUTH_MIN_VALID_MAX_SEC:g} seconds")
    async with locked(grant):
        doc = load(grant)
        minimum = qualify(doc) if qualify is not None and doc else None
        tokens = doc.get("tokens") or {}
        if not tokens.get("access_token"):
            raise grant.refusal((doc.get("sign_in") or {}).get("reason") or "not_signed_in")
        wanted = union(minimum, required_scope)
        missing = _note_scope_request(doc, wanted)
        if missing:
            _save_sign_in(grant, doc)
            raise grant.refusal("insufficient_scope", missing)
        if fresh(tokens, rejected, min_valid):
            return doc
        if not tokens.get("refresh_token"):
            if fresh(tokens, rejected):
                raise grant.refusal("min_valid_unmet", f"the token cannot be shown to stay valid for {min_valid:g}s")
            raise required(grant, doc, "token_rejected" if rejected == tokens["access_token"] else "expired")
        marked = on_refresh is not None and on_refresh(doc, "sent")
        if marked:
            _save_sign_in(grant, doc)  # the consumer's mark is durable before the request leaves
        try:
            response, sent_at = await refresh_request(doc, scope=tokens.get("scope") if qualify is not None else None)
            _refresh_failure(grant, doc, response, consent_bound=qualify is not None)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            if marked and on_refresh is not None and isinstance(exc, NOT_SENT):
                on_refresh(doc, "unsent")  # it never left: the consumer may lift its mark in the same write
            if qualify is not None and (not isinstance(exc, httpx.HTTPStatusError)
                                        or exc.response.status_code == 429 or exc.response.status_code >= 500):
                _save_sign_in(grant, {**doc, "last_error": {"reason": "provider_unreachable", "at": utc_now_iso()}})
            elif marked:
                _save_sign_in(grant, doc)
            raise
        # An omitted response scope inherits the scope in the refresh request. MCP without an explicit
        # refresh scope retains its prior negotiated scope and the RFC 8707 resource.
        doc["tokens"] = response_token(response, tokens.get("refresh_token"), tokens.get("scope"),
                                       sent_at=sent_at, refusal=grant.refusal)
        if on_refresh is not None:
            on_refresh(doc, "renewed")
        doc.pop("last_error", None)  # only a successful provider response clears the transient check
        if qualify is not None:
            try:
                minimum = qualify(doc)
            except SignInRequired as exc:
                # Keep the actual provider response privately for revocation; never return the bearer.
                raise required(grant, doc, exc.reason, exc.detail)
        if rejected and doc["tokens"]["access_token"] == rejected:
            raise required(grant, doc, "token_rejected")
        missing = _note_scope_request(doc, union(minimum, required_scope))
        _settle_scope_requirement(doc, doc["tokens"])
        if fresh(doc["tokens"], None) and (doc.get("sign_in") or {}).get("state") == SIGN_IN_REQUIRED:
            doc["sign_in"] = {"state": SIGNED_IN, "since": utc_now_iso()}
        _save_sign_in(grant, doc)
        if missing:
            raise grant.refusal("insufficient_scope", missing)
        if not fresh(doc["tokens"], None, min_valid):
            reason = "min_valid_unmet" if fresh(doc["tokens"], None) else "expired"
            raise grant.refusal(reason, f"the refreshed token cannot meet {min_valid:g}s of validity")
        return doc


def union(*scopes: Optional[str]) -> Optional[str]:
    """Scope words in first-seen order, with no repeats."""
    words = [word for scope in scopes if scope for word in scope.split()]
    return " ".join(dict.fromkeys(words)) or None


def check_bearer(tokens: Mapping[str, Any], ceiling: Collection[str]) -> None:
    """Qualify a consent-bound bearer by literal wire words, not comparison spellings."""
    access = tokens.get("access_token")
    if access is not None and (not isinstance(access, str) or not access or any(not 33 <= ord(c) <= 126 for c in access)):
        raise SignInRequired("invalid_token_response", "this service did not return a usable bearer token")
    if not ceiling or not isinstance(tokens.get("scope"), str):
        raise SignInRequired("provider_scope_unconfirmed", "the permission scope of this sign-in is not known; sign in again")
    if set(tokens["scope"].split(" ")) - {"", *ceiling}:
        raise SignInRequired("provider_scope_mismatch", "the provider issued permissions outside this sign-in; review permissions and sign in again")
    if str(tokens.get("token_type") or "").casefold() != "bearer":
        raise SignInRequired("invalid_token_response", "this service did not return a bearer token")


def _scope_words(scope: Optional[str], snapshot: Optional[Mapping[str, Any]] = None) -> set[str]:
    """Compare scope words without rewriting them; a consumer's snapshot may declare other spellings of one
    permission (``scope_spellings``, alias -> declared word). The declaration is the consumer's, never a
    provider name here; without it the words compare literally."""
    words = set((scope or "").split())
    spellings = (snapshot or {}).get("scope_spellings")
    if isinstance(spellings, dict):
        words = {spellings[word] if isinstance(spellings.get(word), str) and spellings[word] else word for word in words}
    return words


def _missing_scope(requested: Optional[str], tokens: Dict[str, Any],
                   snapshot: Optional[Mapping[str, Any]] = None) -> str:
    granted = _scope_words(tokens.get("scope"), snapshot)
    return " ".join(word for word in (requested or "").split() if not _scope_words(word, snapshot) <= granted)


def _note_scope_request(doc: Dict[str, Any], requested: Optional[str]) -> str:
    missing = _missing_scope(requested, doc.get("tokens") or {}, doc)
    if missing:
        prior = doc.get("required_scope")
        doc["required_scope"] = union(prior, requested)
        doc["required_scope_since"] = doc.get("required_scope_since") if prior else utc_now_iso()
    return missing


def _settle_scope_requirement(doc: Dict[str, Any], tokens: Dict[str, Any], *,
                              snapshot: Optional[Mapping[str, Any]] = None) -> None:
    """One newly issued token must satisfy the whole open request, not successive subsets."""
    if fresh(tokens, None) and not _missing_scope(doc.get("required_scope"), tokens, snapshot or doc):
        for key in ("required_scope", "required_scope_since", "challenge"):
            doc.pop(key, None)


# -- the two sign-in steps ----------------------------------------------------------------------


def _server_port() -> int:
    """The live main-server port (its port file), else the configured one."""
    from ouroboros.config import AGENT_SERVER_PORT, resolve_port_file

    with contextlib.suppress(OSError, ValueError):
        port = int(resolve_port_file().read_text(encoding="utf-8").strip())
        if port > 0:
            return port
    return int(os.environ.get("OUROBOROS_SERVER_PORT") or AGENT_SERVER_PORT)


def redirect_uri(return_base: str, rule: str) -> str:
    """Where the provider sends the owner's browser back, by the client's ``rule``: the builtin return page
    under ``return_base`` (the browser's origin with the deployment prefix) when it is loopback or HTTPS
    (``loopback_or_https``, MCP) or loopback only (``loopback``, a desktop-kind client), else under this
    server's loopback address, where a browser elsewhere stops and the owner pastes the address; or a
    web-kind client's registered address as written (``exact:<address>``, HTTPS or loopback)."""
    if rule.startswith("exact:"):
        address = rule[len("exact:"):].strip()
        parsed = urllib.parse.urlparse(address)
        if not ((parsed.scheme == "https" and parsed.hostname)
                or (parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK)):
            raise ValueError(f"return address {address!r} is neither HTTPS nor loopback")
        return address
    if rule not in ("loopback_or_https", "loopback"):
        raise ValueError(f"unknown return rule {rule!r}")
    base = (return_base or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(base)
    loopback = parsed.scheme in ("http", "https") and (parsed.hostname or "") in _LOOPBACK
    if not (loopback or (rule == "loopback_or_https" and parsed.scheme == "https" and parsed.hostname)):
        base = f"http://127.0.0.1:{_server_port()}"
    return base + RETURN_PATH


def _return_recipient(kind: str, name: Any, service: Any) -> bool:
    """Core recipient syntax mirrored by the builtin return page, not a grant or an installed-address check."""
    if not isinstance(name, str):
        return False
    if kind == "mcp":
        return (0 < len(name) <= 64 and name == name.strip("._")
                and all(c.isalnum() or c in "_.-" for c in name))
    return (canonical_skill_name(name) == name
            and isinstance(service, str) and re.fullmatch(r"[a-z0-9_]{1,40}", service) is not None)


def state(kind: str, name: str, nonce: str, **extra: str) -> str:
    """The ``state`` a provider carries back: the routing envelope ``{v: 1, <kind>: <name>, nonce, ...}``
    (unpadded base64url JSON) naming the return's one recipient; ``sign_in`` also names its ``svc``."""
    if (kind not in _KINDS or not isinstance(nonce, str) or not nonce or set(extra) & {*_RETURN_KINDS, "v", "nonce"}
            or not _return_recipient(kind, name, extra.get("svc"))):
        raise ValueError(f"no {kind!r} sign-in state for {name!r}")
    raw = json.dumps({"v": 1, kind: name, "nonce": nonce, **extra}, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def envelope(state: str, kind: Optional[str] = None) -> Dict[str, Any]:
    """The routing envelope a returned ``state`` carries (route syntax only: ``complete`` checks the nonce):
    one that names ``kind`` with its fields, or any known kind when none is given; anything else (a skill's
    own envelope, another recipient's) is a ``ValueError``."""
    refused = "not a sign-in return" + (f" for {kind!r}" if kind else "")
    if not isinstance(state, str):
        raise ValueError(refused)
    state = re.sub(r"[ \t\n\r\f]", "", state)  # the browser's atob accepts ASCII whitespace
    if "=" in state and len(state) % 4:
        raise ValueError(refused)
    try:
        raw = base64.b64decode(state + "=" * (-len(state) % 4), altchars=b"-_", validate=True)
        decoded = json.loads(raw.decode("utf-8-sig"))
    except ValueError as exc:
        raise ValueError(refused) from exc
    if not isinstance(decoded, dict):
        raise ValueError(refused)
    version, nonce = decoded.get("v"), decoded.get("nonce")
    named = [key for key in _RETURN_KINDS if key in decoded]
    # JS JSON numbers do not distinguish integer and float; Python bool must not pass as numeric 1.
    if (type(version) not in (int, float) or version != 1 or not isinstance(nonce, str) or not nonce
            or len(named) != 1 or named[0] not in _KINDS or kind not in (None, named[0])
            or not _return_recipient(named[0], decoded[named[0]], decoded.get("svc"))):
        raise ValueError(refused)
    return decoded


def parse_return(address: str) -> Dict[str, str]:
    """The provider's return parameters from the address its browser stopped on (a paste)."""
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(str(address or "").strip()).query, keep_blank_values=True)
    if any(len(query.get(key, ())) > 1 for key in _RETURN_FIELDS):
        raise ValueError("the address carries an ambiguous sign-in return")
    fields = {key: query[key][0] for key in _RETURN_FIELDS if key in query}
    if not fields.get("state") or bool(fields.get("code")) == bool(fields.get("error")):
        raise ValueError("the address carries no complete sign-in return (state and code or error)")
    return fields


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
    if isinstance(raw.get("userinfo_endpoint"), str) and raw["userinfo_endpoint"]:
        endpoints["userinfo_endpoint"] = raw["userinfo_endpoint"]  # OpenID Connect: whose account signed in
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
                    extra_params: Mapping[str, str], state: Callable[[str], str], *, issuer: str,
                    resource: Optional[str] = None, facts: Optional[Mapping[str, Any]] = None,
                    expected_attempt: str = "") -> str:
    """Start one sign-in and return the provider's authorize address for the owner to open: PKCE S256 and a
    fresh nonce, for which ``state`` makes the routing ``state`` (``functools.partial(state, kind, name,
    **extra)``). ``resource`` (RFC 8707) is sent again at the exchange and at every refresh; ``extra_params``
    (a login hint, a provider's declared parameters) may not set the engine's own. The pending sign-in, with
    the issuer and ``endpoints`` it is made for, is saved in the grant under its lock (re-read first: a call
    may have refreshed meanwhile); a later one replaces it, and the grant's tokens, issuer and endpoints stay
    until it completes. ``facts`` are the consumer's own pending facts (the chosen account, say), kept beside."""
    from mcp.client.auth.oauth2 import PKCEParameters

    if set(extra_params) & _OWN_PARAMS:
        raise ValueError(f"authorize parameters {sorted(set(extra_params) & _OWN_PARAMS)} are the engine's own")
    address = checked(str(endpoints.get("authorization_endpoint") or ""))
    pkce, nonce = PKCEParameters.generate(), secrets.token_urlsafe(24)
    params = {"response_type": "code", "client_id": client["client_id"], "redirect_uri": client["redirect_uri"],
              "state": state(nonce), "code_challenge": pkce.code_challenge, "code_challenge_method": "S256",
              **({"scope": scope} if scope else {}), **({"resource": resource} if resource else {}), **extra_params}
    parts = urllib.parse.urlsplit(address)
    replaced = _OWN_PARAMS | params.keys()
    kept = [part for part in parts.query.split("&") if part
            and urllib.parse.unquote_plus(part.partition("=")[0]) not in replaced]
    url = parts._replace(query="&".join([*kept, urllib.parse.urlencode(params)])).geturl()
    async with locked(grant):
        doc = load(grant)
        reserved = (doc.get("pending") or {}) if expected_attempt else {}
        if expected_attempt and reserved.get("attempt_id") != expected_attempt:
            raise grant.refusal("sign_in_superseded", "this sign-in is no longer current")
        doc["pending"] = {**reserved, **(facts or {}), "nonce": nonce, "code_verifier": pkce.code_verifier,
                          "client": client, "scope": scope, "issuer": issuer, "resource": resource,
                          "metadata": dict(endpoints), "started": reserved.get("started") or utc_now_iso()}
        _save_sign_in(grant, doc)
    return url


Identity = Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[Tuple[Grant, Dict[str, Any]]]]


def _save_sign_in(grant: Grant, doc: Dict[str, Any]) -> None:
    if save(grant, doc) is False:
        raise grant.refusal("sign_in_not_filed", "the sign-in could not be saved in its grant")


def _pending_for(grant: Grant, doc: Dict[str, Any], nonce: str) -> Dict[str, Any]:
    pending = doc.get("pending") or {}
    issued = pending.get("nonce")
    if not isinstance(issued, str) or not secrets.compare_digest(issued.encode(), nonce.encode(errors="surrogatepass")):
        raise grant.refusal("sign_in_superseded", "this return belongs to an older or unknown sign-in")
    return pending


def _return_code(grant: Grant, returned: Mapping[str, Any], pending: Dict[str, Any]) -> str:
    issuer, metadata, iss = pending.get("issuer") or "", pending.get("metadata") or {}, returned.get("iss")
    if iss is not None and iss != issuer:  # RFC 9207: exactly the issuer the verifier was made for
        raise grant.refusal("issuer_mismatch", f"{iss} is not {issuer}")
    if iss is None and metadata.get("authorization_response_iss_parameter_supported"):
        raise grant.refusal("issuer_missing", "the provider promised an iss parameter and sent none")
    code, error = returned.get("code"), returned.get("error")
    if bool(code) == bool(error) or (code and not isinstance(code, str)):
        raise grant.refusal("invalid_return", "the provider's return needs a code or a refusal")
    if error:
        raise grant.refusal("denied", returned.get("error_description") or error)
    return code


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


async def _exchange_for_return(grant: Grant, returned: Mapping[str, Any], nonce: str,
                               check: Optional[Callable[[Dict[str, Any]], None]]):
    """Record dispatch before the provider call; retrieve its saved answer, never resend that code."""
    async with locked(grant):
        doc = load(grant)
        pending = _pending_for(grant, doc, nonce)
        if check is not None:
            check(pending)
        code = _return_code(grant, returned, pending)
        digest = _digest(code)
        exchange = pending.get("exchange")
        if exchange:
            if exchange.get("code_sha256") != digest or not exchange.get("tokens"):
                raise grant.refusal("exchange_not_reusable", "no saved exchange for this return; check status before starting again")
            return pending, exchange["tokens"]
        pending["exchange"] = {"code_sha256": digest}
        _save_sign_in(grant, doc)

    client = pending.get("client") or {}
    data = {"grant_type": "authorization_code", "code": code,
            "redirect_uri": client.get("redirect_uri", ""), "code_verifier": pending.get("code_verifier", ""),
            **({"resource": pending["resource"]} if pending.get("resource") else {})}
    async with http_client() as http:
        sent_at = time.time()
        response = await token_request(http, (pending.get("metadata") or {}).get("token_endpoint", ""), client, data)
    if response.status_code != 200:
        raise grant.refusal("exchange_rejected", oauth_error(response))
    tokens = response_token(response, None, pending.get("scope"), sent_at=sent_at, refusal=grant.refusal)
    async with locked(grant):
        doc = load(grant)
        pending = _pending_for(grant, doc, nonce)
        if check is not None:
            check(pending)
        pending["exchange"]["tokens"] = tokens
        _save_sign_in(grant, doc)
    return pending, tokens


def _file_consent(target: Grant, doc: Dict[str, Any], signed: Dict[str, Any], nonce: str) -> Dict[str, Any]:
    _settle_scope_requirement(doc, signed["tokens"], snapshot=signed)
    doc.update(signed, consent_nonce=nonce, sign_in={"state": SIGNED_IN, "since": utc_now_iso()})
    _save_sign_in(target, doc)
    return doc


def _seal_filing(grant: Grant, current: Dict[str, Any], pending: Dict[str, Any], nonce: str, *, retain: bool) -> None:
    if retain:
        exchange = pending["exchange"]
        current["last_filed"] = {"nonce_sha256": _digest(nonce), "code_sha256": exchange["code_sha256"],
                                 "filing_target": exchange["filing_target"], "issuer": pending.get("issuer"),
                                 "attempt_id": pending.get("attempt_id"), "metadata": {
                                     "authorization_response_iss_parameter_supported": pending["metadata"].get(
                                         "authorization_response_iss_parameter_supported")}}
    current.pop("pending", None)
    _save_sign_in(grant, current)


async def _replay_filing(grant: Grant, current: Dict[str, Any], returned: Mapping[str, Any], nonce: str,
                         target_for: Callable[[str], Grant], check: Optional[Callable[[Dict[str, Any]], None]]):
    """Called with the pending lock held. A pointer proves no effect until the account agrees."""
    pending = current.get("pending") or {}
    if pending:
        if pending.get("nonce") != nonce:
            return None
        if check is not None:
            check(pending)
        receipt = pending.get("exchange") or {}
        source = pending
    else:
        receipt = current.get("last_filed") or {}
        if receipt.get("nonce_sha256") != _digest(nonce):
            return None
        source = receipt
    name = receipt.get("filing_target")
    if not name:
        return None
    if not secrets.compare_digest(receipt["code_sha256"], _digest(_return_code(grant, returned, source))):
        raise grant.refusal("exchange_not_reusable", "no saved exchange for this return")
    target = target_for(name)
    async with locked(target):
        filed = load(target)
        if filed.get("consent_nonce") != nonce:
            raise grant.refusal("sign_in_not_filed", "the earlier save is no longer confirmed; check the account")
        if check is not None:
            check(filed)
        if pending:
            _seal_filing(grant, current, pending, nonce, retain=True)
        return filed


async def complete(grant: Grant, returned: Mapping[str, Any], *, identity: Optional[Identity] = None,
                   check: Optional[Callable[[Dict[str, Any]], None]] = None,
                   target_for: Optional[Callable[[str], Grant]] = None,
                   before_filing: Optional[Callable[[Dict[str, Any], Grant], None]] = None,
                   kept: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """File under pending → account locks. Replays of a confirmed split write do no provider work. ``kept``: facts
    an account filed under another grant keeps from its first filing, written only where it lacks them."""
    nonce = envelope(returned.get("state") or "")["nonce"]

    async def replay():
        if target_for is None:
            return None
        async with locked(grant):
            return await _replay_filing(grant, load(grant), returned, nonce, target_for, check)

    if target_for is not None and (filed := await replay()) is not None:
        return filed
    try:
        pending, tokens = await _exchange_for_return(grant, returned, nonce, check)
        target, facts = (grant, {}) if identity is None else await identity(tokens, pending)
        signed = {"issuer": pending.get("issuer"), "resource": pending.get("resource"),
                  "metadata": pending.get("metadata") or {}, "client": pending.get("client") or {},
                  "tokens": tokens, **facts}
        async with locked(grant):
            current = load(grant)
            pending = _pending_for(grant, current, nonce)
            if check is not None:
                check(pending)
            if before_filing is not None:
                before_filing(pending, target)
            if target.path == grant.path:  # MCP: the target and its pending operation are one file
                current.pop("pending")
                return _file_consent(grant, current, signed, nonce)
            async with locked(target):
                exchange = pending["exchange"]
                filed = load(target)
                filing_target = target.path.stem if target_for is not None else target.path.name
                if exchange.get("filing_target"):
                    if exchange["filing_target"] != filing_target or filed.get("consent_nonce") != nonce:
                        raise grant.refusal("sign_in_not_filed", "the account's earlier save was not confirmed")
                else:
                    exchange["filing_target"] = filing_target
                    _save_sign_in(grant, current)
                    first = {name: value for name, value in (kept or {}).items() if name not in filed}
                    filed = _file_consent(target, filed, {**first, **signed}, nonce)
                _seal_filing(grant, current, pending, nonce, retain=target_for is not None)
                return filed
    except SignInRequired as exc:
        if exc.reason == "sign_in_superseded" and target_for is not None and (filed := await replay()) is not None:
            return filed  # another return completed the exact same operation while this one waited
        raise


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
