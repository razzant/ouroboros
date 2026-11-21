"""What a private OpenAI-compatible connection may send while its program is not identified.

``llm_ollama`` reads a connection's metadata before every generation. When that read identifies
nothing, the facts kept here decide, and they name a base URL in two ways, never one for the other:

* its **address** (``scope_id``), the normalized origin and path prefix. An address where Ollama was
  seen holds the send, whatever key, query, header or model asks, until the connection's own read
  identifies its program. Broad on purpose: it withholds on doubt.
* its **endpoint** (``endpoint_id``), every part that decides where a request built on it goes,
  userinfo, the whole path and the query included. Permission to send ``/v1`` belongs to one
  connection at one endpoint: the owner's revocable consent for it, or a route a server is on record as
  answering under an earlier body (``answered_before``), kept only while that connection still serves there
  (``retire_prior_routes``). Narrow on purpose: another tenant behind one origin inherits nothing.

A Settings Test, a successful answer or a label admits nothing, and no admission lifts a hold.

Both names are stored as digests alone: the evidence store keeps the address id with the type,
version, time and source of what was read there, and a saved consent or prior route is the endpoint
id. No key, header, userinfo, query or path text is written with either.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, Iterable, Mapping
from urllib.parse import urlsplit

from ouroboros.llm_ollama_wire import OllamaRouteUnsettledError

HOLDS = "ollama_addresses"  # address -> the Ollama read there: {type, version, observed_at, source}
PRIOR_ROUTES = "compatible_prior_routes"  # connection id -> the endpoint a server answered it at under an earlier body
CHECKS = "compatible_checks"  # connection at an endpoint -> what its own last check identified
TOLD = "compatible_cautions_told"  # digest of a route (connection, endpoint, admission) whose caution a chat has heard -> when
CONSENT_FIELD = "continue_unconfirmed"  # a compatible connection's field (``provider_models.PROVIDER_KINDS``)
CONSENT_LABEL = "Continue via the compatible API without confirmation that history is preserved"
UNIDENTIFIED = "unknown"
#: A stored word, never a URL or an id: consent explicitly withdrawn where a launch environment still holds one
#: (``provider_connections.canonical_builtin_consent`` — a blank save cannot outweigh a non-empty environment).
#: ``consent_token`` takes it for no consent; ``web/modules/connection_server_check.js`` pairs it.
CONSENT_REVOKED = "revoked"
#: What a Settings Test that reached ``/v1`` on such a server did not confirm, said beside its outcome.
TEST_CAUTION = "Server type not confirmed: Ouroboros cannot tell whether it keeps the whole history."
_ID = re.compile(r"[0-9a-f]{24}")


class UnidentifiedRouteWithheld(OllamaRouteUnsettledError):
    """Nothing was sent: the address is held, or the connection has no admission. Only the owner changes it."""
    kind, retryable = "bad_request", False
    route_withheld = True  # its words are the reason of each task they stop (``loop_llm_call.note_route_caution``)

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.route_caution = message


class EvidenceStoreUnavailable(OSError):
    """An existing evidence store could not be read or parsed. A decision that needs the store defers with
    this; it never proceeds on an imagined empty one. An absent file is a legitimately empty store, not this."""


def store_checked(root: Any = None) -> Mapping[str, Any]:
    """The store for a DECISION (a send's admission, the migration snapshot, a retirement, Settings' facts):
    ``{}`` only where no file exists, ``EvidenceStoreUnavailable`` when one exists and cannot be read or parsed.
    Observation writers keep the lenient shared read (``_store``): a mark that cannot be stored tells again."""
    from ouroboros import capability_evidence as ce
    from ouroboros.utils import json_snapshot

    try:
        return json_snapshot(ce._store_path(root if root is not None else ce.canonical_evidence_root()),
                             strict=True) or {}
    except ValueError as exc:
        raise EvidenceStoreUnavailable(str(exc)) from exc


def _wire_path(path: str) -> str:
    """The path as the transport sends it: httpx's own normalization of the dot segments, applied before a
    request leaves, so equivalent spellings of one prefix are one address — and one endpoint — on the wire.
    Percent encoding stays as written (``raw_path``): the transport's projection, not a re-reading of the RFC."""
    if "/." not in path:
        return path
    import httpx

    return httpx.URL("http://h" + ("" if path.startswith("/") else "/") + path).raw_path.decode("ascii", "replace")


def _one_host(host: str) -> str:
    """An IP literal in its one spelling, whichever form the OS takes was written (``127.1``, hex, an
    IPv4-mapped IPv6 address as its IPv4); a name as written, never resolved."""
    from ouroboros.server_window import host_address

    address = host_address(host)
    if address is None:
        return host
    return f"[{address.compressed}]" if address.version == 6 else str(address)


def _http_parts(base_url: Any) -> tuple:
    """``(scheme, hostname, port, url)`` of an http(s) base URL, the default port as its number; ``()`` for none."""
    try:
        url = urlsplit(str(base_url or "").strip())
        scheme, host = url.scheme.lower(), (url.hostname or "").lower()
        port = url.port or {"http": 80, "https": 443}.get(scheme)
    except ValueError:
        return ()
    return (scheme, host, port, url) if scheme in ("http", "https") and host else ()


def address_scope(base_url: Any) -> str:
    """The normalized origin and path prefix requests built on ``base_url`` reach, '' for no http(s) address.

    One host name in any case, with or without its root dot, and one IP address in any spelling are one
    origin; a default port is its number, and a trailing ``/v1`` or slash is no prefix. Two names or
    addresses stay two origins even when both reach this machine (``localhost``, ``127.0.0.1``, ``::1``):
    each may be another server. The prefix is the path as the transport sends it (``_wire_path``): equivalent
    spellings of one prefix hold one address. Userinfo, query and fragment are not part of an address."""
    from ouroboros.server_window import server_root

    parts = _http_parts(base_url)
    if not parts or not parts[1].rstrip("."):
        return ""
    scheme, host, port, url = parts
    return f"{scheme}://{_one_host(host.rstrip('.'))}:{port}{server_root(_wire_path(url.path))}"


def scope_id(base_url: Any) -> str:
    """The stored name of that address: a digest, so no path token or host is written in clear."""
    scope = address_scope(base_url)
    return hashlib.sha256(scope.encode("utf-8")).hexdigest()[:24] if scope else ""


def endpoint_id(base_url: Any) -> str:
    """The stored name of the destination a permission covers, '' for no http(s) address: a digest of every
    part of the base URL that decides where a request built on it goes.

    Only what no server can tell apart is one endpoint: the scheme and host in any case, a default port
    and its number, and one trailing slash of a path no query follows (the client appends its own).
    Userinfo, the host as spelled, the path as the transport sends it (``_wire_path``) with its ``/v1``
    and the query, an empty one too, each make another endpoint: another account, tenant or server behind one origin, where
    ``address_scope`` sees one address. A fragment is never sent. The digest is not the address id of
    the same URL."""
    parts = _http_parts(base_url)
    if not parts:
        return ""
    scheme, host, port, url = parts
    asked = "?" in str(base_url).partition("#")[0]  # a stock client appends its path after a query, even an empty one
    wire = _wire_path(url.path)
    named = ("endpoint", scheme, url.netloc.rpartition("@")[0], host, str(port),
             wire if asked else wire.removesuffix("/"), "?" + url.query if asked else "")
    return hashlib.sha256("\0".join(named).encode("utf-8")).hexdigest()[:24]


def shown_address(base_url: Any) -> str:
    """The server an owner-facing sentence names: scheme, host and port. A path prefix can carry a tenant or a
    token, so words never show it (the ids keep it, digested); userinfo and query never reach the words either."""
    from ouroboros.request_wire_contract import normalize_endpoint

    url = urlsplit(normalize_endpoint(base_url))
    return f"{url.scheme}://{url.netloc}" if url.scheme and url.netloc else "its address"


def consent_token(value: Any) -> str:
    """A consent as it is saved: the id of the endpoint it covers. A base URL (a form's, an operator's
    variable) becomes its id, an id stays, the explicit revocation word (``CONSENT_REVOKED``) and anything
    else is no consent."""
    text = str(value or "").strip()
    if text == CONSENT_REVOKED:
        return ""
    return text if _ID.fullmatch(text) else endpoint_id(text)


def consented(base_url: Any, consent: Any) -> bool:
    """The owner's consent names an endpoint: it stands only for the one the connection has now."""
    endpoint = endpoint_id(base_url)
    return bool(endpoint) and consent_token(consent) == endpoint


def _connection(target: Mapping[str, Any]) -> str:
    from ouroboros.provider_models import connection_id
    return connection_id(str(target.get("provider") or "openai-compatible"), str(target.get("connection") or ""))


def _check_key(*parts: str) -> str:
    """A connection at an endpoint, and what else a row is kept for, as a digest: nothing stored names either."""
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]


def _store() -> dict:
    from ouroboros import capability_evidence as ce
    return ce._read(ce.canonical_evidence_root())


def _put(namespace: str, key: str, value: dict, root: Any = None) -> bool:
    from ouroboros import capability_evidence as ce

    root = root or ce.canonical_evidence_root()
    try:
        store_checked(root)
    except EvidenceStoreUnavailable:
        return False  # an unreadable store is never "repaired" by overwriting it: the write defers with it
    return ce._store_evidence(root, namespace, key, value)


def note_ollama(target: Mapping[str, Any], version: str, source: str) -> bool:
    """Keep that Ollama answered at this address, from a send, a Settings Test or a saved native route.

    Written when the address has no such row or reports another version; a read without a version
    never replaces one. Returns whether the address holds its row now."""
    from ouroboros.utils import utc_now_iso

    scope = scope_id(target.get("base_url"))
    held = (_store().get(HOLDS) or {}).get(scope)
    if not scope or isinstance(held, dict) and not (version and held.get("version") != version):
        return bool(scope)
    return _put(HOLDS, scope, {"type": "ollama", "version": version, "observed_at": utc_now_iso(), "source": source})


def note_check(target: Mapping[str, Any], program: str, version: str = "") -> None:
    """What this connection's own check before a send identified at its endpoint, with the version it read
    when one is given, rewritten only when either changes (Settings)."""
    from ouroboros.utils import utc_now_iso

    endpoint = endpoint_id(target.get("base_url"))
    key = _check_key(_connection(target), endpoint)
    held = (_store().get(CHECKS) or {}).get(key) or {}
    if endpoint and (held.get("type"), held.get("version", "")) != (program, version):
        _put(CHECKS, key, {"type": program, **({"version": version} if version else {}), "observed_at": utc_now_iso()})


def _admission(connection: str, base_url: Any, consent: Any, store: Mapping[str, Any]) -> str:
    endpoint = endpoint_id(base_url)
    if consented(base_url, consent):
        return "consent"
    prior = (store.get(PRIOR_ROUTES) or {}).get(connection)
    return "prior" if endpoint and isinstance(prior, dict) and prior.get("endpoint") == endpoint else ""


def held(address: str) -> UnidentifiedRouteWithheld:
    """The hold's own words: Ollama was read at the address earlier, and nothing names this connection's program
    now. They say what was seen there, never what this server is."""
    return UnidentifiedRouteWithheld("ollama_guard_held", (
        f"Ollama was seen at {address} before, but this connection's program was not identified for this "
        "request, so nothing was sent. Let its metadata be read (GET /api/version under its base URL) or use "
        "another configured route."))


def unidentified(target: Mapping[str, Any], address: str) -> tuple[str, str]:
    """A fresh read left no information about this connection's program.

    Returns the caution a permitted ``/v1`` send carries with the route it is told for (``first_telling``), or
    raises what withholds it: the address hold first, which no admission lifts, then a missing admission. An
    evidence store that cannot be read withholds the send too: its holds and permissions are unknown, never
    imagined empty."""
    try:
        store = store_checked()
    except EvidenceStoreUnavailable as exc:
        raise OllamaRouteUnsettledError(
            "The route evidence store could not be read, so its holds and permissions are unknown; "
            "request not sent") from exc
    if (store.get(HOLDS) or {}).get(scope_id(target.get("base_url"))):
        raise held(address)
    connection = _connection(target)
    admission = _admission(connection, target.get("base_url"), target.get(CONSENT_FIELD), store)
    named = str(target.get("connection") or "")
    if not admission:
        from ouroboros.provider_models import PROVIDER_KINDS
        where = (f"{CONSENT_FIELD} in the {named} row of OUROBOROS_CONNECTIONS" if named
                 else PROVIDER_KINDS["openai-compatible"].flat_key(CONSENT_FIELD))
        raise UnidentifiedRouteWithheld("ollama_guard_consent", (
            f"The server at {address} does not show which program it runs, so Ouroboros cannot confirm that it "
            f"keeps the whole history; nothing was sent. To allow it, turn on “{CONSENT_LABEL}” for this "
            f"connection in Settings → Accounts and keys (headless: set {where} to this connection's base URL)."))
    # The words show the server's origin and name the connection, never its path prefix or query: the route a chat
    # hears them for is the connection at its whole endpoint under its admission, minted beside them (``first_telling``).
    which = f"the {named} connection" if named else "this connection"
    allowed = (f"by your consent for {which}" if admission == "consent"
               else f"because {which} is on record as answering a request before this check existed")
    return (f"The server at {address} did not show which program it runs. This request uses OpenAI-compatible "
            f"/v1 {allowed}; Ouroboros cannot confirm that the server keeps the whole history: it may shorten a "
            "long conversation without saying so."), _check_key(connection, endpoint_id(target.get("base_url")), admission)


def caution_told(route: str) -> bool:
    """Whether a chat line already carried this route's caution (a read; ``mark_caution_told`` writes it)."""
    return route in (_store().get(TOLD) or {})


def mark_caution_told(route: str) -> None:
    """Record the route's caution as told. A mark that could not be stored tells again."""
    from ouroboros.utils import utc_now_iso

    _put(TOLD, route, {"told_at": utc_now_iso()})


def first_telling(route: str) -> bool:
    """Whether a chat hears a permitted send's caution for the first time, marking it told when so: it is said
    once, at the first such use of a route (``unidentified``), and Settings keeps saying it. Kept by the route's
    digest: the words cannot tell two routes apart. A caller whose line must reach the chat before the route
    counts as told splits the two (``caution_told`` / ``mark_caution_told``): an undelivered word tells again."""
    if caution_told(route):
        return False
    mark_caution_told(route)
    return True


def answered_before(settings: Mapping[str, Any], root: Any,
                    resolve: Callable[..., Dict[str, Any]]) -> list[tuple[str, Any]]:
    """``(connection id, base URL)`` of the compatible connections a server is on record as answering under an
    earlier body, where ``settings`` send them now: the one proof that a route predates this check. The record
    is an ACCEPTED request — a positive token count under the very same bound target; a refusal-only history
    proves nothing about what a server keeps. What it establishes is bounded the same way: the configured
    target matched an earlier body's acceptance at the migration that read it, never continuity since.

    ``resolve`` builds one route's target from the same ``settings`` — the start that asks
    passes ``llm_routing.resolve_target``, kept a parameter so this leaf never imports the router.

    The record is a body's own: since 22.10.2026 each kept the size a server accepted or refused under the route
    that was sent (``context_window.record_observation``), named by a digest of its kind, model, address and
    credential (``bound_target_scope``). A model route ``settings`` name is on record only under that same digest,
    so nothing a start is merely given (a settings document, a launch variable, a settings path) is prior by
    being there: a connection never answered, or one whose model, address or key is not the recorded one, asks
    consent. None once this check has written of a route in the store: a record may then be this body's own,
    sent under a consent. Raises ``EvidenceStoreUnavailable`` when an existing store cannot be read: the
    snapshot defers rather than take an imagined empty store for the absence of every earlier route."""
    from ouroboros.context_window import bound_target_scope
    from ouroboros.llm_attempt import ConnectionUnavailable
    from ouroboros.provider_connections import model_routes
    from ouroboros.provider_models import connection_id, parse_model_route

    store = store_checked(root)
    if any(store.get(mine) for mine in (PRIOR_ROUTES, CHECKS, TOLD)):
        return []
    answered = {scope for scope, row in (store.get("main_windows") or {}).items()
                if isinstance(row, dict) and any(
                    isinstance(obs, dict) and int(obs.get("accepted") or 0) > 0
                    for obs in row.get("observations") or [])}
    routes: dict = {}
    for model in model_routes(settings) if answered else ():
        route = parse_model_route(model)
        if route.kind != "openai-compatible" or not route.valid:
            continue
        try:
            target = resolve(model, dict(settings))
        except ConnectionUnavailable:
            continue  # a connection that cannot serve now has no route to keep
        if bound_target_scope(target, model) in answered:
            routes[connection_id(route.kind, route.connection)] = target["base_url"]
    return list(routes.items())


def record_prior_routes(routes: Iterable[tuple[str, Any]], root: Any = None) -> int:
    """The one snapshot of routes that predate this check (``answered_before``), each ``(connection id, base
    URL)`` kept as the endpoint it had. Returns how many are kept; one whose row could not be stored is not,
    and asks the owner's consent."""
    from ouroboros.utils import utc_now_iso

    kept = 0
    for connection, base_url in routes:
        endpoint = endpoint_id(base_url)
        if endpoint and _put(PRIOR_ROUTES, connection, {"endpoint": endpoint, "recorded_at": utc_now_iso()}, root):
            kept += 1
    return kept


def retire_prior_routes(serving: Callable[[], Iterable[tuple[str, Any]]], root: Any = None, *,
                        reviving: Callable[[], Iterable[tuple[str, Any]]] | None = None) -> int:
    """A prior route lives only while its connection serves at the endpoint it was kept for: every other one
    is deleted, for good. ``serving()`` is ``(connection id, base URL)`` of the compatible connections that can
    serve now (``provider_connections.compatible_addresses``), asked only where a route is kept.

    So a connection that was removed, switched off or moved, and one made again under its name at the same
    address, has no prior route and asks the owner's consent; a rotated key leaves its connection serving where
    it was. ``reviving()`` (a settings write, before its bytes land) is what the document it writes will serve:
    only a route that write would bring back into force is deleted then, so no other write waits on the store.
    Returns how many were retired; raises ``OSError`` when the store could not be read or could not take the
    deletion: an unreadable store defers the retirement, it does not read as having nothing to retire."""
    from ouroboros import capability_evidence as ce

    root = root or ce.canonical_evidence_root()
    if not store_checked(root).get(PRIOR_ROUTES):
        return 0
    endpoints = {connection: endpoint_id(base_url) for connection, base_url in serving()}
    revived = None if reviving is None else {connection: endpoint_id(base_url) for connection, base_url in reviving()}

    def retired(rows: Mapping[str, Any]) -> list:
        ends = {connection: row.get("endpoint") if isinstance(row, dict) else "" for connection, row in rows.items()}
        return [connection for connection, end in ends.items()
                if not (end and endpoints.get(connection) == end) and (revived is None or end and revived.get(connection) == end)]

    if not retired(store_checked(root).get(PRIOR_ROUTES) or {}):
        return 0
    with ce._store_transaction(root):
        data = ce._load(root)
        rows = data.get(PRIOR_ROUTES) or {}
        gone = retired(rows)
        for connection in gone:
            del rows[connection]
        if gone and not ce._save(root, data):
            raise OSError("the retired prior routes could not be saved")
    return len(gone)


def connection_facts(connection: str, base_url: Any, consent: Any) -> dict:
    """What Settings shows of a private compatible connection, from stored facts alone: its admission, the
    Ollama read at its address and what its own last check identified at its endpoint, an Ollama whose
    version is too old for its native chat with the release to update to (``llm_ollama.update_to``, the
    transport's own threshold). Nothing here reads the network. A store that cannot be read blocks no page:
    the consent the settings themselves prove still shows, and the stored facts are named unreadable rather
    than imagined empty."""
    from ouroboros.llm_ollama import update_to
    from ouroboros.server_window import private_host

    scope = scope_id(base_url)
    if not scope or not private_host(base_url):
        return {}
    try:
        store = store_checked()
    except EvidenceStoreUnavailable:
        return {"unidentified": {"admission": "consent" if consented(base_url, consent) else "",
                                 "evidence": "unreadable"}}
    facts: dict = {"admission": _admission(connection, base_url, consent, store)}
    for name, row in (("ollama", (store.get(HOLDS) or {}).get(scope)),
                      ("checked", (store.get(CHECKS) or {}).get(_check_key(connection, endpoint_id(base_url))))):
        if isinstance(row, dict):
            facts[name] = {key: row[key] for key in ("type", "version", "observed_at") if row.get(key)}
            if row.get("type") == "ollama" and (needed := update_to(str(row.get("version") or ""))):
                facts[name]["update_to"] = needed
    return {"unidentified": facts}
