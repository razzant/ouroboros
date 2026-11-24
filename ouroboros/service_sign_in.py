"""Sign-ins of skills to the services they declare, held by the core.

A skill declares a service (``service_sign_in_declaration``); the owner signs in for that skill, choosing
one of the issuer's clients first, and the grant lands in one file per account,
``state/sign_in/grants/<skill>/<service>/<account>.json`` (0600; the account key hashes the issuer and the
subject the provider names). That file is the owner's per-skill Allow: only the owner's completed sign-in
for that skill, or the skill handing over a grant it already held, writes it. Skill processes ask the Host
Service for an access token (``gateway/host_sign_in.py``); the refresh token, the client secret and other
skills' grants never leave the host. Client registrations bind to a discovered issuer or an approved
explicit map; a grant keeps its issuing client and destinations, an account its ``connection_id`` until disconnect. A new sign-in answers for the registration (for a client the host registers, the provider) and permissions
shown, and with its return for the skill's current bytes (``_declared`` ``admit``); leaving one needs neither. Disconnect is
local; revoke forgets an account only once the provider confirmed every token, and the owner's either first leaves a
non-secret row in the skill's own ``sign_in_withdrawn.json``. When both locks are needed, the service's pending lock
precedes the account's.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import math
import os
import pathlib
import re
import secrets
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import httpx

from ouroboros import config, oauth_engine
from ouroboros import service_sign_in_declaration as declaration
from ouroboros.contracts.skill_manifest import canonical_skill_name
from ouroboros.oauth_engine import SIGN_IN_REQUIRED, SIGNED_IN, SignInRequired
from ouroboros.utils import atomic_write_json, utc_now_iso

_ACCOUNT = re.compile(r"a-[0-9a-f]{20}")
_PROVIDER_ROLES = {"authorization": "authorization_endpoint", "token": "token_endpoint",
                   "identity": "userinfo_endpoint", "revocation": "revocation_endpoint"}


class SignInNotFound(LookupError):
    """An undeclared service, an unknown skill or account: nothing to sign in to or read. ``reason`` says which to the one reader that
    tells a skill its next step (``gateway/host_sign_in.py``): unknown_service, unknown_account, account_required, not_signed_in or
    no_usable_account, empty when the finder did not say; ``accounts`` are the saved accounts its own read found, ``(key, label)``."""

    def __init__(self, message: str = "", reason: str = "", accounts: Iterable[Tuple[str, str]] = ()) -> None:
        super().__init__(message)
        self.reason, self.accounts = reason, tuple(accounts)


_SIGN_IN_DATA_DIR: contextvars.ContextVar[Optional[pathlib.Path]] = contextvars.ContextVar(
    "sign_in_data_dir", default=None)


@contextlib.contextmanager
def at_data_dir(data_dir: pathlib.Path):
    """Bind the store to the trusted host which authenticated this request."""
    token = _SIGN_IN_DATA_DIR.set(pathlib.Path(data_dir).resolve())
    try:
        yield
    finally:
        _SIGN_IN_DATA_DIR.reset(token)


def _sign_in_data_dir() -> pathlib.Path:
    root = _SIGN_IN_DATA_DIR.get()
    return root if root is not None else config.DATA_DIR


def _service_sign_in_root() -> pathlib.Path:
    return _sign_in_data_dir() / "state" / "sign_in"


def _grant(skill: str, service: str, name: str) -> oauth_engine.Grant:
    if not (canonical_skill_name(skill) == skill and declaration._ID.fullmatch(service)
            and (name == "pending" or _ACCOUNT.fullmatch(name))):
        raise SignInNotFound(f"no sign-in {skill}/{service}/{name}")
    return oauth_engine.Grant(path=_service_sign_in_root() / "grants" / skill / service / f"{name}.json",
                              lock=f"sign_in/grants/{skill}/{service}/{name}.lock",
                              binding={"skill": skill, "service": service}, data_dir=_sign_in_data_dir(), strict_read=True)


def _clients_file() -> oauth_engine.Grant:
    return oauth_engine.Grant(path=_service_sign_in_root() / "clients.json", lock="sign_in/clients.lock",
                              data_dir=_sign_in_data_dir(), strict_read=True)


def account_key(issuer: str, subject: str, *, skill: str = "") -> str:
    """``a-`` and 20 hex of sha256(issuer NUL subject); an unconfirmed import also mixes in its skill."""
    raw = "\0".join((issuer, skill, subject) if skill else (issuer, subject))
    return "a-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _connection_id() -> str:
    return "cn-" + secrets.token_hex(16)


def _connection(doc: Dict[str, Any]) -> Dict[str, Any]:
    """A confirmed account's ``connection_id`` (opaque, not secret): refreshes and its subject's later consent keep
    it, with any client; a sign-in after Disconnect starts another (``complete``, ``_file_handover``)."""
    known = doc.get("connection_id")
    return {"connection_id": known} if known and (doc.get("account") or {}).get("verified") is True else {}


def _unconfirmed_move(doc: Dict[str, Any]) -> bool:
    return bool(doc) and (doc.get("account") or {}).get("verified") is not True and (
        doc.get("consent") or {}).get("via") == "handover"


def _declared(skill_name: str, service_id: str, admit: bool = False) -> declaration.Service:
    """The service a skill declares. ``admit``: also that its current bytes may act now, as the Host Service admits them in every
    review mode; a new sign-in and its return ask, leaving one (cancel, disconnect, revoke) never does."""
    from ouroboros.deployment_extensions import admission_refusal
    from ouroboros.skill_loader import find_skill

    if canonical_skill_name(skill_name) != skill_name:
        raise SignInNotFound("unknown skill", "unknown_service")
    skill = find_skill(_sign_in_data_dir(), skill_name)
    service = declaration.declared(skill).get(service_id) if skill is not None else None
    if service is None:
        raise SignInNotFound(f"skill {skill_name!r} declares no service {service_id!r}", "unknown_service")
    if admit and (refusal := admission_refusal(skill)):
        raise SignInRequired("skill_" + refusal, "the skill's current files may not act now")
    return service


def declared_service(skill_name: str, service_id: str) -> Optional[declaration.Service]:
    try:
        return _declared(skill_name, service_id)
    except SignInNotFound:
        return None


def _words(scope: Any) -> List[str]:
    return [word for word in str(scope or "").split() if word]


def _json(response: httpx.Response) -> Dict[str, Any]:
    try:
        body = response.json() if response.status_code == 200 else {}
    except ValueError:
        body = {}
    return body if isinstance(body, dict) else {}


def _epoch(value: Any) -> Optional[float]:
    """An expiry as epoch seconds, from seconds or an ISO 8601 time (skills store either)."""
    if value in (None, ""):
        return None
    try:
        expires = float(value)
        return expires if not isinstance(value, bool) and math.isfinite(expires) else 0.0
    except (TypeError, ValueError, OverflowError):
        from datetime import datetime

        with contextlib.suppress(OverflowError, OSError, ValueError):
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    return 0.0  # unreadable: treated as expired, so the host refreshes it


def _explicit_issuer(endpoints: Dict[str, str]) -> str:
    parsed = declaration.provider_url(endpoints["authorization"], allow_loopback=True)
    return f"{parsed.scheme}://{parsed.netloc}"


def _issuer_of(service: declaration.Service) -> str:
    # In explicit mode this keeps the existing account namespace/display value. It is not the binding.
    return service.issuer or _explicit_issuer(dict(service.endpoints))


def _provider_binding(service: declaration.Service) -> Dict[str, Any]:
    return {"issuer": _issuer_of(service), "endpoints": None if service.issuer else dict(service.endpoints)}


def _owner_binding(issuer: str, endpoints: Any) -> Dict[str, Any]:
    if endpoints is None:
        return {"issuer": declaration.canonical_issuer(issuer), "endpoints": None}
    approved = declaration.explicit_endpoints(endpoints)
    if issuer != _explicit_issuer(approved):
        raise ValueError("the issuer does not describe this set of explicit endpoints")
    return {"issuer": issuer, "endpoints": approved}


def _matches_provider(row: Dict[str, Any], binding: Dict[str, Any]) -> bool:
    # Old issuer-only rows prove neither mode. In particular, a missing key is not discovery.
    return "endpoints" in row and all(row.get(key) == binding[key] for key in ("issuer", "endpoints"))


def _client_rows(skill: str, binding: Dict[str, Any], rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources = ("owner", "registered", f"handover:{skill}")
    return [row for row in rows if row.get("source") in sources and _matches_provider(row, binding)]


def _account_files(skill: str, service: str) -> List[pathlib.Path]:
    folder = _grant(skill, service, "pending").path.parent
    try:
        return sorted(path for path in folder.iterdir() if path.suffix == ".json" and _ACCOUNT.fullmatch(path.stem))
    except FileNotFoundError:
        return []
    except OSError:
        raise SignInRequired("sign_in_storage_unavailable", "the saved accounts could not be listed") from None


def clients() -> List[Dict[str, Any]]:
    rows = oauth_engine.load(_clients_file()).get("clients", [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SignInRequired("sign_in_storage_unavailable", "the saved clients could not be read")
    return rows


def _public_client(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": row.get("id", ""), "label": row.get("label", ""), "issuer": row.get("issuer", ""),
            "client_id_tail": str(row.get("client_id") or "")[-4:], "return": row.get("return", "loopback"),
            "source": row.get("source", "owner"), "secret_set": bool(row.get("client_secret"))}


# -- the issuer and its clients -------------------------------------------------------------------


async def _endpoints(service: declaration.Service, http: httpx.AsyncClient) -> Tuple[Dict[str, Any], str, Any]:
    """Resolve one mode. Discovered destinations come only from authenticated issuer metadata."""
    if service.issuer:
        metadata, endpoints, issuer = await oauth_engine.discover_issuer(
            service.issuer, service.issuer, service.issuer, http)
        try:
            matches = declaration.canonical_issuer(issuer) == service.issuer
        except ValueError:
            matches = False
        if not matches:
            raise SignInRequired("issuer_mismatch", "discovery did not confirm the approved issuer")
    else:
        metadata, issuer = None, _issuer_of(service)
        endpoints = {_PROVIDER_ROLES[role]: address for role, address in service.endpoints.items()}
        endpoints["authorization_response_iss_parameter_supported"] = False
    if not endpoints.get("userinfo_endpoint"):
        raise SignInRequired("identity_unavailable", "the provider names no account endpoint")
    if not endpoints.get("authorization_endpoint") or not endpoints.get("token_endpoint"):
        raise SignInRequired("invalid_provider_metadata", "the provider names no complete sign-in addresses")
    try:
        for key in ("authorization_endpoint", "token_endpoint", "userinfo_endpoint", "revocation_endpoint",
                    "registration_endpoint"):
            if key in endpoints:
                address = endpoints[key]
                declaration.provider_url(address, allow_loopback=not service.issuer)
                oauth_engine.checked(address)
    except ValueError as exc:
        raise SignInRequired("invalid_provider_metadata", "the provider names an unsupported address") from exc
    return endpoints, issuer, metadata


def _client_row(skill: str, binding: Dict[str, Any], client: str, *, registering: bool = False) -> Dict[str, Any]:
    rows = _client_rows(skill, binding, clients())
    if registering and rows:
        raise SignInRequired("client_required", "a client is now available; choose it on the card")
    if client:
        rows = [row for row in rows if row.get("id") == client] or [row for row in rows if row.get("label") == client]
        if not rows:
            raise SignInRequired("unknown_client", "this client is not available for this service")
    if len(rows) > 1:
        raise SignInRequired("client_required", "choose one client by its ID")
    return rows[0] if rows else {}


def _client_id_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _return_rule(value: Any) -> str:
    if value in (None, "", "loopback"):
        return "loopback"
    if not isinstance(value, str) or not value.startswith("exact:"):
        raise ValueError("a client's return must be loopback or an exact address")
    address = oauth_engine.redirect_uri("", value)
    declaration.provider_url(address, allow_loopback=True)
    return "exact:" + address


async def client_save(issuer: str, client: Dict[str, Any], *, endpoints: Any = None) -> Dict[str, Any]:
    """Save a client at the owner door; an edit does not change its provider or source."""
    binding = _owner_binding(issuer, endpoints)
    ret = _return_rule(client["return"]) if "return" in client else None
    label, client_id = _client_id_text(client.get("label")), _client_id_text(client.get("client_id"))
    row_id, secret = client.get("id"), client.get("client_secret")
    if any(client.get(key) is not None and not isinstance(client[key], str)
           for key in ("id", "label", "client_id", "client_secret")):
        raise ValueError("client fields must be text")
    if not label or not (client_id or row_id):
        raise ValueError("a client needs a label and a client ID")
    async with oauth_engine.locked(_clients_file()):
        rows = clients()
        if row_id:
            row = next((saved for saved in rows if saved.get("id") == row_id), None)
            if row is None or not _matches_provider(row, binding):
                raise ValueError("this client does not belong to the selected provider; add it separately")
            if row.get("source") != "owner" and (
                    (client_id and client_id != row.get("client_id")) or (secret and secret != row.get("client_secret"))):
                raise ValueError("add a separate owner client to change this registration")
        else:
            row = {**binding, "id": "c-" + secrets.token_hex(6), "source": "owner", "added_at": utc_now_iso()}
            rows.append(row)
        row.update(label=label[:80], client_id=client_id or row.get("client_id"),
                   **{"return": ret if ret is not None else row.get("return", "loopback")})
        if secret:
            row["client_secret"] = secret
        oauth_engine.save(_clients_file(), {"clients": rows})
    return _public_client(row)


async def client_remove(client: str) -> bool:
    """Remove one client row; existing grants keep their own copy and keep working."""
    async with oauth_engine.locked(_clients_file()):
        rows = clients()
        kept = [row for row in rows if row.get("id") != client]
        oauth_engine.save(_clients_file(), {"clients": kept})
    return len(kept) != len(rows)


def _engine_client(row: Dict[str, Any], issuer: str, redirect: str, metadata: Any) -> Dict[str, Any]:
    methods = getattr(metadata, "token_endpoint_auth_methods_supported", None) or []
    method = "none" if not row.get("client_secret") else (
        "client_secret_basic" if "client_secret_post" not in methods and "client_secret_basic" in methods
        else "client_secret_post")
    if row.get("source") == "registered":
        method = row.get("token_endpoint_auth_method") or method
    return {"id": row.get("id", ""), "label": row.get("label", ""), "client_id": row.get("client_id", ""),
            "client_secret": row.get("client_secret") or None, "token_endpoint_auth_method": method,
            "redirect_uri": redirect, "issuer": issuer, "return": row.get("return", "loopback"),
            "provider_binding": {"issuer": row["issuer"], "endpoints": row["endpoints"]}}


def _check_binding(service: Optional[declaration.Service], snapshot: Dict[str, Any]) -> None:
    """Use the immutable, approved registration and its resolved destinations, never today's picker row."""
    client, metadata = snapshot.get("client"), snapshot.get("metadata")
    issuer = snapshot.get("issuer")
    try:
        if not isinstance(client, dict) or not client.get("client_id") or not isinstance(metadata, dict):
            raise ValueError
        saved = client.get("provider_binding")
        if not isinstance(saved, dict) or "endpoints" not in saved or not issuer or client.get("issuer") != issuer:
            raise ValueError
        expected = _provider_binding(service) if service is not None else _owner_binding(saved["issuer"], saved["endpoints"])
        if saved != expected:
            raise ValueError
        if service is not None and snapshot.get("identity_fields") != dict(service.identity):
            raise ValueError
        explicit = saved["endpoints"]
        if explicit is not None:
            if issuer != saved["issuer"] or any(metadata.get(key) != explicit.get(role)
                                                for role, key in _PROVIDER_ROLES.items()):
                raise ValueError
        elif declaration.canonical_issuer(issuer) != saved["issuer"]:
            raise ValueError
        for role, key in _PROVIDER_ROLES.items():
            if role != "revocation" or key in metadata:
                declaration.provider_url(metadata.get(key), allow_loopback=explicit is not None)
                oauth_engine.checked(metadata[key])
    except (KeyError, TypeError, ValueError):
        raise SignInRequired("provider_approval_required", "use a client approved for this provider and sign in again") from None


def _attempt_scope_words(pending: Dict[str, Any]) -> List[str]:
    return pending.get("scope_words", _words(pending.get("scope")))  # older attempts: only the literal request


def _qualify(service: declaration.Service, doc: Dict[str, Any]) -> Optional[str]:
    _check_binding(service, doc)
    scopes = doc.get("scopes")
    recorded = scopes.get("requested") if isinstance(scopes, dict) else None
    known = isinstance(recorded, list) and all(isinstance(w, str) and w.split() == [w] for w in recorded)
    allowed = service.literal_scopes().intersection(recorded if known else ())
    if (tokens := doc.get("tokens") or {}).get("access_token"):
        oauth_engine.check_bearer(tokens, allowed)
    # A handover can serve its old scopes without adding the card's identity requirements.
    if (doc.get("consent") or {}).get("via") == "handover":
        names = oauth_engine._scope_words(" ".join(allowed), doc)
        return oauth_engine.union(*(s for s in service.scopes if oauth_engine._scope_words(s, doc) <= names))
    return oauth_engine.union(*service.scopes)


def _single_account(skill: str, service: declaration.Service) -> str:
    if service.accounts != "one":
        return ""
    files = _account_files(skill, service.id)
    if len(files) > 1:
        raise SignInRequired("account_selection_required", "disconnect the other accounts on the skill's card")
    return files[0].stem if files else ""


# -- the two sign-in steps ------------------------------------------------------------------------


async def begin(skill: str, service_id: str, *, client: str = "", optional_scopes: Iterable[str] = (),
                login_hint: str = "", account: str = "", return_base: str = "", via: str = "card",
                register_client: bool = False, scope_words: Optional[Iterable[str]] = None,
                registration: str = "", shown_issuer: str = "") -> Dict[str, Any]:
    """Reserve one attempt for the exact permissions and saved registration the owner was shown (``oauth_engine.registration_ref``;
    none when the host registers one: the provider's issuer shown instead), before provider preparation. A skill whose current bytes may not act starts none."""
    service = _declared(skill, service_id, True)
    binding = _provider_binding(service)
    if register_client and client:
        raise ValueError("choose a saved client or register one, not both")
    row = _client_row(skill, binding, client, registering=register_client)
    redirect = oauth_engine.redirect_uri(return_base, "loopback")
    if not row and (not register_client or not service.issuer):
        raise SignInRequired("no_client", f"add a client for {binding['issuer']} that returns to {redirect}")
    optional = set(optional_scopes)
    chosen = [item.scope for item in service.optional_scopes if item.scope in optional]
    scope = oauth_engine.union(*service.scopes, *chosen)  # an unchecked permission stays off
    requested = _words(scope)
    spellings = service.spellings_for(requested)
    ceiling = [*requested, *spellings]
    if set(scope_words or ()) != set(ceiling):
        raise SignInRequired("sign_in_choices_changed", "review the current permissions before starting sign-in")
    if row and registration != oauth_engine.registration_ref(row, return_base):
        raise SignInRequired("sign_in_choices_changed", "review the saved client before starting sign-in")
    if (shown_issuer or not row) and shown_issuer != binding["issuer"]:  # a client the host registers answers for the provider shown; a saved one, for its registration
        raise SignInRequired("sign_in_choices_changed", "review the provider before starting sign-in")
    pending = _grant(skill, service_id, "pending")
    attempt = "s-" + secrets.token_hex(16)
    async with oauth_engine.locked(pending):
        _declared(skill, service_id, True)  # a deletion or a switch-off while this waited for the lock ends it here
        current = oauth_engine.load(pending)
        active = _single_account(skill, service)
        if active and account and active != account:
            raise SignInRequired("account_mismatch", "disconnect the current account before choosing another")
        account = account or active
        if account and not _ACCOUNT.fullmatch(account):
            raise SignInNotFound("unknown account")
        if account and not _load_account(_grant(skill, service_id, account)):
            raise SignInNotFound(f"no account {account!r} for {skill}/{service_id}")
        current["pending"] = {"attempt_id": attempt, "account": account or None, "started": utc_now_iso()}
        current.pop("last_filed", None)
        oauth_engine._save_sign_in(pending, current)

    async with oauth_engine.http_client() as http:
        endpoints, issuer, metadata = await _endpoints(service, http)
        if not row:
            if not getattr(metadata, "registration_endpoint", None):
                raise SignInRequired("no_client", f"add a client for {issuer} that returns to {redirect}")
            row = await _register_if_empty(skill, service_id, binding, metadata, redirect, http, issuer=issuer)
    redirect = oauth_engine.redirect_uri(return_base, row.get("return") or "loopback")
    params = {**service.authorize_params, **({"login_hint": login_hint} if login_hint else {})}
    issuing_client = _engine_client(row, issuer, redirect, metadata)
    identity_fields = dict(service.identity)
    _check_binding(service, {"issuer": issuer, "client": issuing_client, "metadata": endpoints,
                             "identity_fields": identity_fields})
    url = await oauth_engine.authorize(
        pending, endpoints, issuing_client, scope, params,
        functools.partial(oauth_engine.state, "sign_in", skill, svc=service_id), issuer=issuer,
        expected_attempt=attempt, facts={"optional_chosen": chosen, "identity_fields": identity_fields, "via": via,
                                         "scope_words": ceiling, "scope_spellings": spellings})
    return {"ok": True, "url": url, "redirect_uri": redirect, "client": {"id": row.get("id"), "label": row.get("label")},
            "scope": scope or "", "attempt_id": attempt}


async def cancel(skill: str, service_id: str, *, attempt_id: str = "") -> Dict[str, Any]:
    _declared(skill, service_id)
    grant = _grant(skill, service_id, "pending")
    async with oauth_engine.locked(grant):
        doc = oauth_engine.load(grant)
        pending = doc.get("pending") or {}
        if pending and (pending.get("attempt_id") or "") != attempt_id:
            raise SignInRequired("sign_in_superseded", "this cancel does not identify the current sign-in")
        if pending:
            doc.pop("pending")
            oauth_engine._save_sign_in(grant, doc)
    return {"ok": True, "cancelled": bool(pending), **({"attempt_id": attempt_id} if attempt_id else {})}


async def _register_if_empty(skill: str, service_id: str, binding: Dict[str, Any], metadata: Any, redirect: str,
                             http: httpx.AsyncClient, *, issuer: str) -> Dict[str, Any]:
    """Serialize the empty-list check and registration with saves, imports and other registrations."""
    grant = _clients_file()
    async with oauth_engine.locked(grant):
        try:  # the skill and its binding as they are now, after the wait for this lock
            current = _provider_binding(_declared(skill, service_id, True))
        except SignInNotFound:
            raise SignInRequired("sign_in_skill_gone", "the skill no longer declares this sign-in; nothing was saved") from None
        if current != binding:
            raise SignInRequired("sign_in_choices_changed", "the skill's provider changed meanwhile; nothing was saved")
        rows = clients()
        if _client_rows(skill, binding, rows):
            raise SignInRequired("client_required", "a client is now available; choose it on the card")
        try:
            registered = await oauth_engine.register_client(metadata, redirect, http, issuer=issuer)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise  # the request never left: nothing was asked of the provider
        except httpx.HTTPError as exc:  # it may have registered a client this installation never received
            raise SignInRequired("registration_unconfirmed", "the provider may have registered a client that was never received; no new client is available here") from exc
        method = registered["token_endpoint_auth_method"]
        secret = registered.get("client_secret")
        if (not registered.get("client_id") or method not in ("none", "client_secret_basic", "client_secret_post")
                or (method != "none" and not secret)):
            raise SignInRequired("registration_rejected", "the provider returned an unsupported client registration")
        row = {**binding, "id": "c-" + secrets.token_hex(6), "label": "Registered by Ouroboros",
               "client_id": registered["client_id"], "client_secret": secret if method != "none" else None,
               "token_endpoint_auth_method": method, "return": "exact:" + redirect,
               "source": "registered", "added_at": utc_now_iso()}
        oauth_engine._save_sign_in(grant, {"clients": [*rows, row]})
    return row


async def _identity_info(tokens: Dict[str, Any], snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The identity endpoint's account; ``{}`` when it names none, ``None`` when it refused the bearer (401)."""
    async with oauth_engine.http_client() as http:
        response = await http.get(oauth_engine.checked(snapshot["metadata"]["userinfo_endpoint"]), headers={
            "Authorization": f"Bearer {tokens['access_token']}", "Accept": "application/json"})
    if response.status_code == 429 or response.status_code >= 500:
        response.raise_for_status()
    return None if response.status_code == 401 else _json(response)


def _subject(info: Dict[str, Any], identity: Dict[str, str]) -> str:
    subject = info.get(identity["subject"])
    return str(subject) if (isinstance(subject, str) and subject.strip()) or type(subject) is int else ""


async def _identity(tokens: Dict[str, Any], pending: Dict[str, Any], *, skill: str,
                    service: declaration.Service) -> Tuple[oauth_engine.Grant, Dict[str, Any]]:
    ceiling = _attempt_scope_words(pending)
    oauth_engine.check_bearer(tokens, ceiling)
    if not oauth_engine.fresh(tokens, None):
        raise SignInRequired("expired", "the provider returned an expired token; sign in again")
    fields = pending["identity_fields"]
    try:
        info = await _identity_info(tokens, pending) or {}
    except httpx.HTTPError as exc:  # the provider's answer to the code is already saved with this attempt; only the account read has no answer yet
        raise SignInRequired("answer_saved", "the provider's answer is saved, but the account it names could not be read yet; finishing this same return reads it again "
                             "and does not exchange the code a second time") from exc
    subject = _subject(info, fields)
    if not subject:
        raise SignInRequired("identity_unavailable", "the provider did not confirm which account signed in")
    key = account_key(pending["issuer"], subject)
    asked = pending.get("account")  # ``one``: the unconfirmed moved row this sign-in was sent to repair, as now saved
    moved = _load_account(_grant(skill, service.id, asked)) if service.accounts == "one" and asked and asked != key else {}
    replaces = {"key": asked, "generation": (moved.get("origin") or {}).get("generation")} if _unconfirmed_move(moved) else None
    return _grant(skill, service.id, key), {
        "account": {"key": key, "subject": subject, "label": str(info.get(fields["label"]) or subject), "verified": True},
        "client_ref": pending["client"].get("id"), "last_error": None, "identity_fields": fields, "replaces": replaces,
        "attempt_id": pending.get("attempt_id"),
        "scopes": {"requested": ceiling, "optional_chosen": pending.get("optional_chosen") or [],
                   "granted": _words(tokens.get("scope"))},
        "scope_spellings": pending.get("scope_spellings") or {},
        "consent": {"via": pending.get("via") or "card", "at": utc_now_iso()}}


def _before_filing(skill: str, service: declaration.Service, pending: Dict[str, Any], target: oauth_engine.Grant) -> None:
    """A sign-in sent to repair one account never rewrites another saved one. Another subject files as a new
    account (``many``) or, for ``one``, in place of the unconfirmed moved row it was sent to (``_retire_replaced``);
    it is never matched to that row by label or email."""
    service = _declared(skill, service.id)
    key, asked, active = target.path.stem, pending.get("account"), _single_account(skill, service)
    if key in (pending.get("excluded_accounts") or []):
        raise SignInRequired("sign_in_superseded", "this account was disconnected after the attempt began")
    if service.accounts == "one":
        others = [value for value in {active, asked} if value and value != key
                  and not _unconfirmed_move(_load_account(_grant(skill, service.id, value)))]
    else:
        others = [asked] if asked and asked != key and _load_account(target) else []
    if others:
        raise SignInRequired("account_mismatch", "sign in as the current account, or disconnect it before switching")


async def _retire_replaced(skill: str, service: declaration.Service, doc: Dict[str, Any]) -> None:
    """``one``: a filed confirmed sign-in withdraws, as Disconnect does, only the unconfirmed moved row it was sent
    to repair (``replaces``: that row and the write the sign-in saw), and only while that row is still that write."""
    replaced = doc.get("replaces") or {}
    if service.accounts != "one" or not replaced.get("key") or (doc.get("account") or {}).get("verified") is not True:
        return

    def same(held: Dict[str, Any]) -> bool:
        return _unconfirmed_move(held) and (held.get("origin") or {}).get("generation") == replaced.get("generation")

    with contextlib.suppress(SignInNotFound):
        await withdraw(skill, service.id, replaced["key"], revoke_at_provider=False, expect=same)


async def complete(returned: Dict[str, Any], *, attempt_id: str = "") -> Dict[str, Any]:
    envelope = oauth_engine.envelope(returned.get("state") or "", "sign_in")
    skill, service_id, nonce = envelope["sign_in"], envelope["svc"], envelope["nonce"]
    service = _declared(skill, service_id)
    grant = _grant(skill, service_id, "pending")

    def check(snapshot: Dict[str, Any]) -> None:
        if attempt_id and attempt_id != snapshot.get("attempt_id"):
            raise SignInRequired("sign_in_superseded", "this return is for another sign-in")
        current = _declared(skill, service_id, "nonce" in snapshot)  # an open attempt needs the skill admitted; replaying a filed return does not
        _check_binding(current, snapshot)
        if "nonce" in snapshot and set(_attempt_scope_words(snapshot)) - current.literal_scopes():
            raise SignInRequired("sign_in_superseded", "the service's permissions changed; start sign-in again")

    try:
        doc = await oauth_engine.complete(grant, returned, identity=functools.partial(_identity, skill=skill, service=service),
                                          check=check, target_for=functools.partial(_grant, skill, service_id),
                                          before_filing=functools.partial(_before_filing, skill, service),
                                          kept={"connection_id": _connection_id()})
    except SignInRequired as exc:
        # A refusal may quote the public id only if the supplied nonce identifies it. Do not guess on a stale return.
        with contextlib.suppress(SignInRequired):
            pending = oauth_engine.load(grant).get("pending") or {}
            if pending.get("nonce") == nonce and pending.get("attempt_id"):
                exc.attempt_id = pending["attempt_id"]
        raise
    await _retire_replaced(skill, service, doc)
    row = _account_row(doc, service, [])
    return {"ok": True, "skill": skill, "service": service_id, "label": service.label, "state": row["state"],
            **({"reason": row["reason"]} if row.get("reason") else {}), "attempt_id": doc["attempt_id"],
            "account": {"key": row["key"], "label": row["label"]}, "scope": " ".join(row["scope"]),
            "missing_required": row["missing_required"]}


# -- the skill's tokens ---------------------------------------------------------------------------


def _account_problem(doc: Dict[str, Any], service: declaration.Service) -> str:
    tokens = doc.get("tokens") or {}
    try:
        minimum = _qualify(service, doc)
    except SignInRequired as exc:
        return exc.reason
    if not tokens.get("access_token"):
        return (doc.get("sign_in") or {}).get("reason") or "not_signed_in"
    if _first_proof(doc) == "unconfirmed":  # a moved grant's first renewal: unconfirmed and never repeated
        return "handover_unconfirmed"
    if oauth_engine._missing_scope(minimum, tokens, doc):
        return "insufficient_scope"
    if not (oauth_engine.fresh(tokens, None) or tokens.get("refresh_token")):
        return "expired"
    return ""


def _bound_account(grant: oauth_engine.Grant, doc: Dict[str, Any]) -> Dict[str, Any]:
    account = doc.get("account")
    if doc and (not isinstance(account, dict) or account.get("key") != grant.path.stem):
        raise SignInRequired("sign_in_storage_unavailable", "the saved account could not be identified")
    return doc


def _load_account(grant: oauth_engine.Grant) -> Dict[str, Any]:
    return _bound_account(grant, oauth_engine.load(grant))


def _account_documents(skill: str, service: str) -> List[Dict[str, Any]]:
    return [doc for path in _account_files(skill, service)
            if (doc := _load_account(_grant(skill, service, path.stem)))]


def _for_account(exc: SignInRequired, doc: Dict[str, Any]) -> SignInRequired:
    row = doc.get("account") or {}
    exc.account, exc.account_label = row.get("key", ""), row.get("label", "")
    return exc


def _pick(skill: str, service: declaration.Service, account: str) -> oauth_engine.Grant:
    if account and not _ACCOUNT.fullmatch(account):
        raise SignInNotFound("unknown account", "unknown_account")
    if account and service.accounts != "one":
        grant = _grant(skill, service.id, account)
        if not _load_account(grant):
            raise SignInNotFound(f"no account {account!r} for {skill}/{service.id}", "unknown_account")
        return grant
    rows = _account_documents(skill, service.id)
    eligible = [doc for doc in rows if not _account_problem(doc, service)]
    if service.accounts == "one" and len(eligible) > 1:
        raise SignInRequired("account_selection_required", "disconnect the other accounts on the skill's card")
    if account:
        if not any(doc["account"]["key"] == account for doc in rows):
            raise SignInNotFound(f"no account {account!r} for {skill}/{service.id}", "unknown_account")
        return _grant(skill, service.id, account)
    if len(eligible) == 1:
        return _grant(skill, service.id, eligible[0]["account"]["key"])
    if not eligible and len(rows) == 1:
        # Explain why there is no eligible grant without selecting it for a provider call.
        raise _for_account(SignInRequired(_account_problem(rows[0], service)), rows[0])
    shown = [(doc["account"]["key"], doc["account"].get("label", "")) for doc in (eligible or rows)][:8]  # what this pick read: no second scan
    if eligible:
        raise SignInNotFound("name the account", "account_required", shown)
    raise SignInNotFound(f"no eligible account is signed in to {service.id}", "no_usable_account" if rows else "not_signed_in", shown)


def _first_proof(doc: Dict[str, Any]) -> Optional[str]:
    """A moved grant's first-refresh phase: ``due`` (its refresh token not yet proven), ``renewed`` (answered and
    saved, its identity read pending), ``unconfirmed`` (that refresh left without an answer, or its saved answer is no
    longer fresh: never repeated, never handed out); None once proven or not moved."""
    phase = (doc.get("origin") or {}).get("first_refresh") if (doc.get("consent") or {}).get("via") == "handover" else None
    if phase == "sent" or (phase == "renewed" and not oauth_engine.fresh(doc.get("tokens") or {}, None)):
        return "unconfirmed"
    return phase


def _first_refresh_mark(doc: Dict[str, Any], step: str) -> bool:
    """``current_grant``'s refresh hook for that phase: the first refresh leaves once, marked before it leaves; one that
    never left (``unsent``) is due again, while a timeout after sending stays unconfirmed."""
    phase = _first_proof(doc)
    if phase is None:
        return False
    if step in ("renewed", "unsent"):
        doc["origin"]["first_refresh"] = "renewed" if step == "renewed" else "due"
    elif phase == "unconfirmed":
        raise SignInRequired("handover_unconfirmed", "this moved sign-in's first renewal got no clear answer; use the skill's sign-in card")
    elif phase == "renewed":
        raise SignInRequired("sign_in_storage_unavailable", "this moved sign-in's renewal still waits for its account check")
    else:
        doc["origin"]["first_refresh"] = "sent"
    return True


async def _prove_refresh(grant: oauth_engine.Grant, doc: Dict[str, Any]) -> bool:
    """A moved grant's first refreshed bearer reaches the skill only once the provider names the account the move
    confirmed (the move proved its handed access token, not its refresh token). For a saved refresh only this
    identity read repeats (429/5xx raise and keep it). The answer decides only the exact saved state it was read
    for: False when a newer refresh, consent or disconnect replaced that state, so the caller reads it again."""
    subject = _subject(await _identity_info(doc["tokens"], doc) or {}, doc["identity_fields"])
    async with oauth_engine.locked(grant):
        current = _load_account(grant)
        if any(current.get(name) != doc.get(name) for name in ("tokens", "origin", "consent", "connection_id")):
            return False
        if subject and subject == (current.get("account") or {}).get("subject"):
            current["origin"] = {k: v for k, v in current["origin"].items() if k != "first_refresh"}
            oauth_engine._save_sign_in(grant, current)
            return True
        raise oauth_engine.required(grant, {k: v for k, v in current.items() if k != "tokens"},
                                    "identity_mismatch" if subject else "identity_unavailable")


async def token(skill: str, service_id: str, account: str = "", *, rejected: Optional[str] = None,
                required_scope: str = "", min_valid: float = 0, connection_id: str = "") -> Dict[str, Any]:
    """The skill's bearer for one account. An expected ``connection_id`` is compared under the account's lock before any
    refresh: only a confirmed account with exactly that one matches, never a missing one."""
    service = _declared(skill, service_id)
    grant = _pick(skill, service, account)

    def qualify(doc: Dict[str, Any]) -> Optional[str]:
        if connection_id and _connection(doc).get("connection_id") != connection_id:
            raise SignInRequired("connection_changed", "this account is not the connection the skill expects")
        current = _declared(skill, service_id)
        minimum = _qualify(current, _bound_account(grant, doc))
        declared = oauth_engine._scope_words(" ".join(current.declared_scopes()), doc)
        return oauth_engine.union(minimum, *(word for word in _words(required_scope)
                                             if oauth_engine._scope_words(word, doc) <= declared))

    try:
        for _ in range(3):  # at most: the first refresh, its identity read, then the ordinary read
            saved = _load_account(grant)
            phase = _first_proof(saved) if (saved.get("tokens") or {}).get("access_token") else None
            if phase == "unconfirmed":  # never handed out, with an explicit account or without one
                raise SignInRequired("handover_unconfirmed", "this moved sign-in's first renewal is not confirmed; use the skill's sign-in card")
            if phase == "renewed":
                qualify(saved)  # the saved grant still fits today's declaration and choices before any provider call
                if not await _prove_refresh(grant, saved):
                    continue  # a newer refresh, consent or disconnect replaced what was read
            doc = await oauth_engine.current_grant(grant, rejected, min_valid=min_valid, qualify=qualify,
                                                   on_refresh=_first_refresh_mark)
            if _first_proof(doc) not in ("renewed", "unconfirmed"):
                break
        else:
            raise SignInRequired("sign_in_storage_unavailable", "the sign-in changed while its account was being confirmed")
    except SignInRequired as exc:
        exc.account = grant.path.stem
        with contextlib.suppress(SignInRequired):
            exc.account_label = (_load_account(grant).get("account") or {}).get("label", "")
        raise
    tokens = doc["tokens"]
    return {"access_token": tokens["access_token"], "token_type": tokens["token_type"],
            "expires_at": tokens.get("expires_at"), "scope": tokens["scope"],
            "account": {"key": grant.path.stem, "label": (doc.get("account") or {}).get("label", "")}, **_connection(doc)}


async def withdraw(skill: str, service_id: str, account: str, *, revoke_at_provider: bool,
                   consent: str = "", expect: Optional[Callable[[Dict[str, Any]], bool]] = None) -> Dict[str, Any]:
    """Withdraw one skill grant, whether or not the skill may start a sign-in now. Disconnect and Revoke both answer for the consent
    shown (``oauth_engine.consent_ref``). Disconnect forgets the account here and asks nobody; Revoke asks the provider and forgets it
    only once the provider confirmed every token: any other outcome keeps it, tokens included, for the owner to check at the provider
    and then Disconnect."""
    if account and not _ACCOUNT.fullmatch(account):
        raise SignInNotFound("unknown account")
    control = _grant(skill, service_id, "pending")
    async with oauth_engine.locked(control):
        if not account:
            files = _account_files(skill, service_id)
            if len(files) != 1:
                raise SignInNotFound("name the account" if files else "no account to withdraw")
            account = files[0].stem
        grant = _grant(skill, service_id, account)
        async with oauth_engine.locked(grant):
            doc = _load_account(grant)
            if not doc:
                raise SignInNotFound(f"no account for {skill}/{service_id}")
            if expect is not None and not expect(doc):
                return {"ok": True, "withdrawn": False}
            if expect is None and consent != oauth_engine.consent_ref(doc):  # a public action answers for the consent shown
                raise SignInRequired("consent_changed", "this account was signed in again after it was shown; review it first")
            current, key = oauth_engine.load(control), grant.path.stem
            current["withdrawn_accounts"] = list(dict.fromkeys([*(current.get("withdrawn_accounts") or []), key]))
            pending = current.get("pending") or {}
            target = (pending.get("exchange") or {}).get("filing_target")
            if pending and (target == key or pending.get("account") == key):
                current.pop("pending", None)
            elif pending and not target:
                pending["excluded_accounts"] = list(dict.fromkeys([*(pending.get("excluded_accounts") or []), key]))
            if (current.get("last_filed") or {}).get("filing_target") == key:
                current.pop("last_filed", None)
            outcome = await _request_revocation(doc) if revoke_at_provider else {}
            if revoke_at_provider and not outcome.get("revoked"):  # not confirmed or not asked: kept, nothing is forgotten
                return {"ok": False, "code": "SIGN_IN_UNAVAILABLE", "error": outcome["reason"], **outcome}
            if expect is None:  # the owner's break reaches the skill's own folder before the host forgets anything
                await _record_withdrawal(skill, service_id, doc)
            # Write the no-refile fact before forgetting the account. A failure at unlink is not a success.
            oauth_engine._save_sign_in(control, current)
            grant.path.unlink(missing_ok=True)
            return {"ok": True, **outcome}


async def _request_revocation(doc: Dict[str, Any]) -> Dict[str, Any]:
    """``{"revoked": True}`` only when the provider confirmed every token. Else why not: ``revocation_not_asked`` (nothing was sent: no
    approved binding, revocation address or token) or ``revocation_not_confirmed`` (something was asked and not every answer was a
    confirmation: a refusal, a lost answer or only part of them). Neither is a refusal by the provider; neither forgets the account."""
    try:
        _check_binding(None, doc)  # use this grant's own historical binding, never a replacement endpoint
    except SignInRequired:
        return {"reason": "revocation_not_asked", "revocation_reason": "provider_approval_required"}
    endpoint, tokens = doc["metadata"].get("revocation_endpoint"), doc.get("tokens") or {}
    if not endpoint or not (tokens.get("refresh_token") or tokens.get("access_token")):
        return {"reason": "revocation_not_asked", "revocation_reason": "no_endpoint" if not endpoint else "no_tokens"}
    if await oauth_engine.revoke(endpoint, doc["client"], tokens) is True:
        return {"revoked": True}
    return {"reason": "revocation_not_confirmed", "revocation_reason": "not_confirmed"}


WITHDRAWN_FILE = "sign_in_withdrawn.json"  # in the skill's own state folder, where it still reads it after a core rollback


def _withdrawals_file(skill: str) -> oauth_engine.Grant:
    from ouroboros.skill_loader import skill_state_dir_path

    return oauth_engine.Grant(path=skill_state_dir_path(_sign_in_data_dir(), skill) / WITHDRAWN_FILE,
                              lock=f"skills/{skill}/sign_in_withdrawn.lock", data_dir=_sign_in_data_dir(), strict_read=True)


def _withdrawals(skill: str) -> Optional[List[Dict[str, Any]]]:
    """The owner's recorded breaks for the skill; ``None`` when the record cannot be read, never an empty list."""
    try:
        doc = oauth_engine.load(_withdrawals_file(skill))
    except SignInRequired:
        return None
    rows = doc.get("accounts") if doc else []  # no record yet is no break; a record of another form is unknown, never empty
    return rows if (not doc or doc.get("v") == 1) and isinstance(rows, list) and all(isinstance(row, dict) for row in rows) else None


def _broken(skill: str, service_id: str, item: Dict[str, Any]) -> bool:
    """A recorded break of the handed grant's account, by its legacy key or subject. A record that cannot be read is no
    knowledge either way, so a refusal is then never ``none``."""
    rows, names = _withdrawals(skill), (("legacy_key", item.get("legacy_key")), ("subject", item.get("subject")))
    return rows is None or any(row.get("service") == service_id and any(
        value and row.get(field) == value for field, value in names) for row in rows)


async def _record_withdrawal(skill: str, service_id: str, doc: Dict[str, Any]) -> None:
    """Append the owner's break as one non-secret row; a repeated break adds its own. A record that cannot be read is
    never replaced and a failed write is never confirmed: either fails the withdrawal before the host forgets anything."""
    account, origin = doc.get("account") or {}, doc.get("origin") or {}
    row = {"service": service_id, "account": account.get("key", ""),
           **({"subject": account["subject"]} if account.get("verified") is True and account.get("subject") else {}),
           **{name: origin[name] for name in ("legacy_key", "legacy_generation") if origin.get(name)},
           **({"connection_id": doc["connection_id"]} if doc.get("connection_id") else {}),
           "via": (doc.get("consent") or {}).get("via", ""), "at": utc_now_iso()}
    target = _withdrawals_file(skill)
    async with oauth_engine.locked(target):
        rows = _withdrawals(skill)
        if rows is None:
            raise SignInRequired("sign_in_storage_unavailable", "the skill's record of withdrawals cannot be read; nothing was forgotten")
        if not any({k: v for k, v in old.items() if k != "at"} == {k: v for k, v in row.items() if k != "at"} for old in rows):
            target.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(target.path, {"v": 1, "accounts": [*rows, row]})
            with contextlib.suppress(OSError):
                os.chmod(target.path, 0o600)


# -- what the owner and the skill see -------------------------------------------------------------


def _all_grants() -> List[Dict[str, Any]]:
    folder = _service_sign_in_root() / "grants"
    docs = []
    for path in sorted(folder.glob("*/*/a-*.json")) if folder.is_dir() else []:
        skill, service = path.parent.parent.name, path.parent.name
        if not (canonical_skill_name(skill) == skill and declaration._ID.fullmatch(service) and _ACCOUNT.fullmatch(path.stem)):
            continue
        if doc := _load_account(_grant(skill, service, path.stem)):
            docs.append(doc)
    return docs


def _account_row(doc: Dict[str, Any], service: declaration.Service, everyone: List[Dict[str, Any]]) -> Dict[str, Any]:
    account, status = doc.get("account") or {}, doc.get("sign_in") or {}
    tokens, client = doc.get("tokens") or {}, doc.get("client") or {}
    problem = _account_problem(doc, service)
    granted = _words(tokens.get("scope")) if problem not in (
        "provider_approval_required", "provider_scope_mismatch", "provider_scope_unconfirmed", "invalid_token_response") else []
    # Lost tokens are not current access, even if the original response is still known.
    comparable = oauth_engine._scope_words(" ".join(granted), doc)
    row = {"key": account.get("key", ""), "label": account.get("label", ""), "subject": account.get("subject", ""),
           "verified": account.get("verified") is True, "state": SIGN_IN_REQUIRED if problem else SIGNED_IN,
           "since": status.get("since"), "scope": granted,
           "client": {"id": client.get("id", ""), "label": client.get("label", "")},
           "missing_required": [s for s in service.scopes if not oauth_engine._scope_words(s, doc) <= comparable],
           "not_allowed": [o.scope for o in service.optional_scopes if o.scope in ((doc.get("scopes") or {}).get(
               "optional_chosen") or []) and not oauth_engine._scope_words(o.scope, doc) <= comparable],
           "via": (doc.get("consent") or {}).get("via", ""), "siblings": [
               {"skill": other.get("skill"), "service": other.get("service")} for other in everyone
               if other.get("skill") != doc.get("skill") and other.get("issuer") == doc.get("issuer")
               and client.get("provider_binding") is not None
               and (other.get("client") or {}).get("provider_binding") == client["provider_binding"]
               and (other.get("client") or {}).get("client_id") == client.get("client_id")
               and account.get("verified") is True and (other.get("account") or {}).get("verified") is True
               and (other.get("account") or {}).get("subject") == account.get("subject")]}
    if doc.get("attempt_id"):
        row["attempt_id"] = doc["attempt_id"]
    row["consent"] = oauth_engine.consent_ref(doc)  # what Disconnect and Revoke confirm against
    if problem:
        row["reason"] = problem
    if requirement := doc.get("required_scope"):
        row.update(required_scope=requirement, required_scope_since=doc.get("required_scope_since"))
        if not problem and (missing := oauth_engine._missing_scope(requirement, tokens, doc)):
            row.update(reason="insufficient_scope", missing_step_up=missing.split())
    if doc.get("last_error"):
        row["last_error"] = doc["last_error"]
    return row


def accounts(skill: str, service: declaration.Service) -> List[Dict[str, Any]]:
    rows = []
    for path in _account_files(skill, service.id):
        doc = _load_account(_grant(skill, service.id, path.stem))
        if not doc:
            raise SignInRequired("sign_in_storage_unavailable", "the account list changed while it was being read")
        generation = (doc.get("origin") or {}).get("legacy_generation")
        rows.append({**_account_row(doc, service, []), **({"legacy_generation": generation} if generation else {}),
                     **_connection(doc)})
    return rows


def status(*, return_base: str = "") -> Dict[str, Any]:
    """Every installed skill's declared services, its accounts, the pending sign-in and the issuers' clients;
    never a token, secret or verifier."""
    from ouroboros.deployment_extensions import admission_refusal
    from ouroboros.skill_loader import discover_skills

    rows, everyone, known = [], _all_grants(), clients()
    for skill in discover_skills(_sign_in_data_dir()):
        if canonical_skill_name(skill.name) != skill.name:
            continue
        admission = admission_refusal(skill)
        for service in declaration.declared(skill).values():
            pending = oauth_engine.load(_grant(skill.name, service.id, "pending")).get("pending") or {}
            binding = _provider_binding(service)
            rows.append({
                "skill": skill.name, "service": service.id, "label": service.label, "issuer": binding["issuer"],
                **({"endpoints": binding["endpoints"]} if binding["endpoints"] is not None else {}),
                "scopes": list(service.scopes), "scope_labels": dict(service.scope_labels),
                "scope_spellings": {word: list(aliases) for word, aliases in service.scope_spellings.items()},
                "optional_scopes": [{"scope": o.scope, "label": o.label, "default": o.default}
                                    for o in service.optional_scopes],
                "accounts_mode": service.accounts, "optional": service.optional, "notes": dict(service.notes),
                "admission": admission,
                "accounts": [_account_row(doc, service, everyone) for doc in everyone
                             if doc.get("skill") == skill.name and doc.get("service") == service.id],
                "pending": ({"started": pending.get("started"), "attempt_id": pending.get("attempt_id"),
                             "client": (pending.get("client") or {}).get("label", "")} if pending.get("nonce") else None),
                "clients": [{**_public_client(row), "registration": oauth_engine.registration_ref(row, return_base), "used_by": [
                    {"skill": d.get("skill"), "account": (d.get("account") or {}).get("label", "")} for d in everyone
                    if (d.get("client") or {}).get("id") == row.get("id")]}
                    for row in _client_rows(skill.name, binding, known)]})
    return {"ok": True, "services": rows, "return_address": oauth_engine.redirect_uri(return_base, "loopback")}


# -- moving a skill's own sign-ins to the host -----------------------------------------------------


def _handover_receipt(service: declaration.Service, legacy: str, account: Dict[str, Any], result: str,
                      legacy_generation: Optional[str], connection: Optional[str]) -> Dict[str, Any]:
    """``connection``: the one recorded with that transfer (``_settle``), never one read from a later sign-in."""
    row = _account_row(account, service, [])
    return {"legacy_key": legacy, "result": result, "account": {"key": row["key"], "label": row["label"]},
            "state": row["state"], "via": row["via"], **({"reason": row["reason"]} if row.get("reason") else {}),
            **({"legacy_generation": legacy_generation} if legacy_generation else {}),
            **({"connection_id": connection} if connection else {})}


def _confirmed_handover(doc: Dict[str, Any], slot: Dict[str, Any]) -> bool:
    generation = slot.get("generation")
    return bool(doc and generation) and (slot.get("filed") is True or (doc.get("origin") or {}).get("generation") == generation)


def _settle_published(skill: str, service_id: str, slot: Dict[str, Any]) -> bool:
    """An attempt whose account write landed before its answer was lost becomes the confirmed transfer."""
    attempt = slot.get("attempt") or {}
    key = attempt.get("target")
    filed = _load_account(_grant(skill, service_id, key)) if key else {}
    if not key or (filed.get("origin") or {}).get("generation") != attempt["generation"]:
        return False
    _settle(slot, attempt, key, filed)
    return True


def _settle(slot: Dict[str, Any], attempt: Dict[str, Any], key: str, filed: Dict[str, Any]) -> None:
    """The confirmed transfer: its generations, account, the token it took (``source``) and the connection it filed."""
    slot.clear()
    slot.update(generation=attempt["generation"], legacy_generation=attempt.get("legacy_generation"), target=key,
                filed=True, source=attempt["source"], **_connection(filed))


def _terms(row: Dict[str, Any], item: Dict[str, Any], prior: Optional[str]) -> Dict[str, Any]:
    """The conditions a transfer attempt is claimed on; only the same grant on the same ones resumes it."""
    scope = (item.get("tokens") or {}).get("scope")
    return {"client_ref": row.get("id"), "ceiling": _words(scope) if isinstance(scope, str) else [], "prior": prior,
            "legacy_generation": item.get("legacy_generation")}


def _compatible(skill: str, service: declaration.Service, item: Dict[str, Any], slot: Dict[str, Any]) -> bool:
    """Whether the slot's attempt is the one this grant would resume (``_terms``), read without the provider."""
    wanted, binding = _client_id_text(item.get("client_id")), _provider_binding(service)
    row = next((row for row in clients() if row.get("source") == f"handover:{skill}" and _matches_provider(row, binding)
                and row.get("client_id") == wanted), None) if wanted else None
    prior = slot.get("target") if "previous_legacy_generation" in item and slot.get("generation") else None
    return row is not None and all((slot.get("attempt") or {}).get(name) == value
                                   for name, value in _terms(row, item, prior).items())


async def _note_attempt(control: oauth_engine.Grant, slot_key: str, generation: str, **facts: Any) -> None:
    """Record facts of the transfer in progress (``None`` removes one); a newer attempt supersedes this one."""
    async with oauth_engine.locked(control):
        current = oauth_engine.load(control)
        attempt = ((current.get("handovers") or {}).get(slot_key) or {}).get("attempt") or {}
        if attempt.get("generation") != generation:
            raise SignInRequired("handover_superseded", "a newer transfer of this sign-in started")
        for name, value in facts.items():
            if value is None:
                attempt.pop(name, None)
            else:
                attempt[name] = value
        oauth_engine._save_sign_in(control, current)


async def _renew_once(control: oauth_engine.Grant, slot_key: str, generation: str, snapshot: Dict[str, Any],
                      record: Dict[str, Any], ceiling: List[str]) -> Tuple[Dict[str, Any], str]:
    """The transfer's one refresh: marked ``sent`` before it leaves; the provider's confirmed answer is kept here
    before any identity read, so the same attempt resumes from it. Only a request that never left clears ``sent``;
    any other end without tokens (an unclear answer, a cancel) leaves it, and a repeat of that grant is refused while
    the slot keeps this attempt."""
    await _note_attempt(control, slot_key, generation, refresh="sent")
    try:
        response, sent_at = await oauth_engine.refresh_request({**snapshot, "tokens": record}, scope=record["scope"])
    except oauth_engine.NOT_SENT:
        await _note_attempt(control, slot_key, generation, refresh=None)  # the request never left
        raise
    if response.status_code in (400, 401) and oauth_engine.oauth_error(response) in ("invalid_client", "invalid_grant"):
        await _note_attempt(control, slot_key, generation, refresh="rejected")
        return {}, "refresh_rejected"
    response.raise_for_status()  # an answer that is neither tokens nor a refusal stays ``sent``
    renewed = oauth_engine.response_token(response, record["refresh_token"], record["scope"], sent_at=sent_at)
    oauth_engine.check_bearer(renewed, ceiling)
    await _note_attempt(control, slot_key, generation, refresh="renewed", tokens=renewed)
    return renewed, ""


async def _import_one(skill: str, service: declaration.Service, issuer: str, endpoints: Dict[str, Any],
                      version: str, item: Dict[str, Any], *, row: Optional[Dict[str, Any]], metadata: Any,
                      recover: bool = False, offline: bool = False, own: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``pending.json`` keeps one slot per legacy key: ``generation``, ``legacy_generation``, ``target`` and ``filed``
    name its confirmed transfer; ``attempt`` the one in progress: its handed refresh token's digest (``source``), its
    terms (``client_ref``, ``ceiling``, ``prior``, ``legacy_generation``) and its one refresh (``refresh``: ``sent``,
    then ``renewed`` with the provider's answer kept here, or ``rejected``). Only the same grant on the same terms
    resumes that attempt; while the slot keeps it, its refresh token is not exchanged again, and no attempt for another
    token displaces one that left. A confirmed transfer also keeps its token's digest (``source``) and the connection it
    filed. ``recover`` starts no attempt and files only a kept answer; ``offline`` stops where the provider is first
    needed (``provider_needed``); ``own`` receives this request's attempt for ``_refused``."""
    legacy, tokens = str(item.get("legacy_key") or ""), item.get("tokens") or {}
    if not legacy or not isinstance(tokens, dict) or not isinstance(tokens.get("refresh_token"), str) or not tokens["refresh_token"]:
        raise SignInRequired("incomplete", "the old sign-in is incomplete")
    local_gen, previous = item.get("legacy_generation"), item.get("previous_legacy_generation")
    renewal = "previous_legacy_generation" in item
    if (any(value is not None and not declaration._text(value, 128) for value in (local_gen, previous))
            or (renewal and (not local_gen or local_gen == previous))):
        raise SignInRequired("handover_generation_changed", "the transfer does not identify a new generation")
    control = _grant(skill, service.id, "pending")
    slot_key, source = oauth_engine._digest(legacy), oauth_engine._digest(tokens["refresh_token"])
    prior_key = prior_origin = prior_connection = None
    async with oauth_engine.locked(control):
        try:
            declared_now = _declared(skill, service.id, True)
        except SignInNotFound:
            raise SignInRequired("sign_in_skill_gone", "the skill no longer declares this service; this grant was not imported") from None
        if declared_now != service:
            raise SignInRequired("sign_in_choices_changed", "the service declaration changed while preparing this transfer")
        current = oauth_engine.load(control)
        slot = current.setdefault("handovers", {}).setdefault(slot_key, {})
        if _settle_published(skill, service.id, slot):
            oauth_engine._save_sign_in(control, current)
        if slot.get("generation"):
            key = slot.get("target")
            if key and key in (current.get("withdrawn_accounts") or []):
                raise SignInRequired("account_disconnected", "this transferred account was disconnected here")
            account = _load_account(_grant(skill, service.id, key)) if key else {}
            if not _confirmed_handover(account, slot):
                raise SignInRequired("handover_unconfirmed", "the earlier transfer could not be confirmed; use the skill's sign-in card")
            _check_binding(service, account)
            identity = account["account"]
            confirmed = identity.get("subject") if identity.get("verified") is True else None
            if confirmed and item.get("subject") and item["subject"] != confirmed:
                raise SignInRequired("identity_mismatch", "the old account does not match the confirmed identity")
            if local_gen is None or local_gen == slot.get("legacy_generation"):
                if recover and (slot.get("source") != source or local_gen != slot.get("legacy_generation")):  # only the exact kept sign-in
                    raise SignInRequired("handover_unconfirmed", "the kept transfer is not this exact sign-in; use the skill's sign-in card")
                result = "already_present" if (account.get("consent") or {}).get("via") == "handover" else "kept_newer"
                return _handover_receipt(service, legacy, account, result, local_gen, slot.get("connection_id"))
            if not renewal or previous != slot.get("legacy_generation"):
                raise SignInRequired("handover_generation_changed", "the earlier transfer changed; use the skill's sign-in card")
            if not confirmed or item.get("subject") != confirmed:
                raise SignInRequired("identity_unavailable", "the transfer has no confirmed continuity; use the skill's sign-in card")
            prior_connection = slot.get("connection_id")  # the one its transfer recorded: never read from a later sign-in
            if not prior_connection or _connection(account).get("connection_id") != prior_connection or (
                    item.get("connection_id") and item["connection_id"] != prior_connection):
                raise SignInRequired("connection_changed", "the account is no longer the connection this transfer replaces")
            prior_key, prior_origin = key, slot["generation"]
        elif renewal:
            raise SignInRequired("handover_unconfirmed", "the prior transfer is not saved; use the skill's sign-in card")
        attempt = slot.get("attempt") or {}
        if attempt.get("target") in (current.get("withdrawn_accounts") or []):  # Disconnect wins over a pending move
            raise SignInRequired("account_disconnected", "this transferred account was disconnected here")
        if attempt.get("source") == source and attempt.get("refresh") == "sent":
            raise SignInRequired("handover_unconfirmed", "this sign-in's earlier renewal got no clear answer; use the skill's sign-in card")
        if attempt.get("refresh") and attempt.get("source") != source:  # an exchange that left stays until filed: never displaced
            raise SignInRequired("handover_unconfirmed", "an earlier transfer of this account is unresolved; recover it or use the skill's sign-in card")
        if recover and not (attempt.get("source") == source and attempt.get("refresh") in ("renewed", "rejected")):
            raise SignInRequired("handover_not_started", "nothing of this transfer is kept here; recover starts none")
        if offline:
            raise SignInRequired("provider_needed", "its kept answer is filed with the provider")
        if row is None:
            raise SignInRequired("unknown_client", "this skill has no matching imported client")
        client = _engine_client(row, issuer, "", metadata)
        snapshot = {"issuer": issuer, "metadata": endpoints, "client": client, "identity_fields": dict(service.identity)}
        _check_binding(service, snapshot)
        known_scope = tokens.get("scope")
        ceiling = _words(known_scope) if isinstance(known_scope, str) else []
        if not ceiling:
            raise SignInRequired("provider_scope_unconfirmed", "the old sign-in's permissions are not known")
        record = {"access_token": tokens.get("access_token") or None, "refresh_token": tokens["refresh_token"],
                  "scope": known_scope, "token_type": "Bearer", "expires_at": _epoch(tokens.get("expires_at"))}
        oauth_engine.check_bearer(record, service.literal_scopes())
        snapshot["scope_spellings"] = {alias: word for alias, word in service.spellings_for(service.literal_scopes()).items()
                                       if alias in ceiling}
        if service.accounts == "one" and (current.get("pending") or {}).get("nonce"):
            raise SignInRequired("sign_in_pending", "finish the owner's current sign-in first")
        terms = _terms(row, item, prior_key)
        if attempt.get("source") == source and attempt.get("refresh") in ("renewed", "rejected"):
            if any(attempt.get(name) != value for name, value in terms.items()):  # never another transfer's answer
                raise SignInRequired("handover_unconfirmed", "this sign-in was renewed for other transfer terms; use the skill's sign-in card")
        else:
            attempt = slot["attempt"] = {"generation": "h-" + secrets.token_hex(16), "source": source, **terms}
            oauth_engine._save_sign_in(control, current)  # claim before any provider call
    generation, state = attempt["generation"], attempt.get("refresh")
    if own is not None:
        own["generation"] = generation  # this request's own attempt: what it leaves decides a refusal (``_refused``)
    reason = "refresh_rejected" if state == "rejected" else ""
    record, renewed = ({} if reason else attempt.get("tokens") or record), state == "renewed"
    if not reason and not renewed and not oauth_engine.fresh(record, None):
        (record, reason), renewed = await _renew_once(control, slot_key, generation, snapshot, record, ceiling), True
    info = await _identity_info(record, snapshot) if oauth_engine.fresh(record, None) else {}
    if info is None and not renewed and record["expires_at"] is None:  # a refused bearer of unknown age: its refresh
        (record, reason), renewed = await _renew_once(control, slot_key, generation, snapshot, record, ceiling), True
        info = await _identity_info(record, snapshot) if oauth_engine.fresh(record, None) else {}
    if renewed and not reason and not oauth_engine.fresh(record, None):  # a renewed answer is read, never exchanged again
        raise SignInRequired("handover_unconfirmed", "the renewed sign-in expired before its account was confirmed; use the skill's sign-in card")
    if info is None:  # the provider refused the bearer itself: no pass moves, confirmed or not
        raise SignInRequired("token_rejected", "the provider refused the handed sign-in; use the skill's sign-in card")
    fields = snapshot["identity_fields"]
    subject, label = _subject(info, fields), str(info.get(fields["label"]) or item.get("label") or legacy)
    if subject and item.get("subject") and str(item["subject"]) != subject:
        raise SignInRequired("identity_mismatch", "the provider confirmed a different account")
    key = account_key(issuer, subject) if subject else account_key(issuer, legacy, skill=skill)
    if renewal and (not subject or key != prior_key):
        raise SignInRequired(reason or "identity_unavailable", "the new sign-in did not confirm the prior account")
    proven = {} if renewed or not subject else {"first_refresh": "due"}  # the move proved its bearer, not its refresh token
    return await _file_handover(skill, service, control, slot_key, generation, key, legacy, prior_connection, {
        **snapshot, "client_ref": row.get("id"), "account": {"key": key, "label": label, "subject": subject,
                                                              "verified": bool(subject)},
        "tokens": record, "scopes": {"requested": ceiling, "optional_chosen": [], "granted": _words(record.get("scope"))},
        "sign_in": {"state": SIGN_IN_REQUIRED if reason else SIGNED_IN, "since": utc_now_iso(),
                    **({"reason": reason} if reason else {})}, "consent": {"via": "handover", "at": utc_now_iso()},
        "origin": {"skill_version": version, "legacy_key": legacy, "generation": generation,
                   "legacy_generation": local_gen, "at": utc_now_iso(), **proven}}, prior_origin=prior_origin)


async def _file_handover(skill: str, service: declaration.Service, control: oauth_engine.Grant, slot_key: str,
                         generation: str, key: str, legacy: str, expected: Any, candidate: Dict[str, Any], *,
                         prior_origin: Optional[str] = None) -> Dict[str, Any]:
    async with oauth_engine.locked(control):
        current = oauth_engine.load(control)
        slot = (current.get("handovers") or {}).get(slot_key) or {}
        attempt = slot.get("attempt") or {}
        if attempt.get("generation") != generation:
            raise SignInRequired("handover_superseded", "a newer transfer of this sign-in started")
        if key in (current.get("withdrawn_accounts") or []):
            raise SignInRequired("account_disconnected", "the transfer is no longer eligible to file here")
        active = _single_account(skill, service)
        if (active and active != key) or (service.accounts == "one" and not active
                                          and (current.get("pending") or {}).get("nonce")):
            raise SignInRequired("account_mismatch", "the owner already chose an account for this service")
        _check_binding(_declared(skill, service.id), candidate)
        target = _grant(skill, service.id, key)
        async with oauth_engine.locked(target):
            held, local_gen = _load_account(target), attempt.get("legacy_generation")
            replace = bool(prior_origin and (held.get("consent") or {}).get("via") == "handover")
            if prior_origin and (not held or held["account"].get("verified") is not True
                                 or held["account"].get("subject") != candidate["account"]["subject"]):
                raise SignInRequired("handover_unconfirmed", "the prior confirmed account is no longer readable")
            if replace and (held.get("origin") or {}).get("generation") != prior_origin:
                raise SignInRequired("handover_generation_changed", "the prior account write changed")
            if expected and _connection(held).get("connection_id") != expected:  # under this lock; a missing one never matches
                raise SignInRequired("connection_changed", "the account is no longer the connection this transfer replaces")
            if held:
                _check_binding(service, held)
            filed, result = held, "kept_newer"
            if not held or replace:
                attempt["target"] = key
                oauth_engine._save_sign_in(control, current)  # pointer first; the account proves publication after a lost reply
                filed, result = {**candidate, "connection_id": held.get("connection_id") or _connection_id()}, "imported"
                oauth_engine._save_sign_in(target, filed)
            _settle(slot, attempt, key, filed)
            oauth_engine._save_sign_in(control, current)
            return _handover_receipt(service, legacy, filed, result, local_gen, slot.get("connection_id"))


async def _handover_clients(skill: str, service_id: str, binding: Dict[str, Any], items: Iterable[Dict[str, Any]]) -> tuple:
    """Write origin and provider binding in one operation; never borrow an owner client's credential."""
    source = f"handover:{skill}"
    added, unavailable = [], set()
    async with oauth_engine.locked(_clients_file()):
        try:  # the skill and its binding as they are now, after the wait for this lock
            current = _provider_binding(_declared(skill, service_id, True))
        except SignInNotFound:
            raise SignInRequired("sign_in_skill_gone", "the skill no longer declares this sign-in; nothing was saved") from None
        if current != binding:
            raise SignInRequired("sign_in_choices_changed", "the skill's provider changed meanwhile; nothing was saved")
        rows = clients()
        before = len(rows)
        if binding["endpoints"] is not None and not any(
                row.get("source") == "owner" and _matches_provider(row, binding) for row in rows):
            raise SignInRequired("provider_approval_required", "approve these explicit addresses on the skill's card")
        available = {row["client_id"]: row for row in rows
                     if row.get("source") == source and _matches_provider(row, binding) and row.get("client_id")}
        for item in items:
            client_id = _client_id_text(item.get("client_id"))
            result = {"client_id": client_id[-4:]}
            secret = item.get("client_secret")
            if not client_id or (secret is not None and not isinstance(secret, str)):
                added.append({**result, "result": "refused", "reason": "incomplete"})
                if client_id:
                    unavailable.add(client_id)
                continue
            try:
                ret = _return_rule(item.get("return_hint"))
            except ValueError:
                added.append({**result, "result": "refused", "reason": "invalid_return"})
                unavailable.add(client_id)
                continue
            previous = available.get(client_id)
            if previous is not None:
                if (previous.get("client_secret") or None) != (secret or None):
                    added.append({**result, "result": "refused", "reason": "registration_changed"})
                    unavailable.add(client_id)
                else:
                    added.append({**result, "result": "already_present"})
                continue
            label = f"Imported from {_client_id_text(item.get('label_hint')) or skill}"
            row = {**binding, "id": "c-" + secrets.token_hex(6), "source": source, "added_at": utc_now_iso(),
                   "label": label[:80], "client_id": client_id, "client_secret": secret or None, "return": ret}
            rows.append(row)
            available[client_id] = row
            added.append({**result, "result": "imported", "client": row["id"]})
        if len(rows) != before:
            oauth_engine.save(_clients_file(), {"clients": rows})
    return added, {key: row for key, row in available.items() if key not in unavailable}


async def _refused(skill: str, service: declaration.Service, item: Dict[str, Any], reason: str,
                   own: Optional[str] = None, recover: bool = False) -> Dict[str, Any]:
    """A refused grant's receipt with what this host knows of its handed refresh token, read under the service's lock
    (``exchange``): ``none`` when an ordinary request never sent it (refused before its exchange, or its own refresh never
    left) and the slot records nothing else for it: no other attempt, no transfer of this generation or token, no break
    of its account, a readable record of breaks; ``saved`` when only an unreachable provider stopped the grant's own
    compatible attempt that keeps the provider's answer (``recover`` files it); else ``unknown``, never repeated. Only
    this request's own unsent claim is removed; ``recover`` never answers ``none``."""
    from ouroboros.store import StoreBusy

    legacy, tokens = str(item.get("legacy_key") or ""), item.get("tokens")
    refresh = tokens.get("refresh_token") if isinstance(tokens, dict) else None
    answer: Dict[str, Any] = {"legacy_key": legacy, "result": "refused", "reason": reason, "exchange": "unknown"}
    if not legacy or not isinstance(refresh, str) or not refresh or reason == "account_disconnected":
        return answer
    source = oauth_engine._digest(refresh)
    try:
        control = _grant(skill, service.id, "pending")
        async with oauth_engine.locked(control):
            current = oauth_engine.load(control)
            slot = (current.get("handovers") or {}).get(oauth_engine._digest(legacy)) or {}
            settled = _settle_published(skill, service.id, slot)
            attempt = slot.get("attempt") or {}
            mine, left = bool(own) and attempt.get("generation") == own, attempt.get("refresh")
            kept = left == "rejected" or (left == "renewed" and oauth_engine.fresh(attempt.get("tokens") or {}, None))
            if reason == "provider_unreachable" and kept and (mine or (
                    not own and attempt.get("source") == source and _compatible(skill, service, item, slot))):
                answer["exchange"] = "saved"
            elif not recover and (not attempt or (mine and not left)) and not (
                    slot.get("generation") and slot.get("legacy_generation") == item.get("legacy_generation")
                    or slot.get("generation") and slot.get("source") in (None, source) or slot.get("target") in (current.get("withdrawn_accounts") or [])
                    or _broken(skill, service.id, item)):
                answer["exchange"] = "none"
                if mine:
                    slot.pop("attempt")  # this request's own claim, never sent
            if settled or (mine and answer["exchange"] == "none"):
                oauth_engine._save_sign_in(control, current)
    except (SignInRequired, SignInNotFound, StoreBusy, OSError, AttributeError, TypeError):
        answer["exchange"] = "unknown"  # nothing proven
    return answer


async def handover(skill: str, service_id: str, *, clients_in: Iterable[Dict[str, Any]] = (),
                   grants: Iterable[Dict[str, Any]] = (), version: str = "", recover: bool = False) -> Dict[str, Any]:
    """Take over the skill's own registrations and grants; eligibility does not extend to other skills. Every refused grant,
    an unreachable provider's included, says what this host knows of its token (``_refused``). ``recover`` repeats a lost
    request: it imports no client and starts no transfer; the slots answer first, before any provider call, and only a
    kept provider answer waits for the provider."""
    service = _declared(skill, service_id)
    binding = _provider_binding(service)
    client_items, grant_items = [] if recover else list(clients_in), list(grants)
    answers: Dict[int, Dict[str, Any]] = {}
    for index, item in enumerate(grant_items if recover else []):
        try:
            answers[index] = await _import_one(skill, service, "", {}, version, item, row=None, metadata=None,
                                               recover=True, offline=True)
        except SignInRequired as exc:
            if exc.reason != "provider_needed":
                answers[index] = await _refused(skill, service, item, exc.reason, recover=True)
    waiting = [(index, item) for index, item in enumerate(grant_items) if index not in answers]
    added: List[Dict[str, Any]] = []
    if waiting or client_items:
        try:
            async with oauth_engine.http_client() as http:
                endpoints, issuer, metadata = await _endpoints(service, http)
            added, available = await _handover_clients(skill, service_id, binding, client_items)
        except (SignInRequired, httpx.HTTPError, ValueError) as exc:
            reason = exc.reason if isinstance(exc, SignInRequired) else "provider_unreachable"
            added = [{"client_id": _client_id_text(item.get("client_id"))[-4:], "result": "refused", "reason": reason}
                     for item in client_items]
            for index, item in waiting:
                answers[index] = await _refused(skill, service, item, reason, recover=recover)
            waiting = []
    for index, item in waiting:
        own: Dict[str, Any] = {}
        try:
            answers[index] = await _import_one(skill, service, issuer, endpoints, version, item, recover=recover, own=own,
                                               row=available.get(_client_id_text(item.get("client_id"))), metadata=metadata)
        except SignInRequired as exc:
            answers[index] = await _refused(skill, service, item, exc.reason, own.get("generation"), recover)
        except (httpx.HTTPError, ValueError):
            answers[index] = await _refused(skill, service, item, "provider_unreachable", own.get("generation"), recover)
    return {"ok": True, "clients": added, "grants": [answers[index] for index in range(len(grant_items))]}


# -- lifecycle and masking ------------------------------------------------------------------------


def secret_values() -> tuple:
    """Every live sign-in secret (tokens, client secrets, PKCE verifiers), for masking every sign-in text."""
    root, values = _service_sign_in_root(), []
    for path in sorted(root.glob("grants/*/*/*.json")) if root.is_dir() else []:
        values.extend(oauth_engine.secret_values(path))
    with contextlib.suppress(SignInRequired):
        values.extend(row["client_secret"] for row in clients() if isinstance(row.get("client_secret"), str) and row["client_secret"])
    return tuple(dict.fromkeys(values))
