"""A skill's declared sign-in to a service: the top-level manifest key ``sign_in`` (CREATING_SKILLS).

Each entry names one service the skill signs in to: its issuer (metadata by RFC 8414 / OpenID Connect
discovery) or explicit endpoints, the scopes it needs and the optional ones the owner may choose, extra
authorize parameters, whether one account or many sign in, and the skill's own words for the owner (notes
shown verbatim). The words are the reviewed skill's: no provider name lives in core code. A declaration
carries no credentials; clients are the owner's, per installation (``service_sign_in``). An invalid entry
is ignored with its problem, which preflight reports (``sign_in_schema``); old cores keep the key unused
in ``raw_extra``.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import re
import urllib.parse
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

_ID = re.compile(r"[a-z0-9_]{1,40}")
_SCOPE_WORD = re.compile(r"[\x21\x23-\x5b\x5d-\x7e]{1,256}")  # RFC 6749 scope-token
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# What the engine and the service set themselves; a declaration may not.
OWNED_PARAMS = frozenset({"response_type", "client_id", "redirect_uri", "state", "code_challenge",
                          "code_challenge_method", "scope", "login_hint", "resource"})
NOTE_KEYS = ("client", "refused", "refresh_rejected")
IDENTITY_KEYS = frozenset({"subject", "label"})  # the identity endpoint's account ID and display name: the only fields a sign-in reads
_REQUIRED_ENDPOINTS = {"authorization", "token", "identity"}
_ENDPOINT_KEYS = _REQUIRED_ENDPOINTS | {"revocation"}
_URL_CHARS = re.compile(r"[A-Za-z0-9._~:/?\[\]@!$&'()*+,;=%-]+")
_DNS_HOST = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\.?")


@dataclasses.dataclass(frozen=True)
class OptionalScope:
    scope: str
    label: str
    default: bool


@dataclasses.dataclass(frozen=True)
class Service:
    id: str
    label: str
    issuer: str
    endpoints: Mapping[str, str]
    identity: Mapping[str, str]
    scopes: Tuple[str, ...]
    scope_labels: Mapping[str, str]
    optional_scopes: Tuple[OptionalScope, ...]
    authorize_params: Mapping[str, str]
    accounts: str
    optional: bool
    notes: Mapping[str, str]
    # Other spellings a provider may return for one declared scope (alias words are never requested).
    scope_spellings: Mapping[str, Tuple[str, ...]] = dataclasses.field(default_factory=dict)

    def declared_scopes(self) -> set:
        return {*self.scopes, *(item.scope for item in self.optional_scopes)}

    def spellings_for(self, words: Iterable[str]) -> Dict[str, str]:
        return {alias: word for word in words for alias in self.scope_spellings.get(word, ())}

    def literal_scopes(self) -> set[str]:
        words = {word for scope in self.declared_scopes() for word in scope.split()}
        return {*words, *self.spellings_for(words)}


def provider_url(value: Any, *, allow_loopback: bool = False) -> urllib.parse.SplitResult:
    """Validate a provider address without changing its path or credentials' destination."""
    error = "provider addresses must be HTTPS without user information or fragments"
    if allow_loopback:
        error += " (HTTP is allowed on loopback)"
    if not isinstance(value, str) or not _URL_CHARS.fullmatch(value) or re.search(r"%(?![0-9A-Fa-f]{2})", value):
        raise ValueError(error)
    try:
        parts = urllib.parse.urlsplit(value)
        host, port = parts.hostname or "", parts.port
        if ":" in host:
            if "%" in host:
                raise ValueError(error)
            ipaddress.IPv6Address(host)
            authority = f"[{host}]"
        elif _DNS_HOST.fullmatch(host):
            authority = host
        else:
            raise ValueError(error)
    except ValueError:
        raise ValueError(error) from None
    authority += "" if port is None else f":{port}"
    if (parts.netloc.lower() != authority or port == 0
            or not (parts.scheme == "https" or (allow_loopback and parts.scheme == "http" and host in _LOOPBACK))
            or any(urllib.parse.unquote(segment) in {".", ".."} for segment in parts.path.split("/"))):
        raise ValueError(error)
    return parts


def canonical_issuer(value: Any) -> str:
    """Compare HTTPS issuer keys; retain the published spelling separately for OAuth's return check."""
    parts = provider_url(value)
    if "?" in value:
        raise ValueError("an issuer must not have a query")
    authority = parts.netloc.lower()
    if parts.port == 443:
        authority = authority.rsplit(":", 1)[0]
    return parts._replace(netloc=authority, path="" if parts.path == "/" else parts.path).geturl()


def explicit_endpoints(value: Any) -> Dict[str, str]:
    """One complete role-to-address map; explicit addresses are compared as written."""
    if not isinstance(value, dict) or not _REQUIRED_ENDPOINTS <= value.keys() <= _ENDPOINT_KEYS:
        raise ValueError("explicit endpoints need authorization, token and identity; revocation is optional")
    for address in value.values():
        provider_url(address, allow_loopback=True)
    return dict(value)


def _text(value: Any, limit: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _scope_text(value: Any) -> bool:
    return _text(value, 256) and all(_SCOPE_WORD.fullmatch(word) for word in value.split())


def _service(raw: Any) -> Tuple[Optional[Service], List[str]]:
    if not isinstance(raw, dict):
        return None, ["an entry is not a mapping"]
    sid, problems = raw.get("id"), []
    name = sid if isinstance(sid, str) else "?"
    if not isinstance(sid, str) or not _ID.fullmatch(sid):
        problems.append("id must match [a-z0-9_]{1,40}")
    if "client_id" in raw or "client_secret" in raw:
        problems.append("a declaration carries no credentials (client_id, client_secret)")
    if not _text(raw.get("label"), 80):
        problems.append("label is required (at most 80 characters)")
    issuer, endpoints = "", {}
    try:
        if raw.get("issuer") not in (None, ""):
            if raw.get("endpoints") not in (None, {}):
                raise ValueError("use issuer discovery or explicit endpoints, not both")
            issuer = canonical_issuer(raw["issuer"])
        else:
            endpoints = explicit_endpoints(raw.get("endpoints"))
    except ValueError as exc:
        problems.append(str(exc))
    scopes = raw.get("scopes")
    if not isinstance(scopes, list) or not 1 <= len(scopes) <= 64 or not all(_scope_text(s) for s in scopes):
        problems.append("scopes: 1 to 64 non-empty strings of at most 256 characters")
        scopes = []
    optional, choices = [], raw.get("optional_scopes")
    if choices is not None and not isinstance(choices, list):
        problems.append("optional_scopes must be a list")
        choices = []
    for item in choices or []:
        if not (isinstance(item, dict) and _scope_text(item.get("scope")) and _text(item.get("label"), 120)
                and isinstance(item.get("default", False), bool)):
            problems.append("optional_scopes: each {scope, label (at most 120 characters), default: bool}")
            break
        optional.append(OptionalScope(item["scope"], item["label"], item.get("default", False)))
    spellings: Dict[str, Tuple[str, ...]] = {}
    raw_spellings = raw.get("scope_spellings")
    if raw_spellings is not None:
        declared_words = {word for scope in (*scopes, *(item.scope for item in optional)) for word in scope.split()}
        if not isinstance(raw_spellings, dict):
            problems.append("scope_spellings: a mapping of a declared scope to its other spellings")
        else:
            seen: set = set()
            for scope, aliases in raw_spellings.items():
                if not isinstance(scope, str) or scope not in declared_words:
                    problems.append("scope_spellings: each key is a scope this service declares")
                    continue
                if not isinstance(aliases, list) or not 1 <= len(aliases) <= 8 or not all(isinstance(a, str) and _SCOPE_WORD.fullmatch(a) for a in aliases):
                    problems.append("scope_spellings: 1 to 8 non-empty strings of at most 256 characters")
                    continue
                for alias in aliases:
                    if alias in declared_words or alias in seen:
                        problems.append("scope_spellings: a spelling is one word, not a declared scope, and used once")
                    seen.add(alias)
                spellings[scope] = tuple(aliases)
    params = raw.get("authorize_params") or {}
    if not isinstance(params, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in params.items()):
        problems.append("authorize_params: a mapping of strings")
        params = {}
    if set(params) & OWNED_PARAMS:
        problems.append(f"authorize_params may not set {', '.join(sorted(set(params) & OWNED_PARAMS))}")
    labels, notes = raw.get("scope_labels") or {}, raw.get("notes") or {}
    identity = {} if raw.get("identity") is None else raw["identity"]  # an explicit empty or false value is reported, never read as the default
    if not isinstance(labels, dict) or not all(_text(v, 120) for v in labels.values()):
        problems.append("scope_labels: each label at most 120 characters")
    if not isinstance(notes, dict) or set(notes) - set(NOTE_KEYS) or not all(_text(v, 600) for v in notes.values()):
        problems.append(f"notes: only {', '.join(NOTE_KEYS)}, each at most 600 characters")
    if not isinstance(identity, dict) or set(identity) - IDENTITY_KEYS or not all(_text(v, 80) for v in identity.values()):
        problems.append("identity: only subject and label, each the name of a field the identity endpoint answers (at most 80 characters)")
    if raw.get("accounts", "many") not in ("one", "many") or not isinstance(raw.get("optional", False), bool):
        problems.append("accounts is one or many; optional is a boolean")
    if problems:
        return None, [f"sign_in {name}: {problem}" for problem in problems]
    return Service(id=sid, label=raw["label"].strip(), issuer=issuer,
                   endpoints=dict(endpoints), identity={"subject": "sub", "label": "email", **identity},
                   scopes=tuple(scopes), scope_labels=dict(labels), optional_scopes=tuple(optional),
                   authorize_params=dict(params), accounts=raw.get("accounts", "many"),
                   optional=raw.get("optional", False), notes=dict(notes), scope_spellings=spellings), []


def parse(raw: Any) -> Tuple[Dict[str, Service], List[str]]:
    """The valid services of a ``sign_in`` value and the problems of every invalid entry."""
    if raw is None:
        return {}, []
    if not isinstance(raw, list):
        return {}, ["sign_in must be a list of services"]
    services: Dict[str, Service] = {}
    problems: List[str] = []
    for entry in raw:
        service, found = _service(entry)
        if service is not None and service.id in services:
            found = [f"sign_in {service.id}: the id is declared twice"]
        problems.extend(found)
        if service is not None and not found:
            services[service.id] = service
    return services, problems


def declared(skill: Any) -> Dict[str, Service]:
    """The services an installed skill's reviewed manifest declares (invalid entries left out)."""
    if getattr(skill, "load_error", ""):
        return {}
    raw_extra = getattr(getattr(skill, "manifest", None), "raw_extra", None) or {}
    return parse(raw_extra.get("sign_in"))[0]
