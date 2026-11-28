"""Named API connections: the ``OUROBOROS_CONNECTIONS`` rows and how a route reads
the connection it names.

A route string names its connection (``provider_models.parse_model_route``): a
kind's built-in connection reads that kind's flat settings, a named one reads its
row, which carries the same fields (``provider_models.PROVIDER_KINDS``). The only
way to switch a built-in connection off is a secret-free override row
``{"kind": K, "name": "default", "enabled": false}``. Nothing here picks, orders or
rotates connections: an unknown, disabled or mismatched one is refused before
dispatch (``llm_attempt.ConnectionUnavailable``), never served by another key.
The Settings projections and the save-time shape check sit beside that reader.

Revision. Which saved connection an attempt dispatched on is a fact a later turn can compare (a late answer's
letter: ``extension_late_result``): ``connection_dispatch_account``, a random revision the settings writer
recorded for that very document in ``state/connection_revisions.json`` (``record_connection_revisions``); never
derived from a credential or the file, and unknown without such a record.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ouroboros.provider_models import (
    BUILTIN_CONNECTION, CONNECTIONS_KEY, DIRECT_PROVIDER_DEFAULTS, MODEL_SETTING_KEYS, OPENAI_RESPONSES_DIRECT_DEFAULTS,
    COMPATIBLE_LEGACY_PAIR, PROVIDER_CREDENTIAL_GROUPS, PROVIDER_KINDS, RouteParts, choice_error, compatible_endpoint,
    compatible_endpoint_configured, connection_id, openai_api_surface, parse_model_route,
)
from ouroboros.settings_integrity import runtime_setting

log = logging.getLogger(__name__)
Reader = Callable[..., Any]  # read(flat_setting_key, default) -> the raw saved value
_ROW_META = ("kind", "name", "label", "enabled")
_SURFACE_KEY = PROVIDER_KINDS["openai"].flat_key("api_surface")
# The built-in compatible address: the dedicated key and the legacy one it falls back to (``compatible_endpoint``).
_COMPATIBLE_ADDRESS = PROVIDER_KINDS["openai-compatible"].flat_key("base_url")
_COMPATIBLE_ADDRESS_KEYS = (_COMPATIBLE_ADDRESS, COMPATIBLE_LEGACY_PAIR[_COMPATIBLE_ADDRESS])


def connection_rows(raw: Any) -> tuple[dict, ...]:
    """The saved rows (a JSON string, e.g. an operator's variable, is parsed); non-objects drop."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else []
        except ValueError:
            return ()
    return tuple(dict(row) for row in raw if isinstance(row, dict)) if isinstance(raw, (list, tuple)) else ()


def settings_view(settings: Mapping[str, Any] | None = None) -> tuple[Reader, tuple[dict, ...]]:
    """``(read, rows)`` of an explicit settings mapping, else of this process's runtime view."""
    if settings is None:
        return runtime_setting, connection_rows(runtime_setting(CONNECTIONS_KEY, None))
    return settings.get, connection_rows(settings.get(CONNECTIONS_KEY))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _enabled(row: Mapping[str, Any]) -> bool:
    return str(row.get("enabled", True)).strip().lower() not in {"false", "0", "no", "off"}


def find_row(rows: Iterable[Mapping[str, Any]], kind: str, name: str) -> dict | None:
    """The first saved row of ``kind`` named ``name`` (``default`` = the built-in override row)."""
    return next((dict(row) for row in rows if _text(row.get("kind")) == kind and _text(row.get("name")) == name), None)


def _ready(kind: str, get: Callable[..., Any]) -> bool:
    return any(all(_text(get(field)) for field in group) for group in PROVIDER_KINDS[kind].ready)


def unavailable(route: RouteParts, reason: str) -> Exception:
    from ouroboros.llm_attempt import ConnectionUnavailable

    return ConnectionUnavailable(connection_id(route.kind, route.connection), reason)


def connection_values(route: RouteParts, read: Reader, rows: Iterable[Mapping[str, Any]]) -> Callable[..., Any]:
    """Raw field reader ``get(row_field, default="")`` for the connection ``route`` names.

    The built-in connection reads its kind's flat settings through ``read``; a named
    one reads its row, which must exist for that kind, be enabled and carry its
    credential or endpoint. Anything else raises ``ConnectionUnavailable`` with the
    reason ``missing``, ``disabled``, ``kind_mismatch`` or ``invalid``."""
    rows = tuple(rows)
    spec = PROVIDER_KINDS.get(route.kind)
    if spec is None or not route.valid:
        raise unavailable(route, "invalid")
    if not route.connection:
        if not _enabled(find_row(rows, route.kind, BUILTIN_CONNECTION) or {}):
            raise unavailable(route, "disabled")
        return lambda field, default="": read(spec.flat_key(field), default) if spec.flat_key(field) else default
    row = find_row(rows, route.kind, route.connection)
    if row is None:
        named_elsewhere = any(_text(other.get("name")) == route.connection for other in rows)
        raise unavailable(route, "kind_mismatch" if named_elsewhere else "missing")
    if not _enabled(row):
        raise unavailable(route, "disabled")
    if not _ready(route.kind, row.get):
        raise unavailable(route, "missing")
    return lambda field, default="": row.get(field, default)


def _has_credential(route: RouteParts, read: Reader, rows: tuple[dict, ...]) -> bool:
    """Whether the connection carries its credential/endpoint, enabled or not."""
    if route.connection:
        row = find_row(rows, route.kind, route.connection)
        return row is not None and _ready(route.kind, row.get)
    if route.kind == "openai-compatible":  # the built-in pair keeps the legacy OPENAI_* derivation
        return compatible_endpoint_configured(lambda key: _text(read(key, "")))
    spec = PROVIDER_KINDS[route.kind]
    return _ready(route.kind, lambda field: read(spec.flat_key(field), ""))


def connection_configured(route: RouteParts, read: Reader, rows: Iterable[Mapping[str, Any]]) -> bool:
    """Whether the connection ``route`` names can serve now: enabled, with its credential/endpoint."""
    from ouroboros.llm_attempt import ConnectionUnavailable

    rows = tuple(rows)
    try:
        connection_values(route, read, rows)
    except ConnectionUnavailable:
        return False
    return _has_credential(route, read, rows)


def connection_routes(rows: Iterable[Mapping[str, Any]]) -> list[RouteParts]:
    """Every connection in owner order: per kind its built-in one, then its named rows as saved."""
    rows = tuple(rows)
    routes: list[RouteParts] = []
    for kind in PROVIDER_KINDS:
        routes.append(RouteParts(kind, "", ""))
        for row in rows:
            name = _text(row.get("name"))
            parsed = parse_model_route(f"{kind}@{name}::")
            if (_text(row.get("kind")) == kind and parsed.valid and parsed.connection
                    and all(route.connection != name or route.kind != kind for route in routes)):
                routes.append(RouteParts(kind, name, ""))
    return routes


def configured_connections(settings: Mapping[str, Any] | None = None) -> list[tuple[str, str]]:
    """``(kind, name)`` of every connection that can serve now, in owner order (``name`` '' = built-in)."""
    read, rows = settings_view(settings)
    return [(route.kind, route.connection) for route in connection_routes(rows)
            if connection_configured(route, read, rows)]


def connection_display(kind: str, name: str, rows: Iterable[Mapping[str, Any]]) -> str:
    """The picker label of a connection: ``OpenRouter``, ``OpenRouter · Claude credits``."""
    spec = PROVIDER_KINDS[kind]
    if not name:
        return spec.label
    return f"{spec.label} · {_text((find_row(rows, kind, name) or {}).get('label')) or name}"


def connections_meta(settings: Mapping[str, Any], document: Mapping[str, Any]) -> list[dict]:
    """``GET /api/settings`` ``_meta.connections``: every connection as the picker shows it.

    ``source`` says where its credential is read from now: the saved document,
    the environment (a key the document leaves unset), or nowhere yet. A compatible
    endpoint that refused prompt-cache hints and was learned without them carries
    ``cache_hints: {refused, learned_at}`` (``request_wire_contract.endpoint_cache_hint_refusal``);
    a private one carries ``unidentified``: its ``/v1`` admission at the endpoint it has (``consent``, ``prior``
    or none), the Ollama read at its address and what its own last check identified there
    (``llm_ollama_guard.connection_facts``)."""
    read, rows = settings_view(settings)
    saved = document if isinstance(document, Mapping) else {}
    saved_read, saved_rows = saved.get, connection_rows(saved.get(CONNECTIONS_KEY))
    out = []
    for route in connection_routes(rows):
        name = route.connection
        row = find_row(rows, route.kind, name or BUILTIN_CONNECTION) or {}
        configured = _has_credential(route, read, rows)
        in_document = (find_row(saved_rows, route.kind, name) is not None if name
                       else _has_credential(route, saved_read, saved_rows))
        label = _text(row.get("label")) if name else ""
        out.append({
            "kind": route.kind, "name": name, "label": label, "builtin": not name, "enabled": _enabled(row),
            "configured": configured,
            "source": "none" if not configured else "settings" if in_document else "environment",
            "id": connection_id(route.kind, name), "display": connection_display(route.kind, name, rows),
            **(_compatible_facts(route, read, row) if route.kind == "openai-compatible" and configured else {}),
        })
    return out


def direct_provider_defaults(provider: str, settings: Mapping[str, Any] | None = None, connection: str | None = "") -> dict:
    """The factory profile a direct provider's connection proposes (slots, catalog, deep review, Provider Test):
    an OpenAI connection explicitly on Responses proposes the Responses profile; every other one, a never-chosen
    or unreadable OpenAI surface included, keeps its kind's table."""
    if provider == "openai" and connection is not None:
        from ouroboros.llm_attempt import ConnectionUnavailable

        read, rows = settings_view(settings)
        try:
            surface = openai_api_surface(connection_values(RouteParts("openai", connection, ""), read, rows)("api_surface"))
        except (ConnectionUnavailable, ValueError):
            surface = ""
        if surface == "responses":
            return OPENAI_RESPONSES_DIRECT_DEFAULTS
    return DIRECT_PROVIDER_DEFAULTS.get(provider, {})


def seed_new_api_surfaces(after: dict, before: Mapping[str, Any]) -> list[str]:
    """Save a genuinely new OpenAI connection on Responses (the owner's choice of 16.11.2026); return what was seeded.

    New means it could not serve before this write and can after it, with no API chosen: the built-in key absent
    from the settings and environment the write started from (and not the legacy compatible pair's key), or a
    named row the write adds. A connection that already existed keeps what it reads: a never-chosen API stays
    Chat Completions, whatever the edit."""
    from ouroboros.provider_models import legacy_openai_base

    seeded: list[str] = []
    read_before, rows_before = settings_view(before)
    read_after, rows_after = settings_view(after)
    builtin = RouteParts("openai", "", "")
    if (not _text(after.get(_SURFACE_KEY)) and not legacy_openai_base(after)
            and _has_credential(builtin, read_after, rows_after) and not _has_credential(builtin, read_before, rows_before)):
        after[_SURFACE_KEY] = "responses"
        seeded.append(_SURFACE_KEY)
    rows = after.get(CONNECTIONS_KEY)
    for row in rows if isinstance(rows, list) else ():
        name = _text(row.get("name")) if isinstance(row, dict) else ""
        if (name and name != BUILTIN_CONNECTION and _text(row.get("kind")) == "openai"
                and not _text(row.get("api_surface")) and find_row(rows_before, "openai", name) is None):
            row["api_surface"] = "responses"
            seeded.append(connection_id("openai", name))
    return seeded


def _compatible_address(route: RouteParts, read: Reader, row: Mapping[str, Any]) -> str:
    """Where a compatible connection sends: a named row's own address, the built-in pair's (the legacy one too)."""
    return _text(row.get("base_url")) if route.connection else compatible_endpoint(lambda key: _text(read(key, "")))[1]


def compatible_addresses(settings: Mapping[str, Any] | None = None) -> list[tuple[str, str]]:
    """``(connection id, base URL)`` of every compatible connection that can serve now, in owner order."""
    read, rows = settings_view(settings)
    return [(connection_id(route.kind, route.connection),
             _compatible_address(route, read, find_row(rows, route.kind, route.connection) or {}))
            for route in connection_routes(rows)
            if route.kind == "openai-compatible" and connection_configured(route, read, rows)]


def served_compatible_addresses(document: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """``compatible_addresses`` of what the loader reads once ``document`` is the saved one (``None``: no file):
    that document under the launch environment, without the values this process tree mirrored there from an
    earlier document (``settings_integrity.skill_environ``). Those stand in until the next projection, and
    would show a connection the document has just dropped as still served."""
    from ouroboros import config
    from ouroboros.settings_integrity import settings_read_environ, skill_environ

    with settings_read_environ(skill_environ()):
        return compatible_addresses(config.settings_of_document(document, present=document is not None))


def canonical_builtin_consent(settings: dict, written: Mapping[str, Any]) -> None:
    """The built-in compatible key's consent as a write saves it, in place (``llm_ollama_guard.consent_token``):
    the id of the endpoint it names, never that URL. Only what ``written`` names: a value the document
    or the launch holds stays as its author wrote it. A row's is made so by ``normalize_connections``.

    A blank consent a write saves is a withdrawal, and a blank saved value can never outweigh a non-empty
    launch environment (``config.settings_of_document``): where the launch itself holds a consent — not this
    tree's own projection of a saved one — the write stores the explicit revocation word instead
    (``CONSENT_REVOKED``), which every reader reads as no consent, so a launch-environment consent has an
    honest status and a way back. A write that leaves no address in force takes the stored consent with it,
    so a removed connection never leaves one dangling; rotating the key under a standing address keeps it —
    consent belongs to the endpoint, not the credential."""
    from ouroboros.llm_ollama_guard import CONSENT_FIELD, CONSENT_REVOKED, consent_token
    from ouroboros.settings_integrity import projected_proposals, read_environ

    key = PROVIDER_KINDS["openai-compatible"].flat_key(CONSENT_FIELD)

    def blanked() -> None:
        environ = read_environ()
        launch = environ.get(key)
        settings[key] = CONSENT_REVOKED if _text(launch) and launch != projected_proposals(environ).get(key) else ""

    if key in written:
        text = _text(settings.get(key))
        if text == CONSENT_REVOKED:
            return  # an explicit revocation travels unchanged
        address_touched = any(name in written for name in _COMPATIBLE_ADDRESS_KEYS)
        if text and not (address_touched and not compatible_endpoint(lambda name: _text(settings.get(name)))[1]):
            settings[key] = consent_token(text)
        else:
            blanked()  # withdrawn, or granted beside an address this same write empties: nothing stands to cover
        return
    # A write that empties the address in force (either pair) takes the stored consent with it, so a removed
    # connection never leaves one dangling; rotating the key under a standing address keeps it — consent
    # belongs to the endpoint, not the credential.
    if (any(name in written for name in _COMPATIBLE_ADDRESS_KEYS)
            and not compatible_endpoint(lambda name: _text(settings.get(name)))[1]):
        blanked()


def _compatible_facts(route: RouteParts, read: Reader, row: Mapping[str, Any]) -> dict:
    from ouroboros.llm_ollama_guard import CONSENT_FIELD, connection_facts
    from ouroboros.request_wire_contract import endpoint_cache_hint_refusal

    get = row.get if route.connection else lambda field: read(PROVIDER_KINDS[route.kind].flat_key(field), "")
    base_url = _compatible_address(route, read, row)
    learned = endpoint_cache_hint_refusal(route.kind, base_url)
    return {**({"cache_hints": {"refused": True, "learned_at": learned}} if learned else {}),
            **connection_facts(connection_id(route.kind, route.connection), base_url, get(CONSENT_FIELD))}


def _subagent_routes(raw: Any) -> list[tuple[str, str]]:
    """``(row id, target)`` of each API row of a saved ``OUROBOROS_SUBAGENTS`` catalog, read leniently."""
    try:
        payload = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw
    except ValueError:
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    return [(_text(item.get("subagent_id")), _text((item.get("route") or {}).get("target_id")))
            for item in items or () if isinstance(item, dict) and isinstance(item.get("route"), dict)
            and item["route"].get("kind") in {"api_model", "api_chat"}]


def _named_routes(settings: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """``(setting, entry, model route)`` of every route the settings name: each role, fallback and API agent row."""
    from ouroboros.configured_subagents import SUBAGENTS_SETTING
    from ouroboros.model_slots import MODEL_ROLE_SETTINGS

    named = []
    for key in dict.fromkeys((*MODEL_ROLE_SETTINGS.values(), *MODEL_SETTING_KEYS)):
        models = [model.strip() for model in str(settings.get(key) or "").split(",") if model.strip()]
        # A chain entry is named by its 1-based position, so the receipt can say "Fallback 2".
        named += [(key, str(index + 1) if len(models) > 1 or "FALLBACK" in key else "", model)
                  for index, model in enumerate(models)]
    return named + [(SUBAGENTS_SETTING, row_id, target) for row_id, target in _subagent_routes(settings.get(SUBAGENTS_SETTING))]


def model_routes(settings: Mapping[str, Any]) -> list[str]:
    """Every model route the settings name, once, in their order."""
    return list(dict.fromkeys(model for _setting, _entry, model in _named_routes(settings)))


def connection_references(settings: Mapping[str, Any]) -> list[dict]:
    """Role, fallback and agent rows that name a connection which cannot serve them:
    removed, disabled, of another kind or misspelled. They keep their value and refuse."""
    from ouroboros.llm_attempt import ConnectionUnavailable

    read, rows = settings_view(settings)
    out = []
    for setting, entry, model in _named_routes(settings):
        route = parse_model_route(model)
        if route.kind not in PROVIDER_KINDS:
            continue
        try:
            connection_values(route, read, rows)
        except ConnectionUnavailable as exc:
            out.append({"setting": setting, **({"entry": entry} if entry else {}), "route": model,
                        "connection": exc.connection, "reason": exc.reason})
    return out


def normalize_connections(raw: Any) -> tuple[list[dict], str]:
    """``(rows, error)`` for a saved ``OUROBOROS_CONNECTIONS``: the shape every reader
    relies on, or the first problem as one owner-facing sentence (nothing saved)."""
    from ouroboros.llm_ollama_guard import CONSENT_FIELD, consent_token
    from ouroboros.llm_routing import _OR_PROVIDER_PRESETS

    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else []
        except ValueError:
            return [], f"{CONNECTIONS_KEY} must be a JSON list of connection rows."
    if not isinstance(raw, list):
        return [], f"{CONNECTIONS_KEY} must be a list of connection rows."
    rows, seen = [], set()
    for index, row in enumerate(raw):
        where = f"{CONNECTIONS_KEY}[{index}]"
        if not isinstance(row, dict):
            return [], f"{where} must be an object."
        kind, name = _text(row.get("kind")), _text(row.get("name"))
        spec = PROVIDER_KINDS.get(kind)
        if spec is None:
            return [], f"{where}: unknown kind {kind!r}; use one of {', '.join(PROVIDER_KINDS)}."
        if not (name == BUILTIN_CONNECTION or parse_model_route(f"{kind}@{name}::").valid):
            return [], f"{where}: name must be 1-32 lowercase letters, digits, '-' or '_', starting with a letter or digit."
        if (kind, name) in seen:
            return [], f"{where}: {connection_id(kind, name)} is defined twice."
        seen.add((kind, name))
        if not isinstance(row.get("enabled", True), bool):
            return [], f"{where}: enabled must be true or false."
        fields = {field for field, _flat, _secret in spec.fields}
        allowed = {"kind", "name", "enabled"} if name == BUILTIN_CONNECTION else (
            set(_ROW_META) | fields | ({"provider_routing"} if kind == "openrouter" else set()))
        unknown = sorted(set(row) - allowed)
        if unknown:
            return [], (f"{where}: the built-in {spec.label} connection's row carries only kind, name and enabled; "
                        f"its credential stays in its own setting (got {', '.join(unknown)})."
                        if name == BUILTIN_CONNECTION else f"{where}: unknown field(s) {', '.join(unknown)}.")
        for field in fields | {"label"}:
            if field in row and not isinstance(row[field], (str, bool) if field == "verify_ssl_certs" else str):
                return [], f"{where}: {field} must be text."
            if error := choice_error(kind, field, row.get(field)):
                return [], f"{where}: {field} {error}."
        routing = row.get("provider_routing")
        if routing not in (None, "") and not isinstance(routing, dict) and not (
                isinstance(routing, str) and (routing.strip().lower() in _OR_PROVIDER_PRESETS
                                              or isinstance(_json_object(routing), dict))):
            return [], f"{where}: provider_routing must be a preset ({', '.join(_OR_PROVIDER_PRESETS)}) or a JSON object."
        # A consent is saved as the id of the endpoint it names, never that URL.
        consent = {CONSENT_FIELD: consent_token(row[CONSENT_FIELD])} if _text(row.get(CONSENT_FIELD)) else {}
        rows.append({**row, **consent, "kind": kind, "name": name})
    return rows, ""


def _json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


# --- the saved connection an attempt dispatched on -----------------------------------------------------------------
# A provider name proves no connection and a credential may not become an identity (no digest of one is kept). The
# fact is the settings writer's own record: under its lock, once its bytes landed, it names that document and gives
# each connection the document saves a random revision. A connection keeps its revision through a write that leaves
# its saved fields as they were in the document the record named, so a restart or another process reads the same one;
# any change gives a new one. A document the record does not name (an edit outside this writer, a writer that keeps no
# record, one saved before the first recorded save) names no connection: unknown stays unknown. An endpoint Ouroboros
# supplies (a kind's official address, a shipped default) is no saved field.

_REVISION = re.compile(r"[0-9a-f]{32}")
_REVISION_RECORD_VERSION = 1
_RECORD: tuple = ()  # (the record file's identity, its parsed record)


def settings_snapshot() -> tuple | None:
    """The saved settings document's file identity (path, inode, modification time, size): what tells the document a
    record names from any later one, never a revision itself. A replacement or an edit in place moves it, a restart or
    another process reads it unchanged; None without a document. No device number: a volume's may differ across boots
    and containers while the file stays the same."""
    from ouroboros import config

    try:
        found = config.SETTINGS_PATH.stat()
    except OSError:
        return None
    return str(config.SETTINGS_PATH), found.st_ino, found.st_mtime_ns, found.st_size


def _revisions_path() -> Path:
    from ouroboros import config

    return config.DATA_DIR / "state" / "connection_revisions.json"


def _revision_record() -> dict:
    """The writer's record, reread only when its file changed; {} when absent or of another shape."""
    global _RECORD
    path = _revisions_path()
    try:
        found = path.stat()
    except OSError:
        return {}
    identity = (str(path), found.st_ino, found.st_mtime_ns, found.st_size)
    held = _RECORD
    if held and held[0] == identity:
        return held[1]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not (isinstance(record, dict) and record.get("version") == _REVISION_RECORD_VERSION
            and isinstance(record.get("document"), list) and isinstance(record.get("revisions"), dict)):
        record = {}
    _RECORD = (identity, record)
    return record


def _saved_fields(document: Mapping[str, Any], route: RouteParts) -> Any:
    """What ``document`` itself saves for one connection, compared in memory and never kept: a named connection's row,
    a built-in one's non-blank flat settings with its override row; None when it saves none of its own.
    A display label is not part of the transport or its payer."""
    rows = connection_rows(document.get(CONNECTIONS_KEY))
    row = find_row(rows, route.kind, route.connection or BUILTIN_CONNECTION)
    fields = {key: value for key, value in row.items() if key != "label"} if row else None
    if route.connection:
        return fields
    saved = {key: document[key] for key in PROVIDER_CREDENTIAL_GROUPS[route.kind]
             if document.get(key) not in (None, "")}
    return (saved, fields) if saved else None


def connection_dispatch_account(route: RouteParts, snapshot: tuple) -> str:
    """``api:<connection>#<revision>``: the saved connection a live resolve of ``route`` read, when every read fell
    between two equal ``settings_snapshot`` readings (the caller's check) and the settings writer's record names that
    very document with a revision for this connection. '' otherwise: no record, a record of another document, a
    malformed one or one without this connection, no saved document read (none, or unreadable), or an operator's
    variable that could stand in for one of the connection's settings. Nothing then proves which credential the call
    read, and its letter stays readable by its id."""
    from ouroboros.route_choice import route_settings_view
    from ouroboros.settings_integrity import skill_environ

    view = route_settings_view()
    if not (view and view[0]):
        return ""
    environ = skill_environ()
    keys = (CONNECTIONS_KEY, *(() if route.connection else PROVIDER_CREDENTIAL_GROUPS.get(route.kind, ())))
    if any(str(environ.get(key) or "").strip() for key in keys):
        return ""
    name = connection_id(route.kind, route.connection)
    record = _revision_record()
    revision = record["revisions"].get(name) if record.get("document") == list(snapshot) else None
    return f"api:{name}#{revision}" if isinstance(revision, str) and _REVISION.fullmatch(revision) else ""


def api_connection_for_model(model: str, use_local: bool) -> str:
    """The model's API connection picker id, or empty for a local or subscription model."""
    route = parse_model_route(model)
    return connection_id(route.kind, route.connection) if (
        not use_local and route.valid and not route.local and route.kind in PROVIDER_KINDS) else ""


def live_connection_account(model: str) -> str:
    """A saved API connection's non-secret revision, or empty when its identity is unproven."""
    snapshot = settings_snapshot()
    if snapshot is None:
        return ""
    account = connection_dispatch_account(parse_model_route(model), snapshot)
    return account if settings_snapshot() == snapshot else ""


def connection_account_matches(account: Any, model: str, use_local: bool) -> bool:
    name = api_connection_for_model(model, use_local)
    prefix = f"api:{name}#"
    return bool(name and isinstance(account, str) and account.startswith(prefix) and _REVISION.fullmatch(account[len(prefix):]))


def plan_connection_revisions(read: tuple | None, before: Mapping[str, Any], proposed: Mapping[str, Any]) -> dict:
    """The locked writer plans revisions before publishing, so consent can name the same random identity the
    dispatch reader will see. Only unchanged fields of a witnessed document keep an earlier revision."""
    from ouroboros import config

    prior = _revision_record() if read is not None else {}
    earlier = prior["revisions"] if read is not None and prior.get("document") == list(read) else {}
    old, new = config.coerce_settings_raw(before), config.coerce_settings_raw(proposed)
    revisions = {}
    for route in connection_routes(connection_rows(new.get(CONNECTIONS_KEY))):
        fields = _saved_fields(new, route)
        if fields is None:
            continue
        name = connection_id(route.kind, route.connection)
        kept = earlier.get(name) if _saved_fields(old, route) == fields else None
        revisions[name] = kept if isinstance(kept, str) and _REVISION.fullmatch(kept) else uuid.uuid4().hex
    return {"before": earlier, "after": revisions}


def record_connection_revisions(written: Mapping[str, Any], revisions: dict) -> None:
    """Publish the planned revisions under the writer lock, only for this exact saved document. Never raises."""
    from ouroboros import config
    from ouroboros.utils import atomic_write_json

    try:
        snapshot = settings_snapshot()
        payload = config.serialize_settings(dict(written)).encode("utf-8")
        if snapshot is None or config.SETTINGS_PATH.read_bytes() != payload or settings_snapshot() != snapshot:
            return  # another write already replaced these bytes
        path = _revisions_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"version": _REVISION_RECORD_VERSION, "document": list(snapshot),
                                 "revisions": revisions})
    except Exception:
        log.warning("Settings saved; the record of their connection revisions was not updated", exc_info=True)
