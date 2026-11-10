"""Named API connections: the ``OUROBOROS_CONNECTIONS`` rows and how a route reads
the connection it names.

A route string names its connection (``provider_models.parse_model_route``): a
kind's built-in connection reads that kind's flat settings, a named one reads its
row, which carries the same fields (``provider_models.PROVIDER_KINDS``). The only
way to switch a built-in connection off is a secret-free override row
``{"kind": K, "name": "default", "enabled": false}``. An unknown, disabled or
mismatched connection is refused before dispatch (``llm_attempt.ConnectionUnavailable``),
never served by another key.

The exception is a POOL: a row that lists ``members`` — named connections of the same
kind, each carrying its own account — and serves each call through one of them in turn.
A pool is the owner saying these accounts are interchangeable; a single named connection
is never rotated, and a member that is disabled or carries no credential at selection time
is skipped rather than failing the pool (a provider-level rejection of a member is not
visible here — that is the caller's retry, not a rotation input). Reads that only ask
whether a route can serve (a picker, a reference check, a capability fingerprint) choose
nothing; a structurally wrong member — another kind's connection, or a pool — refuses the
route instead of being skipped.
The Settings projections and the save-time shape check sit beside that reader.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Iterable, Mapping

from ouroboros.provider_models import (
    BUILTIN_CONNECTION, CONNECTIONS_KEY, MODEL_SETTING_KEYS, PROVIDER_KINDS, RouteParts, choice_error,
    compatible_endpoint, compatible_endpoint_configured, connection_id, parse_model_route,
)
from ouroboros.settings_integrity import runtime_setting

Reader = Callable[..., Any]  # read(flat_setting_key, default) -> the raw saved value
_ROW_META = ("kind", "name", "label", "enabled")
# The pool field, and this process's rotation cursor per pool. The cursor is the NAME of the
# member used last: an unchanged member list is walked in order, and an added or removed
# member only moves where the next walk starts.
_POOL_FIELD = "members"
_ROTATION: dict[tuple[str, str], str] = {}
_ROTATION_LOCK = threading.Lock()


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


def pool_members(row: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The account names a pool row lists, in owner order; empty for anything else."""
    if not isinstance(row, Mapping):
        return ()
    raw = row.get(_POOL_FIELD)
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(name for name in (_text(entry) for entry in raw) if name)


def _member_rows(kind: str, row: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> tuple[dict, ...]:
    """The pool's members that can serve now: enabled, credential-complete, and accounts
    themselves (a pool never nests). An unready member is SKIPPED, not fatal — a pool is
    exactly where one account may be out of use or out of credit."""
    rows = tuple(rows)
    out = []
    for name in pool_members(row):
        member = find_row(rows, kind, name)
        if member is None or not _enabled(member) or pool_members(member):
            continue
        if _ready(kind, member.get):
            out.append(member)
    return tuple(out)


def _choose_member(kind: str, name: str, row: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> dict:
    """The member THIS call uses: the one after the last choice, in owner order."""
    candidates = _member_rows(kind, row, rows)
    if not candidates:
        raise unavailable(RouteParts(kind, name, ""), "missing")
    names = [_text(member.get("name")) for member in candidates]
    with _ROTATION_LOCK:
        last = _ROTATION.get((kind, name), "")
        chosen = candidates[(names.index(last) + 1) % len(candidates) if last in names else 0]
        _ROTATION[(kind, name)] = _text(chosen.get("name"))
    return chosen


def _ready(kind: str, get: Callable[..., Any]) -> bool:
    return any(all(_text(get(field)) for field in group) for group in PROVIDER_KINDS[kind].ready)


def unavailable(route: RouteParts, reason: str) -> Exception:
    from ouroboros.llm_attempt import ConnectionUnavailable

    return ConnectionUnavailable(connection_id(route.kind, route.connection), reason)


def connection_check(route: RouteParts, read: Reader, rows: Iterable[Mapping[str, Any]]) -> None:
    """Raise ``ConnectionUnavailable`` unless ``route``'s connection can serve now; choose nothing.

    A read-only question (a picker, a reference check, a capability fingerprint) uses this
    one: a pool validates by having one serveable member, and asking must not spend a
    rotation or move a real call's account."""
    rows = tuple(rows)
    spec = PROVIDER_KINDS.get(route.kind)
    if spec is None or not route.valid:
        raise unavailable(route, "invalid")
    if not route.connection:
        if not _enabled(find_row(rows, route.kind, BUILTIN_CONNECTION) or {}):
            raise unavailable(route, "disabled")
        return
    row = find_row(rows, route.kind, route.connection)
    if row is None:
        named_elsewhere = any(_text(other.get("name")) == route.connection for other in rows)
        raise unavailable(route, "kind_mismatch" if named_elsewhere else "missing")
    if not _enabled(row):
        raise unavailable(route, "disabled")
    if pool_members(row):
        # A member of another kind, or a pool as a member, is an authoring error: refuse the
        # route rather than quietly serving the accounts this layer happens to recognise.
        for member in pool_members(row):
            target = find_row(rows, route.kind, member)
            if target is None:
                if any(_text(other.get("name")) == member for other in rows):
                    raise unavailable(route, "invalid")
                continue
            if pool_members(target):
                raise unavailable(route, "invalid")
        if not _member_rows(route.kind, row, rows):
            raise unavailable(route, "missing")
        return
    if not _ready(route.kind, row.get):
        raise unavailable(route, "missing")


def connection_values(route: RouteParts, read: Reader, rows: Iterable[Mapping[str, Any]],
                      *, choose: bool = True) -> Callable[..., Any]:
    """Raw field reader ``get(row_field, default="")`` for the connection ``route`` names.

    The built-in connection reads its kind's flat settings through ``read``; a named
    one reads its row, which must exist for that kind, be enabled and carry its
    credential or endpoint. A POOL row reads one of its members, rotated per call;
    ``choose=False`` answers with its first serveable member and takes no rotation.
    Anything else raises ``ConnectionUnavailable`` with the reason ``missing``,
    ``disabled``, ``kind_mismatch`` or ``invalid``."""
    rows = tuple(rows)
    connection_check(route, read, rows)
    if not route.connection:
        spec = PROVIDER_KINDS[route.kind]
        return lambda field, default="": read(spec.flat_key(field), default) if spec.flat_key(field) else default
    row = find_row(rows, route.kind, route.connection)
    if row is None:
        raise unavailable(route, "missing")
    if pool_members(row):
        row = (_choose_member(route.kind, route.connection, row, rows) if choose
               else _member_rows(route.kind, row, rows)[0])
    return lambda field, default="": row.get(field, default)


def _has_credential(route: RouteParts, read: Reader, rows: tuple[dict, ...]) -> bool:
    """Whether the connection carries its credential/endpoint, enabled or not."""
    if route.connection:
        row = find_row(rows, route.kind, route.connection)
        if row is None:
            return False
        if pool_members(row):
            return bool(_member_rows(route.kind, row, rows))
        return _ready(route.kind, row.get)
    if route.kind == "openai-compatible":  # the built-in pair keeps the legacy OPENAI_* derivation
        return compatible_endpoint_configured(lambda key: _text(read(key, "")))
    spec = PROVIDER_KINDS[route.kind]
    return _ready(route.kind, lambda field: read(spec.flat_key(field), ""))


def connection_configured(route: RouteParts, read: Reader, rows: Iterable[Mapping[str, Any]]) -> bool:
    """Whether the connection ``route`` names can serve now: enabled, with its credential/endpoint."""
    from ouroboros.llm_attempt import ConnectionUnavailable

    rows = tuple(rows)
    try:
        connection_check(route, read, rows)
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
    """The picker label of a connection: ``OpenRouter``, ``OpenRouter · Claude credits``,
    ``GigaChat · Keys · 2 accounts``."""
    spec = PROVIDER_KINDS[kind]
    if not name:
        return spec.label
    row = find_row(rows, kind, name) or {}
    label = f"{spec.label} · {_text(row.get('label')) or name}"
    members = pool_members(row)
    return f"{label} · {len(members)} account{'s' if len(members) != 1 else ''}" if members else label


def connections_meta(settings: Mapping[str, Any], document: Mapping[str, Any]) -> list[dict]:
    """``GET /api/settings`` ``_meta.connections``: every connection as the picker shows it.

    ``source`` says where its credential is read from now: the saved document,
    the environment (a key the document leaves unset), or nowhere yet. A compatible
    endpoint that refused prompt-cache hints and was learned without them carries
    ``cache_hints: {refused, learned_at}`` (``request_wire_contract.endpoint_cache_hint_refusal``)."""
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
            **({"members": list(pool_members(row))} if pool_members(row) else {}),
            **(_cache_hints(route, read, row) if route.kind == "openai-compatible" and configured else {}),
        })
    return out


def _cache_hints(route: RouteParts, read: Reader, row: Mapping[str, Any]) -> dict:
    from ouroboros.request_wire_contract import endpoint_cache_hint_refusal

    builtin_url = compatible_endpoint(lambda key: _text(read(key, "")))[1]
    learned = endpoint_cache_hint_refusal(route.kind, _text(row.get("base_url")) if route.connection else builtin_url)
    return {"cache_hints": {"refused": True, "learned_at": learned}} if learned else {}


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


def connection_references(settings: Mapping[str, Any]) -> list[dict]:
    """Role, fallback and agent rows that name a connection which cannot serve them:
    removed, disabled, of another kind or misspelled. They keep their value and refuse."""
    from ouroboros.llm_attempt import ConnectionUnavailable
    from ouroboros.configured_subagents import SUBAGENTS_SETTING

    from ouroboros.model_slots import MODEL_ROLE_SETTINGS

    read, rows = settings_view(settings)
    named = []
    for key in dict.fromkeys((*MODEL_ROLE_SETTINGS.values(), *MODEL_SETTING_KEYS)):
        models = [model.strip() for model in str(settings.get(key) or "").split(",") if model.strip()]
        # A chain entry is named by its 1-based position, so the receipt can say "Fallback 2".
        named += [(key, str(index + 1) if len(models) > 1 or "FALLBACK" in key else "", model)
                  for index, model in enumerate(models)]
    named += [(SUBAGENTS_SETTING, row_id, target) for row_id, target in _subagent_routes(settings.get(SUBAGENTS_SETTING))]
    out = []
    for setting, entry, model in named:
        route = parse_model_route(model)
        if route.kind not in PROVIDER_KINDS:
            continue
        try:
            connection_check(route, read, rows)
        except ConnectionUnavailable as exc:
            out.append({"setting": setting, **({"entry": entry} if entry else {}), "route": model,
                        "connection": exc.connection, "reason": exc.reason})
    return out


def _pool_row_problem(where: str, kind: str, name: str, row: Mapping[str, Any]) -> str:
    """Why this saved row cannot be a pool, or '' (a pool lists named accounts and holds no credential)."""
    if name == BUILTIN_CONNECTION:
        return f"{where}: the built-in {PROVIDER_KINDS[kind].label} connection cannot be a pool."
    members = row.get(_POOL_FIELD)
    if not isinstance(members, list) or not members:
        return f"{where}: members must be a non-empty list of connection names of kind {kind}."
    unknown = sorted(set(row) - {"kind", "name", "label", "enabled", _POOL_FIELD})
    if unknown:
        return (f"{where}: a pool row carries only kind, name, label, enabled and members "
                f"(got {', '.join(unknown)}); each account is its own member row.")
    if "label" in row and not isinstance(row.get("label"), str):
        return f"{where}: label must be text."
    for entry in members:
        if not isinstance(entry, str) or not parse_model_route(f"{kind}@{entry.strip()}::").valid:
            return (f"{where}: every member must be a connection name "
                    f"(1-32 lowercase letters, digits, '-' or '_', starting with a letter or digit).")
    if len({_text(entry) for entry in members}) != len(members):
        return f"{where}: members lists the same connection twice."
    return ""


def _pool_reference_problem(rows: list[dict]) -> str:
    """Why a saved pool names something it cannot rotate over, or '' (its members are accounts of one kind)."""
    for row in rows:
        members = pool_members(row)
        if not members:
            continue
        pool = connection_id(_text(row.get("kind")), _text(row.get("name")))
        for member in members:
            if member == BUILTIN_CONNECTION:
                return (f"{pool}: a member must be a named connection; move the built-in account "
                        f"into its own row first.")
            target = find_row(rows, _text(row.get("kind")), member)
            if target is None:
                elsewhere = any(_text(other.get("name")) == member for other in rows)
                return (f"{pool}: member {member!r} is "
                        f"{'a connection of another kind' if elsewhere else 'not a saved connection'}.")
            if pool_members(target):
                return f"{pool}: member {member!r} is itself a pool; a pool's members are accounts."
    return ""


def normalize_connections(raw: Any) -> tuple[list[dict], str]:
    """``(rows, error)`` for a saved ``OUROBOROS_CONNECTIONS``: the shape every reader
    relies on, or the first problem as one owner-facing sentence (nothing saved)."""
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
        if _POOL_FIELD in row:
            problem = _pool_row_problem(where, kind, name, row)
            if problem:
                return [], problem
            rows.append({**row, "kind": kind, "name": name})
            continue
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
        rows.append({**row, "kind": kind, "name": name})
    if problem := _pool_reference_problem(rows):
        return [], problem
    return rows, ""


def _json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None
