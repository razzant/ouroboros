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
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Mapping

from ouroboros.provider_models import (
    BUILTIN_CONNECTION, CONNECTIONS_KEY, MODEL_SETTING_KEYS, PROVIDER_KINDS, RouteParts, choice_error,
    compatible_endpoint, compatible_endpoint_configured, connection_id, parse_model_route,
)
from ouroboros.settings_integrity import runtime_setting

Reader = Callable[..., Any]  # read(flat_setting_key, default) -> the raw saved value
_ROW_META = ("kind", "name", "label", "enabled")


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
            connection_values(route, read, rows)
        except ConnectionUnavailable as exc:
            out.append({"setting": setting, **({"entry": entry} if entry else {}), "route": model,
                        "connection": exc.connection, "reason": exc.reason})
    return out


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
    return rows, ""


def _json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None
