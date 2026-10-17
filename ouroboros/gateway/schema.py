"""Executable Gateway ABI (ABI-3, owner Q7=A): derived JSON Schema + ingress validation.

The descriptive TypedDicts in ``gateway/contracts.py`` stay the one SSOT;
schemas are DERIVED from them here (never hand-written beside them), so a
contract edit cannot leave a stale schema behind. Validation is INGRESS-ONLY —
client→server WS messages and the typed HTTP request bodies:

- open-world: unknown keys are ALLOWED (additive contract evolution);
- declared keys are type-checked; required keys must be present;
- egress frames and history REPLAY are never validated — stored legacy rows
  (the ABI-3 "stored" axis, e.g. an archived ``telegram_chat_id``) must keep
  replaying, so validation lives strictly on the inbound seams.

``GATEWAY_ABI_VERSION`` is the ABI version carrier, deliberately decoupled
from the product version: it moves when the wire contract breaks (this is the
7.0 break), not on ordinary releases.

``web/modules/api_types.js`` ``GATEWAY_CONTRACT_VERSION`` is NOT a mirror of
this constant and is not becoming one, whatever its name suggests (RES-15a):
it is a RELEASE version carrier, pinned equal to the ``VERSION`` file by
``tests/test_gateway_parity.py`` and rewritten beside ``pyproject.toml`` and
``web/package.json`` by ``ouroboros.tools.release_sync``. The JSDoc typedefs
in that file ARE the exact browser mirror of ``gateway.contracts`` — the
ABI-3 deferrals were paid off in the F3.3 sweep; only the version constant
answers a different question.

``contract_document`` derives the published ABI (``GET /api/contract``) from the
same owners: the router-verified ``endpoint_index`` tuples and their shape
bindings, the WS envelope index, and these derived schemas. It is a projection
for clients, never runtime egress enforcement.
"""

from __future__ import annotations

import importlib
import inspect
import re
from typing import Any, Dict, List, Literal, Union, get_args, get_origin, get_type_hints

GATEWAY_ABI_VERSION = "7.0"

_NoneType = type(None)


def _is_typed_dict(tp: Any) -> bool:
    return isinstance(tp, type) and hasattr(tp, "__required_keys__") and hasattr(tp, "__optional_keys__")


def _strip_qualifier(tp: Any) -> Any:
    """Unwrap ``Required[...]`` / ``NotRequired[...]`` (requiredness is read
    from the TypedDict's own key sets, not from the hint)."""
    origin = get_origin(tp)
    name = getattr(origin, "__name__", "") or getattr(tp, "__name__", "")
    if name in ("Required", "NotRequired"):
        return get_args(tp)[0]
    return tp


def _type_schema(tp: Any) -> Dict[str, Any]:
    """JSON Schema for one annotation from the contracts subset."""
    tp = _strip_qualifier(tp)
    if tp is Any:
        return {}
    if tp is None or tp is _NoneType:
        return {"type": "null"}
    if tp is str:
        return {"type": "string"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if _is_typed_dict(tp):
        return json_schema_for(tp)
    if tp is dict:
        return {"type": "object"}
    if tp is list:
        return {"type": "array"}
    origin = get_origin(tp)
    if origin is Literal:
        return {"enum": list(get_args(tp))}
    if origin is Union:
        return {"anyOf": [_type_schema(arg) for arg in get_args(tp)]}
    if origin is dict:
        return {"type": "object"}
    if origin in (list, tuple):
        args = get_args(tp)
        if args and args[0] is not Any:
            return {"type": "array", "items": _type_schema(args[0])}
        return {"type": "array"}
    # Fail-open on an unmapped annotation: an ABI addition must never make the
    # validator reject payloads the contract meant to accept.
    return {}


_SCHEMA_CACHE: Dict[Any, Dict[str, Any]] = {}


def _qualifier_name(hint: Any) -> str:
    origin = get_origin(hint)
    return getattr(origin, "__name__", "") or getattr(origin, "_name", "") or ""


def _required_keys(typed_dict: Any) -> frozenset:
    """Requiredness resolved from the hints, not ``__required_keys__`` alone.

    Under PEP 563 (``from __future__ import annotations`` in contracts.py) the
    class-time ``__required_keys__`` sees only STRING annotations and cannot
    detect ``NotRequired``/``Required`` qualifiers. An explicit qualifier found
    in the resolved hints wins; an unqualified key keeps the requiredness the
    runtime computed from its DECLARING class's totality (PEP 655), which
    ``__required_keys__`` gets right on every version, including inherited keys
    on 3.11 where ``typing.TypedDict`` keeps no ``__orig_bases__``."""
    hints = get_type_hints(typed_dict, include_extras=True)
    runtime_required = set(getattr(typed_dict, "__required_keys__", ()))
    required: set = set()
    for key, hint in hints.items():
        qualifier = _qualifier_name(hint)
        if qualifier == "Required" or (qualifier != "NotRequired" and key in runtime_required):
            required.add(key)
    return frozenset(required)


def json_schema_for(typed_dict: Any) -> Dict[str, Any]:
    """The derived JSON Schema (draft-agnostic subset) for one contract TypedDict."""
    cached = _SCHEMA_CACHE.get(typed_dict)
    if cached is not None:
        return cached
    hints = get_type_hints(typed_dict, include_extras=True)
    properties = {name: _type_schema(hint) for name, hint in hints.items()}
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        # Open ABI evolution: unknown keys pass (additive fields, newer clients).
        "additionalProperties": True,
    }
    required = sorted(_required_keys(typed_dict))
    if required:
        schema["required"] = required
    _SCHEMA_CACHE[typed_dict] = schema
    return schema


def _json_type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True


def _validate(value: Any, schema: Dict[str, Any], path: str) -> List[str]:
    errors: List[str] = []
    if not schema:
        return errors
    if "enum" in schema:
        if value not in schema["enum"]:
            allowed = ", ".join(str(v) for v in schema["enum"])
            errors.append(f"{path} must be one of: {allowed}")
        return errors
    if "anyOf" in schema:
        for candidate in schema["anyOf"]:
            if not _validate(value, candidate, path):
                return []
        errors.append(f"{path} matches no allowed shape")
        return errors
    expected = schema.get("type")
    if expected is not None and not _json_type_ok(value, expected):
        errors.append(f"{path} must be a JSON {expected}")
        return errors
    if expected == "object":
        properties = schema.get("properties") or {}
        for key in schema.get("required", ()):  # missing-required messages keep
            if key not in value:                # the historical "<key> is required"
                errors.append(f"{key} is required" if not path else f"{path}.{key} is required")
        for key, sub in properties.items():
            if key in value:
                errors.extend(_validate(value[key], sub, f"{path}.{key}" if path else key))
    elif expected == "array" and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(_validate(item, schema["items"], f"{path}[{index}]"))
    return errors


def validate_ingress(payload: Any, typed_dict: Any) -> List[str]:
    """Errors validating one INBOUND payload against a contract TypedDict.

    Empty list = admitted. Unknown keys never error (open evolution); declared
    keys must carry their declared JSON types; required keys must be present.
    """
    if not isinstance(payload, dict):
        return ["payload must be a JSON object"]
    return _validate(payload, json_schema_for(typed_dict), "")


# Body words of ``endpoint_index.Shape``: anything else names a contract TypedDict (JSON).
_BODY_MEDIA = {
    "json": "application/json", "html": "text/html", "sse": "text/event-stream",
    "file": "application/octet-stream", "zip": "application/zip", "js": "application/javascript",
    "empty": "", "any": "*/*", "multipart": "multipart/form-data",
    "form": "application/x-www-form-urlencoded",
}
CONTRACT_MODULES = (
    "ouroboros.gateway.contracts", "ouroboros.gateway.client_contracts",
    "ouroboros.gateway.management_contracts", "ouroboros.gateway.history_contracts",
    "ouroboros.gateway.ui_i18n_contracts", "ouroboros.gateway.mcp_import_contracts",
    "ouroboros.gateway.model_route_contracts", "ouroboros.gateway.call_contracts",
)
_CONTRACT_CACHE: Dict[str, Any] = {}


def contract_typed_dict(name: str) -> Any:
    """The contract TypedDict a shape names, from the gateway contract modules (KeyError if none)."""
    for module_name in CONTRACT_MODULES:
        candidate = getattr(importlib.import_module(module_name), name, None)
        if _is_typed_dict(candidate):
            return candidate
    raise KeyError(f"no gateway contract TypedDict named {name!r}")


def _contract_body(spec: str, names: set) -> Dict[str, Any]:
    media: List[str] = []
    schemas: List[str] = []
    for alternative in spec.split("|"):
        kind, _, item = alternative.partition("/")
        if kind in _BODY_MEDIA:
            media.append(_BODY_MEDIA[kind])
            schemas.extend([item] if item else [])
        else:
            media.append("application/json")
            schemas.append(kind)
    names.update(schemas)
    return {"content_type": " | ".join(dict.fromkeys(media)), "schemas": schemas}


def _contract_http_entry(token: str, shape: Any, public_paths: frozenset, names: set) -> Dict[str, Any]:
    method, path = token.split(" ", 1)
    entry: Dict[str, Any] = {
        "method": method, "path": path,
        "path_params": re.findall(r"\{([^}:]+)(?::[^}]*)?\}", path),
        "auth": "public" if path in public_paths else "gate",
        "typed": shape is not None,
    }
    if shape is None:
        return entry
    if shape.query:
        entry["query"] = shape.query.split()
    if shape.headers:
        entry["headers"] = shape.headers.split()
    if shape.request:
        entry["request"] = _contract_body(shape.request, names)
    responses: Dict[str, Any] = {}
    for group in shape.responses.split():
        statuses, body = group.split(":", 1)
        for status in statuses.split(","):
            responses[status] = _contract_body(body, names)
    entry["responses"] = responses
    if shape.note:
        entry["note"] = shape.note
    return entry


def _described_schema(name: str) -> Dict[str, Any]:
    typed_dict = contract_typed_dict(name)
    schema = dict(json_schema_for(typed_dict))  # a copy: the ingress cache stays description-free
    doc = typed_dict.__dict__.get("__doc__")
    if doc:
        schema["description"] = inspect.cleandoc(doc)
    return schema


def contract_document() -> Dict[str, Any]:
    """``GatewayContractResponse``: the published gateway ABI, derived once per process."""
    if "document" in _CONTRACT_CACHE:
        return _CONTRACT_CACHE["document"]
    from ouroboros import get_version
    from ouroboros.gateway import endpoint_index as index
    from ouroboros.gateway.contracts import WS_MESSAGE_TYPES
    from ouroboros.server_auth import _PUBLIC_HTTP_PATHS

    names: set = set()
    public_paths = frozenset(_PUBLIC_HTTP_PATHS)
    http = [_contract_http_entry(token, index.HTTP_SHAPES.get(token), public_paths, names)
            for token in index.HTTP_ENDPOINTS if not token.startswith("WS ")]
    http += [_contract_http_entry(token, shape, public_paths, names) for token, shape in index.AUTH_ENDPOINTS.items()]
    messages = []
    for ws_type in WS_MESSAGE_TYPES:
        inbound, outbound = index.WS_SHAPES[ws_type]
        names.update(name for name in (inbound, outbound) if name)
        messages.append({"type": ws_type, **({"inbound": inbound} if inbound else {}),
                         **({"outbound": outbound} if outbound else {})})
    from ouroboros.gateway.call_contracts import PEER_CONTROLS, SERVER_CONTROLS, socket_contract

    names.update(cls.__name__ for cls in (*PEER_CONTROLS.values(), *SERVER_CONTROLS.values()))
    names.update(("CallSocketContract", "HostCallOpenRequest", "HostCallOpenResponse"))
    document = {
        "call": socket_contract(),
        "abi_version": GATEWAY_ABI_VERSION,
        "runtime_version": get_version(),
        "core_paths": list(index.CORE_PATHS),
        "interface_paths": list(index.INTERFACE_PATHS),
        "http": http,
        "ws": {"path": "/ws", "messages": messages, "extension_namespace": index.EXTENSION_WS_NAMESPACE},
        "schemas": {name: _described_schema(name) for name in sorted(names)},
    }
    _CONTRACT_CACHE["document"] = document
    return document


__all__ = [
    "CONTRACT_MODULES",
    "GATEWAY_ABI_VERSION",
    "contract_document",
    "contract_typed_dict",
    "json_schema_for",
    "validate_ingress",
]
