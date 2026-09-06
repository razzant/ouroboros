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
"""

from __future__ import annotations

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
    """Requiredness resolved from the hints, not ``__required_keys__``.

    Under PEP 563 (``from __future__ import annotations`` in contracts.py) the
    class-time ``__required_keys__`` on Python 3.10 sees only STRING
    annotations, cannot detect ``NotRequired``/``Required`` qualifiers, and
    therefore lies (every key of a total class reads as required). Walk the
    resolved hints instead: an explicit qualifier wins; otherwise the key's
    DECLARING class's totality decides, exactly as PEP 655 specifies."""
    hints = get_type_hints(typed_dict, include_extras=True)
    bases = [b for b in getattr(typed_dict, "__orig_bases__", ())
             if _is_typed_dict(b)]
    required: set = set()
    inherited: set = set()
    for base in bases:
        required |= _required_keys(base)
        inherited |= set(get_type_hints(base))
    total = bool(getattr(typed_dict, "__total__", True))
    for key, hint in hints.items():
        if key in inherited:
            continue  # the declaring base's totality already decided
        qualifier = _qualifier_name(hint)
        if qualifier == "NotRequired":
            required.discard(key)
        elif qualifier == "Required" or total:
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


__all__ = [
    "GATEWAY_ABI_VERSION",
    "json_schema_for",
    "validate_ingress",
]
