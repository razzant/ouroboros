"""Pure direct-OpenAI Chat function/custom wire codec.

Ouroboros keeps one canonical function-tool transcript. This module projects a
copy into OpenAI Chat custom tools, binds the exact original JSON Schemas to the
physical catalog, and converts custom calls back without introducing a second
stored transcript or a Responses dependency.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ouroboros.request_wire_contract import (
    canonical_json,
    canonical_sha256,
    tool_choice_family,
    tool_choice_names,
)
from ouroboros.request_wire_custom_validation import (
    CustomArgumentValidationReceipt,
    issue_custom_argument_validation_receipt,
)
from ouroboros.request_wire_receipts import WireCandidateManifest

# OpenAI's bound on a Chat custom tool's description, in Unicode code points (what len() counts): a direct
# api.openai.com send on 2026-11-15 accepted 4,096 and refused 4,097 as `string_above_max_length` on
# `tools[0].custom.description`, the same for non-ASCII and non-BMP text. Set too low, field prose that fits is
# dropped; set too high, that refusal fails the call: the dialect ladder does not read it as a dialect refusal,
# and Main reads the code as an input too large (`context_budget.CONTEXT_OVERFLOW_CODES`), which no shrinking of
# the transcript cures. Re-probe before changing it (the OpenAI provider canary sends the registry through here).
MAX_CUSTOM_DESCRIPTION_CHARS = 4096
_SCHEMA_CLAUSE = "Input must be one JSON object matching this schema: "

JSON_OBJECT_LARK_GRAMMAR = r"""start: object
?value: object
      | array
      | ESCAPED_STRING -> string
      | NUMBER         -> number
      | "true"         -> true
      | "false"        -> false
      | "null"         -> null
array: "[" [value ("," value)*] "]"
pair: ESCAPED_STRING ":" value
object: "{" [pair ("," pair)*] "}"
ESCAPED_STRING: /"(\\["\\\/bfnrt]|\\u[0-9a-fA-F]{4}|[^"\\\x00-\x1f])*"/
NUMBER: /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/
%import common.WS
%ignore WS
"""

_CUSTOM_FORMAT = {
    "type": "grammar",
    "grammar": {
        "syntax": "lark",
        "definition": JSON_OBJECT_LARK_GRAMMAR,
    },
}
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_SCHEMA_PROSE_KEYS = frozenset({
    "$comment",
    "default",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
})
_SCHEMA_VALUE_KEYWORDS = frozenset({
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
})
_SCHEMA_ARRAY_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_MAP_KEYWORDS = frozenset({
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
})


class CustomToolProjectionError(ValueError):
    """The canonical function catalog cannot be represented without truncating structure."""


@dataclass(frozen=True)
class BoundToolSchema:
    name: str
    schema_json: str
    schema_sha256: str

    def schema(self) -> Dict[str, Any]:
        value = json.loads(self.schema_json)
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class ProjectedCustomToolCatalog:
    wire_tools_json: str
    schemas: Tuple[BoundToolSchema, ...]
    catalog_sha256: str
    serialized_bytes: int

    def wire_tools(self) -> list[Dict[str, Any]]:
        value = json.loads(self.wire_tools_json)
        return list(value) if isinstance(value, list) else []

    def schema_binding(self, name: str) -> Optional[BoundToolSchema]:
        return next((item for item in self.schemas if item.name == name), None)

    @property
    def tool_names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.schemas)


@dataclass(frozen=True)
class ProjectedCustomToolRequest:
    catalog: ProjectedCustomToolCatalog
    tool_choice: Any


def _compact_schema_map(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return deepcopy(value)
    return {
        str(name): _compact_schema(schema)
        for name, schema in sorted(value.items(), key=lambda pair: str(pair[0]))
    }


def _compact_dependencies(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return deepcopy(value)
    compacted = {}
    for name, dependency in sorted(value.items(), key=lambda pair: str(pair[0])):
        if isinstance(dependency, (Mapping, bool)):
            compacted[str(name)] = _compact_schema(dependency)
        else:
            compacted[str(name)] = deepcopy(dependency)
    return compacted


def _compact_schema(value: Any) -> Any:
    """Remove schema prose without interpreting literal JSON values as schemas."""
    if isinstance(value, bool):
        return value
    if not isinstance(value, Mapping):
        return deepcopy(value)
    compacted = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        key_text = str(key)
        if key_text in _SCHEMA_PROSE_KEYS or key_text.lower().startswith("x-"):
            continue
        if key_text in _SCHEMA_VALUE_KEYWORDS:
            compacted[key_text] = _compact_schema(item)
        elif key_text in _SCHEMA_ARRAY_KEYWORDS and isinstance(item, list):
            compacted[key_text] = [_compact_schema(schema) for schema in item]
        elif key_text in _SCHEMA_MAP_KEYWORDS:
            compacted[key_text] = _compact_schema_map(item)
        elif key_text == "items" and isinstance(item, list):
            compacted[key_text] = [_compact_schema(schema) for schema in item]
        elif key_text == "items":
            compacted[key_text] = _compact_schema(item)
        elif key_text == "dependencies":
            compacted[key_text] = _compact_dependencies(item)
        else:
            compacted[key_text] = deepcopy(item)
    return compacted


def _build_projected_catalog(
    projected: Sequence[Mapping[str, Any]],
    bindings: Sequence[BoundToolSchema],
) -> ProjectedCustomToolCatalog:
    projected_sorted = sorted(
        projected,
        key=lambda item: str((item.get("custom") or {}).get("name") or ""),
    )
    bindings_sorted = sorted(bindings, key=lambda item: item.name)
    if not projected_sorted or not bindings_sorted:
        raise CustomToolProjectionError("custom projection requires at least one function tool")
    wire_tools_json = canonical_json(projected_sorted)
    catalog_sha256 = canonical_sha256({
        "wire_tools": projected_sorted,
        "schemas": [
            {"name": item.name, "schema_sha256": item.schema_sha256}
            for item in bindings_sorted
        ],
    })
    return ProjectedCustomToolCatalog(
        wire_tools_json=wire_tools_json,
        schemas=tuple(bindings_sorted),
        catalog_sha256=catalog_sha256,
        serialized_bytes=len(wire_tools_json.encode("utf-8")),
    )


def _description_with_schema(description: Any, schema: Mapping[str, Any]) -> str:
    """The lead and the schema the model reads, within OpenAI's ``custom.description`` bound.

    A tool whose whole lead and whole schema fit sends the schema as the function dialect does, field
    descriptions and defaults included, its text as written: the bound counts code points, so a non-ASCII
    description costs what it holds, not six characters of escape for each. A larger one sends the structure
    alone, its lead whole or clipped beside it; a structure over the bound by itself cannot be represented.
    """
    if isinstance(description, str):
        prose = " ".join(description.split())
    elif description is None:
        prose = ""
    else:
        prose = canonical_json(description)
    whole = (prose + "\n" if prose else "") + _SCHEMA_CLAUSE + json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    if len(whole) <= MAX_CUSTOM_DESCRIPTION_CHARS:
        return whole
    schema_clause = _SCHEMA_CLAUSE + canonical_json(_compact_schema(schema))
    if len(schema_clause) > MAX_CUSTOM_DESCRIPTION_CHARS:
        raise CustomToolProjectionError(
            "tool schema structure exceeds OpenAI custom.description limit"
        )
    if not prose:
        return schema_clause
    available = MAX_CUSTOM_DESCRIPTION_CHARS - len(schema_clause) - 1
    if available <= 0:
        return schema_clause
    if len(prose) > available:
        if available < 3:
            return schema_clause
        prose = prose[:available - 3].rstrip() + "..."
    return prose + "\n" + schema_clause


def project_function_tools_to_openai_custom(
    tools: Sequence[Mapping[str, Any]],
) -> ProjectedCustomToolCatalog:
    """Project an all-function catalog and bind its exact original schemas."""
    projected = []
    bindings = []
    seen = set()
    for tool in tools:
        if not isinstance(tool, Mapping) or str(tool.get("type") or "function") != "function":
            raise CustomToolProjectionError("custom projection accepts function tools only")
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise CustomToolProjectionError("function tool is missing its function object")
        name = str(function.get("name") or "").strip()
        if not _TOOL_NAME_RE.fullmatch(name) or name in seen:
            raise CustomToolProjectionError("function tool name is missing, duplicate, or invalid")
        seen.add(name)
        schema = function.get("parameters")
        if not isinstance(schema, Mapping):
            schema = {"type": "object", "properties": {}}
        schema_dict = dict(schema)
        schema_json = canonical_json(schema_dict)
        schema_sha256 = canonical_sha256(schema_dict)
        bindings.append(BoundToolSchema(name, schema_json, schema_sha256))
        projected.append({
            "type": "custom",
            "custom": {
                "name": name,
                "description": _description_with_schema(function.get("description"), schema_dict),
                "format": _CUSTOM_FORMAT,
            },
        })
    return _build_projected_catalog(projected, bindings)


def project_tool_choice_for_openai_custom(
    tool_choice: Any,
    catalog: ProjectedCustomToolCatalog,
) -> Any:
    names = set(catalog.tool_names)
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip()
        lowered = normalized.lower()
        if lowered in {"auto", "none", "required"}:
            return lowered
        if lowered == "any":
            return "required"
        name = normalized
    elif isinstance(tool_choice, Mapping):
        choice_type = str(tool_choice.get("type") or "").strip().lower()
        if choice_type in {"auto", "none", "required"}:
            return choice_type
        if choice_type == "any":
            return "required"
        nested = tool_choice.get("function") or tool_choice.get("custom") or {}
        nested_name = nested.get("name") if isinstance(nested, Mapping) else None
        name = str(nested_name or tool_choice.get("name") or "").strip()
    else:
        raise CustomToolProjectionError("unsupported custom tool choice")
    if name not in names:
        raise CustomToolProjectionError("named custom tool choice is absent from catalog")
    return {"type": "custom", "custom": {"name": name}}


def _filter_custom_catalog(
    catalog: ProjectedCustomToolCatalog,
    allowed_names: Sequence[str],
) -> ProjectedCustomToolCatalog:
    allowed = set(allowed_names)
    projected = [
        tool
        for tool in catalog.wire_tools()
        if str((tool.get("custom") or {}).get("name") or "") in allowed
    ]
    bindings = [binding for binding in catalog.schemas if binding.name in allowed]
    if {binding.name for binding in bindings} != allowed:
        raise CustomToolProjectionError("allowed custom tool choice names an unknown tool")
    return _build_projected_catalog(projected, bindings)


def project_function_tool_request_to_openai_custom(
    tools: Sequence[Mapping[str, Any]],
    tool_choice: Any,
) -> ProjectedCustomToolRequest:
    """Project one physical custom-tool request, including allowed-tool narrowing."""
    catalog = project_function_tools_to_openai_custom(tools)
    if tool_choice_family(tool_choice) != "allowed_tools":
        return ProjectedCustomToolRequest(
            catalog=catalog,
            tool_choice=project_tool_choice_for_openai_custom(tool_choice, catalog),
        )
    allowed_names = tool_choice_names(tool_choice)
    if not allowed_names:
        raise CustomToolProjectionError("allowed custom tool choice requires tool names")
    allowed_block = tool_choice.get("allowed_tools")
    mode_value = (
        allowed_block.get("mode") if isinstance(allowed_block, Mapping) else None
    ) or tool_choice.get("mode") or "auto"
    mode = str(mode_value).strip().lower()
    if mode not in {"auto", "required"}:
        raise CustomToolProjectionError("allowed custom tool choice mode must be auto or required")
    return ProjectedCustomToolRequest(
        catalog=_filter_custom_catalog(catalog, allowed_names),
        tool_choice=mode,
    )


def _validated_exact_identity(
    value: Any,
    *,
    context: str,
    field: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CustomToolProjectionError(f"{context} tool call {field} is missing or empty")
    if value != value.strip():
        raise CustomToolProjectionError(
            f"{context} tool call {field} has surrounding whitespace"
        )
    return value


def _validated_call_id(
    call: Any,
    seen_call_ids: set[str],
    *,
    context: str,
    expected_type: str,
) -> str:
    if not isinstance(call, Mapping):
        raise CustomToolProjectionError(f"{context} tool call must be a mapping")
    if call.get("type") != expected_type:
        raise CustomToolProjectionError(
            f"{context} tool call type must be exactly {expected_type}"
        )
    call_id = _validated_exact_identity(
        call.get("id"),
        context=context,
        field="id",
    )
    if call_id in seen_call_ids:
        raise CustomToolProjectionError(f"{context} contains a duplicate tool call id")
    seen_call_ids.add(call_id)
    return call_id


def reencode_canonical_calls_for_openai_custom(
    tool_calls: Sequence[Mapping[str, Any]],
    catalog: ProjectedCustomToolCatalog,
) -> list[Dict[str, Any]]:
    """Translate a stored canonical function-call copy into custom wire calls."""
    names = set(catalog.tool_names)
    prepared_calls = []
    seen_call_ids = set()
    for call in tool_calls:
        call_id = _validated_call_id(
            call,
            seen_call_ids,
            context="canonical",
            expected_type="function",
        )
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise CustomToolProjectionError("canonical tool call lacks a function object")
        name = _validated_exact_identity(
            function.get("name"),
            context="canonical",
            field="name",
        )
        if name not in names:
            raise CustomToolProjectionError("canonical tool call is absent from custom catalog")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = canonical_json(arguments if arguments is not None else {})
        prepared_calls.append((call_id, name, arguments))
    return [{
        "id": call_id,
        "type": "custom",
        "custom": {"name": name, "input": arguments},
    } for call_id, name, arguments in prepared_calls]


def project_messages_for_openai_custom(
    messages: Sequence[Mapping[str, Any]],
    catalog: ProjectedCustomToolCatalog,
) -> list[Dict[str, Any]]:
    """Project only assistant tool-call records; ordinary role=tool results stay unchanged."""
    projected = []
    for message in messages:
        copy = dict(message)
        calls = copy.get("tool_calls")
        if copy.get("role") == "assistant" and isinstance(calls, list) and calls:
            copy["tool_calls"] = reencode_canonical_calls_for_openai_custom(calls, catalog)
        projected.append(copy)
    return projected


def normalize_openai_custom_tool_calls(
    tool_calls: Sequence[Mapping[str, Any]],
    candidate: WireCandidateManifest,
) -> Tuple[list[Dict[str, Any]], Tuple[CustomArgumentValidationReceipt, ...]]:
    """Normalize only against the exact factory-owned physical custom candidate."""
    if not isinstance(candidate, WireCandidateManifest):
        raise CustomToolProjectionError(
            "custom response requires a factory-issued wire candidate"
        )
    profile = candidate.accepted_profile
    catalog = candidate.custom_catalog
    if (
        profile.provider != "openai"
        or profile.api_surface != "chat.completions"
        or profile.tool_dialect != "openai_chat_custom"
        or catalog is None
    ):
        raise CustomToolProjectionError(
            "custom response requires a direct OpenAI Chat custom candidate"
        )
    prepared_calls = []
    seen_call_ids = set()
    for call in tool_calls:
        call_id = _validated_call_id(
            call,
            seen_call_ids,
            context="custom response",
            expected_type="custom",
        )
        custom = call.get("custom")
        if not isinstance(custom, Mapping):
            raise CustomToolProjectionError("response custom tool call lacks a custom object")
        name = _validated_exact_identity(
            custom.get("name"),
            context="custom response",
            field="name",
        )
        raw_input = custom.get("input")
        if not isinstance(raw_input, str):
            raw_input = ""
        prepared_calls.append((call_id, name, raw_input))

    canonical_calls = []
    receipts = []
    for call_id, name, raw_input in prepared_calls:
        receipt = issue_custom_argument_validation_receipt(
            candidate=candidate,
            tool_call_id=call_id,
            tool_name=name,
            raw_input=raw_input,
        )
        receipts.append(receipt)
        canonical_calls.append({
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": raw_input},
        })
    return canonical_calls, tuple(receipts)


def custom_receipts_prove_wire_acceptance(
    receipts: Sequence[CustomArgumentValidationReceipt],
) -> bool:
    """Schema-invalid JSON may be corrected; malformed/unknown calls prove no wire contract."""
    return bool(receipts) and all(receipt.proves_wire_acceptance for receipt in receipts)
