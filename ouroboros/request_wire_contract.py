"""Typed, route-scoped evidence for provider request-wire compatibility.

This module is deliberately transport-neutral. It describes one exact physical
route, the small set of repairs that route accepted, the caller-owned physical
attempt budget, and the receipt required before any repair becomes durable. It
never chooses another model, provider, or API surface and never executes
free-form provider prose.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import string
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.utils import (
    is_credential_header_name,
    is_secret_key_name,
    read_json_dict,
    update_json_locked,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ouroboros.request_wire_receipts import WireCompatibilityReceipt

REQUEST_WIRE_SCHEMA_VERSION = 1
REQUEST_WIRE_TTL_SEC = 14 * 24 * 3600.0
REQUEST_WIRE_MAX_CLOCK_SKEW_SEC = 300.0
REQUEST_WIRE_STATE_FILE = "request_wire_compatibility.json"

OPTIONAL_SAMPLING_FIELDS = ("temperature", "top_p", "top_k")
OPTIONAL_REQUEST_FIELDS = OPTIONAL_SAMPLING_FIELDS + (
    "response_format",
    "reasoning_effort",
    "output_config",
    "thinking",
    "parallel_tool_calls",
)
NESTED_REASONING_FIELD = "extra_body.reasoning"
NESTED_THINKING_FIELD = "extra_body.thinking"
ANTHROPIC_REASONING_FIELDS = ("thinking", "output_config")
DROP_FIELDS = frozenset((*OPTIONAL_REQUEST_FIELDS, NESTED_REASONING_FIELD))

TOOL_DIALECTS = frozenset({
    "none",
    "function",
    "openai_chat_custom",
    "server",
    "mixed",
})
TOOL_CHOICE_FAMILIES = frozenset({
    "omitted", "auto", "none", "required", "any", "allowed_tools", "named", "other",
})
FUNCTION_STRICTNESS_VALUES = frozenset({"none", "plain", "strict", "mixed"})
_REGISTERED_DIALECT_TRANSITIONS = frozenset({
    ("tool", "function", "openai_chat_custom"),
    ("tool", "openai_chat_custom", "function"),
})

WIRE_REASON_CODES = frozenset({
    "requested_wire_form",
    "provider_recovery_succeeded",
    "provider_prescribed_value",
    "provider_required_reasoning",
    "provider_unsupported_field",
    "provider_rejected_tool_dialect",
    "provider_metadata_constraint",
    "task_local_availability_fallback",
})

WIRE_SEMANTIC_KINDS = frozenset({
    "chat_message",
    "function_tool_call",
    "custom_tool_call_json_object",
    "anthropic_message",
    "anthropic_tool_use",
})

_SECRET_QUERY_NAMES = frozenset({
    "key",
    "sig",
    "signature",
    "credential",
    "credentials",
})
_TOOL_CHOICE_UNSET = object()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def physical_candidate_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize the exact final payload with the physical-send digest basis."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def physical_candidate_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(physical_candidate_bytes(payload)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and not (set(text.lower()) - set(string.hexdigits.lower()))


def _is_secret_query_name(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return normalized in _SECRET_QUERY_NAMES or is_secret_key_name(normalized)


def normalize_endpoint(value: Any) -> str:
    """Return one absolute, credential-free endpoint identity.

    An omitted URL means the official default endpoint for the named provider.
    Non-secret query selectors remain part of the identity; credentials,
    userinfo, and fragments do not. Invalid or relative URLs return ``""`` so
    they can never acquire durable authority.
    """
    raw = str(value or "").strip()
    if not raw:
        return "provider-default"
    try:
        parsed = urlsplit(raw)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        hostname = parsed.hostname.lower()
        if ":" in hostname:
            hostname = f"[{hostname}]"
        port_value = parsed.port
        port = f":{port_value}" if port_value is not None else ""
        path = (parsed.path or "").rstrip("/")
        safe_query = sorted(
            (str(key), str(item))
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not _is_secret_query_name(key)
        )
        return urlunsplit((
            parsed.scheme.lower(),
            hostname + port,
            path,
            urlencode(safe_query),
            "",
        ))
    except (TypeError, ValueError):
        return ""


def _nested_value_identity(payload: Mapping[str, Any], dotted_path: str) -> Dict[str, Any]:
    """Bind both presence and value for one exact value-profile field."""
    node: Any = payload
    for part in str(dotted_path or "").split("."):
        if not isinstance(node, Mapping) or part not in node:
            return {"present": False}
        node = node[part]
    return {"present": True, "value": node}


def reasoning_carrier(payload: Mapping[str, Any]) -> str:
    if "reasoning_effort" in payload:
        return "reasoning_effort"
    output_config = payload.get("output_config")
    thinking = payload.get("thinking")
    if isinstance(thinking, Mapping):
        thinking_type = str(thinking.get("type") or "").strip().lower()
        if thinking_type == "disabled":
            return "anthropic.disabled"
        if isinstance(output_config, Mapping) and "effort" in output_config:
            return "anthropic.adaptive"
        return f"anthropic.{thinking_type}" if thinking_type else "anthropic.thinking"
    extra_body = payload.get("extra_body")
    if isinstance(extra_body, Mapping) and isinstance(extra_body.get("reasoning"), Mapping):
        return NESTED_REASONING_FIELD
    if isinstance(extra_body, Mapping) and isinstance(extra_body.get("thinking"), Mapping):
        return NESTED_THINKING_FIELD
    return ""


def payload_effort(payload: Mapping[str, Any]) -> str:
    direct = str(payload.get("reasoning_effort") or "").strip().lower()
    if direct:
        return direct
    output_config = payload.get("output_config")
    if isinstance(output_config, Mapping):
        effort = str(output_config.get("effort") or "").strip().lower()
        if effort:
            return effort
    thinking = payload.get("thinking")
    if isinstance(thinking, Mapping) and str(thinking.get("type") or "").lower() == "disabled":
        return "none"
    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, Mapping):
        return ""
    reasoning = extra_body.get("reasoning")
    if isinstance(reasoning, Mapping):
        return str(reasoning.get("effort") or "").strip().lower()
    thinking = extra_body.get("thinking")
    if isinstance(thinking, Mapping) and str(thinking.get("type") or "").lower() == "disabled":
        return "none"
    return ""


def infer_tool_dialect(payload: Mapping[str, Any]) -> str:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return "none"
    dialects = set()
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        tool_type = str(tool.get("type") or "").strip().lower()
        if tool_type == "function" or (not tool_type and tool.get("name") and tool.get("input_schema")):
            dialects.add("function")
        elif tool_type == "custom" and isinstance(tool.get("custom"), Mapping):
            dialects.add("openai_chat_custom")
        else:
            dialects.add("server")
    if len(dialects) == 1:
        return next(iter(dialects))
    return "mixed" if dialects else "none"


def tool_choice_family(choice: Any) -> str:
    if choice is None:
        return "omitted"
    if isinstance(choice, str):
        normalized = choice.strip().lower()
        return normalized if normalized in {"auto", "none", "required", "any"} else "named"
    if isinstance(choice, Mapping):
        normalized = str(choice.get("type") or "").strip().lower()
        if normalized in {"auto", "none", "required", "any", "allowed_tools"}:
            return normalized
        if normalized in {"function", "custom", "tool"}:
            return "named"
        if choice.get("name") or choice.get("function") or choice.get("custom"):
            return "named"
    return "other"


def _tool_choice_entry_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Mapping):
        return ""
    for key in ("function", "custom", "tool"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and str(nested.get("name") or "").strip():
            return str(nested.get("name") or "").strip()
    return str(value.get("name") or "").strip()


def tool_choice_names(choice: Any) -> Tuple[str, ...]:
    """Return exact named/allowed tool names without persisting their schemas."""
    family = tool_choice_family(choice)
    if family == "named":
        name = _tool_choice_entry_name(choice)
        return (name,) if name else ()
    if family != "allowed_tools" or not isinstance(choice, Mapping):
        return ()
    allowed = choice.get("allowed_tools")
    if isinstance(allowed, Mapping):
        raw_tools = allowed.get("tools") or allowed.get("allowed_tools") or []
    else:
        raw_tools = allowed or choice.get("tools") or []
    if not isinstance(raw_tools, (list, tuple)):
        return ()
    return tuple(sorted({
        name for item in raw_tools for name in [_tool_choice_entry_name(item)] if name
    }))


def tool_choice_requires_call(choice: Any) -> bool:
    family = tool_choice_family(choice)
    if family in {"required", "any", "named"}:
        return True
    if family != "allowed_tools" or not isinstance(choice, Mapping):
        return False
    allowed = choice.get("allowed_tools")
    mode = str(
        (allowed.get("mode") if isinstance(allowed, Mapping) else None)
        or choice.get("mode")
        or "auto"
    ).strip().lower()
    return mode == "required"


def tool_choice_allowed_mode(choice: Any) -> str:
    """Return the narrow canonical allowed-tools mode, else ``""``."""
    if tool_choice_family(choice) != "allowed_tools" or not isinstance(choice, Mapping):
        return ""
    allowed = choice.get("allowed_tools")
    mode = str(
        (allowed.get("mode") if isinstance(allowed, Mapping) else None)
        or choice.get("mode")
        or "auto"
    ).strip().lower()
    return mode if mode in {"auto", "required"} else ""


def tool_choice_projection_matches(
    canonical_choice: Any,
    physical_choice: Any,
    *,
    physical_dialect: str,
    physical_catalog_names: Sequence[str],
) -> bool:
    """Check the closed function/custom projection of canonical choice semantics."""
    family = tool_choice_family(canonical_choice)
    physical_family = tool_choice_family(physical_choice)
    names = tool_choice_names(canonical_choice)
    catalog = tuple(physical_catalog_names)
    if physical_dialect != "openai_chat_custom":
        return (
            family,
            names,
            tool_choice_allowed_mode(canonical_choice),
            tool_choice_requires_call(canonical_choice),
            canonical_json(canonical_choice) if family == "other" else "",
        ) == (
            physical_family,
            tool_choice_names(physical_choice),
            tool_choice_allowed_mode(physical_choice),
            tool_choice_requires_call(physical_choice),
            canonical_json(physical_choice) if physical_family == "other" else "",
        )
    if family == "allowed_tools":
        mode = tool_choice_allowed_mode(canonical_choice)
        return bool(mode and names and physical_family == mode and catalog == names)
    if family == "any":
        return physical_family == "required"
    if family == "named":
        return (
            physical_family == "named"
            and tool_choice_names(physical_choice) == names
            and set(names).issubset(catalog)
        )
    return physical_family == family and (not names or set(names).issubset(catalog))


def _tool_choice_digest(choice: Any) -> str:
    family = tool_choice_family(choice)
    if family not in {"named", "allowed_tools", "other"}:
        return ""
    return canonical_sha256({
        "family": family,
        "names": tool_choice_names(choice),
        "requires_call": tool_choice_requires_call(choice),
        "allowed_mode": tool_choice_allowed_mode(choice),
        "raw": choice if family == "other" else None,
    })


def function_strictness(payload: Mapping[str, Any]) -> str:
    strict_values = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, Mapping):
            strict_values.append(bool(function.get("strict")))
    if not strict_values:
        return "none"
    if all(strict_values):
        return "strict"
    return "mixed" if any(strict_values) else "plain"


def _routing_digest(payload: Mapping[str, Any]) -> str:
    extra_body = payload.get("extra_body")
    provider = extra_body.get("provider") if isinstance(extra_body, Mapping) else None
    return canonical_sha256(provider) if isinstance(provider, Mapping) else ""


def _reasoning_options_digest(payload: Mapping[str, Any]) -> str:
    extra_body = payload.get("extra_body")
    reasoning = extra_body.get("reasoning") if isinstance(extra_body, Mapping) else None
    if not isinstance(reasoning, Mapping):
        return ""
    options = {str(key): value for key, value in reasoning.items() if str(key) != "effort"}
    return canonical_sha256(options) if options else ""


def _contract_headers_digest(target: Mapping[str, Any]) -> str:
    # Callers provide only headers whose values alter the provider contract
    # (version/beta/tenant), never the full physical header set. This second
    # guard shares the repository credential-name classifier so late key loading
    # or rotation still cannot fragment durable evidence.
    headers = target.get("contract_headers")
    if not isinstance(headers, Mapping):
        return ""
    safe = {
        str(key).strip().lower(): value
        for key, value in headers.items()
        if not is_credential_header_name(key)
    }
    return canonical_sha256(safe) if safe else ""


@dataclass(frozen=True)
class RequestWireProfile:
    provider: str
    endpoint_sha256: str
    api_surface: str
    model: str
    reasoning_carrier: str
    tool_dialect: str
    tool_choice: str
    tool_choice_sha256: str
    function_strictness: str
    provider_routing_sha256: str
    reasoning_options_sha256: str
    contract_headers_sha256: str
    evidence_scope: str = "capability"
    value_fields: Tuple[str, ...] = ()
    value_sha256: str = ""
    schema_version: int = REQUEST_WIRE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != REQUEST_WIRE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported request-wire profile schema")
        if not self.provider or self.provider == "unknown" or self.provider != self.provider.strip().lower():
            raise ValueError("request-wire profile requires a provider")
        if not self.api_surface or self.api_surface != self.api_surface.strip().lower():
            raise ValueError("request-wire profile requires an API surface")
        if not self.model or self.model != self.model.strip():
            raise ValueError("request-wire profile requires a resolved model")
        if not _valid_sha256(self.endpoint_sha256):
            raise ValueError("request-wire profile requires a normalized endpoint digest")
        if self.tool_dialect not in TOOL_DIALECTS:
            raise ValueError(f"unsupported tool dialect: {self.tool_dialect}")
        if self.tool_choice not in TOOL_CHOICE_FAMILIES:
            raise ValueError(f"unsupported tool-choice family: {self.tool_choice}")
        if self.function_strictness not in FUNCTION_STRICTNESS_VALUES:
            raise ValueError(f"unsupported function strictness: {self.function_strictness}")
        if self.evidence_scope not in {"capability", "value"}:
            raise ValueError(f"unsupported evidence scope: {self.evidence_scope}")
        if (
            not isinstance(self.value_fields, tuple)
            or any(not isinstance(path, str) or not path for path in self.value_fields)
            or tuple(sorted(set(self.value_fields))) != self.value_fields
        ):
            raise ValueError("value evidence fields must be unique, sorted paths")
        if self.evidence_scope == "value":
            if not self.value_fields or not _valid_sha256(self.value_sha256):
                raise ValueError("value evidence requires fields and a value digest")
        elif self.value_fields or self.value_sha256:
            raise ValueError("capability evidence cannot carry value identity")
        for digest in (
            self.tool_choice_sha256,
            self.provider_routing_sha256,
            self.reasoning_options_sha256,
            self.contract_headers_sha256,
        ):
            if digest and not _valid_sha256(digest):
                raise ValueError("malformed request-wire profile digest")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "endpoint_sha256": self.endpoint_sha256,
            "api_surface": self.api_surface,
            "model": self.model,
            "reasoning_carrier": self.reasoning_carrier,
            "tool_dialect": self.tool_dialect,
            "tool_choice": self.tool_choice,
            "tool_choice_sha256": self.tool_choice_sha256,
            "function_strictness": self.function_strictness,
            "provider_routing_sha256": self.provider_routing_sha256,
            "reasoning_options_sha256": self.reasoning_options_sha256,
            "contract_headers_sha256": self.contract_headers_sha256,
            "evidence_scope": self.evidence_scope,
            "value_fields": list(self.value_fields),
            "value_sha256": self.value_sha256,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())


def build_request_wire_profile(
    target: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    api_surface: str,
    evidence_scope: str = "capability",
    value_fields: Sequence[str] = (),
    reasoning_carrier_override: Optional[str] = None,
    tool_dialect_override: Optional[str] = None,
    function_strictness_override: Optional[str] = None,
    tool_choice_override: Any = _TOOL_CHOICE_UNSET,
) -> RequestWireProfile:
    """Build one credential-free profile from the request that will be sent."""
    scope = str(evidence_scope or "").strip().lower()
    if scope not in {"capability", "value"}:
        raise ValueError(f"unsupported evidence scope: {scope}")
    surface = str(api_surface or "").strip().lower()
    provider = str(target.get("provider") or "").strip().lower()
    model = str(target.get("resolved_model") or "").strip()
    payload_model = payload.get("model")
    endpoint = normalize_endpoint(target.get("base_url"))
    if not provider or not model or not surface or not endpoint:
        raise ValueError("incomplete request-wire route identity")
    if not isinstance(payload_model, str) or payload_model != model:
        raise ValueError("physical payload model differs from resolved route model")
    value_paths = tuple(sorted({str(item) for item in value_fields if str(item)}))
    if scope == "value" and not value_paths:
        raise ValueError("value evidence requires at least one value field")
    values = {path: _nested_value_identity(payload, path) for path in value_paths}
    dialect = str(tool_dialect_override or infer_tool_dialect(payload))
    choice = (
        payload.get("tool_choice")
        if tool_choice_override is _TOOL_CHOICE_UNSET
        else tool_choice_override
    )
    return RequestWireProfile(
        provider=provider,
        endpoint_sha256=canonical_sha256(endpoint),
        api_surface=surface,
        model=model,
        reasoning_carrier=str(
            reasoning_carrier_override
            if reasoning_carrier_override is not None
            else reasoning_carrier(payload)
        ),
        tool_dialect=dialect,
        tool_choice=tool_choice_family(choice),
        tool_choice_sha256=_tool_choice_digest(choice),
        function_strictness=str(
            function_strictness_override
            if function_strictness_override is not None
            else function_strictness(payload)
        ),
        provider_routing_sha256=_routing_digest(payload),
        reasoning_options_sha256=_reasoning_options_digest(payload),
        contract_headers_sha256=_contract_headers_digest(target),
        evidence_scope=scope,
        value_fields=value_paths,
        value_sha256=canonical_sha256(values) if scope == "value" else "",
    )


def rebuild_request_wire_profile(
    profile: RequestWireProfile,
    target: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    tool_choice_override: Any = _TOOL_CHOICE_UNSET,
) -> RequestWireProfile:
    """Rebuild ``profile`` identity from one exact physical pre-step payload."""
    return build_request_wire_profile(
        target,
        payload,
        api_surface=profile.api_surface,
        evidence_scope=profile.evidence_scope,
        value_fields=profile.value_fields,
        tool_choice_override=tool_choice_override,
    )


def validate_wire_action(action: Any) -> Dict[str, Any]:
    """Return one normalized closed-vocabulary action, else ``{}``."""
    if not isinstance(action, Mapping):
        return {}
    kind = str(action.get("kind") or "").strip()
    reason_code = str(action.get("reason_code") or "provider_recovery_succeeded").strip()
    if reason_code not in WIRE_REASON_CODES:
        return {}
    if kind == "drop_field":
        if set(action) - {"kind", "fields", "reason_code"}:
            return {}
        raw_fields = action.get("fields")
        if not isinstance(raw_fields, (list, tuple)) or not raw_fields:
            return {}
        fields = [str(value).strip() for value in raw_fields]
        if any(not value or value not in DROP_FIELDS for value in fields):
            return {}
        return {
            "kind": kind,
            "fields": sorted(set(fields)),
            "reason_code": reason_code,
        }
    if kind == "replace_dialect":
        if set(action) - {"kind", "axis", "from", "to", "reason_code"}:
            return {}
        axis = str(action.get("axis") or "").strip()
        source = str(action.get("from") or "").strip()
        target = str(action.get("to") or "").strip()
        if (axis, source, target) not in _REGISTERED_DIALECT_TRANSITIONS:
            return {}
        return {
            "kind": kind,
            "axis": axis,
            "from": source,
            "to": target,
            "reason_code": reason_code,
        }
    if kind != "set_value" or str(action.get("field") or "") != "effort":
        return {}
    if set(action) - {"kind", "field", "mode", "from", "to", "reason_code"}:
        return {}
    from ouroboros.config import EFFORT_SCALE, effort_rank

    mode = str(action.get("mode") or "").strip()
    source = str(action.get("from") or "").strip().lower()
    target = str(action.get("to") or "").strip().lower()
    if mode not in {"exact", "floor", "ceiling"}:
        return {}
    if source not in EFFORT_SCALE or target not in EFFORT_SCALE or source == target:
        return {}
    if mode == "floor" and effort_rank(target) <= effort_rank(source):
        return {}
    if mode == "ceiling" and effort_rank(target) >= effort_rank(source):
        return {}
    return {
        "kind": kind,
        "field": "effort",
        "mode": mode,
        "from": source,
        "to": target,
        "reason_code": reason_code,
    }


def validate_wire_action_for_profile(
    profile: RequestWireProfile,
    action: Any,
) -> Dict[str, Any]:
    """Validate one action against the exact source request shape."""
    normalized = validate_wire_action(action)
    if not normalized:
        return {}
    kind = normalized["kind"]
    if kind == "replace_dialect":
        return normalized if normalized["from"] == profile.tool_dialect else {}
    if kind == "set_value":
        return normalized if profile.reasoning_carrier else {}
    fields = set(normalized["fields"])
    carrier = profile.reasoning_carrier
    carrier_requirements = {
        "reasoning_effort": "reasoning_effort",
        NESTED_REASONING_FIELD: NESTED_REASONING_FIELD,
    }
    for field_name, required_carrier in carrier_requirements.items():
        if field_name in fields and carrier != required_carrier:
            return {}
    if fields.intersection(ANTHROPIC_REASONING_FIELDS):
        if carrier == "anthropic.adaptive":
            fields.update(ANTHROPIC_REASONING_FIELDS)
        elif carrier == "anthropic.disabled" and fields == {"thinking"}:
            pass
        elif carrier.startswith("anthropic.") and fields == {"thinking"}:
            pass
        else:
            return {}
    normalized["fields"] = sorted(fields)
    return normalized


def validate_durable_wire_action_for_profile(
    profile: RequestWireProfile,
    action: Any,
) -> Dict[str, Any]:
    """Validate an action that may become future dispatch authority.

    Explicit ``none`` and the task-local availability reason are deliberately
    accepted by the generic validator for the current conversation, but they
    must never survive a restart or become the next request's first choice.
    """
    normalized = validate_wire_action_for_profile(profile, action)
    if not normalized:
        return {}
    if normalized.get("reason_code") == "task_local_availability_fallback":
        return {}
    if normalized.get("kind") == "set_value" and normalized.get("to") == "none":
        return {}
    return normalized


def _freeze_wire_action(action: Mapping[str, Any]) -> Mapping[str, Any]:
    """Take recursive immutable custody of one normalized authority action."""

    def _freeze(value: Any) -> Any:
        if isinstance(value, Mapping):
            return MappingProxyType({key: _freeze(item) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(_freeze(item) for item in value)
        return value

    return _freeze(action)


def _wire_action_dict(action: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a detached JSON-shaped copy at a non-authority boundary."""

    def _thaw(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: _thaw(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_thaw(item) for item in value]
        return value

    return _thaw(action)


def apply_effort_action(requested: str, action: Mapping[str, Any]) -> str:
    """Apply one evidence-bounded effort change."""
    normalized = validate_wire_action(action)
    current = str(requested or "").strip().lower()
    if normalized.get("kind") != "set_value" or not current:
        return current
    from ouroboros.config import effort_rank

    source = normalized["from"]
    target = normalized["to"]
    mode = normalized["mode"]
    if mode == "exact":
        return target if current == source else current
    current_rank = effort_rank(current)
    target_rank = effort_rank(target)
    if mode == "ceiling" and current_rank > target_rank:
        return target
    if mode == "floor" and 0 <= current_rank < target_rank:
        return target
    return current


@dataclass(frozen=True)
class PendingWireAction:
    """One cognition-preserving action eligible for success-confirmed learning."""

    profile: RequestWireProfile
    action: Mapping[str, Any]

    def __post_init__(self) -> None:
        normalized = validate_durable_wire_action_for_profile(self.profile, self.action)
        if not normalized:
            raise ValueError("invalid request-wire action for profile")
        object.__setattr__(self, "action", _freeze_wire_action(normalized))


@dataclass(frozen=True)
class EphemeralWireAdjustment:
    """Task-local adjustment that can never be passed to the durable writer."""

    profile: RequestWireProfile
    action: Mapping[str, Any]

    def __post_init__(self) -> None:
        normalized = validate_wire_action_for_profile(self.profile, self.action)
        if not normalized:
            raise ValueError("invalid task-local request-wire adjustment")
        object.__setattr__(self, "action", _freeze_wire_action(normalized))


@dataclass(frozen=True)
class WireStoreCommitResult:
    committed: bool
    error_code: str = ""
    actions_by_profile: Mapping[str, Tuple[Dict[str, Any], ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredWireAction:
    action: Mapping[str, Any]
    observed_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _freeze_wire_action(self.action))


def _state_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / REQUEST_WIRE_STATE_FILE


def canonical_wire_evidence_root() -> pathlib.Path:
    """Resolve the one host-level evidence root, never a per-task child drive."""
    from ouroboros.usage_ledger import _drive_root

    return _drive_root(None)


def _age_seconds(timestamp: Any, *, now: Any = None) -> float:
    parsed = parse_deadline_ts(str(timestamp or ""))
    if parsed is None:
        return float("inf")
    current = now if now is not None else utc_now()
    age = (current - parsed).total_seconds()
    if age < -REQUEST_WIRE_MAX_CLOCK_SKEW_SEC:
        return float("inf")
    return max(0.0, age)


def wire_action_identity(action: Mapping[str, Any]) -> str:
    """Return the one storage/resolution identity for a normalized action.

    A newer observation for the same effort source supersedes an older one;
    keeping contradictory historical targets would manufacture a conflict
    after provider drift instead of learning the latest successful contract.
    """
    if action.get("kind") == "set_value":
        return f"set:{action.get('field')}:{action.get('from')}"
    if action.get("kind") == "drop_field":
        return "drop:" + ",".join(action.get("fields") or [])
    return f"dialect:{action.get('axis')}:{action.get('from')}"


def _fresh_records(
    entry: Any,
    profile: RequestWireProfile,
    *,
    now: Any = None,
) -> list[StoredWireAction]:
    if not isinstance(entry, Mapping) or entry.get("profile") != profile.as_dict():
        return []
    records = entry.get("records")
    if not isinstance(records, list):
        return []
    fresh = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        action = validate_durable_wire_action_for_profile(profile, record.get("action"))
        observed_at = str(record.get("observed_at") or "")
        if not action or _age_seconds(observed_at, now=now) >= REQUEST_WIRE_TTL_SEC:
            continue
        fresh.append(StoredWireAction(action=action, observed_at=observed_at))
    return fresh


def _read_wire_action_records_at(
    drive_root: Any,
    profile: RequestWireProfile,
    *,
    now: Any = None,
) -> Tuple[StoredWireAction, ...]:
    """Private explicit-root seam for hermetic tests."""
    try:
        data = read_json_dict(_state_path(drive_root)) or {}
        schema_version = data.get("schema_version")
        if (
            type(schema_version) is not int
            or schema_version != REQUEST_WIRE_SCHEMA_VERSION
        ):
            return ()
        entries = data.get("profiles")
        entry = entries.get(profile.fingerprint) if isinstance(entries, Mapping) else None
        return tuple(_fresh_records(entry, profile, now=now))
    except Exception:
        return ()


def read_wire_action_records(
    profile: RequestWireProfile,
    *,
    now: Any = None,
) -> Tuple[StoredWireAction, ...]:
    return _read_wire_action_records_at(canonical_wire_evidence_root(), profile, now=now)


def _read_wire_actions_at(
    drive_root: Any,
    profile: RequestWireProfile,
    *,
    now: Any = None,
) -> Tuple[Dict[str, Any], ...]:
    return tuple(_wire_action_dict(record.action) for record in _read_wire_action_records_at(
        drive_root, profile, now=now
    ))


def read_wire_actions(
    profile: RequestWireProfile,
    *,
    now: Any = None,
) -> Tuple[Dict[str, Any], ...]:
    return tuple(_wire_action_dict(record.action) for record in read_wire_action_records(
        profile, now=now
    ))


def _commit_wire_compatibility_at(
    drive_root: Any,
    receipt: "WireCompatibilityReceipt",
) -> WireStoreCommitResult:
    """Private explicit-root commit seam used by the production canonical wrapper and tests."""
    from ouroboros.request_wire_receipts import WireCompatibilityReceipt

    if not isinstance(receipt, WireCompatibilityReceipt):
        return WireStoreCommitResult(False, "invalid_receipt")
    grouped: Dict[str, Tuple[RequestWireProfile, list[Dict[str, Any]]]] = {}
    for item in receipt.pending:
        action = validate_durable_wire_action_for_profile(item.profile, item.action)
        if not action:
            return WireStoreCommitResult(False, "invalid_pending_action")
        current = grouped.get(item.profile.fingerprint)
        if current is None:
            grouped[item.profile.fingerprint] = (item.profile, [action])
        else:
            current[1].append(action)
    if not grouped:
        return WireStoreCommitResult(True)
    result: Dict[str, Tuple[Dict[str, Any], ...]] = {}

    def _mutate(current: Dict[str, Any]) -> Dict[str, Any]:
        if current:
            schema_version = current.get("schema_version")
            if (
                type(schema_version) is not int
                or schema_version != REQUEST_WIRE_SCHEMA_VERSION
            ):
                raise ValueError("unsupported request-wire store schema")
            if not isinstance(current.get("profiles"), dict):
                raise ValueError("malformed request-wire store profiles")
            data = dict(current)
            entries = dict(current["profiles"])
            data["profiles"] = entries
        else:
            data = {"schema_version": REQUEST_WIRE_SCHEMA_VERSION, "profiles": {}}
            entries = data["profiles"]
        for fingerprint, (profile, actions) in grouped.items():
            previous = entries.get(fingerprint)
            fresh = _fresh_records(previous, profile)
            for action in actions:
                identity = wire_action_identity(action)
                fresh = [
                    record for record in fresh
                    if wire_action_identity(record.action) != identity
                ]
                fresh.append(StoredWireAction(action=action, observed_at=receipt.observed_at))
            entries[fingerprint] = {
                "profile": profile.as_dict(),
                "records": [
                    {
                        "action": _wire_action_dict(record.action),
                        "observed_at": record.observed_at,
                    }
                    for record in fresh
                ],
                "observed_at": receipt.observed_at,
            }
            result[fingerprint] = tuple(
                _wire_action_dict(record.action) for record in fresh
            )
        return data

    try:
        update_json_locked(
            _state_path(drive_root),
            _mutate,
            strict_existing_dict=True,
            reject_existing_empty_dict=True,
        )
        return WireStoreCommitResult(True, actions_by_profile=result)
    except TimeoutError:
        log.debug("request-wire evidence lock timed out", exc_info=True)
        return WireStoreCommitResult(False, "lock_timeout")
    except ValueError:
        log.debug("request-wire evidence state rejected", exc_info=True)
        return WireStoreCommitResult(False, "incompatible_state")
    except Exception:
        log.debug("request-wire evidence write failed", exc_info=True)
        return WireStoreCommitResult(False, "write_failed")


def commit_wire_compatibility(receipt: "WireCompatibilityReceipt") -> WireStoreCommitResult:
    """Commit terminal success evidence to the one canonical host store."""
    return _commit_wire_compatibility_at(canonical_wire_evidence_root(), receipt)
