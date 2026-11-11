"""Parser-issued validation receipts for exact OpenAI Chat custom calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping

from ouroboros.request_wire_contract import _valid_sha256, canonical_sha256

if TYPE_CHECKING:
    from ouroboros.request_wire_receipts import WireCandidateManifest


CUSTOM_ARGUMENT_ERROR_CODES = frozenset({
    "arguments_not_object",
    "invalid_json",
    "invalid_schema",
    "schema_validation_failed",
    "unknown_tool",
})
_CUSTOM_ARGUMENT_BINDING_TOKEN = object()


@dataclass(frozen=True)
class CustomArgumentValidationReceipt:
    """Outcome derived by the parser from one candidate-owned schema binding."""

    candidate_sha256: str
    catalog_sha256: str
    tool_call_id: str
    tool_name: str
    schema_sha256: str
    arguments_sha256: str
    decoded_object: bool
    valid: bool
    error_code: str = ""
    _binding_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._binding_token is not _CUSTOM_ARGUMENT_BINDING_TOKEN:
            raise ValueError("custom validation receipt must be parser-issued")
        for digest in (
            self.candidate_sha256,
            self.catalog_sha256,
            self.schema_sha256,
            self.arguments_sha256,
        ):
            if not _valid_sha256(digest):
                raise ValueError("malformed custom validation digest")
        if not self.tool_call_id or not self.tool_name:
            raise ValueError("custom validation requires tool-call identity")
        if self.valid and (not self.decoded_object or self.error_code):
            raise ValueError("valid custom arguments cannot carry an error")
        if not self.valid and self.error_code not in CUSTOM_ARGUMENT_ERROR_CODES:
            raise ValueError("invalid custom arguments require a typed error")
        object.__setattr__(self, "_binding_token", None)

    @property
    def allows_execution(self) -> bool:
        return self.valid

    @property
    def proves_wire_acceptance(self) -> bool:
        return self.decoded_object and self.error_code not in {
            "arguments_not_object",
            "invalid_json",
            "unknown_tool",
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "catalog_sha256": self.catalog_sha256,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "schema_sha256": self.schema_sha256,
            "arguments_sha256": self.arguments_sha256,
            "decoded_object": self.decoded_object,
            "valid": self.valid,
            "error_code": self.error_code,
        }


def _reject_non_json_constant(value: str) -> Any:
    raise ValueError(f"non-JSON numeric constant: {value}")


def _schema_validation_error(
    schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> str:
    """Contain all jsonschema construction/resolution failures at its boundary."""
    from jsonschema import validators
    from referencing import Registry

    try:
        validator_class = validators.validator_for(schema)
        validator_class.check_schema(schema)
        # Resolve in-document/bundled resources, never fetch server-supplied refs.
        validator = validator_class(schema, registry=Registry())
        error = next(iter(validator.iter_errors(arguments)), None)
    except Exception:
        # jsonschema has no public common base for schema, regex, and reference
        # failures. Keep the broad catch around third-party schema work only.
        return "invalid_schema"
    return "" if error is None else "schema_validation_failed"


def issue_custom_argument_validation_receipt(
    *,
    candidate: "WireCandidateManifest",
    tool_call_id: str,
    tool_name: str,
    raw_input: str,
) -> CustomArgumentValidationReceipt:
    """Validate one raw call against its exact candidate-owned catalog binding."""
    from ouroboros.request_wire_receipts import WireCandidateManifest

    if not isinstance(candidate, WireCandidateManifest):
        raise ValueError("custom validation requires a factory-issued wire candidate")
    profile = candidate.accepted_profile
    catalog = candidate.custom_catalog
    if (
        profile.provider != "openai"
        or profile.api_surface != "chat.completions"
        or profile.tool_dialect != "openai_chat_custom"
        or catalog is None
    ):
        raise ValueError("custom validation requires an OpenAI Chat custom candidate")
    if (
        not isinstance(tool_call_id, str)
        or not tool_call_id
        or tool_call_id != tool_call_id.strip()
        or not isinstance(tool_name, str)
        or not tool_name
        or tool_name != tool_name.strip()
        or not isinstance(raw_input, str)
    ):
        raise ValueError("custom validation requires one exact raw call")

    binding = catalog.schema_binding(tool_name)
    schema_sha256 = binding.schema_sha256 if binding is not None else canonical_sha256(None)
    arguments_sha256 = canonical_sha256(raw_input)
    decoded_object = False
    valid = False
    error_code = ""
    try:
        arguments = json.loads(raw_input, parse_constant=_reject_non_json_constant)
    except (TypeError, ValueError, json.JSONDecodeError):
        error_code = "invalid_json"
    else:
        decoded_object = isinstance(arguments, dict)
        if not decoded_object:
            error_code = "arguments_not_object"
        elif binding is None:
            error_code = "unknown_tool"
        else:
            error_code = _schema_validation_error(binding.schema(), arguments)
            valid = not error_code
    return CustomArgumentValidationReceipt(
        candidate_sha256=candidate.candidate_sha256,
        catalog_sha256=catalog.catalog_sha256,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        schema_sha256=schema_sha256,
        arguments_sha256=arguments_sha256,
        decoded_object=decoded_object,
        valid=valid,
        error_code=error_code,
        _binding_token=_CUSTOM_ARGUMENT_BINDING_TOKEN,
    )
