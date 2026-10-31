"""Frozen Presence authority types for old task decoding; no admission compiler."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from ouroboros.contracts.skill_manifest import canonical_skill_name

_TOOL_KINDS = frozenset({"builtin", "extension", "mcp"})
_BINDING_SOURCES = frozenset({"static", "conversation", "actor", "message", "destination", "resource"})
_CONTEXT_BINDING_SOURCES = frozenset({"conversation", "actor", "message", "destination"})
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")



class PresenceStateError(ValueError):
    """Typed, safe refusal for malformed or concurrently changed state."""

    def __init__(self, code: str, field: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "presence_state_invalid")
        self.field = str(field or "presence_state")


def _error(code: str, field: str, message: str = "") -> PresenceStateError:
    return PresenceStateError(code, field, message or f"{code}: {field}")


def _require_exact_string(value: Any, *, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _error("presence_state_invalid_string", field_name, f"{field_name} must be a string")
    if value != value.strip() or (not allow_empty and not value):
        qualifier = "a normalized string" if allow_empty else "a non-empty normalized string"
        raise _error("presence_state_invalid_string", field_name, f"{field_name} must be {qualifier}")
    if "\x00" in value:
        raise _error("presence_state_invalid_string", field_name, f"{field_name} must not contain NUL")
    return value


def _require_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not _HEX_SHA256_RE.fullmatch(value):
        raise _error("presence_state_invalid_fingerprint", field_name)
    return value


def _normalize_operations(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise _error("presence_state_invalid_operations", field_name)
    normalized: set[str] = set()
    for operation in value:
        if not isinstance(operation, str) or not operation.strip():
            raise _error("presence_state_invalid_operation", field_name)
        normalized.add(operation.strip().lower())
    return tuple(sorted(normalized))


def _normalize_path_segments(value: Any, *, field_name: str, required: bool) -> tuple[str | int, ...]:
    if not isinstance(value, tuple) or (required and not value):
        raise _error("presence_state_invalid_binding_path", field_name)
    result: list[str | int] = []
    for segment in value:
        if isinstance(segment, bool) or not isinstance(segment, (str, int)):
            raise _error("presence_state_invalid_binding_path", field_name)
        if isinstance(segment, int):
            if segment < 0:
                raise _error("presence_state_invalid_binding_path", field_name)
        elif not segment or segment != segment.strip() or "\x00" in segment:
            raise _error("presence_state_invalid_binding_path", field_name)
        result.append(segment)
    return tuple(result)


def _freeze_json_value(value: Any, *, field_name: str) -> Any:
    """Validate finite JSON and recursively freeze caller-owned containers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error("presence_state_invalid_static_value", field_name)
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value))
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _error("presence_state_invalid_static_value", field_name)
            copied[key] = _freeze_json_value(item, field_name=f"{field_name}.{key}")
        return MappingProxyType(copied)
    raise _error("presence_state_invalid_static_value", field_name)


def _json_value_payload(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value_payload(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _json_value_payload(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class PresenceToolTarget:
    """One exact built-in, extension, or MCP tool target."""

    kind: str
    name: str
    provider: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _TOOL_KINDS:
            raise _error(
                "presence_state_invalid_tool_kind",
                "target.kind",
                f"target.kind must be one of {sorted(_TOOL_KINDS)}",
            )
        _require_exact_string(self.name, field_name="target.name")
        _require_exact_string(self.provider, field_name="target.provider", allow_empty=True)
        if self.kind == "builtin" and self.provider:
            raise _error(
                "presence_state_invalid_tool_provider",
                "target.provider",
                "built-in tool targets must not declare an external provider",
            )
        if self.kind != "builtin" and not self.provider:
            raise _error(
                "presence_state_missing_tool_provider",
                "target.provider",
                "extension and MCP tool targets require an exact provider identity",
            )


@dataclass(frozen=True)
class PresenceScriptTarget:
    """One exact script entry in one reviewed skill."""

    skill: str
    script: str

    def __post_init__(self) -> None:
        _require_exact_string(self.skill, field_name="target.skill")
        if canonical_skill_name(self.skill) != self.skill:
            raise _error(
                "presence_state_invalid_skill_name",
                "target.skill",
                "target.skill must be a canonical skill name",
            )
        script = _require_exact_string(self.script, field_name="target.script")
        if ":" in script or "\\" in script:
            raise _error(
                "presence_state_invalid_script_path",
                "target.script",
                "target.script must use POSIX separators",
            )
        pure = PurePosixPath(script)
        if (
            script == "."
            or pure.is_absolute()
            or script.startswith("~")
            or ".." in pure.parts
            or pure.as_posix() != script
        ):
            raise _error(
                "presence_state_invalid_script_path",
                "target.script",
                "target.script must be a normalized relative path",
            )


@dataclass(frozen=True)
class PresenceResourceTarget:
    """One logical root with a selected operation subset and exact location."""

    root: str
    operations: tuple[str, ...]
    path_prefix: str = "."
    bucket: str = ""
    skill_name: str = ""

    def __post_init__(self) -> None:
        _require_exact_string(self.root, field_name="target.root")
        object.__setattr__(
            self,
            "operations",
            _normalize_operations(self.operations, field_name="target.operations"),
        )
        path_prefix = _require_exact_string(
            self.path_prefix,
            field_name="target.path_prefix",
        )
        if ":" in path_prefix or "\\" in path_prefix:
            raise _error(
                "presence_state_invalid_path_prefix",
                "target.path_prefix",
                "target.path_prefix must use POSIX separators",
            )
        pure = PurePosixPath(path_prefix)
        if pure.is_absolute() or path_prefix.startswith("~") or ".." in pure.parts:
            raise _error(
                "presence_state_invalid_path_prefix",
                "target.path_prefix",
                "target.path_prefix must stay relative to its logical root",
            )
        if pure.as_posix() != path_prefix:
            raise _error(
                "presence_state_invalid_path_prefix",
                "target.path_prefix",
                "target.path_prefix must be normalized",
            )
        _require_exact_string(self.bucket, field_name="target.bucket", allow_empty=True)
        _require_exact_string(self.skill_name, field_name="target.skill_name", allow_empty=True)
        if bool(self.bucket) != bool(self.skill_name):
            raise _error(
                "presence_state_incomplete_resource_location",
                "target.bucket",
                "target.bucket and target.skill_name must be declared together",
            )
        if self.skill_name and canonical_skill_name(self.skill_name) != self.skill_name:
            raise _error(
                "presence_state_invalid_skill_name",
                "target.skill_name",
                "target.skill_name must be a canonical skill name",
            )


PresenceTarget: TypeAlias = PresenceToolTarget | PresenceScriptTarget | PresenceResourceTarget


@dataclass(frozen=True)
class PresenceArgumentBinding:
    """Bind one concrete argument path to an authenticated structural source."""

    argument_path: tuple[str | int, ...]
    source: str
    source_path: tuple[str | int, ...] = ()
    static_value: Any = None
    resource_request_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "argument_path",
            _normalize_path_segments(
                self.argument_path,
                field_name="binding.argument_path",
                required=True,
            ),
        )
        if not isinstance(self.source, str) or self.source not in _BINDING_SOURCES:
            raise _error(
                "presence_state_invalid_binding_source",
                "binding.source",
                f"binding.source must be one of {sorted(_BINDING_SOURCES)}",
            )
        _require_exact_string(
            self.resource_request_fingerprint,
            field_name="binding.resource_request_fingerprint",
            allow_empty=True,
        )
        object.__setattr__(
            self,
            "source_path",
            _normalize_path_segments(
                self.source_path,
                field_name="binding.source_path",
                required=False,
            ),
        )
        if self.source in _CONTEXT_BINDING_SOURCES:
            if self.resource_request_fingerprint:
                raise _error(
                    "presence_state_unexpected_resource_reference",
                    "binding.resource_request_fingerprint",
                    "context bindings must not name a resource request",
                )
            if self.static_value is not None:
                raise _error(
                    "presence_state_unexpected_static_value",
                    "binding.static_value",
                    "context bindings must not declare a static value",
                )
            return
        if self.source_path:
            raise _error(
                "presence_state_unexpected_source_path",
                "binding.source_path",
                "source_path is available only for authenticated context sources",
            )
        if self.source == "static":
            if self.resource_request_fingerprint:
                raise _error(
                    "presence_state_unexpected_resource_reference",
                    "binding.resource_request_fingerprint",
                    "static bindings must not name a resource request",
                )
            value = _freeze_json_value(self.static_value, field_name="binding.static_value")
            object.__setattr__(self, "static_value", value)
            return
        if self.static_value is not None:
            raise _error(
                "presence_state_unexpected_static_value",
                "binding.static_value",
                "resource bindings must not declare a static value",
            )
        _require_sha256(
            self.resource_request_fingerprint,
            field_name="binding.resource_request_fingerprint",
        )
