"""Strict reviewed presence declarations carried by ordinary skill manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional

from ouroboros.contracts.skill_manifest import SkillManifest, SkillManifestError
from ouroboros.contracts.skill_payload_policy import is_skill_payload_control_filename
from tests.frozen_presence_runtime import PresenceRuntimeDefaults, PresenceRuntimeError

PRESENCE_PROFILE_SCHEMA_VERSION = 1
PRESENCE_CAPABILITY_KINDS = frozenset({"tool", "script", "resource"})

_PROFILE_FIELDS = frozenset(
    {
        "schema_version",
        "instructions",
        "instructions_file",
        "context_topics",
        "runtime_defaults",
        "capability_requests",
    }
)
_RUNTIME_FIELDS = frozenset({"model_slot", "inline_max_rounds", "execution"})
_REQUEST_FIELDS = frozenset(
    {
        "id",
        "kind",
        "purpose",
        "required",
        "operations",
    }
)
_MISSING = object()


class PresenceProfileError(SkillManifestError):
    """A typed, field-addressable refusal for a malformed presence profile."""

    def __init__(self, code: str, field: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "presence_invalid")
        self.field = str(field or "presence")


@dataclass(frozen=True)
class PresenceCapabilityRequest:
    """One portable capability requested by a reviewed behavior skill."""

    request_id: str
    kind: str
    required: bool
    operations: tuple[str, ...] = ()
    purpose: str = ""


@dataclass(frozen=True)
class PresenceProfile:
    """Normalized, review-bound behavior profile for one ordinary skill."""

    schema_version: int
    instructions: str
    instructions_file: Optional[str]
    context_topics: tuple[str, ...]
    runtime_defaults: PresenceRuntimeDefaults
    capability_requests: tuple[PresenceCapabilityRequest, ...]


def _reject_unknown_fields(
    value: Mapping[Any, Any],
    allowed: frozenset[str],
    field: str,
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise PresenceProfileError(
            "presence_unknown_field",
            field,
            f"{field} contains unknown field(s): {', '.join(unknown)}",
        )


def _parse_instructions(
    raw: Mapping[str, Any],
    skill_dir: Path,
    manifest: SkillManifest,
) -> tuple[str, Optional[str]]:
    has_inline = "instructions" in raw
    has_file = "instructions_file" in raw
    if has_inline and has_file:
        raise PresenceProfileError(
            "presence_instructions_conflict",
            "presence.instructions",
            "presence may declare instructions or instructions_file, not both",
        )
    if has_inline:
        value = raw["instructions"]
        if not isinstance(value, str) or not value.strip():
            raise PresenceProfileError(
                "presence_invalid_instructions",
                "presence.instructions",
                "presence.instructions must be a non-empty string",
            )
        return value, None
    if not has_file:
        return "", None

    raw_path = raw["instructions_file"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PresenceProfileError(
            "presence_invalid_instructions_file",
            "presence.instructions_file",
            "presence.instructions_file must be a non-empty relative path",
        )
    path_text = raw_path.strip()
    rel = PurePosixPath(path_text)
    if (
        ":" in path_text
        or "\\" in path_text
        or rel.is_absolute()
        or path_text.startswith("~")
        or any(part in {"", ".", ".."} for part in rel.parts)
    ):
        raise PresenceProfileError(
            "presence_instructions_file_escape",
            "presence.instructions_file",
            "presence.instructions_file must stay inside the reviewed skill payload",
        )
    normalized = PurePosixPath(*rel.parts).as_posix()
    if is_skill_payload_control_filename(normalized):
        raise PresenceProfileError(
            "presence_instructions_file_control_plane",
            "presence.instructions_file",
            "presence.instructions_file must not use a skill lifecycle control file",
        )
    try:
        root = skill_dir.resolve(strict=True)
        target = (root / normalized).resolve(strict=True)
        target.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PresenceProfileError(
            "presence_instructions_file_unreadable",
            "presence.instructions_file",
            "presence.instructions_file must resolve to a readable file inside the reviewed skill payload",
        ) from exc
    if not root.is_dir() or not target.is_file():
        raise PresenceProfileError(
            "presence_instructions_file_unreadable",
            "presence.instructions_file",
            "presence.instructions_file must resolve to a regular file inside the reviewed skill payload",
        )
    try:
        from ouroboros.skill_loader import _iter_payload_files

        reviewed_files = {
            path.resolve()
            for path in _iter_payload_files(
                root,
                manifest_entry=manifest.entry,
                manifest_scripts=manifest.scripts,
            )
        }
    except Exception as exc:
        raise PresenceProfileError(
            "presence_instructions_review_surface_unavailable",
            "presence.instructions_file",
            "presence.instructions_file review coverage could not be established",
        ) from exc
    if target not in reviewed_files:
        raise PresenceProfileError(
            "presence_instructions_file_unreviewed",
            "presence.instructions_file",
            "presence.instructions_file must belong to the reviewed skill payload surface",
        )
    try:
        instructions = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PresenceProfileError(
            "presence_instructions_file_encoding",
            "presence.instructions_file",
            "presence.instructions_file must contain valid UTF-8 text",
        ) from exc
    except OSError as exc:
        raise PresenceProfileError(
            "presence_instructions_file_unreadable",
            "presence.instructions_file",
            "presence.instructions_file could not be read",
        ) from exc
    if not instructions.strip():
        raise PresenceProfileError(
            "presence_invalid_instructions",
            "presence.instructions_file",
            "presence.instructions_file must contain non-empty instructions",
        )
    return instructions, normalized


def _parse_context_topics(value: Any) -> tuple[str, ...]:
    if value is _MISSING:
        return ()
    if not isinstance(value, list):
        raise PresenceProfileError(
            "presence_invalid_context_topics",
            "presence.context_topics",
            "presence.context_topics must be a list",
        )
    topics: list[str] = []
    for index, item in enumerate(value):
        field = f"presence.context_topics[{index}]"
        if not isinstance(item, str) or not item.strip():
            raise PresenceProfileError(
                "presence_invalid_context_topic",
                field,
                f"{field} must be a non-empty string",
            )
        topic = item.strip()
        if topic in topics:
            raise PresenceProfileError(
                "presence_duplicate_context_topic",
                field,
                f"{field} duplicates context topic {topic!r}",
            )
        topics.append(topic)
    return tuple(topics)


def _parse_runtime_defaults(value: Any) -> PresenceRuntimeDefaults:
    if value is _MISSING:
        return PresenceRuntimeDefaults()
    if not isinstance(value, Mapping):
        raise PresenceProfileError(
            "presence_invalid_runtime_defaults",
            "presence.runtime_defaults",
            "presence.runtime_defaults must be a mapping",
        )
    _reject_unknown_fields(value, _RUNTIME_FIELDS, "presence.runtime_defaults")
    kwargs: dict[str, Any] = {}
    for key in _RUNTIME_FIELDS:
        if key in value:
            kwargs[key] = value[key]
    try:
        return PresenceRuntimeDefaults(**kwargs)
    except PresenceRuntimeError as exc:
        field = str(exc.field or "runtime_defaults")
        if not field.startswith("presence."):
            field = f"presence.runtime_defaults.{field}"
        raise PresenceProfileError(exc.code, field, str(exc)) from exc


def _parse_capability_requests(value: Any) -> tuple[PresenceCapabilityRequest, ...]:
    if value is _MISSING:
        return ()
    if not isinstance(value, list):
        raise PresenceProfileError(
            "presence_invalid_capability_requests",
            "presence.capability_requests",
            "presence.capability_requests must be a list",
        )
    requests: list[PresenceCapabilityRequest] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        field = f"presence.capability_requests[{index}]"
        if not isinstance(item, Mapping):
            raise PresenceProfileError(
                "presence_invalid_capability_request",
                field,
                f"{field} must be a mapping",
            )
        _reject_unknown_fields(item, _REQUEST_FIELDS, field)
        request_id = item.get("id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise PresenceProfileError(
                "presence_invalid_capability_id",
                f"{field}.id",
                f"{field}.id must be a non-empty string",
            )
        request_id = request_id.strip()
        if request_id in seen_ids:
            raise PresenceProfileError(
                "presence_duplicate_capability_id",
                f"{field}.id",
                f"{field}.id duplicates capability request {request_id!r}",
            )
        kind = item.get("kind")
        if not isinstance(kind, str) or kind.strip().lower() not in PRESENCE_CAPABILITY_KINDS:
            raise PresenceProfileError(
                "presence_invalid_capability_kind",
                f"{field}.kind",
                f"{field}.kind must be one of {sorted(PRESENCE_CAPABILITY_KINDS)}",
            )
        kind = kind.strip().lower()
        if "required" not in item or not isinstance(item["required"], bool):
            raise PresenceProfileError(
                "presence_invalid_capability_required",
                f"{field}.required",
                f"{field}.required must be a boolean",
            )
        operations: tuple[str, ...] = ()
        if kind == "resource":
            raw_operations = item.get("operations")
            if not isinstance(raw_operations, list) or not raw_operations:
                raise PresenceProfileError(
                    "presence_invalid_resource_operations",
                    f"{field}.operations",
                    f"{field}.operations must be a non-empty list for resource requests",
                )
            normalized: set[str] = set()
            for operation in raw_operations:
                if not isinstance(operation, str) or not operation.strip():
                    raise PresenceProfileError(
                        "presence_invalid_resource_operation",
                        f"{field}.operations",
                        f"{field}.operations entries must be non-empty strings",
                    )
                normalized.add(operation.strip().lower())
            operations = tuple(sorted(normalized))
        elif "operations" in item:
            raise PresenceProfileError(
                "presence_unexpected_operations",
                f"{field}.operations",
                f"{field}.operations is allowed only for resource requests",
            )
        raw_purpose = item.get("purpose", "")
        if not isinstance(raw_purpose, str):
            raise PresenceProfileError(
                "presence_invalid_capability_purpose",
                f"{field}.purpose",
                f"{field}.purpose must be a string when provided",
            )
        requests.append(
            PresenceCapabilityRequest(
                request_id=request_id,
                kind=kind,
                required=item["required"],
                operations=operations,
                purpose=raw_purpose.strip(),
            )
        )
        seen_ids.add(request_id)
    return tuple(requests)


def _canonical_sha256(value: Any, *, domain: bytes) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def presence_request_fingerprint(request: PresenceCapabilityRequest) -> str:
    """Hash only the authority semantics that keep a local selection valid."""
    return _canonical_sha256(
        {
            "id": request.request_id,
            "kind": request.kind,
            "operations": list(request.operations),
        },
        domain=b"presence-request-v1",
    )


def presence_profile_fingerprint(profile: PresenceProfile) -> str:
    """Hash the complete normalized, reviewed behavior profile."""
    return _canonical_sha256(
        {
            "schema_version": profile.schema_version,
            "instructions": profile.instructions,
            "instructions_file": profile.instructions_file,
            "context_topics": list(profile.context_topics),
            "runtime_defaults": {
                "model_slot": profile.runtime_defaults.model_slot,
                "inline_max_rounds": profile.runtime_defaults.inline_max_rounds,
                **({"execution": profile.runtime_defaults.execution} if profile.runtime_defaults.execution != "event" else {}),
            },
            "capability_requests": [
                {
                    "id": request.request_id,
                    "kind": request.kind,
                    "required": request.required,
                    "operations": list(request.operations),
                    "purpose": request.purpose,
                }
                for request in profile.capability_requests
            ],
        },
        domain=b"presence-profile-v1",
    )


def parse_presence_profile(
    manifest: SkillManifest,
    skill_dir: Path,
) -> Optional[PresenceProfile]:
    """Parse the singular ``raw_extra['presence']`` declaration, if present."""
    extras = manifest.raw_extra
    if not isinstance(extras, Mapping) or "presence" not in extras:
        return None
    raw = extras["presence"]
    if not isinstance(raw, Mapping):
        raise PresenceProfileError(
            "presence_invalid",
            "presence",
            "presence must be a mapping",
        )
    if manifest.type not in {"instruction", "extension"}:
        raise PresenceProfileError(
            "presence_unsupported_skill_type",
            "presence",
            "presence is supported only for instruction and extension skills",
        )
    _reject_unknown_fields(raw, _PROFILE_FIELDS, "presence")
    schema_version = raw.get("schema_version", PRESENCE_PROFILE_SCHEMA_VERSION)
    if type(schema_version) is not int or schema_version != PRESENCE_PROFILE_SCHEMA_VERSION:
        raise PresenceProfileError(
            "presence_unsupported_schema_version",
            "presence.schema_version",
            f"presence.schema_version must be exactly {PRESENCE_PROFILE_SCHEMA_VERSION}",
        )
    instructions, instructions_file = _parse_instructions(raw, Path(skill_dir), manifest)
    return PresenceProfile(
        schema_version=schema_version,
        instructions=instructions,
        instructions_file=instructions_file,
        context_topics=_parse_context_topics(raw.get("context_topics", _MISSING)),
        runtime_defaults=_parse_runtime_defaults(raw.get("runtime_defaults", _MISSING)),
        capability_requests=_parse_capability_requests(raw.get("capability_requests", _MISSING)),
    )


__all__ = [
    "PRESENCE_CAPABILITY_KINDS",
    "PRESENCE_PROFILE_SCHEMA_VERSION",
    "PresenceCapabilityRequest",
    "PresenceProfile",
    "PresenceProfileError",
    "parse_presence_profile",
    "presence_profile_fingerprint",
    "presence_request_fingerprint",
]
