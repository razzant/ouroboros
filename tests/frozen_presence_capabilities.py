"""Historical profile-state fixtures; never imported by production admission."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from ouroboros.contracts.skill_manifest import canonical_skill_name
from ouroboros.contracts.task_contract import canonical_tool_name
from ouroboros.contracts.skill_payload_policy import PRESENCE_PROFILE_STATE_FILENAME
from tests.frozen_presence_profile import (
    PresenceCapabilityRequest,
    PresenceProfile,
    presence_profile_fingerprint,
    presence_request_fingerprint,
)
from tests.frozen_presence_runtime import (
    PresenceRuntimeOverrides,
    ResolvedPresenceRuntime,
    resolve_presence_runtime,
)
from ouroboros.utils import update_json_locked


from ouroboros.presence_capabilities import (
    PresenceStateError, _error, _require_exact_string, _require_sha256, _normalize_path_segments, _json_value_payload, PresenceToolTarget, PresenceScriptTarget, PresenceResourceTarget, PresenceArgumentBinding,
)

PRESENCE_STATE_SCHEMA_VERSION = 1

_TOOL_KINDS = frozenset({"builtin", "extension", "mcp"})
_BINDING_SOURCES = frozenset({"static", "conversation", "actor", "message", "destination", "resource"})
_CONTEXT_BINDING_SOURCES = frozenset({"conversation", "actor", "message", "destination"})
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_STATE_FIELDS = frozenset({"schema_version", "selections", "runtime_overrides"})
_SELECTION_FIELDS = frozenset({"request_fingerprint", "target", "argument_bindings"})
_TOOL_TARGET_FIELDS = frozenset({"type", "kind", "name", "provider"})
_SCRIPT_TARGET_FIELDS = frozenset({"type", "skill", "script"})
_RESOURCE_TARGET_FIELDS = frozenset({"type", "root", "operations", "path_prefix", "bucket", "skill_name"})
_RUNTIME_OVERRIDE_FIELDS = frozenset({"model_slot", "inline_max_rounds"})


PresenceTarget: TypeAlias = PresenceToolTarget | PresenceScriptTarget | PresenceResourceTarget


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise _error(
            "presence_state_not_canonical",
            "presence_state",
            "presence state contains a value that cannot be canonically encoded",
        ) from exc
    return text.encode("utf-8")


def _domain_sha256(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\0" + _canonical_bytes(value)).hexdigest()


def _binding_payload(binding: PresenceArgumentBinding) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "argument_path": list(binding.argument_path),
        "source": binding.source,
    }
    if binding.source in _CONTEXT_BINDING_SOURCES:
        payload["source_path"] = list(binding.source_path)
    elif binding.source == "static":
        payload["static_value"] = _json_value_payload(binding.static_value)
    else:
        payload["resource_request_fingerprint"] = binding.resource_request_fingerprint
    return payload


def _binding_sort_key(binding: PresenceArgumentBinding) -> bytes:
    return _canonical_bytes(_binding_payload(binding))


@dataclass(frozen=True)
class PresenceSelection:
    """One confirmed mapping from a portable request to an exact local target."""

    request_fingerprint: str
    target: PresenceTarget
    argument_bindings: tuple[PresenceArgumentBinding, ...] = ()

    def __post_init__(self) -> None:
        _require_sha256(self.request_fingerprint, field_name="selection.request_fingerprint")
        if not isinstance(
            self.target,
            (PresenceToolTarget, PresenceScriptTarget, PresenceResourceTarget),
        ):
            raise _error(
                "presence_state_invalid_target",
                "selection.target",
                "selection.target must be a supported structural target",
            )
        if not isinstance(self.argument_bindings, tuple) or any(
            not isinstance(binding, PresenceArgumentBinding) for binding in self.argument_bindings
        ):
            raise _error(
                "presence_state_invalid_bindings",
                "selection.argument_bindings",
                "selection.argument_bindings must be a tuple of argument bindings",
            )
        seen_paths: set[tuple[str | int, ...]] = set()
        for binding in self.argument_bindings:
            for existing in seen_paths:
                common = min(len(existing), len(binding.argument_path))
                if existing[:common] == binding.argument_path[:common]:
                    raise _error(
                        "presence_state_overlapping_binding",
                        "selection.argument_bindings",
                        "argument binding paths must not overlap",
                    )
            seen_paths.add(binding.argument_path)
        object.__setattr__(
            self,
            "argument_bindings",
            tuple(sorted(self.argument_bindings, key=_binding_sort_key)),
        )


def _target_payload(target: PresenceTarget) -> dict[str, Any]:
    if isinstance(target, PresenceToolTarget):
        return {
            "type": "tool",
            "kind": target.kind,
            "name": target.name,
            "provider": target.provider,
        }
    if isinstance(target, PresenceScriptTarget):
        return {"type": "script", "skill": target.skill, "script": target.script}
    return {
        "type": "resource",
        "root": target.root,
        "operations": list(target.operations),
        "path_prefix": target.path_prefix,
        "bucket": target.bucket,
        "skill_name": target.skill_name,
    }


def _selection_payload(selection: PresenceSelection) -> dict[str, Any]:
    return {
        "request_fingerprint": selection.request_fingerprint,
        "target": _target_payload(selection.target),
        "argument_bindings": [_binding_payload(binding) for binding in selection.argument_bindings],
    }


def presence_selection_fingerprint(selection: PresenceSelection) -> str:
    """Return the full domain-separated fingerprint of one exact selection."""
    if not isinstance(selection, PresenceSelection):
        raise _error(
            "presence_state_invalid_selection",
            "selection",
            "selection must be a PresenceSelection",
        )
    return _domain_sha256("ouroboros.presence.selection.v1", _selection_payload(selection))


def _selection_set_fingerprint(selections: tuple[PresenceSelection, ...]) -> str:
    fingerprints = sorted(presence_selection_fingerprint(item) for item in selections)
    return _domain_sha256("ouroboros.presence.selection_set.v1", fingerprints)


def _selection_sort_key(selection: PresenceSelection) -> str:
    return presence_selection_fingerprint(selection)


@dataclass(frozen=True)
class PresenceState:
    """Complete installation-local mapping, including currently orphaned rows."""

    selections: tuple[PresenceSelection, ...] = ()
    runtime_overrides: PresenceRuntimeOverrides = field(default_factory=PresenceRuntimeOverrides)
    workspace_root: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.selections, tuple) or any(
            not isinstance(selection, PresenceSelection) for selection in self.selections
        ):
            raise _error(
                "presence_state_invalid_selections",
                "presence_state.selections",
                "presence_state.selections must be a tuple of selections",
            )
        request_fingerprints = [item.request_fingerprint for item in self.selections]
        if len(request_fingerprints) != len(set(request_fingerprints)):
            raise _error(
                "presence_state_duplicate_request_mapping",
                "presence_state.selections",
                "presence state may store at most one selection per request fingerprint",
            )
        if not isinstance(self.runtime_overrides, PresenceRuntimeOverrides):
            raise _error(
                "presence_state_invalid_runtime_overrides",
                "presence_state.runtime_overrides",
                "presence_state.runtime_overrides must be PresenceRuntimeOverrides",
            )
        _require_exact_string(self.workspace_root, field_name="presence_state.workspace_root", allow_empty=True)
        if self.workspace_root and not Path(self.workspace_root).is_absolute():
            raise _error(
                "presence_state_invalid_workspace",
                "presence_state.workspace_root",
                "presence_state.workspace_root must be an absolute path",
            )
        object.__setattr__(
            self,
            "selections",
            tuple(sorted(self.selections, key=_selection_sort_key)),
        )


def _runtime_overrides_payload(overrides: PresenceRuntimeOverrides) -> dict[str, Any]:
    return {
        "model_slot": overrides.model_slot,
        "inline_max_rounds": overrides.inline_max_rounds,
    }


def _state_payload(state: PresenceState) -> dict[str, Any]:
    return {
        "schema_version": PRESENCE_STATE_SCHEMA_VERSION,
        "selections": [_selection_payload(item) for item in state.selections],
        "runtime_overrides": _runtime_overrides_payload(state.runtime_overrides),
        # Preserve the stored shape and digest of profiles without a workspace.
        **({"workspace_root": state.workspace_root} if state.workspace_root else {}),
    }


def presence_state_fingerprint(state: PresenceState) -> str:
    """Bind every persisted selection (active or orphaned) and local override."""
    if not isinstance(state, PresenceState):
        raise _error(
            "presence_state_invalid",
            "presence_state",
            "state must be a PresenceState",
        )
    return _domain_sha256("ouroboros.presence.state.v1", _state_payload(state))


def _reject_unknown_fields(
    value: Mapping[Any, Any],
    allowed: frozenset[str],
    *,
    field_name: str,
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise _error(
            "presence_state_unknown_field",
            field_name,
            f"{field_name} contains unknown field(s): {', '.join(unknown)}",
        )


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(
            "presence_state_invalid_object",
            field_name,
            f"{field_name} must be a JSON object",
        )
    if any(not isinstance(key, str) for key in value):
        raise _error(
            "presence_state_invalid_object",
            field_name,
            f"{field_name} keys must be strings",
        )
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    field_name: str,
) -> None:
    _reject_unknown_fields(value, expected, field_name=field_name)
    missing = sorted(expected - set(value))
    if missing:
        raise _error(
            "presence_state_missing_field",
            field_name,
            f"{field_name} is missing required field(s): {', '.join(missing)}",
        )


def _path_from_json(value: Any, *, field_name: str, required: bool) -> tuple[str | int, ...]:
    if not isinstance(value, list):
        raise _error(
            "presence_state_invalid_binding_path",
            field_name,
            f"{field_name} must be a JSON array",
        )
    return _normalize_path_segments(tuple(value), field_name=field_name, required=required)


def _target_from_payload(value: Any, *, field_name: str) -> PresenceTarget:
    raw = _require_mapping(value, field_name=field_name)
    target_type = raw.get("type")
    if target_type == "tool":
        _require_exact_fields(raw, _TOOL_TARGET_FIELDS, field_name=field_name)
        name = raw["name"]
        if raw["kind"] == "builtin" and isinstance(name, str):
            name = canonical_tool_name(name)  # a renamed builtin's saved spelling; a non-string fails typed below
        return PresenceToolTarget(raw["kind"], name, raw["provider"])
    if target_type == "script":
        _require_exact_fields(raw, _SCRIPT_TARGET_FIELDS, field_name=field_name)
        return PresenceScriptTarget(skill=raw["skill"], script=raw["script"])
    if target_type == "resource":
        _require_exact_fields(raw, _RESOURCE_TARGET_FIELDS, field_name=field_name)
        operations = raw["operations"]
        if not isinstance(operations, list):
            raise _error(
                "presence_state_invalid_operations",
                f"{field_name}.operations",
                f"{field_name}.operations must be a JSON array",
            )
        return PresenceResourceTarget(
            root=raw["root"],
            operations=tuple(operations),
            path_prefix=raw["path_prefix"],
            bucket=raw["bucket"],
            skill_name=raw["skill_name"],
        )
    raise _error(
        "presence_state_invalid_target_type",
        f"{field_name}.type",
        f"{field_name}.type must be tool, script, or resource",
    )


def _binding_from_payload(value: Any, *, field_name: str) -> PresenceArgumentBinding:
    raw = _require_mapping(value, field_name=field_name)
    source = raw.get("source")
    if not isinstance(source, str):
        raise _error(
            "presence_state_invalid_binding_source",
            f"{field_name}.source",
            f"{field_name}.source must name a supported structural source",
        )
    if source in _CONTEXT_BINDING_SOURCES:
        expected = frozenset({"argument_path", "source", "source_path"})
    elif source == "static":
        expected = frozenset({"argument_path", "source", "static_value"})
    elif source == "resource":
        expected = frozenset({"argument_path", "source", "resource_request_fingerprint"})
    else:
        raise _error(
            "presence_state_invalid_binding_source",
            f"{field_name}.source",
            f"{field_name}.source must name a supported structural source",
        )
    _require_exact_fields(raw, expected, field_name=field_name)
    return PresenceArgumentBinding(
        argument_path=_path_from_json(
            raw["argument_path"],
            field_name=f"{field_name}.argument_path",
            required=True,
        ),
        source=source,
        source_path=(
            _path_from_json(
                raw["source_path"],
                field_name=f"{field_name}.source_path",
                required=False,
            )
            if source in _CONTEXT_BINDING_SOURCES
            else ()
        ),
        static_value=raw.get("static_value"),
        resource_request_fingerprint=raw.get("resource_request_fingerprint", ""),
    )


def _selection_from_payload(value: Any, *, index: int) -> PresenceSelection:
    field_name = f"presence_state.selections[{index}]"
    raw = _require_mapping(value, field_name=field_name)
    _require_exact_fields(raw, _SELECTION_FIELDS, field_name=field_name)
    bindings_raw = raw["argument_bindings"]
    if not isinstance(bindings_raw, list):
        raise _error(
            "presence_state_invalid_bindings",
            f"{field_name}.argument_bindings",
            f"{field_name}.argument_bindings must be a JSON array",
        )
    bindings = tuple(
        _binding_from_payload(item, field_name=f"{field_name}.argument_bindings[{position}]")
        for position, item in enumerate(bindings_raw)
    )
    return PresenceSelection(
        request_fingerprint=raw["request_fingerprint"],
        target=_target_from_payload(raw["target"], field_name=f"{field_name}.target"),
        argument_bindings=bindings,
    )


def _state_from_payload(value: Any) -> PresenceState:
    raw = _require_mapping(value, field_name="presence_state")
    fields = _STATE_FIELDS | {"workspace_root"} if "workspace_root" in raw else _STATE_FIELDS
    _require_exact_fields(raw, fields, field_name="presence_state")
    version = raw["schema_version"]
    if type(version) is not int or version != PRESENCE_STATE_SCHEMA_VERSION:
        raise _error(
            "presence_state_unsupported_version",
            "presence_state.schema_version",
            f"presence_state.schema_version must be exactly {PRESENCE_STATE_SCHEMA_VERSION}",
        )
    selections_raw = raw["selections"]
    if not isinstance(selections_raw, list):
        raise _error(
            "presence_state_invalid_selections",
            "presence_state.selections",
            "presence_state.selections must be a JSON array",
        )
    overrides_raw = _require_mapping(
        raw["runtime_overrides"],
        field_name="presence_state.runtime_overrides",
    )
    _require_exact_fields(
        overrides_raw,
        _RUNTIME_OVERRIDE_FIELDS,
        field_name="presence_state.runtime_overrides",
    )
    try:
        overrides = PresenceRuntimeOverrides(
            model_slot=overrides_raw["model_slot"],
            inline_max_rounds=overrides_raw["inline_max_rounds"],
        )
    except ValueError as exc:
        raise _error(
            "presence_state_invalid_runtime_overrides",
            "presence_state.runtime_overrides",
            "presence_state.runtime_overrides contains an invalid value",
        ) from exc
    return PresenceState(
        selections=tuple(_selection_from_payload(item, index=index) for index, item in enumerate(selections_raw)),
        runtime_overrides=overrides,
        workspace_root=raw.get("workspace_root", ""),
    )


def _validated_skill_name(skill_name: Any) -> str:
    if not isinstance(skill_name, str) or not skill_name:
        raise _error(
            "presence_state_invalid_skill_name",
            "skill_name",
            "skill_name must be a canonical skill name",
        )
    canonical = canonical_skill_name(skill_name)
    if canonical != skill_name or canonical == "_unnamed":
        raise _error(
            "presence_state_invalid_skill_name",
            "skill_name",
            "skill_name must be a canonical skill name",
        )
    return skill_name


def _state_path(drive_root: Any, skill_name: Any) -> Path:
    name = _validated_skill_name(skill_name)
    try:
        root = Path(drive_root)
    except TypeError as exc:
        raise _error(
            "presence_state_invalid_drive_root",
            "drive_root",
            "drive_root must be path-like",
        ) from exc
    return root / "state" / "skills" / name / PRESENCE_PROFILE_STATE_FILENAME


def load_presence_state(drive_root: Any, skill_name: str) -> PresenceState:
    """Read strict host-owned state; an absent file is an empty state and no mkdir."""
    path = _state_path(drive_root, skill_name)
    if not path.exists():
        return PresenceState()
    if not path.is_file():
        raise _error(
            "presence_state_not_file",
            "presence_state",
            "presence state must be a regular JSON file",
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error(
            "presence_state_malformed_json",
            "presence_state",
            "presence state contains malformed JSON",
        ) from exc
    except OSError as exc:
        raise _error(
            "presence_state_read_failed",
            "presence_state",
            "presence state could not be read",
        ) from exc
    return _state_from_payload(value)


def save_presence_state(
    drive_root: Any,
    skill_name: str,
    state: PresenceState,
    *,
    expected_state_fingerprint: str,
) -> PresenceState:
    """Atomically save state when the locked current fingerprint still matches."""
    if not isinstance(state, PresenceState):
        raise _error(
            "presence_state_invalid",
            "presence_state",
            "state must be a PresenceState",
        )
    expected = _require_sha256(
        expected_state_fingerprint,
        field_name="expected_state_fingerprint",
    )
    path = _state_path(drive_root, skill_name)

    def _replace(current_payload: dict[str, Any]) -> dict[str, Any]:
        current = _state_from_payload(current_payload) if path.exists() else PresenceState()
        if presence_state_fingerprint(current) != expected:
            raise _error(
                "presence_state_conflict",
                "expected_state_fingerprint",
                "presence state changed after it was read; reload and retry",
            )
        return _state_payload(state)

    try:
        updated = update_json_locked(path, _replace, strict_existing_dict=True)
    except PresenceStateError:
        raise
    except TimeoutError as exc:
        raise _error(
            "presence_state_lock_timeout",
            "presence_state",
            "presence state could not acquire its write lock",
        ) from exc
    except ValueError as exc:
        raise _error(
            "presence_state_malformed_json",
            "presence_state",
            "existing presence state is malformed or is not a JSON object",
        ) from exc
    except OSError as exc:
        raise _error(
            "presence_state_write_failed",
            "presence_state",
            "presence state could not be written",
        ) from exc
    return _state_from_payload(updated)


def _selection_matches_request(
    selection: PresenceSelection,
    request: PresenceCapabilityRequest,
) -> bool:
    if request.kind == "tool":
        return isinstance(selection.target, PresenceToolTarget)
    if request.kind == "script":
        return isinstance(selection.target, PresenceScriptTarget)
    if request.kind != "resource" or not isinstance(
        selection.target,
        PresenceResourceTarget,
    ):
        return False
    return not selection.argument_bindings and set(selection.target.operations) == set(request.operations)


def _selection_resource_bindings_resolve(
    selection: PresenceSelection,
    active_by_request: Mapping[str, PresenceSelection],
) -> bool:
    if isinstance(selection.target, PresenceResourceTarget):
        return True
    for binding in selection.argument_bindings:
        if binding.source != "resource":
            continue
        resource = active_by_request.get(binding.resource_request_fingerprint)
        if resource is None or not isinstance(resource.target, PresenceResourceTarget):
            return False
    return True


@dataclass(frozen=True)
class PresenceProfileResolution:
    """Structural resolution of saved state against one reviewed profile."""

    active: tuple[PresenceSelection, ...]
    missing_required: tuple[PresenceCapabilityRequest, ...]
    missing_optional: tuple[PresenceCapabilityRequest, ...]
    orphaned: tuple[PresenceSelection, ...]
    runtime: ResolvedPresenceRuntime
    profile_fingerprint: str
    selection_fingerprint: str
    required_selections_present: bool


def resolve_presence_profile_state(
    profile: PresenceProfile,
    state: PresenceState,
    *,
    global_max_rounds: int | None,  # None = no task round limit; the inline cap stays finite
) -> PresenceProfileResolution:
    """Resolve structural mappings without claiming live target readiness."""
    if not isinstance(profile, PresenceProfile):
        raise _error(
            "presence_state_invalid_profile",
            "profile",
            "profile must be a PresenceProfile",
        )
    if not isinstance(state, PresenceState):
        raise _error(
            "presence_state_invalid",
            "presence_state",
            "state must be a PresenceState",
        )

    requests_by_fingerprint: dict[str, PresenceCapabilityRequest] = {}
    for request in profile.capability_requests:
        fingerprint = presence_request_fingerprint(request)
        if fingerprint in requests_by_fingerprint:
            raise _error(
                "presence_state_duplicate_request_fingerprint",
                "profile.capability_requests",
                "profile capability requests must have unique structural fingerprints",
            )
        requests_by_fingerprint[fingerprint] = request

    candidates: dict[str, PresenceSelection] = {}
    orphaned: list[PresenceSelection] = []
    for selection in state.selections:
        request = requests_by_fingerprint.get(selection.request_fingerprint)
        if request is None or not _selection_matches_request(selection, request):
            orphaned.append(selection)
            continue
        candidates[selection.request_fingerprint] = selection

    active_by_request = dict(candidates)
    for fingerprint, selection in tuple(candidates.items()):
        if not _selection_resource_bindings_resolve(selection, active_by_request):
            active_by_request.pop(fingerprint, None)
            orphaned.append(selection)

    active = tuple(sorted(active_by_request.values(), key=_selection_sort_key))
    active_fingerprints = set(active_by_request)
    missing_required: list[PresenceCapabilityRequest] = []
    missing_optional: list[PresenceCapabilityRequest] = []
    for request in profile.capability_requests:
        if presence_request_fingerprint(request) in active_fingerprints:
            continue
        (missing_required if request.required else missing_optional).append(request)

    missing_required_tuple = tuple(missing_required)
    return PresenceProfileResolution(
        active=active,
        missing_required=missing_required_tuple,
        missing_optional=tuple(missing_optional),
        orphaned=tuple(sorted(orphaned, key=_selection_sort_key)),
        runtime=resolve_presence_runtime(
            profile.runtime_defaults,
            state.runtime_overrides,
            global_max_rounds=global_max_rounds,
        ),
        profile_fingerprint=presence_profile_fingerprint(profile),
        selection_fingerprint=_selection_set_fingerprint(active),
        required_selections_present=not missing_required_tuple,
    )


__all__ = [
    "PRESENCE_STATE_SCHEMA_VERSION",
    "PresenceArgumentBinding",
    "PresenceProfileResolution",
    "PresenceResourceTarget",
    "PresenceScriptTarget",
    "PresenceSelection",
    "PresenceState",
    "PresenceStateError",
    "PresenceTarget",
    "PresenceToolTarget",
    "load_presence_state",
    "presence_selection_fingerprint",
    "presence_state_fingerprint",
    "resolve_presence_profile_state",
    "save_presence_state",
]
