"""Immutable positive authority compiled from one reviewed presence profile."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ouroboros.dialogue_provenance import (  # the provenance predicate; authority re-exports it
    PRESENCE_OWN_WORK_SCOPE,
    presence_caller_binding,
    presence_metadata_binding as presence_metadata_binding,
    presence_effective_hops,
    presence_effective_related,
    presence_queue_task,
    presence_record_binding,
    presence_related_work,
    presence_target_record,
)
from ouroboros.presence_capabilities import (
    PresenceArgumentBinding,
    PresenceProfileResolution,
    PresenceResourceTarget,
    PresenceScriptTarget,
    PresenceToolTarget,
)
from ouroboros.tool_capabilities import COGNITIVE_MEMORY_TOOL_NAMES

PRESENCE_CEILING_SCHEMA_VERSION = 1
_SHA256_LEN = 64
# The own-work baseline (owner Q1/Q2): a Presence mind may find, read and message
# the independent work it started from ANY conversation of its own binding. The two
# readers arrive bound to that scope (the host overwrites the argument); steer_task
# is narrowed by its handler for every Presence caller. A profile that selected one
# of these names keeps that grant, so a selected global reader stays global.
_OWN_WORK_BASELINE = (("get_task_result", True), ("recent_tasks", True), ("steer_task", False))


class PresenceAuthorityError(ValueError):
    """A host-minted presence authority block is invalid or incomplete."""

    def __init__(self, code: str, field: str) -> None:
        self.code = str(code or "presence_authority_invalid")
        self.field = str(field or "capability_ceiling")
        super().__init__(f"{self.code}: {self.field}")


def _text(value: Any, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or value != value.strip() or (not empty and not value):
        raise PresenceAuthorityError("presence_authority_invalid_string", field)
    return value


def _sha(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) != _SHA256_LEN or any(ch not in "0123456789abcdef" for ch in text):
        raise PresenceAuthorityError("presence_authority_invalid_fingerprint", field)
    return text


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PresenceAuthorityError("presence_authority_not_canonical", "capability_ceiling") from exc


def _digest(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "digest"}
    return hashlib.sha256(b"ouroboros.presence.ceiling.v1\0" + _canonical(body)).hexdigest()


def _binding_payload(binding: PresenceArgumentBinding) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "argument_path": list(binding.argument_path),
        "source": binding.source,
    }
    if binding.source in {"conversation", "actor", "message", "destination"}:
        payload["source_path"] = list(binding.source_path)
    elif binding.source == "static":
        payload["static_value"] = json.loads(_canonical(binding.static_value))
    else:
        payload["resource_request_fingerprint"] = binding.resource_request_fingerprint
    return payload


@dataclass(frozen=True)
class PresenceToolGrant:
    """One exact tool name and its host-authored argument bindings."""

    name: str
    bindings: tuple[PresenceArgumentBinding, ...] = ()

    def __post_init__(self) -> None:
        _text(self.name, "tool_grant.name")
        if not isinstance(self.bindings, tuple) or any(
            not isinstance(binding, PresenceArgumentBinding) for binding in self.bindings
        ):
            raise PresenceAuthorityError("presence_authority_invalid_bindings", "tool_grant.bindings")


@dataclass(frozen=True)
class PresenceResourceGrant:
    """One exact logical root, operation set, and confined relative prefix."""

    root: str
    operations: tuple[str, ...]
    path_prefix: str = "."
    bucket: str = ""
    skill_name: str = ""

    def __post_init__(self) -> None:
        _text(self.root, "resource_grant.root")
        if not isinstance(self.operations, tuple) or not self.operations or any(
            not isinstance(item, str) or not item for item in self.operations
        ):
            raise PresenceAuthorityError("presence_authority_invalid_operations", "resource_grant.operations")
        if tuple(sorted(set(self.operations))) != self.operations:
            raise PresenceAuthorityError("presence_authority_invalid_operations", "resource_grant.operations")
        _text(self.path_prefix, "resource_grant.path_prefix")
        _text(self.bucket, "resource_grant.bucket", empty=True)
        _text(self.skill_name, "resource_grant.skill_name", empty=True)


@dataclass(frozen=True)
class PresenceCapabilityCeiling:
    """Frozen maximum carried by a presence turn and every descendant."""

    skill_name: str
    skill_content_hash: str
    profile_fingerprint: str
    state_fingerprint: str
    selection_fingerprint: str
    model_slot: str
    inline_max_rounds: int
    tool_grants: tuple[PresenceToolGrant, ...]
    resource_grants: tuple[PresenceResourceGrant, ...]
    digest: str

    def __post_init__(self) -> None:
        _text(self.skill_name, "capability_ceiling.skill_name")
        for field in (
            "skill_content_hash",
            "profile_fingerprint",
            "state_fingerprint",
            "selection_fingerprint",
            "digest",
        ):
            _sha(getattr(self, field), f"capability_ceiling.{field}")
        if self.model_slot not in {"main", "light"}:
            raise PresenceAuthorityError("presence_authority_invalid_model_slot", "capability_ceiling.model_slot")
        if isinstance(self.inline_max_rounds, bool) or not isinstance(self.inline_max_rounds, int) or self.inline_max_rounds < 1:
            raise PresenceAuthorityError("presence_authority_invalid_rounds", "capability_ceiling.inline_max_rounds")
        if len({grant.name for grant in self.tool_grants}) != len(self.tool_grants):
            raise PresenceAuthorityError("presence_authority_duplicate_tool", "capability_ceiling.tool_grants")


def presence_ceiling_payload(ceiling: PresenceCapabilityCeiling) -> dict[str, Any]:
    """Return the strict JSON carrier stored inside an existing task contract."""

    payload: dict[str, Any] = {
        "schema_version": PRESENCE_CEILING_SCHEMA_VERSION,
        "skill_name": ceiling.skill_name,
        "skill_content_hash": ceiling.skill_content_hash,
        "profile_fingerprint": ceiling.profile_fingerprint,
        "state_fingerprint": ceiling.state_fingerprint,
        "selection_fingerprint": ceiling.selection_fingerprint,
        "runtime": {
            "model_slot": ceiling.model_slot,
            "inline_max_rounds": ceiling.inline_max_rounds,
        },
        "tools": [
            {"name": grant.name, "bindings": [_binding_payload(item) for item in grant.bindings]}
            for grant in ceiling.tool_grants
        ],
        "resources": [
            {
                "root": grant.root,
                "operations": list(grant.operations),
                "path_prefix": grant.path_prefix,
                "bucket": grant.bucket,
                "skill_name": grant.skill_name,
            }
            for grant in ceiling.resource_grants
        ],
    }
    payload["digest"] = _digest(payload)
    return payload


def _binding_from_payload(value: Any, field: str) -> PresenceArgumentBinding:
    if not isinstance(value, Mapping):
        raise PresenceAuthorityError("presence_authority_invalid_binding", field)
    source = value.get("source")
    argument_path = value.get("argument_path")
    if not isinstance(argument_path, list):
        raise PresenceAuthorityError("presence_authority_invalid_binding", f"{field}.argument_path")
    kwargs: dict[str, Any] = {
        "argument_path": tuple(argument_path),
        "source": source,
    }
    if source in {"conversation", "actor", "message", "destination"}:
        source_path = value.get("source_path")
        if set(value) != {"argument_path", "source", "source_path"} or not isinstance(source_path, list):
            raise PresenceAuthorityError("presence_authority_invalid_binding", field)
        kwargs["source_path"] = tuple(source_path)
    elif source == "static":
        if set(value) != {"argument_path", "source", "static_value"}:
            raise PresenceAuthorityError("presence_authority_invalid_binding", field)
        kwargs["static_value"] = value.get("static_value")
    elif source == "resource":
        if set(value) != {"argument_path", "source", "resource_request_fingerprint"}:
            raise PresenceAuthorityError("presence_authority_invalid_binding", field)
        kwargs["resource_request_fingerprint"] = value.get("resource_request_fingerprint")
    else:
        raise PresenceAuthorityError("presence_authority_invalid_binding", f"{field}.source")
    try:
        return PresenceArgumentBinding(**kwargs)
    except ValueError as exc:
        raise PresenceAuthorityError("presence_authority_invalid_binding", field) from exc


def presence_ceiling_from_payload(value: Any) -> PresenceCapabilityCeiling:
    """Decode and verify the frozen JSON carrier copied through task state."""

    if not isinstance(value, Mapping):
        raise PresenceAuthorityError("presence_authority_invalid_payload", "capability_ceiling")
    expected = {
        "schema_version",
        "skill_name",
        "skill_content_hash",
        "profile_fingerprint",
        "state_fingerprint",
        "selection_fingerprint",
        "runtime",
        "tools",
        "resources",
        "digest",
    }
    if set(value) != expected or value.get("schema_version") != PRESENCE_CEILING_SCHEMA_VERSION:
        raise PresenceAuthorityError("presence_authority_invalid_payload", "capability_ceiling")
    runtime = value.get("runtime")
    tools = value.get("tools")
    resources = value.get("resources")
    if not isinstance(runtime, Mapping) or set(runtime) != {"model_slot", "inline_max_rounds"}:
        raise PresenceAuthorityError("presence_authority_invalid_runtime", "capability_ceiling.runtime")
    if not isinstance(tools, list) or not isinstance(resources, list):
        raise PresenceAuthorityError("presence_authority_invalid_payload", "capability_ceiling")
    tool_grants: list[PresenceToolGrant] = []
    for index, item in enumerate(tools):
        field = f"capability_ceiling.tools[{index}]"
        if not isinstance(item, Mapping) or set(item) != {"name", "bindings"} or not isinstance(item.get("bindings"), list):
            raise PresenceAuthorityError("presence_authority_invalid_tool", field)
        tool_grants.append(
            PresenceToolGrant(
                item.get("name"),
                tuple(
                    _binding_from_payload(binding, f"{field}.bindings[{position}]")
                    for position, binding in enumerate(item["bindings"])
                ),
            )
        )
    resource_grants: list[PresenceResourceGrant] = []
    for index, item in enumerate(resources):
        field = f"capability_ceiling.resources[{index}]"
        if not isinstance(item, Mapping) or set(item) != {
            "root", "operations", "path_prefix", "bucket", "skill_name"
        } or not isinstance(item.get("operations"), list):
            raise PresenceAuthorityError("presence_authority_invalid_resource", field)
        resource_grants.append(
            PresenceResourceGrant(
                item.get("root"),
                tuple(item["operations"]),
                item.get("path_prefix"),
                item.get("bucket"),
                item.get("skill_name"),
            )
        )
    ceiling = PresenceCapabilityCeiling(
        skill_name=value.get("skill_name"),
        skill_content_hash=value.get("skill_content_hash"),
        profile_fingerprint=value.get("profile_fingerprint"),
        state_fingerprint=value.get("state_fingerprint"),
        selection_fingerprint=value.get("selection_fingerprint"),
        model_slot=runtime.get("model_slot"),
        inline_max_rounds=runtime.get("inline_max_rounds"),
        tool_grants=tuple(tool_grants),
        resource_grants=tuple(resource_grants),
        digest=value.get("digest"),
    )
    if _digest(value) != ceiling.digest:
        raise PresenceAuthorityError("presence_authority_digest_mismatch", "capability_ceiling.digest")
    return ceiling


def build_presence_capability_ceiling(
    *,
    skill_name: str,
    skill_content_hash: str,
    state_fingerprint: str,
    resolution: PresenceProfileResolution,
) -> PresenceCapabilityCeiling:
    """Compile active selections; missing required requests refuse admission."""

    if not isinstance(resolution, PresenceProfileResolution) or not resolution.required_selections_present:
        raise PresenceAuthorityError("presence_authority_missing_required", "resolution.missing_required")
    tools: list[PresenceToolGrant] = []
    resources: list[PresenceResourceGrant] = []
    for selection in resolution.active:
        target = selection.target
        if isinstance(target, PresenceToolTarget):
            tools.append(PresenceToolGrant(target.name, selection.argument_bindings))
        elif isinstance(target, PresenceScriptTarget):
            bindings = (
                PresenceArgumentBinding(("skill",), "static", static_value=target.skill),
                PresenceArgumentBinding(("script",), "static", static_value=target.script),
                *selection.argument_bindings,
            )
            tools.append(PresenceToolGrant("skill_exec", tuple(bindings)))
        elif isinstance(target, PresenceResourceTarget):
            resources.append(
                PresenceResourceGrant(
                    target.root,
                    target.operations,
                    target.path_prefix,
                    target.bucket,
                    target.skill_name,
                )
            )
    # The cognitive baseline is part of the ceiling, not of the profile: an
    # admitted conversation is still this mind, so it keeps its own memory in
    # every room instead of losing what the exchange taught it. It adds no
    # authority over settings, delivery, or the filesystem. A name the profile
    # already selected keeps THAT grant, because its host-authored argument
    # bindings are the exact facts apply_presence_argument_bindings overrides
    # the model with; an unselected name arrives with no bindings at all.
    selected = {grant.name for grant in tools}
    tools.extend(
        PresenceToolGrant(name)
        for name in sorted(COGNITIVE_MEMORY_TOOL_NAMES)
        if name not in selected
    )
    scope = (PresenceArgumentBinding(("presence_scope",), "static", static_value=PRESENCE_OWN_WORK_SCOPE),)
    tools.extend(
        PresenceToolGrant(name, scope if scoped else ())
        for name, scoped in _OWN_WORK_BASELINE
        if name not in selected
    )
    provisional = PresenceCapabilityCeiling(
        skill_name=_text(skill_name, "skill_name"),
        skill_content_hash=_sha(skill_content_hash, "skill_content_hash"),
        profile_fingerprint=resolution.profile_fingerprint,
        state_fingerprint=_sha(state_fingerprint, "state_fingerprint"),
        selection_fingerprint=resolution.selection_fingerprint,
        model_slot=resolution.runtime.model_slot,
        inline_max_rounds=resolution.runtime.inline_max_rounds,
        tool_grants=tuple(sorted(tools, key=lambda item: item.name)),
        resource_grants=tuple(sorted(resources, key=lambda item: (item.root, item.path_prefix))),
        digest="0" * _SHA256_LEN,
    )
    payload = presence_ceiling_payload(provisional)
    return PresenceCapabilityCeiling(
        **{**provisional.__dict__, "digest": payload["digest"]}
    )


def presence_ceiling_allows_tool(ceiling: PresenceCapabilityCeiling, tool_name: str) -> bool:
    return any(grant.name == tool_name for grant in ceiling.tool_grants)


def presence_ceiling_allows_delegated_surface(ctx: Any, surface: str) -> bool:
    """Keep mutative descendants inside an explicitly selected logical write root."""

    ceiling = presence_ceiling_from_context(ctx)
    if ceiling is None:
        return True
    required_root = {
        "self_worktree": "system_repo",
        "external_workspace": "active_workspace",
    }.get(str(surface or "").strip())
    return bool(required_root and any(
        grant.root == required_root and "write" in grant.operations
        for grant in ceiling.resource_grants
    ))


def presence_ceiling_allows_delegated_read(ctx: Any) -> bool:
    """Allow harness read access only when Presence selected the whole active repo root."""

    ceiling = presence_ceiling_from_context(ctx)
    if ceiling is None:
        return True
    workspace_active = bool(
        callable(getattr(ctx, "is_workspace_mode", None)) and ctx.is_workspace_mode()
    )
    required_root = "active_workspace" if workspace_active else "system_repo"
    return any(
        grant.root == required_root
        and grant.path_prefix in {"", "."}
        and bool(set(grant.operations) & {"read", "list", "search"})
        for grant in ceiling.resource_grants
    )


def _presence_bound_value(ctx: Any, binding: PresenceArgumentBinding) -> Any:
    if binding.source == "static":
        return json.loads(_canonical(binding.static_value))
    metadata = getattr(ctx, "task_metadata", None)
    presence = metadata.get("presence") if isinstance(metadata, Mapping) else None
    event = presence.get("event") if isinstance(presence, Mapping) else None
    if not isinstance(event, Mapping):
        raise PresenceAuthorityError("presence_authority_event_missing", "task_metadata.presence.event")
    if binding.source == "resource":
        raise PresenceAuthorityError(
            "presence_authority_resource_projection_unsupported",
            "tool_grant.bindings.source",
        )
    value: Any = event.get(binding.source)
    for part in binding.source_path:
        if isinstance(part, int) and isinstance(value, (list, tuple)) and 0 <= part < len(value):
            value = value[part]
        elif isinstance(part, str) and isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            raise PresenceAuthorityError(
                "presence_authority_binding_source_missing",
                f"task_metadata.presence.event.{binding.source}",
            )
    return json.loads(_canonical(value))


def _set_bound_argument(target: Any, path: tuple[str | int, ...], value: Any) -> None:
    current = target
    for position, part in enumerate(path):
        last = position == len(path) - 1
        if isinstance(part, str) and isinstance(current, dict):
            if last:
                current[part] = value
                return
            next_part = path[position + 1]
            child = current.get(part)
            if not isinstance(child, (dict, list)):
                child = [] if isinstance(next_part, int) else {}
                current[part] = child
            current = child
            continue
        if isinstance(part, int) and isinstance(current, list) and part >= 0:
            while len(current) <= part:
                current.append(None)
            if last:
                current[part] = value
                return
            next_part = path[position + 1]
            child = current[part]
            if not isinstance(child, (dict, list)):
                child = [] if isinstance(next_part, int) else {}
                current[part] = child
            current = child
            continue
        raise PresenceAuthorityError("presence_authority_invalid_argument_path", "tool_grant.bindings.argument_path")


def apply_presence_argument_bindings(ctx: Any, tool_name: str, args: Mapping[str, Any]) -> dict[str, Any]:
    """Overwrite model arguments with the host-selected exact event/static facts."""

    ceiling = presence_ceiling_from_context(ctx)
    result = dict(args or {})
    if ceiling is None:
        return result
    grant = next((item for item in ceiling.tool_grants if item.name == tool_name), None)
    if grant is None:
        return result
    for binding in grant.bindings:
        _set_bound_argument(result, binding.argument_path, _presence_bound_value(ctx, binding))
    return result


def presence_ceiling_from_context(ctx: Any) -> PresenceCapabilityCeiling | None:
    """Read the verified carrier from a ToolContext-like object, if present."""

    contract = getattr(ctx, "task_contract", None)
    metadata = getattr(ctx, "task_metadata", None)
    if not isinstance(contract, Mapping) and isinstance(metadata, Mapping):
        contract = metadata.get("task_contract")
    if not isinstance(contract, Mapping):
        return None
    raw = contract.get("capability_ceiling")
    return presence_ceiling_from_payload(raw) if raw is not None else None


def presence_ceiling_allows_binding(ceiling: PresenceCapabilityCeiling, binding: Any) -> bool:
    """Intersect an already-resolved registry target with the positive ceiling."""

    try:
        target = Path(binding.target_path).resolve(strict=False)
        base = Path(binding.base_path).resolve(strict=False)
        logical_base = getattr(binding, "logical_base_path", None)
        if logical_base is not None:
            physical_relative = target.relative_to(base)
            logical_root = Path(logical_base).resolve(strict=False)
            # Preserve the configured container's actual logical location. A
            # default install uses ``~/Ouroboros/Deliverables`` (so the prefix
            # is ``Ouroboros/Deliverables``). An explicitly external container
            # has no user-files-relative spelling, so retain the established
            # logical resource name ``Deliverables`` rather than deriving a
            # potentially arbitrary physical basename. Do not turn either
            # shape into a hard-coded alias that widens or narrows a grant.
            try:
                logical_container = base.relative_to(logical_root)
            except ValueError:
                logical_container = Path("Deliverables")
            relative = logical_container / physical_relative
        else:
            relative = target.relative_to(base)
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    relative_parts = PurePosixPath(relative.as_posix() or ".").parts
    for grant in ceiling.resource_grants:
        if grant.root != str(binding.root) or str(binding.operation) not in grant.operations:
            continue
        # Skill-payload bindings carry the effective physical authority in
        # ``source`` (native, external, clawhub, ...).  A non-empty bucket on a
        # presence grant is an exact selector, not a wildcard; otherwise a
        # native-read overlay could turn a ceiling for one bucket into access
        # to another while the logical root/path still matched.
        if grant.bucket and grant.bucket != str(getattr(binding, "source", "") or ""):
            continue
        if grant.skill_name and grant.skill_name != str(getattr(binding, "skill_name", "") or ""):
            continue
        prefix_parts = () if grant.path_prefix == "." else PurePosixPath(grant.path_prefix).parts
        if relative_parts[: len(prefix_parts)] == prefix_parts:
            return True
    return False


def presence_effective_refusal(ctx: Any, task_id: str, effective: Any, *, drive_root: Any = None,
                               same_tree: bool = False) -> str:
    """Refusal when an admitted ``task_id``'s effective projection reaches work the caller may not address.

    A retry successor replaces the requested record's content, so each hop is judged
    by the same rule before anything is projected; the foreign id is not named.
    """

    for hop in presence_effective_hops(task_id, effective):
        if presence_work_refusal(ctx, hop, drive_root=drive_root, same_tree=same_tree):
            return (
                f"⚠️ PRESENCE_CAPABILITY_BLOCKED: task {task_id}'s effective result continues in work "
                "that was not started from this Presence binding; a Presence turn reads only that work."
            )
    return ""


def presence_work_refusal(ctx: Any, task_id: str, *, drive_root: Any = None, same_tree: bool = False) -> str:
    """Refusal text when a Presence caller may not address ``task_id``, else ``""``.

    A non-Presence caller is never narrowed here; a delegated descendant of a
    Presence-bound task is one through its inherited binding authority.
    ``presence_target_record`` decides, and a record naming another binding refuses.
    ``same_tree`` also admits the caller's own task tree, its root included (its
    lineage reads). ``drive_root`` defaults to the caller's task-status root.
    """

    binding = presence_caller_binding(ctx)
    if binding is None:
        return ""
    if drive_root is None:
        metadata = getattr(ctx, "task_metadata", None)
        drive_root = ((metadata.get("budget_drive_root") if isinstance(metadata, Mapping) else "")
                      or getattr(ctx, "budget_drive_root", "") or ctx.drive_root)
    target = str(task_id or "").strip()
    record = presence_target_record(drive_root, target)
    if isinstance(record, Mapping) and presence_related_work(binding, record):
        return ""
    if same_tree and isinstance(record, Mapping):
        metadata = getattr(ctx, "task_metadata", None)
        own_id = str(getattr(ctx, "task_id", "") or "")
        own_root = str((metadata or {}).get("root_task_id") or own_id) if isinstance(metadata, Mapping) else own_id
        if own_id and (target in {own_id, own_root} or str(record.get("parent_task_id") or "") == own_id
                       or str(record.get("root_task_id") or "") == own_root):
            return ""
    return (
        f"⚠️ PRESENCE_CAPABILITY_BLOCKED: task {target or '?'} is not independent work started from "
        "this Presence binding (same binding_id, any of its conversations); a Presence turn "
        "addresses only that work."
    )


__all__ = [
    "PRESENCE_CEILING_SCHEMA_VERSION",
    "PRESENCE_OWN_WORK_SCOPE",
    "presence_caller_binding",
    "presence_effective_refusal",
    "presence_effective_related",
    "presence_queue_task",
    "presence_record_binding",
    "presence_related_work",
    "presence_target_record",
    "presence_work_refusal",
    "PresenceAuthorityError",
    "PresenceCapabilityCeiling",
    "PresenceResourceGrant",
    "PresenceToolGrant",
    "build_presence_capability_ceiling",
    "apply_presence_argument_bindings",
    "presence_ceiling_allows_binding",
    "presence_ceiling_allows_delegated_surface",
    "presence_ceiling_allows_delegated_read",
    "presence_ceiling_from_context",
    "presence_ceiling_allows_tool",
    "presence_ceiling_from_payload",
    "presence_ceiling_payload",
]
