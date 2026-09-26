"""Tool API v2 access matrix.

This is the single policy shape for LLM-visible tools: a profile asks to run an
operation against a resource root and receives an allow/block decision. The
legacy per-tool checks still provide defense-in-depth while the public API is
migrated to neutral tool names.
"""

from __future__ import annotations

import os  # noqa: F401 — historical facade surface
import pathlib
import re  # noqa: F401 — historical facade surface
from dataclasses import dataclass  # noqa: F401 — historical facade surface
from typing import Any, Iterable, Literal, Optional  # noqa: F401 — historical facade surface

from ouroboros.artifacts import (delegated_capture_read_target,
                                 task_artifact_dir_path, task_id_for_artifacts)
from ouroboros.headless import task_state_dir
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE  # noqa: F401 — historical facade surface
from ouroboros.contracts.task_constraint import VALID_WRITE_SURFACES, normalize_task_constraint  # noqa: F401 — historical facade surface
from ouroboros import deliverables_paths as _deliverables_paths
from ouroboros.shell_parse import is_absolute_path_text
from ouroboros.task_results import validate_task_id
from ouroboros.utils import safe_relpath

_deliverables_root_lexical = _deliverables_paths._deliverables_root_lexical
_deliverables_root_lexical_alias = _deliverables_paths._deliverables_root_lexical_alias
_lexical_path_is_relative_to_casefold = _deliverables_paths._lexical_path_is_relative_to_casefold

# v7 D04 split: the owners below were extracted VERBATIM from this module
# (see each leaf's header); re-exported here so historical imports and
# monkeypatch targets keep working unchanged.
from ouroboros.tool_access_types import (  # noqa: F401 — re-exported moved surface
    Operation,
    ResolvedResourceBinding,
    ResourceRoot,
    SUBAGENT_CAPABILITIES,
    SubagentCapability,
    ToolAccessDecision,
    ToolProfile,
    _ALL_ROOTS,
    _POLICY,
    _READONLY_RESOURCE_ROOTS,
    _READ_OPS,
    _WRITE_LIKE_OPS,
    _SUBAGENT_CAPABILITY_TO_OPERATION,
    _TOP_LEVEL_PRINCIPAL_POLICY,
    _TOP_LEVEL_PRINCIPAL_PROFILES,
)
from ouroboros.tool_access_paths import (  # noqa: F401 — re-exported moved surface
    _deliverables_root,
    _path_is_relative_to_casefold,
    _user_files_root,
    canonical_data_root,
    normalize_root,
    normalize_root_relative,
    normalize_runtime_data_path,
    path_is_relative_to,
    paths_overlap_casefold,
    workspace_mode_block_reason,
)
from ouroboros.tool_access_roots import (  # noqa: F401 — re-exported moved surface
    _is_subagent_ctx,
    _skill_payload_base,
    active_tool_profile,
    binding_targets_system_repo,
    is_external_workspace,
    load_bound_skill,
    predicted_subagent_profile,
    project_room_lens_dir,
    resource_root_path,
)
from ouroboros.tool_access_user_files import (  # noqa: F401 — re-exported moved surface
    UserFilesPathBlockedError,
    _delegated_capture_read_hint,
    _subagent_projects_read_hint,
    resolve_user_file_path,
    user_files_path_block_reason,
)


# Orchestrator READ-ONLY roots (subagent projects, the Deliverables container): only ever
# {read,list,search}, never a shell cwd, read-only even for operator_control. A child never
# reads `subagent_projects` (parallel candidates stay independent); it reads owner-visible
# `deliverables` and, by the lineage rule below, its parent's and root's task files (T4=A).

def summarize_subagent_profile(profile: ToolProfile, *, effective_lane: str = "") -> str:
    """Compact, human-readable summary of a subagent's EFFECTIVE tool profile
    (v6.57.0, 1.6): shell yes/no, writable roots, and model lane — derived from the
    _POLICY matrix (the same SSOT active_tool_profile resolves), so the parent sees
    at schedule time (and the child sees first line of its context) what the child
    CAN and CANNOT do. Prevents the wasted rounds where a prober child hit
    workspace_blocked on run_script because neither side knew shell was off.
    The second line names what the child can READ (incl. its parent's and root's
    task files, never a sibling's) and which roots stay invisible to it."""
    matrix = _POLICY.get(_effective_policy_profile(profile), {})
    shell_roots = sorted(root for root, ops in matrix.items() if "shell" in ops)
    write_roots = sorted(root for root, ops in matrix.items() if ops & {"write", "edit"})
    read_roots = sorted(root for root, ops in matrix.items() if "read" in ops)
    has_shell = bool(shell_roots)
    bits = [
        f"profile={profile}",
        f"shell={'yes (' + ', '.join(shell_roots) + ')' if has_shell else 'no'}",
        f"writable={', '.join(write_roots) if write_roots else 'none (read-only)'}",
    ]
    lane = str(effective_lane or "").strip()
    if lane:
        bits.append(f"model_lane={lane}")
    lineage = (" (task_drive/artifact_store: its own, its parent's, its root task's and, when its"
               " contract names one, its predecessor's files, never a sibling's)" if "task_drive" in read_roots else "")
    return (
        "child capabilities — " + " · ".join(bits)
        + f"\nreadable={', '.join(read_roots) or 'none'}{lineage}"
        + f" · unreadable={', '.join(sorted(_ALL_ROOTS - set(read_roots))) or 'none'}"
    )


def lineage_task_ids(ctx: Any) -> tuple[str, ...]:
    """Task ids whose ``task_drive``/``artifact_store`` this actor may READ: its own, its parent's
    and its root's (own lineage fields, T4=A #1105) and the ONE predecessor its contract names
    (``task_contract.predecessor_authority.source.task_id``, one hop, #1232; a child carrying the
    envelope reads it too) — never a sibling's, nothing found by walking the disk, malformed ids dropped."""
    meta, contract = (v if isinstance(v, dict) else {} for v in (
        getattr(ctx, "task_metadata", None), getattr(ctx, "task_contract", None)))
    authority = (contract or meta).get("predecessor_authority")
    source = authority.get("source") if isinstance(authority, dict) else None
    ids = [task_id_for_artifacts(ctx)]
    for raw in (meta.get("parent_task_id"), meta.get("root_task_id"),
                source.get("task_id") if isinstance(source, dict) else None):
        try:
            ids.append(validate_task_id(raw))
        except ValueError:
            continue
    return tuple(dict.fromkeys(ids))


def _task_root_drives(ctx: Any) -> list[pathlib.Path]:
    """The data drives a task's own task roots are enumerated on."""
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    drives: list[pathlib.Path] = []
    for raw in (getattr(ctx, "drive_root", ""), *(meta.get(key) for key in (
            "drive_root", "child_drive_root", "headless_child_drive_root"))):
        if not raw:
            continue
        drive = pathlib.Path(raw).resolve(strict=False)
        if drive not in drives:
            drives.append(drive)
    return drives


def lineage_read_base(ctx: Any, root: ResourceRoot, target: pathlib.Path) -> pathlib.Path | None:
    """The lineage ``task_drive``/``artifact_store`` base containing ``target``, or None:
    ``lineage_task_ids`` on the canonical data root and the task's own drives (where a parent's
    files live while the child runs elsewhere), and each id on ITS OWN headless drive (#1260),
    never through a symlinked headless root. Physical containment only; the caller keeps the READ-only gate."""
    if root not in {"task_drive", "artifact_store"} or not hasattr(ctx, "drive_root"):
        return None
    candidate, canonical = pathlib.Path(target).resolve(strict=False), canonical_data_root(ctx)
    drives = [canonical] + [drive for drive in _task_root_drives(ctx) if drive != canonical]
    task_ids = lineage_task_ids(ctx)
    pairs = [(drive, task_id, False) for drive in drives for task_id in task_ids]
    pairs += [(task_state_dir(canonical, task_id) / "data", task_id, True) for task_id in task_ids]
    for drive, task_id, headless in pairs:
        lexical = (drive / "task_drives" / task_id if root == "task_drive"
                   else task_artifact_dir_path(drive, task_id, create=False))
        base = lexical.resolve(strict=False)
        if path_is_relative_to(candidate, base) and (base == lexical or not headless):
            return base
    return None


def _effective_policy_profile(profile: ToolProfile) -> ToolProfile:
    """Map an acting child to the existing full matrix only in Cyber Pro."""
    if profile != "acting_subagent":
        return profile
    try:
        from ouroboros.config import get_runtime_mode
        from ouroboros.runtime_mode_policy import runtime_mode_at_least

        if runtime_mode_at_least(get_runtime_mode(), "cyber_pro"):
            return "operator_control"
    except Exception:
        pass
    return profile


def operation_roots(profile: ToolProfile, operation: Operation) -> str:
    """The roots ``profile`` may ``operation`` — the one vocabulary every refusal names."""
    matrix = _POLICY.get(_effective_policy_profile(profile), {})
    return ", ".join(sorted(root for root, ops in matrix.items() if operation in ops)) or "(none)"


def decide_tool_access(
    *,
    profile: ToolProfile,
    root: ResourceRoot,
    operation: Operation,
) -> ToolAccessDecision:
    effective_profile = _effective_policy_profile(profile)
    allowed = operation in _POLICY.get(effective_profile, {}).get(root, set())
    if allowed:
        return ToolAccessDecision(True, guard=f"{effective_profile}:{root}:{operation}")
    return ToolAccessDecision(
        False,
        reason=f"profile={effective_profile} cannot {operation} root={root}. "
        f"Roots your profile can {operation}: {operation_roots(profile, operation)}.",
        guard=f"{effective_profile}:{root}:{operation}",
    )


def subagent_profile_satisfies(profile: ToolProfile, needs: Iterable[str]) -> tuple[bool, list[str]]:
    """Return whether a tool profile can satisfy each declared schedule-time need.

    The caller supplies a closed-enum list from the schedule_subagent schema. This
    function does no prose inference: it maps each declared need to the existing
    Tool API operation matrix and checks whether the profile can perform that
    operation on at least one root.
    """

    ops_by_root = _POLICY.get(_effective_policy_profile(profile), {})
    available_ops = {op for ops in ops_by_root.values() for op in ops}
    missing: list[str] = []
    for need in needs or []:
        normalized = str(need or "").strip().lower()
        if normalized == "delegate" and profile in {
            "local_readonly_subagent",
            "acting_subagent",
            "workspace_task",
            "external_workspace_task",
            "self_modification",
            "operator_control",
        }:
            continue
        if normalized == "vcs" and profile in {
            "local_readonly_subagent",
            "acting_subagent",
            "workspace_task",
            "external_workspace_task",
            "self_modification",
            "operator_control",
        }:
            continue
        op = _SUBAGENT_CAPABILITY_TO_OPERATION.get(normalized)
        if not op or op not in available_ops:
            missing.append(normalized or str(need))
    return (not missing, missing)


def _process_root_candidates(
    ctx: Any,
    operation: Operation,
    *,
    bucket: str = "",
    skill_name: str = "",
    include_skill: bool = False,
    require_active: bool = True,
) -> list[tuple[ResourceRoot, pathlib.Path, str, str]]:
    """Return side-effect-free ``(root, base, source, skill)`` candidates."""

    profile = active_tool_profile(ctx)
    candidates: list[tuple[ResourceRoot, pathlib.Path, str, str]] = []
    try:
        active = resource_root_path(ctx, "active_workspace")
    except ValueError:
        if require_active:
            raise
    else:
        candidates.append(("active_workspace", active, "active_workspace", ""))
    candidates += [
        ("system_repo", resource_root_path(ctx, "system_repo"), "system_repo", ""),
    ]

    def _add_task_roots(drive: pathlib.Path) -> None:
        task_id = task_id_for_artifacts(ctx)
        candidates.extend([
            ("task_drive", drive / "task_drives" / task_id, "task_drive", ""),
            ("artifact_store", task_artifact_dir_path(drive, task_id, create=False), "artifact_store", ""),
        ])

    if hasattr(ctx, "drive_root"):
        for drive in _task_root_drives(ctx):
            _add_task_roots(drive)
        candidates.append(("user_files", resource_root_path(ctx, "user_files"), "user_files", ""))
    if include_skill:
        base, source, selected_name = _skill_payload_base(
            ctx,
            profile=profile,
            operation=operation,
            location=bucket,
            skill_name=skill_name,
        )
        candidates.append(("skill_payload", base, source, selected_name))
    return [
        (label, pathlib.Path(root).resolve(strict=False), source, selected_name)
        for label, root, source, selected_name in candidates
        if decide_tool_access(profile=profile, root=label, operation=operation).allow
    ]


def _side_effect_free_process_roots(
    ctx: Any,
    operation: Operation,
    *,
    bucket: str = "",
    skill_name: str = "",
    include_skill: bool = False,
) -> list[tuple[str, pathlib.Path]]:
    """Project the shared process inventory without materializing any root."""
    return [
        (label, root)
        for label, root, _source, _name in _process_root_candidates(
            ctx, operation, bucket=bucket, skill_name=skill_name,
            include_skill=include_skill,
        )
    ]


def filesystem_affordance_map(ctx: Any, *, runtime_mode: str = "") -> dict[str, Any]:
    """A compact, side-effect-free projection of filesystem/tool access affordances.

    This is context for the LLM, not a new policy layer. Every fact is derived
    from the Tool API v2 matrix and git-shell policy constants so the model can
    plan inside the same envelope that the dispatcher later enforces.
    """

    profile = active_tool_profile(ctx)
    policy = _POLICY.get(_effective_policy_profile(profile), {})
    # H2 (capinv-447): a root is writable iff a MUTATING operation is granted on
    # it. Grouping "vcs" as write-like claimed writable roots for the read-only
    # child profile (status/diff-only vcs), contradicting summarize_subagent_profile.
    writable_roots = sorted(root for root, ops in policy.items() if ops & _WRITE_LIKE_OPS)
    readonly_roots = sorted(
        root for root, ops in policy.items()
        if ops and not (ops & _WRITE_LIKE_OPS)
    )
    shell_roots = _side_effect_free_process_roots(ctx, "shell")
    service_roots = _side_effect_free_process_roots(ctx, "service")
    try:
        from ouroboros.git_shell_policy import GIT_READONLY_SUBCOMMANDS

        git_readonly_subcommands = sorted(GIT_READONLY_SUBCOMMANDS)
    except Exception:
        git_readonly_subcommands = []
    light_gated_roots: list[str] = []
    if str(runtime_mode or "").strip().lower() == "light":
        for root in ("active_workspace", "system_repo"):
            if root in policy:
                light_gated_roots.append(root)
    # Skill payload needs selectors; every other visible root is projected fail-soft.
    root_paths: dict[str, str] = {}
    for _root_label in sorted(policy):
        if _root_label == "skill_payload":
            continue
        try:
            root_paths[_root_label] = str(resource_root_path(ctx, _root_label))
        except Exception:
            continue
    result = {
        "profile": profile,
        "writable_roots": writable_roots,
        "readonly_roots": readonly_roots,
        "searchable_roots": sorted(root for root, ops in policy.items() if "search" in ops),
        # Environment fact, not another policy gate.
        "invisible_roots": sorted(_ALL_ROOTS - set(policy)),
        "root_paths": root_paths,
        "default_shell_cwd": shell_roots[0][0] if shell_roots else "",
        "allowed_shell_cwd_roots": [label for label, _root in shell_roots],
        "default_service_cwd": service_roots[0][0] if service_roots else "",
        "allowed_service_cwd_roots": [label for label, _root in service_roots],
        "git_readonly_subcommands": git_readonly_subcommands,
        "light_gated_roots": sorted(light_gated_roots),
    }
    if "active_workspace" in policy:
        result["default_root"] = "active_workspace"
    if "skill_payload" in policy:
        result["skill_payload_selector"] = (
            "root=skill_payload requires bucket + skill_name"
        )
        from ouroboros.skill_payload_binding import selected_manifestless_user_repo_name

        if selected := selected_manifestless_user_repo_name(
            ctx,
            canonical_data_root(ctx),
        ):
            result["skill_payload_selector"] += (
                "; omit bucket only for the exact selected manifestless "
                f"user_repo target {selected}"
            )
    _room = project_room_lens_dir(ctx)
    if _room is not None:
        # In this chat the project room is the active/default filesystem focus.
        result["project_room_dir"] = str(_room)
        result["default_shell_cwd"] = f"project room ({_room})"
    return result


def profile_readable_root_paths(ctx: Any, *, operation: Operation = "read") -> list[tuple[str, pathlib.Path]]:
    """Project the ``(label, path)`` pairs this profile may ``operation`` from the
    policy SSOT. ``skill_payload`` needs selectors and is omitted; individual
    resolution is fail-soft so one unavailable root cannot hide the rest."""
    out: list[tuple[str, pathlib.Path]] = []
    try:
        policy = _POLICY.get(_effective_policy_profile(active_tool_profile(ctx)), {})
    except Exception:
        return out
    for root, ops in sorted(policy.items()):
        if operation not in ops or root == "skill_payload":
            continue
        try:
            out.append((root, pathlib.Path(resource_root_path(ctx, root)).resolve(strict=False)))
        except Exception:
            continue
    return out


def shell_cwd_block_message(ctx: Any, cwd: str = "", *, operation: Operation = "shell", error: Exception | None = None) -> str:
    """Actionable fail-closed message for process cwd resolution failures."""

    try:
        allowed = _side_effect_free_process_roots(ctx, operation)
        # Show resolved label=path pairs so the caller can self-correct.
        seen: set[str] = set()
        allowed_entries: list[str] = []
        for label, root in allowed:
            entry = f"{label}={root}"
            if entry not in seen:
                seen.add(entry)
                allowed_entries.append(entry)
    except Exception:
        allowed_entries = []
    hint = (
        "Allowed cwd roots for this tool/profile: " + ", ".join(allowed_entries)
        if allowed_entries else
        "No process cwd root is available to this tool/profile."
    )
    detail = f" ({type(error).__name__}: {error})" if error is not None else ""
    return (
        f"⚠️ SHELL_CWD_BLOCKED: CWD_BLOCKED: cwd {str(cwd or '.')} is outside allowed roots for {operation}{detail}. "
        f"{hint}. Use one of those exact paths as cwd (or root=task_drive/artifact_store/user_files in file tools)."
    )


def light_cognitive_or_root_redirect(tool_name: str, args: dict[str, Any]) -> str | None:
    """Precise light-mode redirect for write attempts that should use a cognitive
    tool or an explicit ``user_files`` root. Returns the message, or ``None``.

    Only ``write_file``/``edit_text`` qualify. Callers invoke this inside the
    light-mode repo-mutation block so a returned message replaces the generic
    LIGHT_MODE_BLOCKED with actionable, non-noisy guidance.
    """
    if tool_name not in ("write_file", "edit_text"):
        return None
    paths: list[str] = []
    primary = str(args.get("path", "") or "")
    if primary:
        paths.append(primary)
    for entry in args.get("files") or []:
        if isinstance(entry, dict) and entry.get("path"):
            paths.append(str(entry.get("path")))
    raw_root = str(args.get("root", "") or "active_workspace")
    try:
        root = normalize_root(raw_root)
    except Exception:
        root = "active_workspace"

    if root == "runtime_data":
        for path_text in paths:
            # Logical resource-path components. Normalize Windows separators to the
            # POSIX convention these tool paths use, then compare parts (not raw
            # separators), so both memory/identity.md and memory\identity.md match.
            parts = pathlib.PurePosixPath(str(path_text or "").replace("\\", "/")).parts
            if len(parts) >= 2 and parts[0].lower() == "memory":
                area = parts[1].lower()
                if area.startswith("identity") or area.startswith("scratchpad") or area == "knowledge":
                    return (
                        "⚠️ COGNITIVE_TOOL_REQUIRED: cognitive memory is not written via "
                        f"{tool_name!r}. Use the dedicated first-class tools (always available in "
                        "light mode): update_identity for memory/identity.md, update_scratchpad for "
                        "memory/scratchpad.md, knowledge_write for memory/knowledge/<topic>.md. They "
                        "apply the correct structure (journaling, timestamped blocks, index "
                        "maintenance). Read the current state before writing (Bible P12)."
                    )

    if root == "active_workspace":
        for path_text in paths:
            # Use pathlib semantics (no hardcoded separators): an expanded path
            # that is absolute and under the owner home should use root=user_files.
            # This is cross-platform (POSIX `/`, `~`, and Windows drive paths).
            try:
                candidate = pathlib.Path(path_text).expanduser()
                if not candidate.is_absolute():
                    continue
                candidate.resolve(strict=False).relative_to(_user_files_root())
            except (ValueError, OSError, RuntimeError):
                continue
            return (
                "⚠️ ROOT_REQUIRED_USER_FILES: an absolute home path "
                f"({path_text!r}) was given but root defaulted to 'active_workspace'. "
                "Pass root='user_files' to write under the owner's home, e.g. "
                "write_file(root='user_files', path='Desktop/file.html', content=...)."
            )
    return None


def _select_process_target(
    ctx: Any,
    cwd: str,
    operation: Operation,
    *,
    bucket: str = "",
    skill_name: str = "",
    materialize: bool,
) -> tuple[ResourceRoot, pathlib.Path, pathlib.Path, str, str, list[tuple[str, pathlib.Path]]]:
    """Select one process target from the shared candidate inventory."""
    profile = active_tool_profile(ctx)
    text = str(cwd or "").strip()
    normalized = text.replace("\\", "/")
    reserved_root = ""
    reserved_subdir = ""
    if normalized and not is_absolute_path_text(text) and not text.startswith("~"):
        head, separator, tail = normalized.partition("/")
        if head in _ALL_ROOTS:
            reserved_root, reserved_subdir = head, tail if separator else ""
    decision = decide_tool_access(
        profile=profile, root=reserved_root, operation=operation,  # type: ignore[arg-type]
    ) if reserved_root else None
    if decision is not None and not decision.allow:
        raise ValueError(decision.reason)
    include_skill = reserved_root == "skill_payload"
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    if include_skill and constraint and constraint.has_selected_skill:
        from ouroboros.contracts.skill_payload_policy import constraint_bucket_skill

        selected_bucket, selected_skill = constraint_bucket_skill(constraint)
        bucket = bucket or selected_bucket
        skill_name = skill_name or selected_skill
    if include_skill and (not str(bucket or "").strip() or not str(skill_name or "").strip()):
        raise ValueError("cwd=skill_payload[/subdir] requires bucket and skill_name")
    candidate_records = _process_root_candidates(
        ctx, operation, bucket=bucket, skill_name=skill_name, include_skill=include_skill,
        require_active=(reserved_root == "active_workspace" or (
            not reserved_root and not is_absolute_path_text(text) and not text.startswith("~")
        )),
    )
    allowed = [(label, root) for label, root, _source, _name in candidate_records]
    if not candidate_records:
        raise ValueError(f"profile={profile} cannot {operation} any process cwd root")
    def _finish(
        record: tuple[ResourceRoot, pathlib.Path, str, str], target: pathlib.Path,
        *, scoped_allowed: list[tuple[str, pathlib.Path]] | None = None,
    ) -> tuple[ResourceRoot, pathlib.Path, pathlib.Path, str, str, list[tuple[str, pathlib.Path]]]:
        label, base, source, selected_name = record
        selected = pathlib.Path(target).resolve(strict=False)
        if label == "user_files":
            reason = user_files_path_block_reason(ctx, selected)
            if reason:
                raise ValueError(reason)
        if materialize and label in {"task_drive", "artifact_store"}:
            try:
                selected.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"could not create {label} cwd {selected}: {exc}") from exc
        return label, base, selected, source, selected_name, scoped_allowed or allowed
    if not text or normalized in {".", "./"}:
        first = candidate_records[0]
        return _finish(first, first[1])
    if reserved_root:
        for record in candidate_records:
            label, base, _source, _name = record
            if label != reserved_root:
                continue
            if label == "user_files" and not reserved_subdir:
                deliverables = _deliverables_root()
                if materialize:
                    deliverables.mkdir(parents=True, exist_ok=True)
                return _finish(record, deliverables)
            target = (base / safe_relpath(reserved_subdir or ".")).resolve(strict=False)
            if not path_is_relative_to(target, base):
                raise ValueError(f"cwd escapes {label}")
            return _finish(record, target)
        raise ValueError(f"profile={profile} cannot {operation} root={reserved_root}")
    raw = pathlib.Path(text).expanduser()
    physical_candidates = (
        [raw.resolve(strict=False)]
        if is_absolute_path_text(text) or text.startswith("~") else
        [(root / safe_relpath(text)).resolve(strict=False)
         for _label, root, _source, _name in candidate_records]
    )
    for target in physical_candidates:
        for record in candidate_records:
            label, base, _source, _name = record
            if path_is_relative_to(target, base):
                try:
                    return _finish(record, target)
                except ValueError:
                    continue
    # External-workspace host scratch remains an exact user-files path, not `/`.
    if is_external_workspace(ctx) and decide_tool_access(
        profile=profile, root="user_files", operation=operation,
    ).allow:
        for target in physical_candidates:
            if not target.is_absolute() or user_files_path_block_reason(ctx, target):
                continue
            record = ("user_files", target, "user_files", "")
            return _finish(record, target, scoped_allowed=[*allowed, ("user_files", target)])
    raise ValueError("cwd is outside allowed roots")


def resolve_shell_cwd(
    ctx: Any,
    cwd: str = "",
    *,
    operation: Operation = "shell",
    bucket: str = "",
    skill_name: str = "",
) -> tuple[pathlib.Path, str, list[tuple[str, pathlib.Path]]]:
    """Compatibility projection of the process target selector."""
    root, _base, target, _source, _selected_name, allowed = _select_process_target(
        ctx, cwd, operation, bucket=bucket, skill_name=skill_name, materialize=True,
    )
    return target, root, allowed


def canonical_repo_relative_path(ctx: Any, root: str, path: str) -> str:
    """Normalize repo paths so guards and mutations judge the same target."""
    if root not in {"active_workspace", "system_repo"}:
        return path
    try:
        base = resource_root_path(ctx, root)  # type: ignore[arg-type]
    except Exception:
        return path
    try:
        return normalize_root_relative(base, path)
    except Exception:
        return path


def _resolve_target_in_selected_base(
    ctx: Any,
    *,
    root: ResourceRoot,
    base_path: pathlib.Path,
    path: str,
    operation: Operation,
) -> pathlib.Path:
    """Resolve one target inside an already-selected physical base."""

    if root == "user_files":
        return resolve_user_file_path(
            ctx,
            path,
            allow_protected_descendants=operation in {"list", "search"},
            operation=operation,
        )
    resolved_base = pathlib.Path(base_path).resolve(strict=False)
    path_text = str(path or ".")
    if root == "runtime_data":
        path_text = normalize_runtime_data_path(resolved_base, path_text)
    elif root in {"active_workspace", "system_repo"}:
        path_text = normalize_root_relative(resolved_base, path_text)
    if root == "active_workspace" and is_absolute_path_text(path_text):
        # An admitted Docker workspace has a second, explicit address space.
        # Map it through the executor owner, then apply the same confinement.
        from ouroboros.workspace_executor import executor_ref_from_ctx, map_backend_path

        executor = executor_ref_from_ctx(ctx)
        if executor is not None and executor.kind == "docker_exec":
            path_text = str(map_backend_path(executor, path_text))
    if is_absolute_path_text(path_text):
        candidate = pathlib.Path(path_text).expanduser()
        # A foreign Windows absolute spelling has no native anchor on POSIX;
        # joining it to cwd would invent a different physical file.
        if candidate.anchor:
            candidate = (resolved_base / candidate).resolve(strict=False)
            try:
                path_text = candidate.relative_to(resolved_base).as_posix()
            except ValueError:
                if operation in _READ_OPS:
                    # Cross-prefix READS: the lineage rule (own id on every drive, the
                    # parent's and the root's task files), then the ORPHAN capture rule,
                    # which the custody authority alone answers.
                    if lineage_read_base(ctx, root, candidate) is not None:
                        return candidate
                    if root == "artifact_store":
                        from ouroboros.delegate_shared import orphan_capture_read_target

                        anchored = orphan_capture_read_target(ctx, candidate)
                        if anchored is not None:
                            return anchored
        if is_absolute_path_text(path_text):
            raise ValueError(
                f"absolute path {path!r} is outside selected root={root} ({resolved_base}). "
                f"Roots your profile can {operation}: "
                f"{operation_roots(active_tool_profile(ctx), operation)}."
            )
    if root == "artifact_store" and operation in _READ_OPS:
        # C1 delegated captures (CR1-2): written on the CANONICAL drive, read from
        # a CHILD drive_root — re-anchor (see `delegated_capture_read_target`).
        anchored = delegated_capture_read_target(
            canonical_data_root(ctx), task_id_for_artifacts(ctx),
            safe_relpath(path_text), resolved_base)
        if anchored is not None:
            return anchored
    resolved = (resolved_base / safe_relpath(path_text)).resolve(strict=False)
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"path escapes {resolved_base}") from exc
    return resolved


def build_resolved_resource_binding(
    ctx: Any,
    *,
    root: str | None = None,
    operation: Operation,
    path: str = ".",
    bucket: str = "",
    skill_name: str = "",
    process_cwd: str | None = None,
) -> ResolvedResourceBinding:
    """Resolve policy, physical base, and target exactly once for one call."""

    if process_cwd is not None:
        selected_root, base, target, source, selected_name, _allowed = _select_process_target(
            ctx,
            process_cwd,
            operation,
            bucket=bucket,
            skill_name=skill_name,
            materialize=True,
        )
        return ResolvedResourceBinding(
            profile=active_tool_profile(ctx),
            root=selected_root,
            operation=operation,
            base_path=pathlib.Path(base).resolve(strict=False),
            target_path=pathlib.Path(target).resolve(strict=False),
            source=source,
            skill_name=selected_name,
            state_drive_root=canonical_data_root(ctx),
        )

    normalized = normalize_root(root)
    profile = active_tool_profile(ctx)
    decision = decide_tool_access(profile=profile, root=normalized, operation=operation)
    if not decision.allow:
        raise ValueError(decision.reason)

    # Migration-additive legacy: edit_text outside a project workspace and explicit
    # runtime_data skill paths historically accepted canonical data-bucket spellings.
    # Resolve those forms here so guard and mutator still share one frozen target.
    workspace_active = False
    try:
        workspace_active = bool(getattr(ctx, "is_workspace_mode")())
    except (AttributeError, TypeError):
        pass
    legacy_data_form = (
        normalized == "runtime_data" and operation in {"write", "edit"}
    ) or (
        normalized == "active_workspace" and operation == "edit" and not workspace_active
        and project_room_lens_dir(ctx) is None
    )
    if legacy_data_form:
        from ouroboros.contracts.skill_payload_policy import (
            SkillPayloadPathError,
            resolve_skill_payload_target,
        )

        legacy_state_root = canonical_data_root(ctx)
        try:
            normalization_root = (
                resource_root_path(ctx, "runtime_data")
                if normalized == "runtime_data"
                else legacy_state_root
            )
            legacy_path = normalize_runtime_data_path(normalization_root, path)
            legacy = resolve_skill_payload_target(legacy_state_root, legacy_path)
        except SkillPayloadPathError:
            legacy = None
        if legacy is not None:
            base, legacy_source, legacy_name = _skill_payload_base(
                ctx,
                profile=profile,
                operation=operation,
                location=legacy.bucket,
                skill_name=legacy.skill,
                allow_missing=operation == "write",
            )
            target = _resolve_target_in_selected_base(
                ctx,
                root="skill_payload",
                base_path=base,
                path=legacy.rel_path,
                operation=operation,
            )
            return ResolvedResourceBinding(
                profile=profile,
                root=normalized,
                operation=operation,
                base_path=base,
                target_path=target,
                source=legacy_source,
                skill_name=legacy_name,
                state_drive_root=legacy_state_root,
            )

    source = str(normalized)
    selected_name = ""
    room = project_room_lens_dir(ctx) if normalized == "active_workspace" else None
    if normalized == "skill_payload":
        selected_bucket = str(bucket or "").strip()
        selected_skill = str(skill_name or "").strip()
        constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
        if constraint and constraint.has_selected_skill:
            from ouroboros.contracts.skill_payload_policy import constraint_bucket_skill

            expected_bucket, expected_skill = constraint_bucket_skill(constraint)
            if (
                not selected_bucket
                and (not selected_skill or selected_skill == expected_skill)
            ):
                selected_bucket = expected_bucket
                selected_skill = selected_skill or expected_skill
            # Physical identity is checked by resolve_skill_payload_base for
            # file, legacy runtime-data and process selectors alike; logical
            # external/native aliases must not look like a different payload.
        from ouroboros.contracts.skill_payload_policy import _is_skill_create_signal

        base, source, selected_name = _skill_payload_base(
            ctx,
            profile=profile,
            operation=operation,
            location=selected_bucket,
            skill_name=selected_skill,
            allow_missing=(
                operation == "write"
                and _is_skill_create_signal(path)
                and selected_bucket.lower() in {"", "external", "user_repo"}
            ),
        )
    elif room is not None:
        base = room
        source = "project_room"
    else:
        base = resource_root_path(ctx, normalized)
    target = _resolve_target_in_selected_base(
        ctx,
        root=normalized,
        base_path=base,
        path=path,
        operation=operation,
    )
    # The physical base follows the container holding the target: an absolute
    # user_files target may land in the configured Deliverables container outside the
    # home (else an exact Presence path-prefix check calls a valid deliverable outside
    # the binding), and a lineage READ lands in the parent's or root's task root.
    logical_base_path = None
    container = None
    if normalized == "user_files":
        try:
            deliverables = _deliverables_root()
            if path_is_relative_to(target, deliverables) or _path_is_relative_to_casefold(target, deliverables):
                container = deliverables
        except (OSError, TypeError, ValueError, RuntimeError):
            pass
    elif operation in _READ_OPS and not path_is_relative_to(target, base):
        container = lineage_read_base(ctx, normalized, target)
    if container is not None:
        logical_base_path = pathlib.Path(base).resolve(strict=False)
        base = container
    return ResolvedResourceBinding(
        profile=profile,
        root=normalized,
        operation=operation,
        base_path=pathlib.Path(base).resolve(strict=False),
        target_path=target,
        source=source,
        skill_name=selected_name,
        state_drive_root=canonical_data_root(ctx),
        logical_base_path=logical_base_path,
    )
def resolve_resource_path(
    ctx: Any,
    *,
    root: ResourceRoot,
    path: str,
    bucket: str = "",
    skill_name: str = "",
) -> pathlib.Path:
    base = resource_root_path(ctx, root, bucket=bucket, skill_name=skill_name)
    return _resolve_target_in_selected_base(
        ctx,
        root=root,
        base_path=base,
        path=path,
        operation="read",
    )
