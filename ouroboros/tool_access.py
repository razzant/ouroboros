"""Tool API v2 access matrix.

This is the single policy shape for LLM-visible tools: a profile asks to run an
operation against a resource root and receives an allow/block decision. The
legacy per-tool checks still provide defense-in-depth while the public API is
migrated to neutral tool names.
"""

from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

from ouroboros.artifacts import (delegated_capture_read_target,
                                 task_artifact_dir_path, task_id_for_artifacts)
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE
from ouroboros.contracts.task_constraint import VALID_WRITE_SURFACES, normalize_task_constraint
from ouroboros.shell_parse import is_absolute_path_text
from ouroboros.utils import safe_relpath


def _user_files_root() -> pathlib.Path:
    """Filesystem base for the ``user_files`` resource root.

    Defaults to the owner's real home. A jailed/benchmark runtime can redirect it
    to a scratch directory via ``OUROBOROS_USER_FILES_ROOT`` so a task physically
    cannot resolve the owner's real home (e.g. ``~/file1.txt`` secret files). Any
    unusable value falls back to the real home — fail-safe, never broadens reach.
    """
    raw = (os.environ.get("OUROBOROS_USER_FILES_ROOT") or "").strip()
    if raw:
        try:
            return pathlib.Path(raw).expanduser().resolve(strict=False)
        except Exception:
            # ANY unusable value (bad path, unknown ``~user`` RuntimeError, odd OS error)
            # fails safe to the real home — the doc's "any unusable value" contract.
            pass
    return pathlib.Path.home().resolve(strict=False)


def _deliverables_root() -> pathlib.Path:
    """Container for UNNAMED user deliverables, JAIL-AWARE: when the user_files home is
    redirected (``OUROBOROS_USER_FILES_ROOT``) and no explicit
    ``OUROBOROS_DELIVERABLES_ROOT`` is set, keep unnamed deliverables INSIDE the jail so a
    bare ``write_file(root='user_files', path='answer.txt')`` stays reachable and in-bounds
    instead of escaping to the real ``~/Ouroboros/Deliverables`` (which the outside-home
    check would then reject). Otherwise the global config default applies.
    """
    from ouroboros.config import get_deliverables_root

    jail = (os.environ.get("OUROBOROS_USER_FILES_ROOT") or "").strip()
    explicit = (os.environ.get("OUROBOROS_DELIVERABLES_ROOT") or "").strip()
    if explicit:
        return pathlib.Path(explicit).expanduser().resolve(strict=False)
    if jail and not explicit:
        return (_user_files_root() / "Deliverables").resolve(strict=False)
    return pathlib.Path(get_deliverables_root()).expanduser().resolve(strict=False)


ToolProfile = Literal[
    "self_modification",
    "workspace_task",
    "external_workspace_task",
    "acting_subagent",
    "skill_repair",
    "local_readonly_subagent",
    "operator_control",
]
ResourceRoot = Literal[
    "active_workspace",
    "system_repo",
    "runtime_data",
    "task_drive",
    "skill_payload",
    "artifact_store",
    "user_files",
    "subagent_projects",
    "deliverables",
]
Operation = Literal[
    "read",
    "list",
    "search",
    "write",
    "edit",
    "shell",
    "vcs",
    "review",
    "delegate",
    "service",
]
SubagentCapability = Literal[
    "write",
    "edit",
    "shell",
    "vcs",
    "review",
    "delegate",
    "service",
]


@dataclass(frozen=True)
class ToolAccessDecision:
    allow: bool
    reason: str = ""
    guard: str = ""


@dataclass(frozen=True)
class ResolvedResourceBinding:
    """One dispatch-selected logical root and its exact physical target."""

    profile: ToolProfile
    root: ResourceRoot
    operation: Operation
    base_path: pathlib.Path
    target_path: pathlib.Path
    source: str
    skill_name: str
    state_drive_root: pathlib.Path


_ALL_ROOTS: frozenset[str] = frozenset({
    "active_workspace",
    "system_repo",
    "runtime_data",
    "task_drive",
    "skill_payload",
    "artifact_store",
    "user_files",
    "subagent_projects",
    "deliverables",
})

# Deferral 1: orchestrator-visible READ-ONLY roots — durable subagent (genesis) projects
# and the unnamed-deliverables container. Only ever granted {read,list,search}; NEVER
# write/edit/shell/vcs (no mutation, no shell-cwd — deliberately absent from
# resolve_shell_cwd candidates) and NEVER to acting/readonly subagents (a child must not
# read sibling projects). operator_control is capped to read-only on these too.
_READONLY_RESOURCE_ROOTS: frozenset[str] = frozenset({"subagent_projects", "deliverables"})
_TOP_LEVEL_PRINCIPAL_PROFILES: frozenset[str] = frozenset({
    "workspace_task",
    "external_workspace_task",
    "self_modification",
})

_READ_OPS = frozenset({"read", "list", "search"})
_USER_FILES_SECRET_COMPONENTS = frozenset({
    ".aws",
    ".azure",
    ".config",
    ".docker",
    ".git",   # v6.52.0: VCS internals hold config + stored credentials
    ".gnupg",
    ".hg",
    ".kube",
    ".local",
    ".netrc",
    ".ssh",
    ".svn",
    "library",
})
_USER_FILES_SECRET_NAMES = frozenset({
    ".env",
    # v6.52.0: credential / shell-init / history dotFILES kept blocked AFTER the bare
    # `startswith('.')` block was dropped (so benign project dotdirs are readable while
    # secret-bearing dotfiles are not).
    ".bash_history",
    ".bash_profile",
    ".bashrc",
    ".dockercfg",
    ".git-credentials",
    ".gitconfig",
    ".htpasswd",
    ".npmrc",
    ".pgpass",
    ".profile",
    ".pypirc",
    ".python_history",
    ".zsh_history",
    ".zprofile",
    ".zshrc",
    "auth.json",
    "credentials",
    "credentials.json",
    "secrets.json",
    "settings.json",
    "token.json",
    "tokens.json",
})
_USER_FILES_SECRET_RE = re.compile(r"(?:^|[._-])(api[_-]?key|credential|password|secret|token)(?:[._-]|$)", re.I)
# v6.52.0 (P1): a SMALL allowlist of benign hidden (dot) project components. The dotfile guard
# is DEFAULT-DENY: a credential blocklist can never be exhaustive (e.g. ~/.terraform.d,
# ~/.cargo/credentials.toml, ~/.oci/config, ~/.pip/pip.conf, ~/.m2/settings.xml, ~/.*_history all
# leak under enumeration), so a dotted component is blocked UNLESS it is one of these known-safe
# project-config dirs/files. This serves the goal (read .github/.vscode/.idea project config)
# without opening the whole in-home dotfile space.
_USER_FILES_ALLOWED_DOTNAMES = frozenset({
    ".github",
    ".gitlab",
    ".circleci",
    ".devcontainer",
    ".vscode",
    ".idea",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    ".dockerignore",
    ".editorconfig",
})

_TOP_LEVEL_PRINCIPAL_POLICY: dict[str, set[str]] = {
    "active_workspace": {"read", "list", "search", "write", "edit", "shell", "vcs", "review", "service"},
    "system_repo": {"read", "list", "search", "write", "edit", "shell", "vcs", "review", "service"},
    "runtime_data": {"read", "list", "search", "write", "edit"},
    "task_drive": {"read", "list", "write", "edit", "shell", "service"},
    "skill_payload": {"read", "list", "search", "write", "edit", "review", "shell"},
    "artifact_store": {"read", "list", "write", "shell", "service"},
    "user_files": {"read", "list", "search", "write", "edit", "shell", "service"},
    "subagent_projects": {"read", "list", "search"},
    "deliverables": {"read", "list", "search"},
}


_POLICY: dict[str, dict[str, set[str]]] = {
    "local_readonly_subagent": {
        # Read-only child VCS names still need their target binding to resolve.
        "active_workspace": set(_READ_OPS) | {"vcs"},
        "system_repo": set(_READ_OPS) | {"vcs"},
        "runtime_data": {"read", "list"},
        "task_drive": {"read", "list"},
        "artifact_store": {"read", "list"},
        # v6.70.0 (owner-approved): read-only scouts sent to review a skill were
        # structurally blind to its payload — a scout literally reported
        # "reviewing blind", and a correct "skill does not exist" answer was
        # indistinguishable from an access block. Payloads are skill CODE
        # (data/skills/...); grants/secrets live in data/state/skills, which
        # stays invisible to this profile.
        "skill_payload": {"read", "list", "search"},
    },
    "skill_repair": {
        "skill_payload": {"read", "list", "search", "write", "edit", "review"},
        "runtime_data": {"read", "list"},
        "task_drive": {"read", "list"},
        "artifact_store": {"read", "list"},
    },
    # Top-level preset names remain observable, but workspace focus never narrows
    # the ordinary principal. Independent path/credential/child/runtime guards
    # still apply after this shared operation matrix.
    "workspace_task": _TOP_LEVEL_PRINCIPAL_POLICY,
    "external_workspace_task": _TOP_LEVEL_PRINCIPAL_POLICY,
    # Mutative (acting) subagents write only inside their isolated active
    # workspace (self_worktree / external_workspace / genesis). No vcs-commit /
    # review here; the parent integrates and commits. self_worktree additionally
    # keeps protected-path discipline active in the registry (it is the system
    # repo). runtime_data stays read-only.
    "acting_subagent": {
        # Acting children write ONLY inside their isolated surface (active_workspace =
        # the self_worktree / external_workspace / genesis). task_drive / artifact_store
        # are read-only here (no extra write surface); the deliverable is a workspace.patch.
        "active_workspace": {"read", "list", "search", "write", "edit", "shell", "vcs", "service"},
        "runtime_data": {"read", "list"},
        "task_drive": {"read", "list"},
        "artifact_store": {"read", "list"},
    },
    "self_modification": _TOP_LEVEL_PRINCIPAL_POLICY,
    # operator_control gets full authority on every mutable root, but the orchestrator
    # read-only roots stay read-only even here (they are deliverables/durable projects,
    # not a control surface).
    "operator_control": {
        **{root: {"read", "list", "search", "write", "edit", "shell", "vcs", "review", "delegate", "service"}
           for root in _ALL_ROOTS if root not in _READONLY_RESOURCE_ROOTS},
        **{root: {"read", "list", "search"} for root in _READONLY_RESOURCE_ROOTS},
    },
}
_SUBAGENT_CAPABILITY_TO_OPERATION: dict[str, Operation] = {
    "write": "write",
    "edit": "edit",
    "shell": "shell",
    "vcs": "vcs",
    "review": "review",
    "delegate": "delegate",
    "service": "service",
}
SUBAGENT_CAPABILITIES: tuple[str, ...] = tuple(_SUBAGENT_CAPABILITY_TO_OPERATION.keys())


def _is_subagent_ctx(ctx: Any) -> bool:
    """True when the task is a delegated subagent (by lineage metadata)."""
    for attr in ("task_metadata", "task_contract"):
        data = getattr(ctx, attr, None)
        if isinstance(data, dict) and str(data.get("delegation_role") or "").strip() == "subagent":
            return True
    return False


def is_external_workspace(ctx: Any) -> bool:
    """True for an EXTERNAL-workspace top-level task (not the system repo).

    External-workspace tasks operate on a pre-existing working tree somewhere on
    the host (container scratch, a repo cloned under ``/tmp`` or ``/build``,
    etc.). They legitimately read, run commands, and use git OUTSIDE the user
    home, while the Ouroboros runtime (system repo + data drive) and
    credential-like files stay protected by the per-path guards. ``self_worktree``
    and ``genesis`` are acting-subagent SURFACES (``acting_subagent`` profile),
    never this profile, so they keep full home/runtime confinement.
    """
    try:
        if not bool(getattr(ctx, "is_workspace_mode", lambda: False)()):
            return False
    except Exception:
        return False
    return str(getattr(ctx, "workspace_mode", "") or "").strip().lower() == "external"


def active_tool_profile(ctx: Any) -> ToolProfile:
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    mode = str(getattr(constraint, "mode", "") or "").strip()
    if mode == LOCAL_READONLY_SUBAGENT_MODE:
        return "local_readonly_subagent"
    if mode == ACTING_SUBAGENT_MODE:
        # Acting subagents require a resolved write surface; otherwise fail
        # closed to read-only rather than inheriting a broader profile.
        surface = str(getattr(constraint, "surface", "") or "").strip()
        if surface in VALID_WRITE_SURFACES:
            return "acting_subagent"
        return "local_readonly_subagent"
    if mode == "skill_repair":
        return "skill_repair"
    # Fail-closed floor (BIBLE P3), checked BEFORE workspace/direct-chat: a
    # delegated subagent without a valid readonly/acting/skill constraint is
    # read-only and must never inherit workspace_task / operator_control /
    # self_modification. The parent remains the sole local writer/committer.
    if _is_subagent_ctx(ctx):
        return "local_readonly_subagent"
    if bool(getattr(ctx, "is_workspace_mode", lambda: False)()):
        # Keep distinct preset names for focus/path diagnostics. Both use the
        # shared ordinary principal; external host-scratch reach is a path fact.
        if is_external_workspace(ctx):
            return "external_workspace_task"
        return "workspace_task"
    if bool(getattr(ctx, "is_direct_chat", False)):
        return "operator_control"
    return "self_modification"


def predicted_subagent_profile(*, write_surface: str = "") -> ToolProfile:
    """The tool profile a scheduled subagent will resolve to, from schedule-time
    inputs only (v6.57.0, 1.6). A valid write_surface → acting_subagent; otherwise
    a read-only subagent. Mirrors active_tool_profile's subagent branches so the
    parent's schedule result and the child's start context can preview the profile
    without a live ctx. NOT authoritative — the supervisor's _resolve_subagent_
    constraint is the real gate; this is a visibility preview."""
    surface = str(write_surface or "").strip()
    if surface and surface in VALID_WRITE_SURFACES:
        return "acting_subagent"
    return "local_readonly_subagent"


def summarize_subagent_profile(profile: ToolProfile, *, effective_lane: str = "") -> str:
    """Compact, human-readable summary of a subagent's EFFECTIVE tool profile
    (v6.57.0, 1.6): shell yes/no, writable roots, and model lane — derived from the
    _POLICY matrix (the same SSOT active_tool_profile resolves), so the parent sees
    at schedule time (and the child sees first line of its context) what the child
    CAN and CANNOT do. Prevents the wasted rounds where a prober child hit
    workspace_blocked on run_script because neither side knew shell was off."""
    matrix = _POLICY.get(profile, {})
    shell_roots = sorted(root for root, ops in matrix.items() if "shell" in ops)
    write_roots = sorted(root for root, ops in matrix.items() if ops & {"write", "edit"})
    has_shell = bool(shell_roots)
    bits = [
        f"profile={profile}",
        f"shell={'yes (' + ', '.join(shell_roots) + ')' if has_shell else 'no'}",
        f"writable={', '.join(write_roots) if write_roots else 'none (read-only)'}",
    ]
    lane = str(effective_lane or "").strip()
    if lane:
        bits.append(f"model_lane={lane}")
    return "child capabilities — " + " · ".join(bits)


def decide_tool_access(
    *,
    profile: ToolProfile,
    root: ResourceRoot,
    operation: Operation,
) -> ToolAccessDecision:
    allowed = operation in _POLICY.get(profile, {}).get(root, set())
    if allowed:
        return ToolAccessDecision(True, guard=f"{profile}:{root}:{operation}")
    allowed_roots = ", ".join(sorted(r for r, ops in _POLICY.get(profile, {}).items() if operation in ops)) or "(none)"
    return ToolAccessDecision(
        False,
        reason=f"profile={profile} cannot {operation} root={root}. Roots your profile can {operation}: {allowed_roots}.",
        guard=f"{profile}:{root}:{operation}",
    )


def subagent_profile_satisfies(profile: ToolProfile, needs: Iterable[str]) -> tuple[bool, list[str]]:
    """Return whether a tool profile can satisfy each declared schedule-time need.

    The caller supplies a closed-enum list from the schedule_subagent schema. This
    function does no prose inference: it maps each declared need to the existing
    Tool API operation matrix and checks whether the profile can perform that
    operation on at least one root.
    """

    ops_by_root = _POLICY.get(profile, {})
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
) -> list[tuple[ResourceRoot, pathlib.Path, str, str]]:
    """Return side-effect-free ``(root, base, source, skill)`` candidates."""

    profile = active_tool_profile(ctx)
    active = resource_root_path(ctx, "active_workspace")
    candidates: list[tuple[ResourceRoot, pathlib.Path, str, str]] = []
    room = project_room_lens_dir(ctx)
    if room is not None:
        candidates.append(("active_workspace", room, "active_workspace", ""))
    candidates += [
        ("active_workspace", active, "active_workspace", ""),
        ("system_repo", resource_root_path(ctx, "system_repo"), "system_repo", ""),
    ]

    def _add_task_roots(drive: pathlib.Path) -> None:
        task_id = task_id_for_artifacts(ctx)
        candidates.extend([
            ("task_drive", drive / "task_drives" / task_id, "task_drive", ""),
            ("artifact_store", task_artifact_dir_path(drive, task_id, create=False), "artifact_store", ""),
        ])

    if hasattr(ctx, "drive_root"):
        _add_task_roots(pathlib.Path(ctx.drive_root).resolve(strict=False))
        meta = getattr(ctx, "task_metadata", {})
        meta = meta if isinstance(meta, dict) else {}
        for key in ("drive_root", "child_drive_root", "headless_child_drive_root"):
            if not meta.get(key):
                continue
            _add_task_roots(pathlib.Path(meta[key]).resolve(strict=False))
    if hasattr(ctx, "drive_root"):
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


def project_room_lens_dir(ctx: Any) -> Optional[pathlib.Path]:
    """Return a direct-chat room's verified project cwd, otherwise ``None``.

    Promoted/workspace/subagent tasks carry their own workspace; only a direct
    chat without one may use the injected existing ``_project_room_dir``.
    """
    if not bool(getattr(ctx, "is_direct_chat", False)):
        return None
    if getattr(ctx, "workspace_root", None):
        return None
    meta = getattr(ctx, "task_metadata", None)
    raw = str(meta.get("_project_room_dir") or "").strip() if isinstance(meta, dict) else ""
    if not raw:
        return None
    try:
        candidate = pathlib.Path(raw).resolve(strict=False)
        return candidate if candidate.is_dir() else None
    except OSError:
        return None


def filesystem_affordance_map(ctx: Any, *, runtime_mode: str = "") -> dict[str, Any]:
    """A compact, side-effect-free projection of filesystem/tool access affordances.

    This is context for the LLM, not a new policy layer. Every fact is derived
    from the Tool API v2 matrix and git-shell policy constants so the model can
    plan inside the same envelope that the dispatcher later enforces.
    """

    profile = active_tool_profile(ctx)
    policy = _POLICY.get(profile, {})
    write_like = {"write", "edit", "shell", "vcs", "service"}
    writable_roots = sorted(root for root, ops in policy.items() if ops & write_like)
    readonly_roots = sorted(
        root for root, ops in policy.items()
        if ops and not (ops & write_like) and ops <= (_READ_OPS | {"review", "delegate"})
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
    _room = project_room_lens_dir(ctx)
    if _room is not None:
        # In this chat the project room is the active/default filesystem focus.
        result["project_room_dir"] = str(_room)
        result["default_shell_cwd"] = f"project room ({_room})"
    return result


def profile_readable_root_paths(ctx: Any) -> list[tuple[str, pathlib.Path]]:
    """Project readable ``(label, path)`` pairs from the policy SSOT.

    ``skill_payload`` needs selectors and is omitted; individual resolution is
    fail-soft so one unavailable root cannot hide the rest.
    """
    out: list[tuple[str, pathlib.Path]] = []
    try:
        policy = _POLICY.get(active_tool_profile(ctx), {})
    except Exception:
        return out
    for root, ops in sorted(policy.items()):
        if "read" not in ops or root == "skill_payload":
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


def normalize_root(root: str | None, *, default: ResourceRoot = "active_workspace") -> ResourceRoot:
    candidate = str(root or default).strip() or default
    if candidate not in _ALL_ROOTS:
        raise ValueError(f"unknown root {candidate!r}; expected one of {sorted(_ALL_ROOTS)}")
    return candidate  # type: ignore[return-value]


def path_is_relative_to(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        pathlib.Path(path).resolve(strict=False).relative_to(pathlib.Path(root).resolve(strict=False))
        return True
    except (OSError, ValueError):
        return False


def normalize_root_relative(root: pathlib.Path, path: str) -> str:
    """Map a model-supplied path to a root-relative string when it redundantly
    encodes the root, so structural/read tools accept the paths an agent
    naturally writes: an absolute path inside the active root (e.g. ``/app/foo``
    under a workspace rooted at ``/app``) and a single redundant root-basename
    prefix (``app/foo``). Returns a RELATIVE string only — it never widens
    access: callers still apply ``safe_relpath`` + a ``relative_to`` confinement
    check, so a genuine escape is still rejected downstream.

    - absolute & inside root  -> stripped to relative
    - absolute & outside root -> returned unchanged (caller's check rejects it)
    - redundant root-basename prefix, existence-guarded -> stripped
    - otherwise -> unchanged
    """

    text = str(path or "").strip().replace("\\", "/")
    if not text or text in (".", "./"):
        return text
    try:
        root_resolved = pathlib.Path(root).resolve(strict=False)
    except (OSError, ValueError):
        return text
    # (A) absolute path that already points inside the root.
    if is_absolute_path_text(text):
        try:
            return pathlib.Path(text).resolve(strict=False).relative_to(root_resolved).as_posix()
        except (OSError, ValueError):
            return text  # outside root -> let the caller's confinement reject it
    # (B) redundant root-basename prefix ('app' or 'app/x' when root basename is
    # 'app'). Strip it UNLESS the root contains a real same-named subdir (then
    # 'app/x' is ambiguously a genuine nested path and is kept). Gating on the
    # absence of that subdir — not on the target existing — lets NEW write/create
    # targets ('app/new.py' -> 'new.py') normalize too, while a real 'app/'
    # subdir is never mis-stripped. Only ever shortens toward root (no escape).
    base = root_resolved.name
    if base and (text == base or text.startswith(base + "/")):
        try:
            if not (root_resolved / base).is_dir():
                return text[len(base):].lstrip("/") or "."
        except (ValueError, OSError):
            # `..`/traversal or stat error: leave unchanged so the caller's
            # confinement produces the canonical (not a generic) error.
            return text
    return text


def _path_is_relative_to_casefold(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path_parts = pathlib.Path(path).resolve(strict=False).parts
        root_parts = pathlib.Path(root).resolve(strict=False).parts
    except (OSError, ValueError):
        return False
    if len(path_parts) < len(root_parts):
        return False
    return tuple(part.casefold() for part in path_parts[: len(root_parts)]) == tuple(
        part.casefold() for part in root_parts
    )


def paths_overlap_casefold(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Return True when two paths overlap under case-insensitive path semantics."""

    return _path_is_relative_to_casefold(left, right) or _path_is_relative_to_casefold(right, left)


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


def workspace_mode_block_reason(ctx: Any) -> str:
    mode = str(getattr(ctx, "workspace_mode", "") or "").strip()
    workspace_root = getattr(ctx, "workspace_root", None)
    if not mode or workspace_root is None:
        return ""
    try:
        workspace = pathlib.Path(workspace_root).resolve(strict=False)
    except (OSError, TypeError, ValueError):
        return "workspace_root is invalid"
    protected_values = (
        ("Ouroboros system repo", getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir", None)),
        ("Ouroboros repo", getattr(ctx, "repo_dir", None)),
        ("Ouroboros data drive", getattr(ctx, "drive_root", None)),
        (
            "Ouroboros parent data drive",
            (getattr(ctx, "task_metadata", {}) or {}).get("budget_drive_root")
            if isinstance(getattr(ctx, "task_metadata", {}), dict)
            else "",
        ),
    )
    for label, value in protected_values:
        if not value:
            continue
        try:
            protected = pathlib.Path(value).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            continue
        if (
            path_is_relative_to(workspace, protected)
            or path_is_relative_to(protected, workspace)
            or paths_overlap_casefold(workspace, protected)
        ):
            return f"workspace_root overlaps the {label}"
    return ""


def _subagent_projects_read_hint(
    ctx: Any,
    resolved: pathlib.Path,
    hard_protected_roots: list[pathlib.Path],
) -> str:
    """A targeted refusal for a user_files path that actually lives inside the
    subagent-projects area: name root=subagent_projects with the exact relative
    path instead of steering the model at roots that cannot reach the target.
    Empty when the target is not there, the active profile cannot read that root,
    or the projects root is misconfigured to overlap a HARD drive (never steer a
    read at the control plane)."""
    try:
        profile_policy = _POLICY.get(active_tool_profile(ctx), {})
        if "read" not in profile_policy.get("subagent_projects", set()):
            return ""
        projects_root = resource_root_path(ctx, "subagent_projects")
        if any(
            path_is_relative_to(projects_root, hard) or _path_is_relative_to_casefold(projects_root, hard)
            for hard in hard_protected_roots
        ):
            return ""
        if not (
            path_is_relative_to(resolved, projects_root)
            or _path_is_relative_to_casefold(resolved, projects_root)
        ):
            return ""
        try:
            rel = str(resolved.relative_to(projects_root))
        except ValueError:
            rel = os.path.relpath(str(resolved), str(projects_root))
        rel = rel if rel not in ("", ".") else "."
        return (
            "this path is inside root=subagent_projects (the durable child-project "
            f"area); read it via root=subagent_projects, path={rel!r} "
            "(read/list/search only — no write/shell there by design: children "
            "write via write_surface=external_workspace, and the host "
            "checkpoint-commits dirty coop trees at root finalization)"
        )
    except Exception:
        return ""


def user_files_path_block_reason(
    ctx: Any,
    candidate: pathlib.Path,
    *,
    allow_protected_descendants: bool = False,
) -> str:
    """Return a block reason when candidate is not an external user file."""

    resolved = pathlib.Path(candidate).expanduser().resolve(strict=False)
    home = _user_files_root()
    outside_home = not path_is_relative_to(resolved, home) and not _path_is_relative_to_casefold(resolved, home)
    # External-workspace tasks may reach host scratch outside home (/tmp, /build,
    # sibling checkouts). The runtime-overlap and credential guards BELOW still
    # run on the full path, so the Ouroboros repo/data drive and secret-like
    # files stay protected even when home confinement is lifted.
    if outside_home and not is_external_workspace(ctx):
        return f"path is outside user home {home}"

    # The Ouroboros runtime/control surface is the system repo PLUS every data
    # drive the task touches: the parent drive (ctx.drive_root) and any child /
    # budget drive carried in task_metadata. External-workspace mode lifts home
    # confinement, so these must be enumerated explicitly here — otherwise a
    # child-drive control path (e.g. <child_drive>/memory) would slip through.
    protected_values: list[Any] = [
        getattr(ctx, "drive_root", None),
        getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir", None),
    ]
    meta = getattr(ctx, "task_metadata", {})
    if isinstance(meta, dict):
        for key in ("drive_root", "child_drive_root", "headless_child_drive_root", "budget_drive_root"):
            if meta.get(key):
                protected_values.append(meta.get(key))
    protected_roots: list[pathlib.Path] = []
    hard_protected_roots: list[pathlib.Path] = []  # the data/repo/budget drives THEMSELVES
    for value in protected_values:
        try:
            root = pathlib.Path(value).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            continue
        protected_roots.append(root)
        hard_protected_roots.append(root)
        parent = root.parent.resolve(strict=False)
        if root.name in {"repo", "data"} and path_is_relative_to(parent, home):
            # The workspace PARENT is a SOFT boundary (keeps user_files out of ~/Ouroboros at large);
            # it is deliberately NOT a hard root, so the Deliverables sibling under it stays allowed.
            protected_roots.append(parent)
    # The configured Deliverables container is an INTENDED user-output root, allowed past the
    # workspace-overlap guard — but ONLY when it is a genuine sibling: a misconfigured
    # OUROBOROS_DELIVERABLES_ROOT that overlaps or contains a HARD data/repo/budget drive must NOT
    # open a bypass. The outside-home, credential, and hidden-name checks still apply regardless.
    in_deliverables = False
    try:
        _deliverables = _deliverables_root()
        _deliverables_safe = not any(
            path_is_relative_to(_deliverables, pr) or _path_is_relative_to_casefold(_deliverables, pr)
            or path_is_relative_to(pr, _deliverables) or _path_is_relative_to_casefold(pr, _deliverables)
            for pr in hard_protected_roots
        )
        if _deliverables_safe and (
            path_is_relative_to(resolved, _deliverables) or _path_is_relative_to_casefold(resolved, _deliverables)
        ):
            in_deliverables = True
    except Exception:
        in_deliverables = False
    if not in_deliverables:
        for protected in protected_roots:
            overlaps_protected = path_is_relative_to(resolved, protected) or _path_is_relative_to_casefold(resolved, protected)
            contains_protected = path_is_relative_to(protected, resolved) or _path_is_relative_to_casefold(protected, resolved)
            if overlaps_protected or (
                not allow_protected_descendants and contains_protected
            ):
                # Name the root that ACTUALLY contains the target (the v6.54.3
                # shell_cwd_block_message lesson applied to this surface): the
                # subagent-projects area lives under the SOFT ~/Ouroboros parent,
                # so every coop-tree read used to get a message naming four roots
                # that cannot reach it while omitting the one that can. MESSAGE
                # ONLY — subagent_projects stays a read-only root (no user_files
                # write carve-out), and a target inside a HARD drive never takes
                # this branch.
                projects_hint = _subagent_projects_read_hint(ctx, resolved, hard_protected_roots)
                if projects_hint:
                    return projects_hint
                return (
                    "path overlaps the Ouroboros repo/runtime workspace; use "
                    "root=active_workspace, root=task_drive, root=artifact_store, "
                    "or root=skill_payload instead"
                )

    try:
        parts = resolved.relative_to(home).parts
    except ValueError:
        parts = resolved.parts
    for part in parts:
        if not part:
            continue
        part_lower = part.lower()
        # v6.52.0 (P1): DEFAULT-DENY hidden (dot) components. Known secret/credential/VCS dirs
        # are always blocked; ANY OTHER dotted component is blocked too UNLESS it is in the small
        # benign allowlist (.github/.vscode/.idea/...). Benign project dotdirs become readable
        # (the owner's goal) while the in-home dotfile space stays safe-by-default — an enumerated
        # blocklist would leak credential stores like ~/.terraform.d, ~/.cargo, ~/.pip, etc.
        if part_lower in _USER_FILES_SECRET_COMPONENTS:
            return "path is hidden or credential-like (secret/credential directory)"
        if part.startswith(".") and part_lower not in _USER_FILES_ALLOWED_DOTNAMES:
            return "path is hidden or credential-like (non-allowlisted hidden component)"
    name = resolved.name
    name_lower = name.lower()
    if (
        name_lower in _USER_FILES_SECRET_NAMES
        or _USER_FILES_SECRET_RE.search(name)
        or name_lower.endswith((".key", ".pem", ".p12", ".pfx"))
    ):
        return "path name is credential-like"

    return ""


class UserFilesPathBlockedError(ValueError):
    """Typed user_files confinement refusal (a POLICY denial, not an I/O failure).

    Subclasses ``ValueError`` so every existing generic handler keeps working;
    the read-surface wrappers (read_file/list_files/search_code) render it with
    the typed ``⚠️ USER_FILES_PATH_BLOCKED`` prefix so the outcome axis can
    partition it into ``execution.policy_denials`` (v6.57.0) instead of the
    generic ``error`` status that falsely degraded a shipped task to
    ``tool_failure`` (the submarine wave-3 incident)."""


def resolve_user_file_path(
    ctx: Any,
    path: str,
    *,
    allow_protected_descendants: bool = False,
    allow_outside_home: bool = False,
) -> pathlib.Path:
    """Resolve a user_files path under the user's home and outside Ouroboros control-plane roots.

    Absolute paths OUTSIDE the user_files home (and the Deliverables container) are
    rejected EARLY with an actionable error instead of resolving to a foreign root
    and failing later with an opaque ``relative_to`` crash (v6.54.3 — the TB2.1
    ``'/app' is not in the subpath of '/root'`` class). ``allow_outside_home=True``
    (the ``query_code`` external-target caller) skips only this EARLY actionable
    check; ``user_files_path_block_reason`` below remains the outside-home
    AUTHORITY, and it permits outside-home only for external-workspace contexts —
    the mode the documented query_code contract (benchmark ``/app``) runs in.
    Neither flag expands authority: a non-external context could not reach
    outside-home before this check existed either."""

    raw_text = str(path or ".").strip() or "."
    try:
        raw = pathlib.Path(raw_text).expanduser()
    except Exception:
        # expanduser() raises RuntimeError for an unknown '~user'; leave it unexpanded —
        # the '~' branch below maps it into the jail home (raw is only used elsewhere for
        # absolute paths, where expanduser is a no-op anyway).
        raw = pathlib.Path(raw_text)
    home = _user_files_root()
    # is_absolute_path_text gives consistent cross-platform absolute detection
    # (drive-less "/x" roots and "C:\\x"/"\\\\unc" are all absolute) so Windows
    # does not silently treat a rooted path as home-relative.
    if is_absolute_path_text(raw_text):
        candidate = raw.resolve(strict=False)
        # External-workspace tasks legitimately reach host scratch outside home
        # (/tmp, /build, sibling checkouts) — for them the generic
        # user_files_path_block_reason below stays the authority, mirroring its
        # own is_external_workspace carve-out.
        if not allow_outside_home and not is_external_workspace(ctx):
            home_resolved = home.resolve(strict=False)
            # Case-insensitive-platform parity with the user_files_path_block_reason
            # authority: a differently-cased safe home path must not be rejected
            # early where the casefold-aware guard would accept it (review round 7).
            inside_home = path_is_relative_to(candidate, home_resolved) or _path_is_relative_to_casefold(
                candidate, home_resolved
            )
            inside_deliverables = False
            if not inside_home:
                try:
                    deliverables_resolved = _deliverables_root().resolve(strict=False)
                    inside_deliverables = path_is_relative_to(
                        candidate, deliverables_resolved
                    ) or _path_is_relative_to_casefold(candidate, deliverables_resolved)
                except (OSError, ValueError):
                    inside_deliverables = False
            if not inside_home and not inside_deliverables:
                raise UserFilesPathBlockedError(
                    "user_files path blocked: absolute path "
                    f"{raw_text!r} is outside the user_files home ({home_resolved}). "
                    "Use root='active_workspace' for workspace paths, or a "
                    "home-relative path (e.g. 'Desktop/file.txt') for user files."
                )
    elif raw_text.startswith("~"):
        # '~' / '~user' must expand to the CONFIGURED user_files home (the jail), NOT the
        # real OS home — otherwise OUROBOROS_USER_FILES_ROOT isolation is bypassed by a
        # '~/...' path. The jail has a single home, so '~user/sub' maps to '<home>/sub'.
        _after = raw_text[1:]
        if _after[:1] in ("/", "\\"):
            _rel = _after[1:]
        elif "/" in _after or "\\" in _after:
            _rel = _after.replace("\\", "/").split("/", 1)[1]
        else:
            _rel = ""  # bare '~' or '~user' -> the home directory itself
        candidate = (home / safe_relpath(_rel)).resolve(strict=False) if _rel else home.resolve(strict=False)
    else:
        # safe_relpath has already normalized any Windows backslash to a POSIX '/', so the
        # directory test below is separator-correct on every platform.
        rel = safe_relpath(raw_text)
        home_candidate = home / rel
        if "/" in rel.strip("/") or home_candidate.exists():
            # An explicit placement (a path WITH a directory — Desktop/..., Downloads/..., a subdir)
            # OR a bare name that ALREADY EXISTS under home (an existing file or directory such as
            # `Desktop`) is honored under the owner home exactly as given. This keeps read/list/search
            # of existing user files and directory names home-relative — only a genuinely NEW unnamed
            # output is containerized.
            candidate = home_candidate.resolve(strict=False)
        else:
            # A bare name with no directory that does NOT already exist under home is an unnamed NEW
            # deliverable: route it into the visible Deliverables container instead of cluttering the
            # home root (a later read of the same bare name resolves there too, staying consistent).
            candidate = (_deliverables_root() / rel).resolve(strict=False)
    reason = user_files_path_block_reason(
        ctx,
        candidate,
        allow_protected_descendants=allow_protected_descendants,
    )
    if reason:
        raise UserFilesPathBlockedError(f"user_files path blocked: {reason}")
    return candidate


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
    if include_skill and (not str(bucket or "").strip() or not str(skill_name or "").strip()):
        raise ValueError("cwd=skill_payload[/subdir] requires bucket and skill_name")
    candidate_records = _process_root_candidates(
        ctx, operation, bucket=bucket, skill_name=skill_name, include_skill=include_skill,
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


def canonical_data_root(ctx: Any) -> pathlib.Path:
    """Return canonical skill data: task budget → context budget → task drive."""
    metadata = getattr(ctx, "task_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    for candidate in (metadata.get("budget_drive_root"), getattr(ctx, "budget_drive_root", "")):
        text = str(candidate or "").strip()
        if text:
            return pathlib.Path(text).resolve(strict=False)
    return pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False)


def normalize_runtime_data_path(data_root: pathlib.Path, path: str) -> str:
    """Normalize historical runtime-data prefixes before physical binding."""
    norm = str(path or ".").strip().replace("\\", "/")
    norm = norm[2:] if norm.startswith("./") else norm
    stripped = norm.lstrip("/")
    root_text = str(pathlib.Path(data_root)).rstrip("/").lstrip("/")
    if root_text and stripped.startswith(root_text):
        return stripped[len(root_text):].lstrip("/") or "."
    if stripped.startswith(".tmp-data-"):
        _prefix, separator, after = stripped.partition("/")
        if separator:
            return after[len("data/"):] if after.startswith("data/") else after
    return norm or "."


def load_bound_skill(binding: ResolvedResourceBinding) -> Any:
    """Load the frozen payload target while preserving lifecycle provenance."""
    from ouroboros.skill_loader import _classify_skill_source, load_skill
    loaded = load_skill(binding.base_path, binding.state_drive_root)
    if loaded is not None:
        loaded.source = _classify_skill_source(
            binding.base_path,
            location=binding.source,
            drive_root=binding.state_drive_root,
        )
    return loaded


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


def _skill_payload_base(
    ctx: Any,
    *,
    profile: ToolProfile,
    operation: Operation,
    location: str,
    skill_name: str,
    allow_missing: bool = False,
) -> tuple[pathlib.Path, str, str]:
    """Select one physical skill package without reading lifecycle state."""
    from ouroboros.skill_loader import (
        _sanitize_skill_name,
        _select_skill_location,
        _skill_location_inventory,
    )
    requested_location = str(location or "").strip().lower()
    allowed_locations = {"external", "clawhub", "ouroboroshub", "native", "user_repo"}
    canonical_name = _sanitize_skill_name(skill_name)
    if not str(skill_name or "").strip() or canonical_name == "_unnamed":
        raise ValueError("root=skill_payload requires a non-empty skill_name")
    state_root = canonical_data_root(ctx)
    candidates = _skill_location_inventory(state_root)
    if not requested_location and operation == "review":
        identity = tuple(item for item in candidates if item.name == canonical_name)
        if not identity:
            raise ValueError(f"skill {canonical_name!r} was not found")
        requested_location = identity[0].location
    elif requested_location not in allowed_locations:
        raise ValueError(
            "root=skill_payload requires bucket/location in "
            "external|clawhub|ouroboroshub|native|user_repo"
        )
    if requested_location in {"native", "user_repo"} and profile not in _TOP_LEVEL_PRINCIPAL_PROFILES:
        raise ValueError(
            f"profile={profile} cannot select skill location={requested_location}"
        )
    if requested_location == "native" and operation in {"write", "edit", "shell"}:
        raise ValueError(
            "installed native skills are read/review only; edit their seed via root=system_repo"
        )

    selected = _select_skill_location(
        candidates,
        name=canonical_name,
        location=requested_location,
        require_unique_identity=operation not in {"read", "list", "search"},
    )
    if selected is not None:
        return selected.skill_dir.resolve(strict=False), selected.location, selected.name
    if (
        operation == "write"
        and allow_missing
        and requested_location in {"external", "clawhub", "ouroboroshub"}
    ):
        return (
            (state_root / "skills" / requested_location / canonical_name).resolve(strict=False),
            requested_location,
            canonical_name,
        )
    raise ValueError(
        f"skill {canonical_name!r} was not found in location {requested_location!r}"
    )


def resource_root_path(
    ctx: Any,
    root: ResourceRoot,
    *,
    bucket: str = "",
    skill_name: str = "",
) -> pathlib.Path:
    if root == "active_workspace":
        active = getattr(ctx, "active_repo_dir", None)
        candidate = None
        if callable(active):
            try:
                candidate = active()
            except Exception:
                candidate = None
        if candidate is None or candidate.__class__.__module__.startswith("unittest.mock"):
            candidate = getattr(ctx, "repo_dir")
        return pathlib.Path(candidate).resolve(strict=False)
    if root == "system_repo":
        return pathlib.Path(getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir")).resolve(strict=False)
    if root == "runtime_data":
        return pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False)
    if root == "task_drive":
        return (pathlib.Path(getattr(ctx, "drive_root")).resolve(strict=False) / "task_drives" / task_id_for_artifacts(ctx)).resolve(strict=False)
    if root == "artifact_store":
        return task_artifact_dir_path(pathlib.Path(getattr(ctx, "drive_root")), task_id_for_artifacts(ctx), create=False).resolve(strict=False)
    if root == "user_files":
        return _user_files_root()
    if root == "subagent_projects":
        from ouroboros.config import get_subagent_projects_root

        return pathlib.Path(get_subagent_projects_root()).expanduser().resolve(strict=False)
    if root == "deliverables":
        return _deliverables_root()
    if root == "skill_payload":
        b = str(bucket or "").strip()
        s = str(skill_name or "").strip()
        if not b or not s:
            raise ValueError("root=skill_payload requires bucket and skill_name")
        base, _source, _name = _skill_payload_base(
            ctx,
            profile=active_tool_profile(ctx),
            operation="read",
            location=b,
            skill_name=s,
        )
        return base
    raise ValueError(f"unknown root {root!r}")


def binding_targets_system_repo(
    ctx: Any, binding: ResolvedResourceBinding | None,
) -> bool:
    """Whether a selected logical root physically lands on Ouroboros source."""

    if binding is None:
        return False
    try:
        return pathlib.Path(binding.base_path).resolve(strict=False) == resource_root_path(
            ctx, "system_repo",
        )
    except (OSError, TypeError, ValueError):
        return False


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
        )
    resolved_base = pathlib.Path(base_path).resolve(strict=False)
    path_text = str(path or ".")
    if root == "runtime_data":
        path_text = normalize_runtime_data_path(resolved_base, path_text)
    elif root in {"active_workspace", "system_repo"}:
        path_text = normalize_root_relative(resolved_base, path_text)
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
    room = (
        project_room_lens_dir(ctx)
        if normalized == "active_workspace" and operation in {"read", "list", "search", "shell"}
        else None
    )
    if normalized == "skill_payload":
        selected_bucket = str(bucket or "").strip()
        selected_skill = str(skill_name or "").strip()
        constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
        if constraint and constraint.mode == "skill_repair":
            from ouroboros.contracts.skill_payload_policy import constraint_bucket_skill

            expected_bucket, expected_skill = constraint_bucket_skill(constraint)
            if (
                not selected_bucket
                and (not selected_skill or selected_skill == expected_skill)
            ):
                selected_bucket = expected_bucket
                selected_skill = selected_skill or expected_skill
            elif (selected_bucket, selected_skill) != (expected_bucket, expected_skill):
                raise ValueError(
                    "SKILL_REDIRECT_BLOCKED: active skill_repair payload is "
                    f"{expected_bucket}/{expected_skill}; cannot select "
                    f"{selected_bucket}/{selected_skill}"
                )
        from ouroboros.contracts.skill_payload_policy import _is_skill_create_signal

        base, source, selected_name = _skill_payload_base(
            ctx,
            profile=profile,
            operation=operation,
            location=selected_bucket,
            skill_name=selected_skill,
            allow_missing=(
                operation == "write"
                and selected_bucket.lower() == "external"
                and _is_skill_create_signal(path)
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
    return ResolvedResourceBinding(
        profile=profile,
        root=normalized,
        operation=operation,
        base_path=pathlib.Path(base).resolve(strict=False),
        target_path=target,
        source=source,
        skill_name=selected_name,
        state_drive_root=canonical_data_root(ctx),
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
